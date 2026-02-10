#!/usr/bin/env python3
"""
Mooer Camera NG - 主入口
重构后的模块化智能视角控制系统
"""

import faulthandler
import fcntl
import os
import subprocess
import sys
import time
from collections import deque

# 启用 faulthandler，在崩溃时打印 Python 堆栈
faulthandler.enable()

from camera_ng import (
    DEFAULT_NUM_STEPS, DEFAULT_TOTAL_ANGLE,
    ROTATION_SPEED, TRACK_CHECK_INTERVAL, DETECTION_INTERVAL,
    PTZ_SPEED, PTZ_FAST_SPEED,
    TRACKER_MAX_AGE, TRACKER_MIN_HITS,
    CAPTURE_WIDTH, CAPTURE_HEIGHT, LOCK_FILE,
    CAMERA_RTSP, CAMERA_RTSP_SUB, STREAM_LOW_LATENCY, DEVICE_SERIAL, ACCESS_TOKEN,
    HAND_SIDE_MODE,
    CameraController, VisionAnalyzer, HandRaiseDetector, XiaoxiaoTTS, AsyncVoiceQueue,
    PersonTracker, TrackingMemory
)
from camera_ng.handlers import (
    HandGestureHandler, HandGesture,
    RecordingManager, SmartShotWorker,
)

TELEGRAM_TARGET = "1115213761"


def run_openclaw_send(cmd: list[str], retries: int = 2, timeout_sec: int = 20) -> bool:
    """发送消息（含超时与重试），避免后台任务长期占用"""
    for attempt in range(1, retries + 1):
        try:
            subprocess.run(cmd, check=True, timeout=timeout_sec)
            return True
        except Exception as e:
            if attempt == retries:
                print(f"❌ OpenClaw 发送失败: {e}")
                return False
            print(f"⚠️ OpenClaw 发送失败，重试 {attempt}/{retries - 1}: {e}")
            time.sleep(0.6)
    return False


def validate_config():
    """验证必要配置是否存在"""
    errors = []
    
    if CAMERA_RTSP is None:
        errors.append("  - camera.rtsp_url: RTSP 流地址未配置")
    if DEVICE_SERIAL is None:
        errors.append("  - camera.device_serial: 设备序列号未配置")
    if ACCESS_TOKEN is None:
        errors.append("  - camera.access_token: 萤石云 AccessToken 未配置")
    
    if errors:
        print("\n" + "=" * 60)
        print("❌ 配置错误：缺少必要的敏感信息")
        print("=" * 60)
        print("\n请在以下位置之一创建 camera-config.yaml 文件：")
        print("  1. ./memory/camera-config.yaml")
        print("  2. ~/.openclaw/workspace/memory/camera-config.yaml")
        print("  3. ~/clawd/memory/camera-config.yaml")
        print("  4. ~/.config/mooer-camera/config.yaml")
        print("\n缺失的配置项：")
        for err in errors:
            print(err)
        print("\ncamera-config.yaml 示例格式：")
        print("""
camera:
  rtsp_url: "rtsp://admin:PASSWORD@IP:554/h264/ch1/main/av_stream"
  device_serial: "YOUR_DEVICE_SERIAL"
  access_token: "YOUR_ACCESS_TOKEN"
  capture:
    seek_time: "00:00:00.5"
    resolution:
      width: 640
      height: 360
    quality: 2
  ptz:
    rotation_speed: 28
""")
        print("=" * 60)
        sys.exit(1)


class SmartCamera:
    """智能摄像头系统（human 专用）"""

    def __init__(self):
        self.camera = CameraController()
        self.vision = VisionAnalyzer()

    def human(self, num_steps: int = DEFAULT_NUM_STEPS,
              total_angle: float = DEFAULT_TOTAL_ANGLE,
              use_gpu: bool = False, smart: bool = True,
              fast: bool = False, keep_stream: bool = True,
              center_and_wait: bool = False) -> bool:
        """执行 human 扫描，找到人返回True"""
        if not self.camera.stream_active:
            print("🎥 启动视频流...")
            if not self.camera.start_stream(use_gpu=use_gpu):
                print("❌ 无法启动视频流")
                return False
            time.sleep(0.5)
        else:
            print("🔄 复用已有视频流...")

        try:
            self.camera.center_and_wait_mode = center_and_wait
            
            if smart:
                result = self.camera.human_steps_smart(self.vision)
            elif fast:
                result = self.camera.human_steps_fast(self.vision)
            else:
                result = self.camera.human_steps(
                    self.vision, num_steps=num_steps, total_angle=total_angle
                )
        finally:
            if not keep_stream:
                self.camera.stop_stream()

        return result
    
    def stop(self):
        """完全停止并清理资源"""
        self.camera.stop_stream()
        print("✅ 智能摄像头系统已停止")
    
    def human_smart_only(self) -> bool:
        """智能扫描（复用已有流）"""
        return self.human(smart=True)


class SingleInstanceLock:
    """单实例文件锁"""

    def __init__(self, lock_file: str = LOCK_FILE):
        self.lock_file = lock_file
        self.fd = None

    def acquire(self) -> bool:
        """获取文件锁"""
        try:
            self.fd = open(self.lock_file, "w")
            fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.fd.write(str(os.getpid()))
            self.fd.flush()
            return True
        except (IOError, OSError):
            if self.fd:
                self.fd.close()
                self.fd = None
            return False

    def release(self):
        """释放文件锁"""
        if self.fd:
            try:
                fcntl.flock(self.fd.fileno(), fcntl.LOCK_UN)
                self.fd.close()
                if os.path.exists(self.lock_file):
                    os.remove(self.lock_file)
            except (IOError, OSError):
                pass
            finally:
                self.fd = None

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError("另一个实例正在运行")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def capture_and_send_current_view(
    camera: CameraController,
    message: str,
    send_to_tg: bool = False,
) -> bool:
    """基于当前画面抓拍；可选发送 Telegram"""
    output_dir = os.path.expanduser("~/Desktop/capture/pictures")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    ms = int((time.time() % 1) * 1000)
    img_path = os.path.join(output_dir, f"{timestamp}_{ms:03d}.jpg")

    camera.capture(output_path=img_path, full_quality=True)
    print(f"📸 已保存高质量抓拍: {img_path}")

    if not send_to_tg:
        print("📭 Telegram 发送已关闭（使用 --tg 开启）")
        return True

    try:
        send_cmd = [
            "openclaw", "message", "send",
            "--channel", "telegram",
            "--target", TELEGRAM_TARGET,
            "--media", img_path,
            "--message", message
        ]
        print("📤 正在通过 OpenClaw 发送照片...")
        ok = run_openclaw_send(send_cmd)
        if ok:
            print("✅ 照片发送成功！")
        return ok
    except Exception as e:
        print(f"❌ 照片发送失败: {e}")
        return False


def send_greeting_voice(tts: XiaoxiaoTTS, message: str, send_to_tg: bool = False) -> bool:
    """发送中文问候语音（可选 Telegram）"""
    try:
        if not send_to_tg:
            print("📭 Telegram 语音发送已关闭（使用 --tg 开启）")
            return True

        voice_path = tts.synthesize(message)
        send_cmd = [
            "openclaw", "message", "send",
            "--channel", "telegram",
            "--target", TELEGRAM_TARGET,
            "--media", voice_path,
            "--message", "右手手势语音问候",
        ]
        print("🔊 正在发送问候语音...")
        ok = run_openclaw_send(send_cmd)
        if ok:
            print("✅ 语音发送成功！")
        return ok
    except Exception as e:
        print(f"❌ 语音发送失败: {e}")
        return False


# 已迁移到 handlers.py


# 已迁移到 handlers.py


def handle_gesture_event(
    event,
    recording_mgr: RecordingManager | None,
    smart_shot_worker: SmartShotWorker,
    current_time: float,
) -> bool:
    """处理手势事件，返回是否成功处理"""
    from camera_ng.handlers import HandGesture

    if event.gesture == HandGesture.LEFT_HAND:
        # 左手控制录像开关
        print(f"\n   ✋ 检测到左手抬起，切换录像状态... ({event.reason})")
        if recording_mgr is not None:
            recording_mgr.toggle(current_time)
            return True
        return False

    elif event.gesture == HandGesture.RIGHT_HAND:
        # 右手触发 Smart-Shot
        print(f"\n   🙋 检测到右手抬起，触发 Smart-Shot... ({event.reason})")
        return smart_shot_worker.submit("右手", event.reason)

    return False


def check_single_instance():
    """检查是否单实例运行"""
    lock = SingleInstanceLock()
    if not lock.acquire():
        print("❌ 错误：另一个 mooer_camera 实例正在运行")
        print(f"   锁文件: {LOCK_FILE}")
        print("   请先停止其他实例再运行")
        sys.exit(1)
    return lock


def track_human_realtime(num_steps: int = DEFAULT_NUM_STEPS,
                         total_angle: float = 360,
                         detection_interval: int = DETECTION_INTERVAL,
                         use_gpu: bool = False,
                         tracking_mode: str = "hybrid",
                         smart_shot: bool = False,
                         quick_mode: bool = False,
                         send_to_tg: bool = False,
                         enable_miss: bool = False,
                         voice_drop_oldest: bool = False,
                         mute_voice: bool = False,
                         low_latency_stream: bool = STREAM_LOW_LATENCY,
                         ptz_speed: int = PTZ_SPEED,
                         ptz_speed_fast: int = PTZ_FAST_SPEED) -> None:
    """实时目标跟踪模式 - 使用重构后的处理器"""
    cam = SmartCamera()
    cam.camera.ptz_speed_default = int(max(1, min(7, ptz_speed)))
    effective_detection_interval = 1 if quick_mode else detection_interval
    tracker = PersonTracker(
        yolo_model="yolov8n",
        confidence=0.5,
        detection_interval=effective_detection_interval
    )

    # 状态管理
    cycle_count = 0
    tracking_mode = (tracking_mode or "hybrid").lower()
    software_assist_active = tracking_mode == "software"
    person_found = False
    analyzing = False
    lost_count = 0
    LOST_THRESHOLD = 5

    fps_history = deque(maxlen=30)
    last_time = time.time()
    offset_history_size = 3 if smart_shot else 5
    offset_x_history = deque(maxlen=offset_history_size)
    offset_y_history = deque(maxlen=offset_history_size)
    recenter_candidate_count = 0
    last_recenter_time = 0.0

    # 云台控制参数
    RECENTER_CONFIRM_FRAMES = 3
    RECENTER_COOLDOWN = 1.4 if smart_shot else 1.2
    BASE_RECENTER_X_THRESHOLD = 0.5
    BASE_RECENTER_Y_THRESHOLD_UP = 0.28
    BASE_RECENTER_Y_THRESHOLD_DOWN = 0.32
    POST_FOUND_REVERSE_STEP_SEC = 0.15
    POST_FOUND_REVERSE_MIN_OFFSET = 0.08
    STABLE_X_DEADBAND = 0.12
    STABLE_Y_DEADBAND = 0.16
    STABLE_HOLD_SEC = 0.45
    STABLE_VEL_THRESHOLD = 0.03
    CALIB_DONE_MIN_INTERVAL_SEC = 2.0
    MICRO_STEP_MAX_SEC = 0.09
    MICRO_STEP_MIN_SEC = 0.02
    MICRO_STEP_Y_THRESHOLD_UP = 0.10
    MICRO_STEP_Y_THRESHOLD_DOWN = 0.12
    FAST_TARGET_VEL_X = 0.06
    FAST_TURN_BOOST = 1.25 if smart_shot else 1.6
    FAST_TURN_MAX_SEC = 0.09 if smart_shot else 0.12
    # 垂直云台可用范围较小（约 20°），Smart-Shot 垂直纠偏要更保守
    SMART_SHOT_FOOT_MARGIN_MIN = 0.05
    SMART_SHOT_HEAD_MARGIN_MIN = 0.08
    SMART_SHOT_MARGIN_GAIN = 1.8
    SMART_SHOT_MARGIN_PUSH_MAX = 0.20
    SMART_SHOT_FULL_BODY_HEIGHT_MAX = 0.78
    SMART_SHOT_HEAD_LOW_TRIGGER = 0.58
    SMART_SHOT_HEAD_LOW_PUSH_GAIN = 1.2
    SMART_SHOT_HEAD_LOW_PUSH_MAX = 0.22

    calibrating_active = False
    stable_candidate_since = 0.0
    last_calibration_done_time = 0.0

    status_voice_last_ts: dict[str, float] = {}

    def broadcast_status(text: str, min_interval_sec: float = 0.0) -> None:
        if mute_voice:
            return
        now = time.time()
        last_ts = status_voice_last_ts.get(text, 0.0)
        if min_interval_sec > 0 and (now - last_ts) < min_interval_sec:
            return
        status_voice_last_ts[text] = now
        if voice_queue is not None:
            voice_queue.enqueue(text)

    def apply_micro_recenter(offset_x: float, offset_y: float, speed_x: float = 0.0) -> bool:
        moved = False
        abs_x = abs(offset_x)
        fast_move = speed_x >= FAST_TARGET_VEL_X and abs_x >= 0.10
        cmd_speed = int(max(1, min(7, ptz_speed_fast if fast_move else ptz_speed)))
        if abs_x >= 0.06:
            step_x = min(MICRO_STEP_MAX_SEC, max(MICRO_STEP_MIN_SEC, abs_x * 0.06))
            if fast_move:
                step_x = min(FAST_TURN_MAX_SEC, step_x * FAST_TURN_BOOST)
            direction_x = "left" if offset_x < 0 else "right"
            if cam.camera.ptz_turn(direction_x, step_x, speed=cmd_speed):
                moved = True

        abs_y = abs(offset_y)
        y_threshold = MICRO_STEP_Y_THRESHOLD_UP if offset_y < 0 else MICRO_STEP_Y_THRESHOLD_DOWN
        if abs_y >= y_threshold:
            step_y_scale = 0.06 if offset_y < 0 else 0.04
            step_y = min(0.06, max(0.02, abs_y * step_y_scale))
            direction_y = "up" if offset_y < 0 else "down"
            if cam.camera.ptz_turn(direction_y, step_y, speed=cmd_speed):
                moved = True

        return moved

    # Smart-Shot 组件
    tts = XiaoxiaoTTS() if not mute_voice else None
    voice_queue = None
    if tts is not None:
        voice_queue = AsyncVoiceQueue(tts=tts, max_queue_size=8, drop_oldest=voice_drop_oldest)
        voice_queue.start()
    hand_detector = HandRaiseDetector(
        infer_imgsz=224 if quick_mode else 256,
        hand_side_mode=HAND_SIDE_MODE,
    ) if smart_shot else None
    gesture_handler = (
        HandGestureHandler(
            detector=hand_detector,
            confirm_frames=1 if quick_mode else 2,
            release_frames=2 if quick_mode else 3,
            cooldown_sec=0.6 if quick_mode else 1.0,
            log_interval_sec=0.5 if quick_mode else 1.0,
            detect_interval_sec=0.25 if quick_mode else 0.55,
        )
        if smart_shot and hand_detector else None
    )
    recording_mgr = (
        RecordingManager(
            rtsp_url=CAMERA_RTSP,
            tts=tts,
            voice_enqueue=(voice_queue.enqueue if voice_queue is not None else None),
            toggle_cooldown_sec=1.5,
            auto_start_on_person_found=False,
        )
        if smart_shot else None
    )
    def smart_shot_task_callback(camera, hand_text, hand_reason, tts_instance):
        """Smart-Shot 任务回调"""
        capture_and_send_current_view(
            camera,
            f"Albert，我检测到你抬{hand_text}，已为你抓拍！📸",
            send_to_tg=send_to_tg,
        )

    smart_shot_worker = (
        SmartShotWorker(
            camera=cam.camera,
            tts=tts,
            telegram_target=TELEGRAM_TARGET,
            voice_enqueue=(voice_queue.enqueue if voice_queue is not None else None),
            max_queue_size=3,
            task_callback=smart_shot_task_callback,
        )
        if smart_shot else None
    )

    if smart_shot and smart_shot_worker:
        smart_shot_worker.start()

    frame_sleep = 0.005 if quick_mode else 0.03
    recenter_pause = 0.2 if quick_mode else 0.5

    print("\n" + "=" * 60)
    print("🔍 启动实时目标跟踪模式 (Real-time + SORT)")
    print("=" * 60)
    print(f"配置: {num_steps}步/{total_angle}°")
    print(f"YOLO检测间隔: 每{effective_detection_interval}帧")
    print(f"跟踪器: SORT (max_age={TRACKER_MAX_AGE}, min_hits={TRACKER_MIN_HITS})")
    print(f"视频解码: {'GPU (CUDA)' if use_gpu else 'CPU'}")
    print(f"🎯 跟踪模式: {tracking_mode}{' (software-assist)' if software_assist_active else ''}")
    print(f"🎛️ PTZ速度: 常规{int(max(1, min(7, ptz_speed)))} / 追赶{int(max(1, min(7, ptz_speed_fast)))}")
    print(f"📺 跟踪流: {'子码流' if CAMERA_RTSP_SUB else '主码流'} | {'低延迟' if low_latency_stream else '常规延迟'}")
    if smart_shot:
        print("📸 Smart-Shot: 右手异步保存高质量抓拍，左手抬起开始/停止录像")
        print("🖼️ 抓拍目录: ~/Desktop/capture/pictures")
        print(f"📨 Telegram 发送: {'开启' if send_to_tg else '关闭（默认）'}")
        if hand_detector is None or hand_detector.model is None:
            print("⚠️ Smart-Shot pose 模型不可用，抬手检测不会触发")
        if tts is None or not tts.is_available():
            print("⚠️ 晓晓 TTS 不可用，右手抬起后不会发送语音")
        print("📬 Smart-Shot 队列策略: drop_oldest（队列满时丢弃最旧任务）")
        print(f"🙋 手势检测频率: 每 {0.20 if quick_mode else 0.40:.2f}s 一次（优先跟随流畅度）")
        print(f"🙋 手势模型输入: {224 if quick_mode else 256}px")
        print(f"🙋 左右手模式: {HAND_SIDE_MODE}（auto/normal/swapped）")
        print("🎬 录像策略: 仅左手抬起开始，丢失目标时自动停止")
    print(f"🔔 目标丢失播报: {'开启' if enable_miss else '关闭（使用 --enable-miss 开启）'}")
    if mute_voice:
        print("🔇 本机语音播报: 关闭（-m/--mute）")
    else:
        print(f"🔊 语音队列: {'drop_oldest' if voice_drop_oldest else 'keep_all'}")
    if quick_mode:
        print("⚡ Quick 模式: 高频检测 + 更低冷静时间")
    if tracking_mode in ("native", "hybrid"):
        print("ℹ️ native/hybrid 依赖摄像头端已开启人像追踪；失效时 hybrid 会自动切回软件扫描")
    print("按 Ctrl+C 停止追踪")
    print("=" * 60 + "\n")

    tracking_rtsp = CAMERA_RTSP_SUB or CAMERA_RTSP
    if not cam.camera.start_stream(use_gpu=use_gpu, rtsp_url=tracking_rtsp, low_latency=low_latency_stream):
        print("❌ 无法启动视频流")
        return

    try:
        while True:
            cycle_count += 1
            current_time = time.time()

            fps_history.append(1.0 / (current_time - last_time + 0.001))
            avg_fps = sum(fps_history) / len(fps_history)
            last_time = current_time

            if not person_found:
                if tracking_mode in ("native", "hybrid") and not software_assist_active:
                    frame = cam.camera.get_frame()
                    if frame is None:
                        time.sleep(frame_sleep)
                        continue

                    tracks = tracker.update(frame)
                    main_person = tracker.get_main_person()
                    if main_person is not None and tracks:
                        person_found = True
                        analyzing = True
                        lost_count = 0
                        recenter_candidate_count = 0
                        calibrating_active = False
                        stable_candidate_since = 0.0
                        print("✅ native 跟踪检测到目标")
                        broadcast_status("目标捕获", min_interval_sec=0.8)
                        cam.camera.tracking_memory.reset()
                        if recording_mgr:
                            recording_mgr.on_person_found()
                        time.sleep(frame_sleep)
                        continue

                    lost_count += 1
                    if tracking_mode == "hybrid" and lost_count >= LOST_THRESHOLD:
                        print("⚠️ native 跟踪未稳定捕获，切换到软件扫描兜底...")
                        software_assist_active = True
                        lost_count = 0
                    time.sleep(frame_sleep)
                    continue

                # 扫描找人
                print(f"\n{'=' * 60}")
                print(f"🔄 第 {cycle_count} 轮 | 执行智能扫描...")
                print(f"{'=' * 60}")

                person_found = cam.human_smart_only()

                if person_found:
                    print("✅ 找到目标！")
                    if software_assist_active:
                        print("🧩 软件扫描兜底成功，回交 native 跟踪")
                    broadcast_status("目标捕获", min_interval_sec=0.8)
                    cam.camera.tracking_memory.reset()

                    if not cam.camera.stream_active:
                        if not cam.camera.start_stream(
                            use_gpu=use_gpu,
                            rtsp_url=tracking_rtsp,
                            low_latency=low_latency_stream,
                        ):
                            return
                        time.sleep(0.5)

                    # 初始化跟踪
                    init_attempts = 0
                    max_init_attempts = 10

                    while init_attempts < max_init_attempts:
                        frame = cam.camera.get_frame()
                        if frame is not None:
                            tracks = tracker.update(frame, force_detect=True)
                            if tracks:
                                analyzing = True
                                lost_count = 0
                                break
                        init_attempts += 1
                        time.sleep(0.1)
                    else:
                        person_found = False
                        continue

                    analyzing = True
                    lost_count = 0
                    if tracking_mode == "hybrid":
                        software_assist_active = False
                    calibrating_active = False
                    stable_candidate_since = 0.0

                    # 找到目标后，按人物偏移做一次反方向微调（替代固定等待）
                    init_main = tracker.get_main_person()
                    if init_main is not None and (tracking_mode == "software" or software_assist_active):
                        init_cx = (init_main.bbox[0] + init_main.bbox[2]) / 2
                        init_offset_x = (init_cx - CAPTURE_WIDTH / 2) / (CAPTURE_WIDTH / 2)
                        if init_offset_x >= POST_FOUND_REVERSE_MIN_OFFSET:
                            cam.camera.ptz_turn("left", POST_FOUND_REVERSE_STEP_SEC)
                            print(f"↩️ 捕获后反向微调: left {POST_FOUND_REVERSE_STEP_SEC:.2f}s")
                        elif init_offset_x <= -POST_FOUND_REVERSE_MIN_OFFSET:
                            cam.camera.ptz_turn("right", POST_FOUND_REVERSE_STEP_SEC)
                            print(f"↪️ 捕获后反向微调: right {POST_FOUND_REVERSE_STEP_SEC:.2f}s")

                    # 找到目标：按配置决定是否自动开始录像
                    if recording_mgr:
                        recording_mgr.on_person_found()
                else:
                    print("未找到，继续扫描...")
                    if not cam.camera.start_stream(
                        use_gpu=use_gpu,
                        rtsp_url=tracking_rtsp,
                        low_latency=low_latency_stream,
                    ):
                        cam.camera.start_stream(
                            use_gpu=use_gpu,
                            rtsp_url=tracking_rtsp,
                            low_latency=low_latency_stream,
                        )
                    time.sleep(0.5)

            elif analyzing:
                # 实时跟踪模式
                frame = cam.camera.get_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue

                tracks = tracker.update(frame)
                main_person = tracker.get_main_person()

                if main_person:
                    # 计算偏移并平滑
                    cx = (main_person.bbox[0] + main_person.bbox[2]) / 2
                    cy = (main_person.bbox[1] + main_person.bbox[3]) / 2
                    offset_x = (cx - CAPTURE_WIDTH/2) / (CAPTURE_WIDTH/2)
                    offset_y = (cy - CAPTURE_HEIGHT/2) / (CAPTURE_HEIGHT/2)

                    offset_x_history.append(offset_x)
                    offset_y_history.append(offset_y)
                    smoothed_offset_x = sum(offset_x_history) / len(offset_x_history)
                    smoothed_offset_y = sum(offset_y_history) / len(offset_y_history)
                    effective_offset_y = smoothed_offset_y

                    if smart_shot:
                        top_margin_ratio = max(0.0, float(main_person.bbox[1]) / CAPTURE_HEIGHT)
                        bottom_margin_ratio = max(0.0, float(CAPTURE_HEIGHT - main_person.bbox[3]) / CAPTURE_HEIGHT)
                        person_height_ratio = max(0.0, float(main_person.bbox[3] - main_person.bbox[1]) / CAPTURE_HEIGHT)

                        # 人像过高通常受视角/FOV限制，继续压垂直云台收益有限且更容易打到上下限位
                        can_push_for_foot = person_height_ratio <= SMART_SHOT_FULL_BODY_HEIGHT_MAX

                        if can_push_for_foot and bottom_margin_ratio < SMART_SHOT_FOOT_MARGIN_MIN:
                            push_down = (SMART_SHOT_FOOT_MARGIN_MIN - bottom_margin_ratio) * SMART_SHOT_MARGIN_GAIN
                            push_down = min(SMART_SHOT_MARGIN_PUSH_MAX, push_down)
                            effective_offset_y = max(effective_offset_y, push_down)

                        if top_margin_ratio < SMART_SHOT_HEAD_MARGIN_MIN:
                            push_up = (SMART_SHOT_HEAD_MARGIN_MIN - top_margin_ratio) * SMART_SHOT_MARGIN_GAIN
                            push_up = min(SMART_SHOT_MARGIN_PUSH_MAX, push_up)
                            effective_offset_y = min(effective_offset_y, -push_up)

                        # 头部明显落在画面下半区时，优先触发向下修正（补偿“头在底部不跟随”）
                        if top_margin_ratio > SMART_SHOT_HEAD_LOW_TRIGGER:
                            head_low_push = (top_margin_ratio - SMART_SHOT_HEAD_LOW_TRIGGER) * SMART_SHOT_HEAD_LOW_PUSH_GAIN
                            head_low_push = min(SMART_SHOT_HEAD_LOW_PUSH_MAX, head_low_push)
                            effective_offset_y = max(effective_offset_y, head_low_push)

                    ptz_control_active = (tracking_mode == "software") or software_assist_active

                    # 更新运动记忆
                    current_angle = cam.camera.tracking_memory.last_angle
                    if smoothed_offset_x < -0.3:
                        current_angle = (current_angle - 20) % 360
                    elif smoothed_offset_x > 0.3:
                        current_angle = (current_angle + 20) % 360
                    cam.camera.tracking_memory.update(current_angle)

                    # 居中逻辑
                    person_width_ratio = (main_person.bbox[2] - main_person.bbox[0]) / CAPTURE_WIDTH
                    dynamic_x_threshold = BASE_RECENTER_X_THRESHOLD + min(0.25, person_width_ratio * 0.35)
                    dynamic_y_threshold = (
                        BASE_RECENTER_Y_THRESHOLD_UP
                        if effective_offset_y < 0
                        else BASE_RECENTER_Y_THRESHOLD_DOWN
                    )
                    need_recenter = (
                        abs(smoothed_offset_x) > dynamic_x_threshold
                        or abs(effective_offset_y) > dynamic_y_threshold
                    )

                    if need_recenter:
                        recenter_candidate_count += 1
                    else:
                        recenter_candidate_count = 0

                    if (
                        ptz_control_active
                        and
                        recenter_candidate_count >= RECENTER_CONFIRM_FRAMES
                        and (current_time - last_recenter_time) >= RECENTER_COOLDOWN
                    ):
                        if not calibrating_active:
                            broadcast_status("校准中", min_interval_sec=0.5)
                        calibrating_active = True
                        stable_candidate_since = 0.0
                        speed_x = 0.0
                        if len(offset_x_history) >= 2:
                            speed_x = abs(offset_x_history[-1] - offset_x_history[-2])
                        print(f"\n   🎯 持续偏移触发微步跟随: 水平{smoothed_offset_x:+.2f}, 垂直{effective_offset_y:+.2f}")
                        apply_micro_recenter(smoothed_offset_x, effective_offset_y, speed_x)
                        recenter_candidate_count = 0
                        last_recenter_time = current_time
                        time.sleep(frame_sleep)

                    # 校准完成判定：进入中心死区并持续稳定一段时间
                    if ptz_control_active and calibrating_active:
                        in_deadband = (
                            abs(smoothed_offset_x) <= STABLE_X_DEADBAND
                            and abs(smoothed_offset_y) <= STABLE_Y_DEADBAND
                        )
                        if len(offset_x_history) >= 2 and len(offset_y_history) >= 2:
                            vel_x = abs(offset_x_history[-1] - offset_x_history[-2])
                            vel_y = abs(offset_y_history[-1] - offset_y_history[-2])
                            low_motion = vel_x <= STABLE_VEL_THRESHOLD and vel_y <= STABLE_VEL_THRESHOLD
                        else:
                            low_motion = False

                        if in_deadband and low_motion:
                            if stable_candidate_since <= 0:
                                stable_candidate_since = current_time
                            elif (
                                (current_time - stable_candidate_since) >= STABLE_HOLD_SEC
                                and (current_time - last_calibration_done_time) >= CALIB_DONE_MIN_INTERVAL_SEC
                            ):
                                broadcast_status("校准完成", min_interval_sec=0.5)
                                last_calibration_done_time = current_time
                                calibrating_active = False
                                stable_candidate_since = 0.0
                        else:
                            stable_candidate_since = 0.0
                    elif not ptz_control_active:
                        calibrating_active = False
                        stable_candidate_since = 0.0
                        recenter_candidate_count = 0

                    # 手势检测
                    if gesture_handler and smart_shot_worker:
                        # 校准阶段暂缓手势检测，优先保证目标跟随实时性
                        if calibrating_active or recenter_candidate_count > 0:
                            lost_count = 0
                            time.sleep(frame_sleep)
                            continue

                        # 目标偏移较大时优先云台跟随，暂缓手势推理，避免拖慢跟踪
                        if abs(smoothed_offset_x) > 0.65 or abs(smoothed_offset_y) > 0.75:
                            lost_count = 0
                            time.sleep(frame_sleep)
                            continue

                        x1, y1, x2, y2 = [int(v) for v in main_person.bbox]
                        box_w = max(1, x2 - x1)
                        box_h = max(1, y2 - y1)
                        pad_x = int(box_w * 0.2)
                        pad_y = int(box_h * 0.25)

                        gx1 = max(0, x1 - pad_x)
                        gy1 = max(0, y1 - pad_y)
                        gx2 = min(frame.shape[1], x2 + pad_x)
                        gy2 = min(frame.shape[0], y2 + pad_y)

                        gesture_frame = frame[gy1:gy2, gx1:gx2]
                        if gesture_frame.size == 0:
                            gesture_frame = frame

                        event = gesture_handler.update(gesture_frame, current_time)

                        # 处理触发事件
                        if event:
                            handle_gesture_event(
                                event,
                                recording_mgr,
                                smart_shot_worker,
                                current_time,
                            )

                    lost_count = 0
                else:
                    # 丢失目标处理
                    recenter_candidate_count = 0
                    offset_x_history.clear()
                    offset_y_history.clear()
                    calibrating_active = False
                    stable_candidate_since = 0.0

                    lost_count += 1

                    if lost_count >= LOST_THRESHOLD:
                        print(f"\n   ⚠️ 丢失目标，重新扫描...")
                        if enable_miss:
                            broadcast_status("目标丢失", min_interval_sec=1.5)
                        if gesture_handler:
                            gesture_handler.reset()
                        if recording_mgr:
                            recording_mgr.on_person_lost()
                        if tracking_mode == "hybrid":
                            software_assist_active = True
                            print("   🧩 hybrid 已切换到软件扫描兜底")
                        analyzing = False
                        person_found = False

                time.sleep(frame_sleep)

            else:
                # 非分析模式，仅检查跟踪状态
                frame = cam.camera.get_frame()
                if frame is not None:
                    tracks = tracker.update(frame)
                    if tracks:
                        lost_count = 0
                    else:
                        lost_count += 1
                        if lost_count >= LOST_THRESHOLD:
                            if enable_miss:
                                broadcast_status("目标丢失", min_interval_sec=1.5)
                            if recording_mgr:
                                recording_mgr.on_person_lost()
                            if tracking_mode == "hybrid":
                                software_assist_active = True
                            person_found = False
                time.sleep(TRACK_CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n停止追踪...")
    finally:
        if voice_queue is not None:
            voice_queue.stop()
        if recording_mgr:
            recording_mgr.cleanup()
        if smart_shot_worker:
            smart_shot_worker.stop()
        cam.camera.stop_stream()
        print("\n追踪已停止")
        print(f"   共执行 {cycle_count} 轮")


def show_help():
    """显示帮助信息"""
    print("Mooer Camera NG - 智能视角控制系统")
    print("\n用法:")
    print("  python3 -m camera_ng --help")
    print("  python3 -m camera_ng <命令> [选项] [步数] [角度]")
    print("  python3 -m camera_ng <命令> --help")
    print("\n可用命令:")
    print("  human [选项] [步数] [角度]  - 多步扫描找人")
    print("  track [选项] [步数] [角度]  - 实时跟踪模式")
    print("  smart-shot [选项]           - 跟踪+右手抓拍(异步保存)+左手录像")
    print("  shot [步数] [角度]          - 拍照并发送")
    print("  prepare-tts [选项]          - 预生成常用本机提示音到 media/tts")
    print("  calibrate                   - 校准云台转速")
    print("\n选项:")
    print("  -h, --help                  - 显示帮助信息")
    print("  -g, --gpu                   - 使用 GPU 硬解")
    print("  -quick, --quick             - 高性能模式（更灵敏，更耗电）")
    print("  --tg, --telegram            - 开启 Telegram 发送（默认关闭）")
    print("  --enable-miss              - 开启“目标丢失”语音播报")
    print("  --voice-drop-oldest         - 语音队列满时丢弃最旧播报")
    print("  --low-latency               - 跟踪流启用低延迟拉流参数")
    print("  --normal-latency            - 跟踪流使用常规拉流参数")
    print("  --ptz-speed <1-7>           - 云台常规速度档位")
    print("  --ptz-speed-fast <1-7>      - 云台追赶速度档位")
    print("  --tracking-mode <mode>      - 跟踪模式: native|hybrid|software")
    print("  -m, --mute                  - 不入队任何本机语音播报")
    print("  --overwrite                 - 仅用于 prepare-tts，覆盖已有音频")
    print("  --speed <度/秒>             - 指定转速")
    print("\n示例:")
    print("  python3 -m camera_ng human          # 默认扫描")
    print("  python3 -m camera_ng track -g       # GPU 实时跟踪")
    print("  python3 -m camera_ng smart-shot -g -quick  # 高灵敏手势抓拍")
    print("  python3 -m camera_ng smart-shot -g --tg     # 开启 Telegram 发送")
    print("  python3 -m camera_ng track -g --low-latency # 低延迟跟踪")
    print("  python3 -m camera_ng track --tracking-mode hybrid")
    print("  python3 -m camera_ng smart-shot -g --tracking-mode native")
    print("  python3 -m camera_ng track --ptz-speed 2 --ptz-speed-fast 5")
    print("  python3 -m camera_ng shot 8 180 --tg         # 拍照并发送")
    print("  python3 -m camera_ng prepare-tts             # 预生成本地提示音")
    print("\nSmart-Shot 行为:")
    print("  - 右手抬起：异步保存高质量照片到 ~/Desktop/capture/pictures/<timestamp>.jpg")
    print("  - 左手抬起：开始/停止录像")
    print("  - 目标捕获/校准中/校准完成：默认播报")
    print("  - 目标丢失：仅 --enable-miss 时播报")
    print("  - 语音播报统一异步队列；可用 --voice-drop-oldest 避免积压")
    print("  - 使用 -m/--mute 可禁用全部本机语音入队")
    print("  - Telegram 默认关闭；加 --tg 才发送")


def main():
    """主入口函数"""
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        show_help()
        sys.exit(0 if sys.argv[1] in ('-h', '--help') else 1)

    cmd = sys.argv[1]
    args = sys.argv[2:]

    if "-h" in args or "--help" in args:
        show_help()
        sys.exit(0)

    # 解析 GPU 选项
    use_gpu = False
    if "-g" in args or "--gpu" in args:
        use_gpu = True
        args = [a for a in args if a not in ["-g", "--gpu"]]

    # 解析高性能选项
    quick_mode = False
    if "-quick" in args or "--quick" in args:
        quick_mode = True
        args = [a for a in args if a not in ["-quick", "--quick"]]

    # 解析 Telegram 发送选项
    send_to_tg = False
    if "--tg" in args or "--telegram" in args:
        send_to_tg = True
        args = [a for a in args if a not in ["--tg", "--telegram"]]

    overwrite_tts = False
    if "--overwrite" in args:
        overwrite_tts = True
        args = [a for a in args if a != "--overwrite"]

    enable_miss = False
    if "--enable-miss" in args:
        enable_miss = True
        args = [a for a in args if a != "--enable-miss"]

    voice_drop_oldest = False
    if "--voice-drop-oldest" in args:
        voice_drop_oldest = True
        args = [a for a in args if a != "--voice-drop-oldest"]

    mute_voice = False
    if "-m" in args or "--mute" in args:
        mute_voice = True
        args = [a for a in args if a not in ["-m", "--mute"]]

    low_latency_stream = STREAM_LOW_LATENCY
    if "--low-latency" in args:
        low_latency_stream = True
        args = [a for a in args if a != "--low-latency"]
    if "--normal-latency" in args:
        low_latency_stream = False
        args = [a for a in args if a != "--normal-latency"]

    ptz_speed = PTZ_SPEED
    if "--ptz-speed" in args:
        idx = args.index("--ptz-speed")
        if idx + 1 < len(args):
            ptz_speed = max(1, min(7, int(args[idx + 1])))
            args = args[:idx] + args[idx + 2:]

    ptz_speed_fast = PTZ_FAST_SPEED
    if "--ptz-speed-fast" in args:
        idx = args.index("--ptz-speed-fast")
        if idx + 1 < len(args):
            ptz_speed_fast = max(1, min(7, int(args[idx + 1])))
            args = args[:idx] + args[idx + 2:]

    tracking_mode = "hybrid"
    if "--tracking-mode" in args:
        idx = args.index("--tracking-mode")
        if idx + 1 < len(args):
            tracking_mode = args[idx + 1].strip().lower()
            args = args[:idx] + args[idx + 2:]
        else:
            print("❌ --tracking-mode 需要参数: native|hybrid|software")
            sys.exit(1)

    if tracking_mode not in {"native", "hybrid", "software"}:
        print(f"❌ 无效 --tracking-mode: {tracking_mode}")
        print("   可选值: native, hybrid, software")
        sys.exit(1)

    # 解析转速选项
    global ROTATION_SPEED
    if "--speed" in args:
        speed_idx = args.index("--speed")
        if speed_idx + 1 < len(args):
            ROTATION_SPEED = float(args[speed_idx + 1])
            print(f"⚙️  使用指定转速: {ROTATION_SPEED}°/s")
            args = args[:speed_idx] + args[speed_idx + 2:]

    num_steps = int(args[0]) if len(args) > 0 and args[0].lstrip("-").isdigit() else DEFAULT_NUM_STEPS
    total_angle = float(args[1]) if len(args) > 1 and args[1].replace(".", "", 1).lstrip("-").isdigit() else DEFAULT_TOTAL_ANGLE

    config_required_cmds = {"human", "shot", "track", "smart-shot"}
    if cmd in config_required_cmds:
        validate_config()

    lock = check_single_instance() if cmd in config_required_cmds else None

    cam = None
    try:
        if cmd == "human":
            cam = SmartCamera()
            result = cam.human(num_steps=num_steps, total_angle=total_angle, use_gpu=use_gpu)
            print(f"\n{'='*60}")
            print(f"扫描结果: {'找到人' if result else '未找到人'}")
            print(f"{'='*60}")
            cam.stop()
            
        elif cmd == "shot":
            cam = SmartCamera()
            result = cam.human(num_steps=num_steps, total_angle=total_angle, 
                             use_gpu=use_gpu, center_and_wait=True)
            if result:
                capture_and_send_current_view(
                    cam.camera,
                    "Albert，我抓拍到你啦！📸💕",
                    send_to_tg=send_to_tg,
                )
                    
            print(f"\n{'='*60}")
            print(f"拍照结果: {'成功' if result else '未找到人'}")
            print(f"{'='*60}")
            cam.stop()
            
        elif cmd == "track":
            track_human_realtime(
                num_steps=num_steps,
                total_angle=total_angle,
                use_gpu=use_gpu,
                tracking_mode=tracking_mode,
                quick_mode=quick_mode,
                send_to_tg=send_to_tg,
                enable_miss=enable_miss,
                voice_drop_oldest=voice_drop_oldest,
                mute_voice=mute_voice,
                low_latency_stream=low_latency_stream,
                ptz_speed=ptz_speed,
                ptz_speed_fast=ptz_speed_fast,
            )

        elif cmd == "smart-shot":
            track_human_realtime(
                num_steps=num_steps,
                total_angle=total_angle,
                use_gpu=use_gpu,
                tracking_mode=tracking_mode,
                smart_shot=True,
                quick_mode=quick_mode,
                send_to_tg=send_to_tg,
                enable_miss=enable_miss,
                voice_drop_oldest=voice_drop_oldest,
                mute_voice=mute_voice,
                low_latency_stream=low_latency_stream,
                ptz_speed=ptz_speed,
                ptz_speed_fast=ptz_speed_fast,
            )

        elif cmd == "prepare-tts":
            tts = XiaoxiaoTTS()
            created, skipped = tts.pregenerate_common_prompts(overwrite=overwrite_tts)
            print("\n" + "=" * 60)
            print("🔊 预生成提示音完成")
            print(f"   目录: {tts.media_dir}")
            print(f"   新生成: {created}")
            print(f"   跳过: {skipped}")
            print("=" * 60)
            
        elif cmd == "calibrate":
            subprocess.run(["python3", "/home/albert/clawd/scripts/calibrate_speed.py"])
        else:
            print(f"未知命令: {cmd}")
            print("支持命令: human, shot, track, smart-shot, prepare-tts, calibrate")
            sys.exit(1)
    finally:
        if lock is not None:
            lock.release()


if __name__ == "__main__":
    main()
