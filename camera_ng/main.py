#!/usr/bin/env python3
"""
Mooer Camera NG - 主入口
重构后的模块化智能视角控制系统
"""

import faulthandler
import fcntl
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque

# 启用 faulthandler，在崩溃时打印 Python 堆栈
faulthandler.enable()

from camera_ng import (
    DEFAULT_NUM_STEPS, DEFAULT_TOTAL_ANGLE,
    ROTATION_SPEED, TRACK_CHECK_INTERVAL, DETECTION_INTERVAL,
    TRACKER_MAX_AGE, TRACKER_MIN_HITS,
    CAPTURE_WIDTH, CAPTURE_HEIGHT, LOCK_FILE,
    CAMERA_RTSP, DEVICE_SERIAL, ACCESS_TOKEN,
    CameraController, VisionAnalyzer, HandRaiseDetector, XiaoxiaoTTS,
    PersonTracker, TrackingMemory
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


# 启动时验证配置
validate_config()


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


def capture_and_send_current_view(camera: CameraController, message: str) -> bool:
    """基于当前画面直接抓拍并发送，不执行找人流程"""
    img_path = camera.capture(full_quality=True)
    print(f"📸 已抓拍当前画面: {img_path}")

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


def send_greeting_voice(tts: XiaoxiaoTTS, message: str) -> bool:
    """发送中文问候语音到与图片相同的 Telegram 目标"""
    try:
        if tts.playback(message):
            print("🔈 已在本机播放问候语音")
        else:
            print("⚠️ 本机语音播放失败（已继续发送 Telegram 语音）")

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


def start_high_quality_recording() -> tuple[subprocess.Popen[str] | None, str | None]:
    """启动高质量录像并返回 ffmpeg 进程与输出路径"""
    try:
        output_dir = os.path.expanduser("~/Desktop/capture")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"{timestamp}.mp4")

        cmd = [
            "ffmpeg",
            "-rtsp_transport",
            "tcp",
            "-i",
            CAMERA_RTSP,
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            "-y",
            output_path,
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        print(f"🎬 左手触发：开始录像 {output_path}")
        return proc, output_path
    except Exception as e:
        print(f"❌ 启动录像失败: {e}")
        return None, None


def stop_high_quality_recording(record_proc: subprocess.Popen[str] | None, output_path: str | None) -> None:
    """停止录像并落盘"""
    if record_proc is None:
        return

    proc = record_proc
    try:
        if proc.poll() is None:
            if proc.stdin is not None:
                proc.stdin.write("q\n")
                proc.stdin.flush()
            proc.wait(timeout=2.0)
    except Exception:
        try:
            proc.terminate()
            proc.wait(timeout=1.5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    finally:
        if output_path:
            print(f"✅ 左手触发：停止录像，已保存 {output_path}")


def local_voice_broadcast(tts: XiaoxiaoTTS | None, text: str) -> None:
    """本机语音播报（后台执行，不阻塞录像/追踪）"""
    if tts is None or not tts.is_available():
        return

    def _worker() -> None:
        try:
            ok = tts.playback(text)
            if ok:
                print(f"🔈 语音播报: {text}")
        except Exception:
            pass

    threading.Thread(target=_worker, daemon=True).start()


def start_smart_shot_worker(
    camera: CameraController,
    tts: XiaoxiaoTTS | None,
    task_queue: queue.Queue,
    stop_event: threading.Event,
) -> threading.Thread:
    """启动 Smart-Shot 后台 worker（串行消费队列任务）"""

    def _worker():
        while not stop_event.is_set():
            try:
                hand_text, hand_reason = task_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                capture_and_send_current_view(
                    camera,
                    f"Albert，我检测到你抬{hand_text}，已为你抓拍！📸",
                )
                if "right" in hand_reason and tts is not None and tts.is_available():
                    send_greeting_voice(tts, "嗨 Albert，你好呀，我看到你举起右手啦。")
            finally:
                task_queue.task_done()

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    return worker


def trigger_smart_shot_async(
    hand_text: str,
    hand_reason: str,
    tts: XiaoxiaoTTS | None,
    task_queue: queue.Queue,
) -> bool:
    """入队 Smart-Shot 任务，主循环不阻塞（满队列时丢弃最旧任务）"""

    if task_queue.full():
        try:
            _ = task_queue.get_nowait()
            task_queue.task_done()
            print("🗑️ Smart-Shot 队列已满，丢弃最旧任务（drop_oldest）")
        except queue.Empty:
            pass

    try:
        task_queue.put_nowait((hand_text, hand_reason))
        if tts is not None and tts.is_available():
            if tts.playback("收到"):
                print("🔈 已本机播报: 收到")
            else:
                print("⚠️ 本机播报“收到”失败")
        return True
    except queue.Full:
        print("⏳ Smart-Shot 队列拥塞，跳过本次触发")
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
                         smart_shot: bool = False,
                         quick_mode: bool = False) -> None:
    """实时目标跟踪模式"""
    cam = SmartCamera()
    effective_detection_interval = 1 if quick_mode else detection_interval
    tracker = PersonTracker(
        yolo_model="yolov8n",
        confidence=0.5,
        detection_interval=effective_detection_interval
    )
    
    cycle_count = 0
    person_found = False
    analyzing = False
    lost_count = 0
    LOST_THRESHOLD = 5
    
    fps_history = deque(maxlen=30)
    last_time = time.time()
    offset_x_history = deque(maxlen=5)
    offset_y_history = deque(maxlen=5)
    recenter_candidate_count = 0
    last_recenter_time = 0.0

    # 抗抖参数：避免“转头/喝水”这类短时姿态变化触发云台
    RECENTER_CONFIRM_FRAMES = 3
    RECENTER_COOLDOWN = 1.2
    BASE_RECENTER_X_THRESHOLD = 0.5
    BASE_RECENTER_Y_THRESHOLD = 0.6

    hand_raise_detector = HandRaiseDetector() if smart_shot else None
    tts = XiaoxiaoTTS() if smart_shot else None
    smart_shot_queue = queue.Queue(maxsize=3) if smart_shot else None
    smart_shot_stop_event = threading.Event() if smart_shot else None
    smart_shot_worker = (
        start_smart_shot_worker(cam.camera, tts, smart_shot_queue, smart_shot_stop_event)
        if smart_shot and smart_shot_queue is not None and smart_shot_stop_event is not None
        else None
    )
    hand_raise_confirm_frames = 1 if quick_mode else 2
    hand_raise_count = 0
    shot_cooldown = 0.6 if quick_mode else 1.0
    last_shot_time = 0.0
    last_hand_log_time = 0.0
    hand_trigger_armed = True
    record_proc: subprocess.Popen[str] | None = None
    record_output_path: str | None = None
    record_toggle_cooldown = 1.5
    last_record_toggle_time = 0.0
    frame_sleep = 0.005 if quick_mode else 0.03
    recenter_pause = 0.2 if quick_mode else 0.5

    print("\n" + "=" * 60)
    print("🔍 启动实时目标跟踪模式 (Real-time + SORT)")
    print("=" * 60)
    print(f"配置: {num_steps}步/{total_angle}°")
    print(f"YOLO检测间隔: 每{effective_detection_interval}帧")
    print(f"跟踪器: SORT (max_age={TRACKER_MAX_AGE}, min_hits={TRACKER_MIN_HITS})")
    print(f"视频解码: {'GPU (CUDA)' if use_gpu else 'CPU'}")
    if smart_shot:
        print("📸 Smart-Shot: 右手抬起抓拍发送，左手抬起开始/停止录像")
        if hand_raise_detector is None or hand_raise_detector.model is None:
            print("⚠️ Smart-Shot pose 模型不可用，抬手检测不会触发")
        if tts is None or not tts.is_available():
            print("⚠️ 晓晓 TTS 不可用，右手抬起后不会发送语音")
        print("📬 Smart-Shot 队列策略: drop_oldest（队列满时丢弃最旧任务）")
    if quick_mode:
        print("⚡ Quick 模式: 高频检测 + 更低冷静时间")
    print("按 Ctrl+C 停止追踪")
    print("=" * 60 + "\n")

    if not cam.camera.start_stream(use_gpu=use_gpu):
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
                print(f"\n{'=' * 60}")
                print(f"🔄 第 {cycle_count} 轮 | 执行智能扫描...")
                print(f"{'=' * 60}")

                person_found = cam.human_smart_only()

                if person_found:
                    print("✅ 找到目标！")
                    cam.camera.tracking_memory.reset()
                    
                    if not cam.camera.stream_active:
                        if not cam.camera.start_stream():
                            return
                        time.sleep(0.5)

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
                else:
                    print("未找到，继续扫描...")
                    if not cam.camera.start_stream():
                        cam.camera.start_stream()
                    time.sleep(0.5)
                    
            elif analyzing:
                # 实时跟踪模式 - 强制异步读取最新帧
                frame = cam.camera.get_frame()
                if frame is None:
                    time.sleep(0.001) # 极短等待，防止空循环
                    continue
                
                # 更新跟踪器 (YOLO推理完全在GPU上，不阻塞拉流线程)
                tracks = tracker.update(frame)
                main_person = tracker.get_main_person()
                
                # 计算并显示 FPS (识别帧率)
                detect_mode = "DETECT" if cycle_count % effective_detection_interval == 0 else "TRACK "
                status = f"\r📊 [{detect_mode}] FPS:{avg_fps:.1f} | Tracks:{len(tracks)}"
                if main_person:
                    cx, cy = main_person.bbox[[0, 2]].mean(), main_person.bbox[[1, 3]].mean()
                    offset_x = (cx - CAPTURE_WIDTH/2) / (CAPTURE_WIDTH/2)
                    status += f" | MainID:{main_person.id} offset:{offset_x:+.2f}"
                print(status, end="", flush=True)
                
                if main_person:
                    cx = (main_person.bbox[0] + main_person.bbox[2]) / 2
                    cy = (main_person.bbox[1] + main_person.bbox[3]) / 2
                    offset_x = (cx - CAPTURE_WIDTH/2) / (CAPTURE_WIDTH/2)
                    offset_y = (cy - CAPTURE_HEIGHT/2) / (CAPTURE_HEIGHT/2)

                    offset_x_history.append(offset_x)
                    offset_y_history.append(offset_y)
                    smoothed_offset_x = sum(offset_x_history) / len(offset_x_history)
                    smoothed_offset_y = sum(offset_y_history) / len(offset_y_history)
                    
                    # 更新运动记忆
                    current_angle = cam.camera.tracking_memory.last_angle
                    if smoothed_offset_x < -0.3:
                        current_angle = (current_angle - 20) % 360
                    elif smoothed_offset_x > 0.3:
                        current_angle = (current_angle + 20) % 360
                    cam.camera.tracking_memory.update(current_angle)
                    
                    # 触发居中逻辑（连续多帧 + 平滑 + 冷却）
                    person_width_ratio = (main_person.bbox[2] - main_person.bbox[0]) / CAPTURE_WIDTH
                    dynamic_x_threshold = BASE_RECENTER_X_THRESHOLD + min(0.25, person_width_ratio * 0.35)
                    need_recenter = (
                        abs(smoothed_offset_x) > dynamic_x_threshold
                        or abs(smoothed_offset_y) > BASE_RECENTER_Y_THRESHOLD
                    )

                    if need_recenter:
                        recenter_candidate_count += 1
                    else:
                        recenter_candidate_count = 0

                    if (
                        recenter_candidate_count >= RECENTER_CONFIRM_FRAMES
                        and (current_time - last_recenter_time) >= RECENTER_COOLDOWN
                    ):
                        print(
                            f"\n   🎯 持续偏移触发居中: 水平{smoothed_offset_x:+.2f}, 垂直{smoothed_offset_y:+.2f}"
                        )
                        cam.camera.center_person(smoothed_offset_x, smoothed_offset_y)
                        recenter_candidate_count = 0
                        last_recenter_time = current_time
                        offset_x_history.clear()
                        offset_y_history.clear()
                        # 仅在调整云台后短暂停顿，其他时间全力跑
                        time.sleep(recenter_pause)

                    if smart_shot and hand_raise_detector is not None and smart_shot_queue is not None:
                        hand_raised, hand_reason = hand_raise_detector.get_hand_raise_state(frame)
                        if hand_raised:
                            hand_raise_count = min(hand_raise_count + 1, hand_raise_confirm_frames)
                        else:
                            hand_raise_count = max(hand_raise_count - 1, 0)
                            if hand_raise_count == 0:
                                hand_trigger_armed = True

                        if (current_time - last_hand_log_time) >= 2.0:
                            print(f"\n   🙋 手势检测: {hand_reason} | 连续帧: {hand_raise_count}/{hand_raise_confirm_frames}")
                            last_hand_log_time = current_time

                        if (
                            hand_trigger_armed
                            and hand_raised
                            and hand_raise_count >= hand_raise_confirm_frames
                            and (current_time - last_shot_time) >= shot_cooldown
                        ):
                            if "left" in hand_reason:
                                if (current_time - last_record_toggle_time) >= record_toggle_cooldown:
                                    if record_proc is None or record_proc.poll() is not None:
                                        record_proc, record_output_path = start_high_quality_recording()
                                        if record_proc is not None:
                                            local_voice_broadcast(tts, "开始录像")
                                    else:
                                        stop_high_quality_recording(record_proc, record_output_path)
                                        local_voice_broadcast(tts, "停止录像")
                                        record_proc = None
                                        record_output_path = None
                                    last_record_toggle_time = current_time
                                hand_raise_count = 0
                                hand_trigger_armed = False
                                continue

                            if "right" not in hand_reason:
                                hand_raise_count = 0
                                hand_trigger_armed = False
                                continue

                            print("\n   🙋 检测到右手抬起，进入 Smart-Shot（不重新找人）...")
                            started = trigger_smart_shot_async(
                                "右手",
                                hand_reason,
                                tts,
                                smart_shot_queue,
                            )
                            if started:
                                last_shot_time = current_time
                                hand_raise_count = 0
                                hand_trigger_armed = False
                    
                    lost_count = 0
                else:
                    recenter_candidate_count = 0
                    offset_x_history.clear()
                    offset_y_history.clear()
                    lost_count += 1
                    if lost_count >= LOST_THRESHOLD:
                        hand_raise_count = 0
                        print(f"\n   ⚠️ 丢失目标，重新扫描...")
                        analyzing = False
                        person_found = False
                
                # 限制 FPS 避免 CPU 忙等
                time.sleep(frame_sleep)
            else:
                frame = cam.camera.get_frame()
                if frame is not None:
                    tracks = tracker.update(frame)
                    if tracks:
                        lost_count = 0
                    else:
                        lost_count += 1
                        if lost_count >= LOST_THRESHOLD:
                            person_found = False
                time.sleep(TRACK_CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n停止追踪...")
    finally:
        if record_proc is not None:
            stop_high_quality_recording(record_proc, record_output_path)
        if smart_shot_stop_event is not None:
            smart_shot_stop_event.set()
        if smart_shot_worker is not None:
            smart_shot_worker.join(timeout=0.5)
        cam.camera.stop_stream()
        print("\n追踪已停止")
        print(f"   共执行 {cycle_count} 轮")


def show_help():
    """显示帮助信息"""
    print("Mooer Camera NG - 智能视角控制系统")
    print("\n用法: python3 -m camera_ng <命令> [选项] [步数] [角度]")
    print("\n可用命令:")
    print("  human [选项] [步数] [角度]  - 多步扫描找人")
    print("  track [选项] [步数] [角度]  - 实时跟踪模式")
    print("  smart-shot [选项]           - 跟踪+右手抬起自动抓拍发送")
    print("  shot [步数] [角度]          - 拍照并发送")
    print("  calibrate                   - 校准云台转速")
    print("\n选项:")
    print("  -h, --help                  - 显示帮助信息")
    print("  -g, --gpu                   - 使用 GPU 硬解")
    print("  -quick, --quick             - 高性能模式（更灵敏，更耗电）")
    print("  --speed <度/秒>             - 指定转速")
    print("\n示例:")
    print("  python3 -m camera_ng human          # 默认扫描")
    print("  python3 -m camera_ng track -g       # GPU 实时跟踪")
    print("  python3 -m camera_ng smart-shot -g -quick  # 高灵敏手势抓拍")
    print("  python3 -m camera_ng shot 8 180     # 拍照模式")


def main():
    """主入口函数"""
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        show_help()
        sys.exit(0 if sys.argv[1] in ('-h', '--help') else 1)

    cmd = sys.argv[1]
    args = sys.argv[2:]

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

    # 解析转速选项
    global ROTATION_SPEED
    if "--speed" in args:
        speed_idx = args.index("--speed")
        if speed_idx + 1 < len(args):
            ROTATION_SPEED = float(args[speed_idx + 1])
            print(f"⚙️  使用指定转速: {ROTATION_SPEED}°/s")
            args = args[:speed_idx] + args[speed_idx + 2:]

    num_steps = int(args[0]) if len(args) > 0 else DEFAULT_NUM_STEPS
    total_angle = float(args[1]) if len(args) > 1 else DEFAULT_TOTAL_ANGLE

    lock = check_single_instance()

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
                capture_and_send_current_view(cam.camera, "Albert，我抓拍到你啦！📸💕")
                    
            print(f"\n{'='*60}")
            print(f"拍照结果: {'成功' if result else '未找到人'}")
            print(f"{'='*60}")
            cam.stop()
            
        elif cmd == "track":
            track_human_realtime(
                num_steps=num_steps,
                total_angle=total_angle,
                use_gpu=use_gpu,
                quick_mode=quick_mode,
            )

        elif cmd == "smart-shot":
            track_human_realtime(
                num_steps=num_steps,
                total_angle=total_angle,
                use_gpu=use_gpu,
                smart_shot=True,
                quick_mode=quick_mode,
            )
            
        elif cmd == "calibrate":
            subprocess.run(["python3", "/home/albert/clawd/scripts/calibrate_speed.py"])
        else:
            print(f"未知命令: {cmd}")
            print("支持命令: human, shot, track, smart-shot, calibrate")
            sys.exit(1)
    finally:
        lock.release()


if __name__ == "__main__":
    main()
