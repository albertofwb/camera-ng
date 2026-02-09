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
    TRACKER_MAX_AGE, TRACKER_MIN_HITS,
    CAPTURE_WIDTH, CAPTURE_HEIGHT, LOCK_FILE,
    CAMERA_RTSP, DEVICE_SERIAL, ACCESS_TOKEN,
    CameraController, VisionAnalyzer,
    PersonTracker, TrackingMemory
)


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
                         use_gpu: bool = False) -> None:
    """实时目标跟踪模式"""
    cam = SmartCamera()
    tracker = PersonTracker(
        yolo_model="yolov8n",
        confidence=0.5,
        detection_interval=detection_interval
    )
    
    cycle_count = 0
    person_found = False
    analyzing = False
    lost_count = 0
    LOST_THRESHOLD = 5
    
    fps_history = deque(maxlen=30)
    last_time = time.time()

    print("\n" + "=" * 60)
    print("🔍 启动实时目标跟踪模式 (Real-time + SORT)")
    print("=" * 60)
    print(f"配置: {num_steps}步/{total_angle}°")
    print(f"YOLO检测间隔: 每{detection_interval}帧")
    print(f"跟踪器: SORT (max_age={TRACKER_MAX_AGE}, min_hits={TRACKER_MIN_HITS})")
    print(f"视频解码: {'GPU (CUDA)' if use_gpu else 'CPU'}")
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
                frame = cam.camera.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                tracks = tracker.update(frame)
                main_person = tracker.get_main_person()
                
                detect_mode = "DETECT" if cycle_count % detection_interval == 0 else "TRACK "
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
                    
                    current_angle = cam.camera.tracking_memory.last_angle
                    if offset_x < -0.3:
                        current_angle = (current_angle - 20) % 360
                    elif offset_x > 0.3:
                        current_angle = (current_angle + 20) % 360
                    cam.camera.tracking_memory.update(current_angle)
                    
                    if abs(offset_x) > 0.5 or abs(offset_y) > 0.6:
                        print(f"\n   调整: 水平{offset_x:+.2f}, 垂直{offset_y:+.2f}")
                        cam.camera.center_person(offset_x, offset_y)
                        time.sleep(0.8)
                    
                    lost_count = 0
                else:
                    lost_count += 1
                    if lost_count >= LOST_THRESHOLD:
                        print(f"\n   丢失目标，重新扫描...")
                        analyzing = False
                        person_found = False
                
                time.sleep(0.01)
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
    print("  shot [步数] [角度]          - 拍照并发送")
    print("  calibrate                   - 校准云台转速")
    print("\n选项:")
    print("  -h, --help                  - 显示帮助信息")
    print("  -g, --gpu                   - 使用 GPU 硬解")
    print("  --speed <度/秒>             - 指定转速")
    print("\n示例:")
    print("  python3 -m camera_ng human          # 默认扫描")
    print("  python3 -m camera_ng track -g       # GPU 实时跟踪")
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
                img_path = cam.camera.capture()
                print(f"📸 已自动抓拍并居中: {img_path}")
                
                try:
                    target = "1115213761"
                    msg = "Albert，我抓拍到你啦！📸💕"
                    send_cmd = [
                        "openclaw", "message", "send",
                        "--channel", "telegram",
                        "--target", target,
                        "--media", img_path,
                        "--message", msg
                    ]
                    print(f"📤 正在通过 OpenClaw 发送照片...")
                    subprocess.run(send_cmd, check=True)
                    print("✅ 照片发送成功！")
                except Exception as e:
                    print(f"❌ 照片发送失败: {e}")
                    
            print(f"\n{'='*60}")
            print(f"拍照结果: {'成功' if result else '未找到人'}")
            print(f"{'='*60}")
            cam.stop()
            
        elif cmd == "track":
            track_human_realtime(num_steps=num_steps, total_angle=total_angle, use_gpu=use_gpu)
            
        elif cmd == "calibrate":
            subprocess.run(["python3", "/home/albert/clawd/scripts/calibrate_speed.py"])
        else:
            print(f"未知命令: {cmd}")
            print("支持命令: human, shot, track, calibrate")
            sys.exit(1)
    finally:
        lock.release()


if __name__ == "__main__":
    main()
