#!/usr/bin/env python3
"""
云台控制器模块 - CameraController 类
处理 PTZ 控制和智能扫描逻辑
"""

import json
import subprocess
import threading
import time
import urllib.request
import urllib.parse
from typing import Optional, TYPE_CHECKING

from .config import (
    CAMERA_RTSP, CAPTURE_SEEK_TIME, CAPTURE_WIDTH, CAPTURE_HEIGHT, CAPTURE_QUALITY,
    DEVICE_SERIAL, ACCESS_TOKEN, ROTATION_SPEED,
    LEFT_LIMIT_STEP_DURATION, TURN_STABILIZE_TIME,
    CENTER_THRESHOLD, MAX_CENTER_ADJUST,
    DETECTION_SLEEP_TIME, TRACK_SCAN_DELAY,
    DIR_LEFT_CODE, DIR_RIGHT_CODE, DIR_UP_CODE, DIR_DOWN_CODE,
    PTZ_ERROR_CODES
)
from .stream import VideoStream
from .tracking import TrackingMemory

if TYPE_CHECKING:
    from .vision import VisionAnalyzer


class CameraController:
    """摄像头控制器 - 支持实时视频流和智能追踪"""

    def __init__(self):
        self.video_stream: Optional[VideoStream] = None
        self.stream_active = False
        self.tracking_memory = TrackingMemory()
        # 垂直边界记录
        self.hit_up_limit = False
        self.hit_down_limit = False
        self._ptz_lock = threading.Lock()
        self.center_and_wait_mode = False

    def start_stream(self, use_gpu: bool = False, force_restart: bool = False) -> bool:
        """启动实时视频流（支持复用）"""
        if self.stream_active and self.video_stream is not None and not force_restart:
            return True
        
        if force_restart and self.stream_active:
            self.stop_stream()

        self.video_stream = VideoStream(
            rtsp_url=CAMERA_RTSP,
            width=CAPTURE_WIDTH,
            height=CAPTURE_HEIGHT,
            buffer_size=3,
            use_gpu=use_gpu
        )

        if self.video_stream.start():
            self.stream_active = True
            time.sleep(0.5)
            return True
        return False
    
    def stop_stream(self, cleanup: bool = True):
        """停止视频流"""
        if self.video_stream:
            self.video_stream.stop()
            if cleanup:
                self.video_stream = None
                self.stream_active = False
            else:
                self.stream_active = False

    def capture(self, output_path: str = "/tmp/mooer_view.jpg") -> str:
        """抓取当前画面（兼容旧接口）"""
        # 优先使用视频流
        if self.stream_active and self.video_stream:
            frame = self.video_stream.get_frame()
            if frame is not None:
                import cv2
                cv2.imwrite(output_path, frame)
                return output_path
        
        # 回退到 ffmpeg 截图
        cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-i", CAMERA_RTSP,
            "-ss", CAPTURE_SEEK_TIME,
            "-vframes", "1",
            "-vf", f"scale={CAPTURE_WIDTH}:{CAPTURE_HEIGHT}",
            "-q:v", str(CAPTURE_QUALITY),
            output_path,
            "-y",
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"截图失败: {result.stderr.decode()}")
        return output_path
    
    def get_frame(self) -> Optional:
        """获取当前帧"""
        if self.video_stream:
            return self.video_stream.get_frame()
        return None

    def ptz_turn(self, direction: str, duration: float) -> bool:
        """云台转动（水平或垂直，阻塞式）"""
        direction_map = {
            "left": DIR_LEFT_CODE,
            "right": DIR_RIGHT_CODE,
            "up": DIR_UP_CODE,
            "down": DIR_DOWN_CODE,
        }

        if direction not in direction_map:
            print(f"⚠️  未知方向: {direction}")
            return False

        dir_code = direction_map[direction]

        try:
            warning = self._ptz_control("start", dir_code)
            if warning:
                return False

            time.sleep(float(duration))
            self._ptz_control("stop", dir_code)
            return True
        except Exception as e:
            print(f"⚠️  PTZ转动异常: {e}")
            return False

    def ptz_turn_async(self, direction: str):
        """异步启动云台转动（非阻塞）"""
        direction_map = {
            "left": DIR_LEFT_CODE,
            "right": DIR_RIGHT_CODE,
            "up": DIR_UP_CODE,
            "down": DIR_DOWN_CODE,
        }

        if direction not in direction_map:
            print(f"⚠️  未知方向: {direction}")
            return False, 0

        dir_code = direction_map[direction]
        warning = self._ptz_control("start", dir_code)
        if warning:
            return False, 0

        return True, dir_code

    def ptz_stop(self, direction: int) -> None:
        """停止指定方向的转动"""
        self._ptz_control("stop", direction)

    def center_person(self, offset_x: float, offset_y: float = 0) -> bool:
        """将人物移到画面中央"""
        adjusted = False
        
        try:
            # 水平调整
            if abs(offset_x) >= CENTER_THRESHOLD:
                angle = offset_x * MAX_CENTER_ADJUST
                direction = "left" if offset_x < 0 else "right"
                duration = abs(angle) / ROTATION_SPEED
                try:
                    self.ptz_turn(direction, duration)
                    adjusted = True
                except Exception as e:
                    print(f"   ⚠️ 水平调整失败: {e}")
            
            # 垂直调整
            VERTICAL_THRESHOLD = 0.3
            if abs(offset_y) >= VERTICAL_THRESHOLD:
                direction = "up" if offset_y < 0 else "down"
                
                if direction == "up" and self.hit_up_limit:
                    pass
                elif direction == "down" and self.hit_down_limit:
                    pass
                else:
                    tilt_duration = min(abs(offset_y) * 0.5, 0.3)
                    try:
                        success = self.ptz_turn(direction, tilt_duration)
                        if not success:
                            if direction == "up":
                                self.hit_up_limit = True
                            else:
                                self.hit_down_limit = True
                        else:
                            adjusted = True
                    except Exception as e:
                        print(f"   ⚠️ 垂直调整失败: {e}")
            
            if adjusted:
                time.sleep(0.5)
        except Exception as e:
            print(f"⚠️  center_person 异常: {e}")
        
        return adjusted

    def _ptz_control(self, action: str, direction: int) -> Optional[str]:
        """调用萤石云 API"""
        with self._ptz_lock:
            data = urllib.parse.urlencode({
                "accessToken": ACCESS_TOKEN,
                "deviceSerial": DEVICE_SERIAL,
                "channelNo": 1,
                "direction": direction,
                "speed": 1,
            })

            url = f"https://open.ys7.com/api/lapp/device/ptz/{action}"
            req = urllib.request.Request(url, data=data.encode(), method="POST")

            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    result = json.loads(resp.read().decode())
                    code = result.get("code")
                    msg = result.get("msg", "")

                    if code == "200":
                        return None

                    if code in PTZ_ERROR_CODES:
                        print(f"⚠️  PTZ [{code}] {PTZ_ERROR_CODES[code]}: {msg}")
                        if code == "20006" or "限位" in msg:
                            return msg
                        return None
                    else:
                        print(f"⚠️  PTZ [{code}] {msg}")
                        if "限位" in msg:
                            return msg
                        return None

            except Exception as e:
                print(f"❌ PTZ 异常: {e}")
                return str(e)

    def goto_left_limit(self, vision: "VisionAnalyzer" = None) -> bool:
        """转到左极限位置"""
        print("\n🎯 转到左极限...")
        print("-" * 50)

        left_steps = 0
        while True:
            success = self.ptz_turn("left", LEFT_LIMIT_STEP_DURATION)
            if not success:
                print(f"🚧 到达左极限")
                break
            left_steps += 1

            if vision:
                time.sleep(DETECTION_SLEEP_TIME)
                try:
                    frame = self.get_frame()
                    if frame is not None:
                        has_person, _ = vision.check_person(frame=frame)
                        if has_person:
                            print("   检测到人物！停止转动")
                            position = vision.analyze_position(frame=frame)
                            offset_x, offset_y = vision.get_person_offset(frame=frame)
                            self.center_person(offset_x, offset_y)
                            return True
                except Exception as e:
                    print(f"⚠️  检测异常: {e}")

            time.sleep(DETECTION_SLEEP_TIME)

        print(f"✅ 左极限定位完成 (转了 {left_steps} 步)")
        print("-" * 50)
        return False

    def human_steps(self, vision: "VisionAnalyzer", num_steps: int = 8, total_angle: float = 180) -> bool:
        """human 多步扫描策略"""
        step_size = total_angle / num_steps
        step_duration = step_size / ROTATION_SPEED

        print(f"\n🔄 启动{num_steps}步扫描...")
        print("=" * 60)

        # 预检查
        frame = self.get_frame()
        if frame is not None and vision.check_person(frame=frame)[0]:
            print("当前位置已有人！")
            position = vision.analyze_position(frame=frame)
            print(f"   语音播报: 找到你了，{position}")
            return True

        # 转到左极限
        if self.goto_left_limit(vision=vision):
            return True

        # 多步扫描
        print(f"\n📍 开始{num_steps}步扫描...")
        print("-" * 60)

        for i in range(num_steps):
            print(f"\n🔍 步骤 {i + 1}/{num_steps}")

            if i > 0:
                print("   → 右转...", end=" ", flush=True)
                self.ptz_turn("right", step_duration)
                print("✅")
                time.sleep(TURN_STABILIZE_TIME)

            print("   📸 检测人物...", end=" ", flush=True)
            frame = self.get_frame()
            if frame is not None and vision.check_person(frame=frame)[0]:
                print("👤 有人！")
                offset_x, offset_y = vision.get_person_offset(frame=frame)
                position = vision.analyze_position(offset_x=offset_x)
                print(f"   语音播报: 找到你了，{position}")
                
                if self.center_and_wait_mode:
                    print("   🎯 正在执行人物居中...")
                    self.center_person(offset_x, offset_y)
                    time.sleep(2.0)
                
                return True

        print("\n" + "=" * 60)
        print(f"⚠️  {num_steps}步扫描未发现人")
        return False

    def human_steps_fast(self, vision: "VisionAnalyzer", detect_interval: float = 0.1) -> bool:
        """快速扫描找人（左极限 → 右极限）"""
        import time

        print(f"\n🚀 启动快速扫描...")
        print("=" * 60)

        # 预检查
        frame = self.get_frame()
        if frame is not None and vision.check_person(frame=frame)[0]:
            print("✅ 当前位置已有人！")
            offset_x, offset_y = vision.get_person_offset(frame=frame)
            position = vision.analyze_position(offset_x=offset_x)
            print(f"   语音播报: 找到你了，{position}")
            self.center_person(offset_x, offset_y)
            return True

        # 步进式转到左极限
        print("🎯 步进转到左极限...")
        step_duration = 0.15
        max_steps = 25
        left_steps = 0

        for i in range(max_steps):
            success = self.ptz_turn("left", step_duration)
            if not success:
                print(f"🚧 到达左极限")
                break
            left_steps += 1

            if i % 2 == 0:
                try:
                    frame = self.get_frame()
                    if frame is not None and vision.check_person(frame=frame)[0]:
                        print(f"✅ 左转途中发现人！")
                        offset_x, offset_y = vision.get_person_offset(frame=frame)
                        position = vision.analyze_position(offset_x=offset_x)
                        print(f"   语音播报: 找到你了，{position}")
                        self.center_person(offset_x, offset_y)
                        return True
                except Exception as e:
                    print(f"⚠️  检测异常: {e}")

        print(f"✅ 左极限定位完成")

        # 步进式向右扫描
        print(f"\n📍 开始向右扫描...")
        print("-" * 40)

        right_steps = 0
        max_right_steps = 80

        for i in range(max_right_steps):
            success = self.ptz_turn("right", step_duration)
            if not success:
                print(f"🚧 到达右极限，扫描完成")
                break

            right_steps += 1

            if i % 2 == 0:
                try:
                    frame = self.get_frame()
                    if frame is not None and vision.check_person(frame=frame)[0]:
                        current_angle = right_steps * step_duration * ROTATION_SPEED
                        print(f"\n✅ 右转途中发现人！")
                        offset_x, offset_y = vision.get_person_offset(frame=frame)
                        position = vision.analyze_position(offset_x=offset_x)
                        print(f"   语音播报: 找到你了，{position}")
                        self.center_person(offset_x, offset_y)
                        return True
                except Exception as e:
                    print(f"⚠️  检测异常: {e}")

        print(f"\n⚠️ 水平扫描完成，未发现人")
        return False

    def human_steps_smart(self, vision: "VisionAnalyzer") -> bool:
        """智能惯性扫描找人"""
        import time

        memory = self.tracking_memory
        step_duration = 0.15

        print(f"\n🧠 启动智能惯性扫描...")
        print("=" * 60)

        if memory.is_fresh() and memory.confidence > 0.3:
            print(f"📍 有有效记忆: 最后角度 {memory.last_angle:.0f}°")
            predicted_dir = memory.get_predicted_direction()

            if predicted_dir == "right":
                print(f"\n🎯 策略: 人向右走，优先向右扫描 →")
                return self._scan_with_fallback(vision, "right", memory.last_angle)
            elif predicted_dir == "left":
                print(f"\n🎯 策略: 人向左走，优先向左扫描 ←")
                return self._scan_with_fallback(vision, "left", memory.last_angle)

        print("📭 无有效记忆或记忆过期")
        print("🎯 策略: 完整扫描")
        return self.human_steps_fast(vision)

    def _scan_with_fallback(self, vision: "VisionAnalyzer", priority_dir: str, start_angle: float) -> bool:
        """优先方向扫描，未找到则回退反向扫描"""
        step_duration = 0.15
        max_steps = 40

        print(f"\n🔍 优先向{priority_dir}扫描...")
        print("-" * 40)

        for i in range(max_steps):
            success = self.ptz_turn(priority_dir, step_duration)
            if not success:
                break

            if i % 2 == 0:
                try:
                    frame = self.get_frame()
                    if frame is not None and vision.check_person(frame=frame)[0]:
                        print(f"\n✅ {priority_dir}向扫描发现人！")
                        offset_x, offset_y = vision.get_person_offset(frame=frame)
                        position = vision.analyze_position(offset_x=offset_x)
                        print(f"   语音播报: 找到你了，{position}")
                        self.center_person(offset_x, offset_y)
                        return True
                except Exception as e:
                    print(f"⚠️  检测异常: {e}")

        # 未找到 -> 回退反向扫描
        opposite = "left" if priority_dir == "right" else "right"
        print(f"\n↩️ 优先方向未找到，回退向{opposite}扫描...")
        
        return False
