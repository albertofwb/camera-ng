#!/usr/bin/env python3
"""
手势和录像管理模块 - 重构后的清晰实现
"""

import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Optional

import numpy as np

from . import CameraController, XiaoxiaoTTS, HandRaiseDetector


class HandGesture(Enum):
    """手势类型"""
    NONE = auto()
    LEFT_HAND = auto()
    RIGHT_HAND = auto()


@dataclass
class GestureEvent:
    """手势事件"""
    gesture: HandGesture
    reason: str
    timestamp: float


class HandGestureHandler:
    """
    手势检测状态机
    管理连续帧检测、冷却时间、触发逻辑
    """

    def __init__(
        self,
        detector: HandRaiseDetector,
        confirm_frames: int = 2,
        release_frames: int = 3,
        cooldown_sec: float = 1.0,
        log_interval_sec: float = 1.0,
        detect_interval_sec: float = 0.25,
    ):
        self.detector = detector
        self.confirm_frames = confirm_frames
        self.release_frames = release_frames
        self.cooldown_sec = cooldown_sec
        self.log_interval_sec = log_interval_sec
        self.detect_interval_sec = detect_interval_sec

        # 状态
        self._consecutive_count = 0
        self._release_count = 0
        self._armed = True
        self._last_trigger_time = 0.0
        self._last_log_time = 0.0
        self._last_detect_time = -1e9
        self._last_hand_raised = False
        self._last_reason = ""

    def update(self, frame: np.ndarray, current_time: float) -> Optional[GestureEvent]:
        """
        更新状态机，返回触发的手势事件（如果有）
        """
        if current_time - self._last_detect_time < self.detect_interval_sec:
            return None

        self._last_detect_time = current_time
        hand_raised, reason = self.detector.get_hand_raise_state(frame)
        self._last_hand_raised = hand_raised
        self._last_reason = reason

        # 状态机转换
        if hand_raised:
            self._consecutive_count += 1
            self._release_count = 0
        else:
            self._consecutive_count = 0
            self._release_count += 1
            if self._release_count >= self.release_frames:
                self._armed = True  # 连续放下后才允许下次触发

        # 检查是否满足触发条件
        can_trigger = (
            self._armed
            and hand_raised
            and self._consecutive_count >= self.confirm_frames
            and (current_time - self._last_trigger_time) >= self.cooldown_sec
        )

        if can_trigger:
            gesture = self._classify_gesture(reason)
            if gesture != HandGesture.NONE:
                self._armed = False
                self._consecutive_count = 0
                self._last_trigger_time = current_time
                return GestureEvent(gesture=gesture, reason=reason, timestamp=current_time)

        return None

    def _classify_gesture(self, reason: str) -> HandGesture:
        """根据检测原因分类手势"""
        if "left" in reason:
            return HandGesture.LEFT_HAND
        elif "right" in reason:
            return HandGesture.RIGHT_HAND
        return HandGesture.NONE

    def should_log(self, current_time: float) -> bool:
        """检查是否应该输出日志"""
        if current_time - self._last_log_time >= self.log_interval_sec:
            self._last_log_time = current_time
            return True
        return False

    def get_status(self) -> tuple[bool, str, int]:
        """获取当前状态用于日志显示"""
        return self._last_hand_raised, self._last_reason, self._consecutive_count

    def reset(self):
        """重置状态（如丢失目标时）"""
        self._consecutive_count = 0
        self._release_count = 0
        self._armed = True


class RecordingManager:
    """
    录像管理器
    管理录像的生命周期：开始、停止、自动根据目标状态控制
    """

    def __init__(
        self,
        rtsp_url: str,
        tts: Optional[XiaoxiaoTTS] = None,
        toggle_cooldown_sec: float = 1.5,
        auto_start_on_person_found: bool = False,
    ):
        self.rtsp_url = rtsp_url
        self.tts = tts
        self.toggle_cooldown_sec = toggle_cooldown_sec
        self.auto_start_on_person_found = auto_start_on_person_found

        self._proc: Optional[subprocess.Popen[str]] = None
        self._output_path: Optional[str] = None
        self._last_toggle_time = 0.0
        self._is_recording = False

    def _start_recording(self) -> bool:
        """启动录像"""
        import os

        try:
            output_dir = os.path.expanduser("~/Desktop/capture")
            os.makedirs(output_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"{timestamp}.mp4")

            cmd = [
                "ffmpeg",
                "-fflags", "+genpts",
                "-use_wallclock_as_timestamps", "1",
                "-rtsp_transport", "tcp",
                "-i", self.rtsp_url,
                "-map", "0:v:0",
                "-map", "0:a?",
                "-vf", "fps=30",
                "-vsync", "cfr",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "20",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "128k",
                "-avoid_negative_ts", "make_zero",
                "-movflags", "+faststart",
                "-y", output_path,
            ]

            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            self._output_path = output_path
            self._is_recording = True
            print(f"🎬 开始录像文件: {output_path}")
            return True
        except Exception as e:
            print(f"❌ 启动录像失败: {e}")
            return False

    def _stop_recording(self):
        """停止录像"""
        if self._proc is None:
            return

        try:
            if self._proc.poll() is None:
                if self._proc.stdin is not None:
                    self._proc.stdin.write("q\n")
                    self._proc.stdin.flush()
                self._proc.wait(timeout=2.0)
        except Exception:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=1.5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        finally:
            if self._output_path:
                print(f"✅ 停止录像: {self._output_path}")
            self._proc = None
            self._output_path = None
            self._is_recording = False

    def toggle(self, current_time: float) -> bool:
        """
        切换录像状态（手动触发）
        返回是否成功执行切换
        """
        if current_time - self._last_toggle_time < self.toggle_cooldown_sec:
            return False  # 冷却中

        self._last_toggle_time = current_time

        if self._is_recording:
            self._stop_recording()
            self._broadcast("停止录像")
        else:
            if self._start_recording():
                self._broadcast("开始录像")

        return True

    def on_person_found(self):
        """当找到目标时自动开始录像"""
        if self.auto_start_on_person_found and not self._is_recording:
            if self._start_recording():
                self._broadcast("找到目标，开始录像")

    def on_person_lost(self):
        """当丢失目标时自动停止录像"""
        if self._is_recording:
            self._stop_recording()
            self._broadcast("丢失目标，停止录像")

    def _broadcast(self, text: str):
        """语音播报"""
        tts = self.tts
        if tts is None or not tts.is_available():
            return

        def _worker():
            try:
                if tts.playback(text):
                    print(f"🔈 已播报: {text}")
                else:
                    print(f"⚠️ 本机未播报: {text}")
            except Exception:
                print(f"⚠️ 本机播报异常: {text}")

        threading.Thread(target=_worker, daemon=True).start()

    def cleanup(self):
        """清理资源"""
        self._stop_recording()

    @property
    def is_recording(self) -> bool:
        return self._is_recording


class SmartShotWorker:
    """
    Smart-Shot 后台任务处理器
    处理拍照和语音发送的队列任务
    """

    def __init__(
        self,
        camera: CameraController,
        tts: Optional[XiaoxiaoTTS],
        telegram_target: str,
        max_queue_size: int = 3,
        ack_cooldown_sec: float = 1.2,
        task_callback: Optional[Callable] = None,
    ):
        self.camera = camera
        self.tts = tts
        self.telegram_target = telegram_target
        self._task_callback = task_callback
        self.ack_cooldown_sec = ack_cooldown_sec
        self._last_ack_time = 0.0
        self.queue: queue.Queue[tuple[str, str]] = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None

    def start(self) -> threading.Thread:
        """启动后台 worker"""
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        return self.worker_thread

    def _worker_loop(self):
        """后台任务循环"""
        while not self.stop_event.is_set():
            try:
                hand_text, hand_reason = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self._process_task(hand_text, hand_reason)
            finally:
                self.queue.task_done()

    def _process_task(self, hand_text: str, hand_reason: str):
        """处理单个任务 - 使用回调避免循环导入"""
        if self._task_callback:
            self._task_callback(self.camera, hand_text, hand_reason, self.tts)

    def submit(self, hand_text: str, hand_reason: str) -> bool:
        """
        提交任务到队列
        如果队列满，丢弃最旧任务
        """
        # 队列满时丢弃最旧任务
        if self.queue.full():
            try:
                _ = self.queue.get_nowait()
                self.queue.task_done()
                print("🗑️ Smart-Shot 队列已满，丢弃最旧任务")
            except queue.Empty:
                pass

        try:
            self.queue.put_nowait((hand_text, hand_reason))
            # 播放收到提示
            now = time.time()
            if (
                self.tts is not None
                and self.tts.is_available()
                and (now - self._last_ack_time) >= self.ack_cooldown_sec
            ):
                if self.tts.playback("收到"):
                    print("🔈 已本机播报: 收到")
                self._last_ack_time = now
            return True
        except queue.Full:
            print("⏳ Smart-Shot 队列拥塞，跳过本次触发")
            return False

    def stop(self):
        """停止 worker"""
        self.stop_event.set()
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=0.5)
