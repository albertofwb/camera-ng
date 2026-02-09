#!/usr/bin/env python3
"""
视频流模块 - VideoStream 类
"""

import os
import threading
import time
from collections import deque
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("❌ OpenCV not available")

from .config import CAPTURE_WIDTH, CAPTURE_HEIGHT, STREAM_BUFFER_SIZE


class VideoStream:
    """
    异步视频流读取器
    使用独立线程持续读取 RTSP 流，提供最新帧
    支持 GPU 硬解 (NVIDIA CUDA)
    """

    def __init__(self, rtsp_url: str, width: int = 640, height: int = 360,
                 buffer_size: int = 3, use_gpu: bool = False):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.buffer_size = buffer_size
        self.use_gpu = use_gpu

        self.cap = None
        self.gpu_mode = False
        self.frame_buffer = deque(maxlen=buffer_size)
        self.latest_frame = None
        self.frame_count = 0
        self.fps = 0

        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._last_frame_time = 0

    def start(self) -> bool:
        """启动视频流"""
        if not CV2_AVAILABLE:
            print("❌ OpenCV not available")
            return False

        # 配置 OpenCV 优化
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'

        # 尝试 GPU 解码
        if self.use_gpu:
            try:
                # 检查是否有 CUDA 支持
                if hasattr(cv2, 'cudacodec'):
                    self.cap = cv2.cudacodec.createVideoReader(self.rtsp_url)
                    self.gpu_mode = True
                    print(f"✅ GPU 硬解已启用 (CUDA)")
                else:
                    print("⚠️ OpenCV 未启用 CUDA 支持，回退到 CPU 解码")
                    self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                    self.gpu_mode = False
            except Exception as e:
                print(f"⚠️ GPU 解码初始化失败: {e}")
                print("   回退到 CPU 解码")
                self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                self.gpu_mode = False
        else:
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            self.gpu_mode = False

        if not self.gpu_mode:
            # CPU 模式：检查 VideoCapture 是否成功
            if not self.cap.isOpened():
                print(f"❌ 无法打开视频流: {self.rtsp_url}")
                return False

            # 设置分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            # 尝试设置缓冲区大小
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # 读取第一帧确认
            ret, frame = self.cap.read()
            if not ret:
                print("❌ 无法读取视频流第一帧")
                return False

            # 调整大小
            frame = cv2.resize(frame, (self.width, self.height))
            self.latest_frame = frame
        else:
            # GPU 模式：读取第一帧确认
            try:
                ret, gpu_frame = self.cap.nextFrame()
                if not ret:
                    print("❌ GPU 模式无法读取视频流第一帧")
                    return False
                # 下载到 CPU
                frame = gpu_frame.download()
                frame = cv2.resize(frame, (self.width, self.height))
                self.latest_frame = frame
            except Exception as e:
                print(f"❌ GPU 模式初始化失败: {e}")
                return False

        # 启动读取线程
        self._running = True
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

        mode_str = "GPU" if self.gpu_mode else "CPU"
        print(f"✅ VideoStream started ({mode_str}): {self.width}x{self.height}")
        return True
    
    def _update(self):
        """后台线程：持续读取帧"""
        frame_times = deque(maxlen=30)

        while self._running:
            if self.cap is None:
                time.sleep(0.01)
                continue

            # GPU 模式
            if self.gpu_mode:
                try:
                    ret, gpu_frame = self.cap.nextFrame()
                    if not ret:
                        continue
                    # 下载到 CPU 并调整大小
                    frame = gpu_frame.download()
                    frame = cv2.resize(frame, (self.width, self.height))
                except Exception as e:
                    print(f"⚠️ GPU 解码错误: {e}")
                    continue
            else:
                # CPU 模式
                if not self.cap.isOpened():
                    time.sleep(0.01)
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    continue

                # 调整大小
                frame = cv2.resize(frame, (self.width, self.height))

            # 更新最新帧
            with self._lock:
                self.latest_frame = frame.copy()
                self.frame_count += 1

            # 计算 FPS
            current_time = time.time()
            if self._last_frame_time > 0:
                frame_times.append(current_time - self._last_frame_time)
                if len(frame_times) >= 10:
                    self.fps = len(frame_times) / sum(frame_times)
            self._last_frame_time = current_time
    
    def get_frame(self) -> Optional[np.ndarray]:
        """获取最新帧"""
        with self._lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """兼容 cv2.VideoCapture 接口"""
        frame = self.get_frame()
        return frame is not None, frame
    
    def stop(self):
        """停止视频流"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.cap:
            if not self.gpu_mode:
                self.cap.release()
            self.cap = None
        self.gpu_mode = False
        print("✅ VideoStream stopped")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
