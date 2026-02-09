#!/usr/bin/env python3
"""
视觉检测模块 - YOLOPersonDetector 和 VisionAnalyzer
重点：YOLO 模型强制使用 device='cuda'
"""

import sys
import threading
from typing import List, Tuple, Optional

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLOv8 not available")

from .config import (
    YOLO_MODEL, YOLO_CONFIDENCE,
    CAPTURE_WIDTH, CAPTURE_HEIGHT
)


class YOLOPersonDetector:
    """YOLOv8 人物检测器 - 强制使用 CUDA GPU"""

    def __init__(self):
        self.model = None
        self._lock = threading.Lock()
        self.device = 'cuda'  # 强制使用 CUDA
        
        if YOLO_AVAILABLE:
            try:
                model_name = f"{YOLO_MODEL}.pt"
                # 强制使用 CUDA 设备加载模型
                self.model = YOLO(model_name, verbose=False)
                # 预热模型，使用 CUDA
                self.model.to(self.device)
                print(f"✅ YOLOv8 loaded ({YOLO_MODEL}) on device: {self.device}")
            except Exception as e:
                print(f"⚠️  YOLOv8 load failed: {e}")
                print(f"   请确保 CUDA 可用: nvidia-smi")

    def check_person(self, image_path: str = None, frame: np.ndarray = None) -> Tuple[bool, str]:
        """检测是否有人（支持图片路径或 numpy 数组）"""
        if self.model is None:
            return False, "YOLO not available"

        with self._lock:
            try:
                input_data = frame if frame is not None else image_path
                if input_data is None:
                    return False, "No input provided"

                # 强制使用 CUDA 设备进行推理
                results = self.model(input_data, verbose=False, device=self.device)
                for result in results:
                    boxes = result.boxes
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            if int(box.cls[0]) == 0:
                                conf = float(box.conf[0])
                                if conf > YOLO_CONFIDENCE:
                                    return True, f"Person detected (conf: {conf:.2f})"
                return False, "No person detected"
            except Exception as e:
                return False, f"Error: {e}"

    def get_person_boxes(self, image_path: str = None, frame: np.ndarray = None) -> List[dict]:
        """返回人物位置框列表（支持图片路径或 numpy 数组）"""
        if self.model is None:
            return []

        with self._lock:
            try:
                input_data = frame if frame is not None else image_path
                if input_data is None:
                    return []
                
                # 强制使用 CUDA 设备进行推理
                results = self.model(input_data, verbose=False, device=self.device)
                boxes = []
                for result in results:
                    for box in result.boxes:
                        if int(box.cls[0]) == 0:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            boxes.append({
                                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                'conf': float(box.conf[0]),
                                'center_x': (x1 + x2) / 2,
                                'center_y': (y1 + y2) / 2
                            })
                return boxes
            except Exception as e:
                print(f"⚠️  get_person_boxes error: {e}")
                return []


class VisionAnalyzer:
    """视觉分析器 - YOLOv8 检测"""

    def __init__(self):
        self.yolo = YOLOPersonDetector()

    def check_person(self, image_path: str = None, frame: np.ndarray = None) -> Tuple[bool, str]:
        """检查是否有人（支持图片路径或 numpy 数组）"""
        if self.yolo.model is None:
            print("❌ YOLO not available, exiting...")
            sys.exit(1)
        
        has_person, info = self.yolo.check_person(image_path=image_path, frame=frame)
        if has_person:
            return True, f"YOLO: {info}"
        else:
            return False, f"YOLO: {info}"

    def get_person_offset(self, image_path: str = None, frame: np.ndarray = None) -> Tuple[float, float]:
        """获取人物相对于画面中心的偏移"""
        try:
            boxes = self.yolo.get_person_boxes(image_path=image_path, frame=frame)
            if not boxes:
                return 0.0, 0.0
            
            best_box = max(boxes, key=lambda b: b['conf'])
            
            center_x, center_y = CAPTURE_WIDTH / 2, CAPTURE_HEIGHT / 2
            person_cx = float(best_box['center_x'])
            person_cy = float(best_box['center_y'])
            
            offset_x = (person_cx - center_x) / center_x
            offset_y = (person_cy - center_y) / center_y
            
            return offset_x, offset_y
        except Exception as e:
            print(f"⚠️  get_person_offset 异常: {e}")
            return 0.0, 0.0

    def analyze_activity(self, image_path: str) -> str:
        """分析人在做什么（保留旧接口）"""
        return "Person detected"

    def analyze_position(self, image_path: str = None, frame: np.ndarray = None, offset_x: float = None) -> str:
        """分析人物在画面中的位置"""
        if offset_x is None:
            offset_x, _ = self.get_person_offset(image_path=image_path, frame=frame)
        
        if offset_x < -0.3:
            return "left side"
        elif offset_x > 0.3:
            return "right side"
        else:
            return "center"
