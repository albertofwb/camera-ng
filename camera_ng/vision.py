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


class HandRaiseDetector:
    """右手抬起手势检测器（基于 YOLOv8 Pose）"""

    LEFT_SHOULDER_IDX = 5
    LEFT_WRIST_IDX = 9
    RIGHT_SHOULDER_IDX = 6
    RIGHT_WRIST_IDX = 10

    def __init__(
        self,
        model_name: str = "yolov8n-pose",
        infer_imgsz: int = 320,
        hand_side_mode: str = "auto",
    ):
        self.model = None
        self._lock = threading.Lock()
        self.device = "cuda"
        self.infer_imgsz = infer_imgsz
        mode = (hand_side_mode or "auto").lower()
        if mode not in ("auto", "normal", "swapped"):
            mode = "auto"
        self.hand_side_mode = mode
        self._auto_swap_score = 0
        self._swap_hands = mode == "swapped"

        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(f"{model_name}.pt", verbose=False)
                self.model.to(self.device)
                print(f"✅ Pose model loaded ({model_name}) on device: {self.device}")
            except Exception as e:
                print(f"⚠️  Pose model load failed: {e}")

    def get_hand_raise_state(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.35,
    ) -> tuple[bool, str]:
        """检测画面主人物是否抬手（右手/左手任一）"""
        if self.model is None:
            return False, "pose model unavailable"

        with self._lock:
            try:
                results = self.model(
                    frame,
                    verbose=False,
                    device=self.device,
                    imgsz=self.infer_imgsz,
                )
                if not results:
                    return False, "no pose result"

                result = results[0]
                keypoints = getattr(result, "keypoints", None)
                if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
                    return False, "no keypoints"

                boxes = result.boxes
                if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
                    return False, "no boxes"

                box_areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
                person_idx = int(box_areas.argmax().item())
                person_box = boxes.xyxy[person_idx]
                box_h = max(float(person_box[3] - person_box[1]), 1.0)
                margin = max(10.0, box_h * 0.08)

                points = keypoints.xy[person_idx]
                kpt_conf = keypoints.conf[person_idx] if keypoints.conf is not None else None

                left_shoulder = points[self.LEFT_SHOULDER_IDX]
                left_wrist = points[self.LEFT_WRIST_IDX]
                right_shoulder = points[self.RIGHT_SHOULDER_IDX]
                right_wrist = points[self.RIGHT_WRIST_IDX]

                if kpt_conf is not None:
                    right_valid = (
                        float(kpt_conf[self.RIGHT_SHOULDER_IDX]) >= conf_threshold
                        and float(kpt_conf[self.RIGHT_WRIST_IDX]) >= conf_threshold
                    )
                    left_valid = (
                        float(kpt_conf[self.LEFT_SHOULDER_IDX]) >= conf_threshold
                        and float(kpt_conf[self.LEFT_WRIST_IDX]) >= conf_threshold
                    )
                else:
                    right_valid = True
                    left_valid = True

                right_raised = False
                left_raised = False
                right_lift = float("-inf")
                left_lift = float("-inf")
                right_shoulder_y = 0.0
                right_wrist_y = 0.0
                left_shoulder_y = 0.0
                left_wrist_y = 0.0
                right_shoulder_x = float(right_shoulder[0])
                left_shoulder_x = float(left_shoulder[0])

                # auto 模式：根据肩膀左右顺序自动判断是否镜像画面
                if self.hand_side_mode == "auto" and right_valid and left_valid:
                    mirrored_sample = right_shoulder_x > left_shoulder_x
                    if mirrored_sample:
                        self._auto_swap_score = min(10, self._auto_swap_score + 1)
                    else:
                        self._auto_swap_score = max(-10, self._auto_swap_score - 1)
                    if self._auto_swap_score >= 3:
                        self._swap_hands = True
                    elif self._auto_swap_score <= -3:
                        self._swap_hands = False
                elif self.hand_side_mode == "normal":
                    self._swap_hands = False

                if right_valid:
                    right_shoulder_y = float(right_shoulder[1])
                    right_wrist_y = float(right_wrist[1])
                    right_lift = right_shoulder_y - right_wrist_y
                    right_raised = right_lift > margin

                if left_valid:
                    left_shoulder_y = float(left_shoulder[1])
                    left_wrist_y = float(left_wrist[1])
                    left_lift = left_shoulder_y - left_wrist_y
                    left_raised = left_lift > margin

                # 调试信息：返回详细的关节点状态
                debug_parts = []
                if right_valid:
                    debug_parts.append(f"RS={right_shoulder_y:.1f},RW={right_wrist_y:.1f}")
                else:
                    debug_parts.append("right_invalid")
                if left_valid:
                    debug_parts.append(f"LS={left_shoulder_y:.1f},LW={left_wrist_y:.1f}")
                else:
                    debug_parts.append("left_invalid")
                debug_parts.append(f"margin={margin:.1f}")
                if kpt_conf is not None:
                    debug_parts.append(f"conf={float(kpt_conf[self.RIGHT_WRIST_IDX]):.2f}")
                debug_parts.append(f"swap={int(self._swap_hands)}")
                debug_info = ",".join(debug_parts)

                if right_raised and left_raised:
                    # 双手同时抬起时，按抬起幅度更大的一侧判定；幅度接近时视为歧义不触发
                    if abs(right_lift - left_lift) < max(6.0, margin * 0.25):
                        return False, f"both hands raised ambiguous ({debug_info})"
                    if right_lift > left_lift:
                        right_raised, left_raised = True, False
                    else:
                        right_raised, left_raised = False, True

                right_label = "left" if self._swap_hands else "right"
                left_label = "right" if self._swap_hands else "left"

                if right_raised:
                    return True, f"{right_label} hand raised ({debug_info})"
                if left_raised:
                    return True, f"{left_label} hand raised ({debug_info})"

                return False, f"hands below threshold ({debug_info})"
            except Exception as e:
                print(f"⚠️  HandRaise detection error: {e}")
                return False, str(e)

    def is_right_hand_raised(self, frame: np.ndarray, conf_threshold: float = 0.2) -> bool:
        """兼容旧接口：任一手抬起返回 True"""
        raised, _ = self.get_hand_raise_state(frame, conf_threshold=conf_threshold)
        return raised
