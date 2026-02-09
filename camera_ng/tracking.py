#!/usr/bin/env python3
"""
目标跟踪模块 - SORTTracker, PersonTracker, TrackingMemory
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from .config import (
    YOLO_MODEL, YOLO_CONFIDENCE,
    TRACKER_MAX_AGE, TRACKER_MIN_HITS,
    DETECTION_INTERVAL, IOU_THRESHOLD
)


@dataclass
class Detection:
    """检测结果数据类"""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    conf: float
    class_id: int = 0
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, 
                (self.bbox[1] + self.bbox[3]) / 2)
    
    @property
    def area(self) -> float:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


@dataclass  
class Track:
    """跟踪目标数据类"""
    id: int
    bbox: np.ndarray
    conf: float
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    
    def update(self, detection: Detection):
        """更新跟踪位置"""
        self.bbox = detection.bbox
        self.conf = detection.conf
        self.hits += 1
        self.time_since_update = 0
        
    def predict(self):
        """预测下一帧位置"""
        self.time_since_update += 1
        self.age += 1


class SORTTracker:
    """
    简化版 SORT 跟踪器
    基于 IOU 匹配的在线多目标跟踪
    """
    
    def __init__(self, max_age: int = 15, min_hits: int = 2, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[Track] = []
        self._next_id = 1
        self.frame_count = 0
    
    @staticmethod
    def iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """计算两个边界框的 IOU"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = bbox1_area + bbox2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def update(self, detections: List[Detection]) -> List[Track]:
        """更新跟踪器"""
        self.frame_count += 1
        
        # 预测现有跟踪位置
        for track in self.tracks:
            track.predict()
        
        # 使用贪心匹配
        matched_tracks = []
        matched_dets = set()
        
        # 计算 IOU 矩阵
        if self.tracks and detections:
            iou_matrix = np.zeros((len(self.tracks), len(detections)))
            for i, track in enumerate(self.tracks):
                for j, det in enumerate(detections):
                    iou_matrix[i, j] = self.iou(track.bbox, det.bbox)
            
            # 贪心匹配
            for _ in range(min(len(self.tracks), len(detections))):
                if iou_matrix.size == 0:
                    break
                
                max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                max_iou = iou_matrix[max_idx]
                
                if max_iou < self.iou_threshold:
                    break
                
                track_idx, det_idx = max_idx
                self.tracks[track_idx].update(detections[det_idx])
                matched_tracks.append(self.tracks[track_idx])
                matched_dets.add(det_idx)
                
                # 移除已匹配
                iou_matrix[track_idx, :] = -1
                iou_matrix[:, det_idx] = -1
        
        # 未匹配的检测 -> 新建跟踪
        for i, det in enumerate(detections):
            if i not in matched_dets:
                new_track = Track(
                    id=self._next_id,
                    bbox=det.bbox.copy(),
                    conf=det.conf,
                    hits=1
                )
                self._next_id += 1
                self.tracks.append(new_track)
                matched_tracks.append(new_track)
        
        # 移除丢失的跟踪
        self.tracks = [t for t in self.tracks 
                      if t.time_since_update <= self.max_age]
        
        # 返回确认的跟踪
        confirmed_tracks = [t for t in matched_tracks 
                           if t.hits >= self.min_hits or t.age < self.min_hits]
        
        return confirmed_tracks


class PersonTracker:
    """
    人物跟踪器封装
    结合 YOLO 检测和 SORT 跟踪，实现高效实时跟踪
    """
    
    def __init__(self, yolo_model: str = "yolov8n", 
                 confidence: float = 0.5,
                 detection_interval: int = 3,
                 max_age: int = 5,
                 min_hits: int = 2):
        self.confidence = confidence
        self.detection_interval = detection_interval
        self.frame_count = 0
        
        # 初始化 YOLO - 强制使用 CUDA
        self.model = None
        self.device = 'cuda'
        if YOLO_AVAILABLE:
            try:
                model_name = f"{yolo_model}.pt"
                self.model = YOLO(model_name, verbose=False)
                self.model.to(self.device)
                print(f"✅ YOLOv8 loaded ({yolo_model}) on {self.device}")
            except Exception as e:
                print(f"⚠️  YOLOv8 load failed: {e}")
        
        # 初始化 SORT 跟踪器
        self.tracker = SORTTracker(max_age=max_age, min_hits=min_hits)
        self.last_tracks = []
        
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """使用 YOLO 检测人物 - 强制使用 CUDA"""
        detections = []
        
        if self.model is None:
            return detections
        
        try:
            results = self.model(frame, verbose=False, device=self.device)
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        if int(box.cls[0]) == 0:  # person class
                            conf = float(box.conf[0])
                            if conf > self.confidence:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                detections.append(Detection(
                                    bbox=np.array([x1, y1, x2, y2]),
                                    conf=conf
                                ))
        except Exception as e:
            print(f"⚠️  Detection error: {e}")
        
        return detections
    
    def update(self, frame: np.ndarray, force_detect: bool = False) -> List[Track]:
        """更新跟踪器"""
        self.frame_count += 1
        
        # 按间隔进行 YOLO 检测
        detections = []
        if force_detect or (self.frame_count % self.detection_interval == 0):
            detections = self.detect(frame)
        
        # 更新跟踪器
        tracks = self.tracker.update(detections)
        self.last_tracks = tracks
        
        return tracks
    
    def get_main_person(self) -> Optional[Track]:
        """获取主要跟踪人物（面积最大）"""
        if not self.last_tracks:
            return None
        
        return max(self.last_tracks, 
                  key=lambda t: (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1]))
    
    def draw_tracks(self, frame: np.ndarray) -> np.ndarray:
        """在帧上绘制跟踪结果"""
        try:
            import cv2
            for track in self.last_tracks:
                x1, y1, x2, y2 = track.bbox.astype(int)
                
                # 绘制边界框
                color = (0, 255, 0) if track.time_since_update == 0 else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # 绘制 ID
                label = f"ID:{track.id} {track.conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        except ImportError:
            pass
        
        return frame


@dataclass
class TrackingMemory:
    """追踪记忆 - 记录人物运动轨迹用于预测"""
    last_angle: float = 0.0
    last_tilt: float = 0.0
    last_seen_time: float = 0.0
    motion_direction: str = "unknown"
    vertical_motion: str = "unknown"
    confidence: float = 0.0
    history: deque = None
    tilt_history: deque = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = deque(maxlen=5)
        if self.tilt_history is None:
            self.tilt_history = deque(maxlen=3)
    
    def update(self, angle: float, tilt: float = 0.0, timestamp: float = None):
        """更新位置记录"""
        if timestamp is None:
            timestamp = time.time()
        
        # 计算水平运动方向
        if self.history:
            prev_angle = self.history[-1]
            diff = (angle - prev_angle) % 360
            if diff > 180:
                diff = diff - 360
            
            if diff > 2:
                self.motion_direction = "right"
            elif diff < -2:
                self.motion_direction = "left"
        
        # 计算垂直运动方向
        if self.tilt_history:
            prev_tilt = self.tilt_history[-1]
            tilt_diff = tilt - prev_tilt
            if tilt_diff > 3:
                self.vertical_motion = "up"
            elif tilt_diff < -3:
                self.vertical_motion = "down"
        
        self.history.append(angle)
        self.tilt_history.append(tilt)
        self.last_angle = angle
        self.last_tilt = tilt
        self.last_seen_time = timestamp
        self.confidence = min(len(self.history) / 3, 1.0)
    
    def get_predicted_direction(self) -> str:
        """获取预测搜索方向"""
        if self.confidence < 0.3:
            return "unknown"
        return self.motion_direction
    
    def get_vertical_prediction(self) -> str:
        """获取垂直预测方向"""
        if len(self.tilt_history) < 2:
            return "unknown"
        return self.vertical_motion
    
    def is_fresh(self, max_age: float = 10.0) -> bool:
        """记忆是否新鲜"""
        return (time.time() - self.last_seen_time) < max_age
    
    def reset(self):
        """重置记忆"""
        self.last_angle = 0.0
        self.last_tilt = 0.0
        self.last_seen_time = 0.0
        self.motion_direction = "unknown"
        self.vertical_motion = "unknown"
        self.confidence = 0.0
        self.history.clear()
        self.tilt_history.clear()
