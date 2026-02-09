#!/usr/bin/env python3
"""
camera_ng 包 - Mooer 智能视角控制系统
"""

from .config import (
    CONFIG, load_config, get_config,
    CAMERA_RTSP, DEVICE_SERIAL, ACCESS_TOKEN,
    CAPTURE_WIDTH, CAPTURE_HEIGHT,
    PHOTO_WIDTH, PHOTO_HEIGHT, PHOTO_QUALITY,
    DEFAULT_NUM_STEPS, DEFAULT_TOTAL_ANGLE,
    YOLO_MODEL, YOLO_CONFIDENCE,
    ROTATION_SPEED, TRACK_CHECK_INTERVAL, DETECTION_INTERVAL,
    TRACKER_MAX_AGE, TRACKER_MIN_HITS,
    LOCK_FILE
)
from .stream import VideoStream
from .vision import YOLOPersonDetector, VisionAnalyzer, HandRaiseDetector
from .tracking import (
    Detection, Track, SORTTracker,
    PersonTracker, TrackingMemory
)
from .controller import CameraController

__version__ = "2.0.0"
__all__ = [
    # Config
    "CONFIG", "load_config", "get_config",
    "CAMERA_RTSP", "DEVICE_SERIAL", "ACCESS_TOKEN",
    "CAPTURE_WIDTH", "CAPTURE_HEIGHT",
    "PHOTO_WIDTH", "PHOTO_HEIGHT", "PHOTO_QUALITY",
    "DEFAULT_NUM_STEPS", "DEFAULT_TOTAL_ANGLE",
    "YOLO_MODEL", "YOLO_CONFIDENCE",
    "ROTATION_SPEED", "TRACK_CHECK_INTERVAL", "DETECTION_INTERVAL",
    "TRACKER_MAX_AGE", "TRACKER_MIN_HITS",
    "LOCK_FILE",
    # Stream
    "VideoStream",
    # Vision
    "YOLOPersonDetector", "VisionAnalyzer", "HandRaiseDetector",
    # Tracking
    "Detection", "Track", "SORTTracker",
    "PersonTracker", "TrackingMemory",
    # Controller
    "CameraController",
]
