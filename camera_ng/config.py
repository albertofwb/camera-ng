#!/usr/bin/env python3
"""
配置模块 - 处理 YAML 配置加载
"""

import os
from typing import Any, Dict

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# 配置文件搜索路径
CONFIG_PATHS = [
    os.path.join(os.getcwd(), "memory/camera-config.yaml"),
    os.path.expanduser("~/.openclaw/workspace/memory/camera-config.yaml"),
    os.path.expanduser("~/clawd/memory/camera-config.yaml"),
    os.path.expanduser("~/clawd/mooer-camera-locate/references/camera-config.yaml"),
    os.path.expanduser("~/.config/mooer-camera/config.yaml"),
    os.path.expanduser("~/.mooer-camera.yaml"),
]


def load_config() -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    if not YAML_AVAILABLE:
        print("⚠️  PyYAML not installed, using default config")
        return {}
    
    for path in CONFIG_PATHS:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    config = yaml.safe_load(f)
                    print(f"✅ Loaded config from {path}")
                    return config or {}
            except Exception as e:
                print(f"⚠️  Failed to load {path}: {e}")
    
    print("⚠️  No config file found, using defaults")
    return {}


def get_config(path: str, default: Any, config: Dict[str, Any] = None) -> Any:
    """从嵌套配置字典获取值"""
    if config is None:
        config = CONFIG
    keys = path.split('.')
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


# 全局配置对象
CONFIG = load_config()

# 文件锁配置
LOCK_FILE = "/tmp/mooer_camera.lock"

# 摄像头配置（敏感信息必须从 YAML 配置读取，无默认值）
CAMERA_RTSP = get_config('camera.rtsp_url', None)  # 必须从配置读取
CAMERA_RTSP_SUB = get_config('camera.rtsp_sub_url', None)
CAPTURE_SEEK_TIME = get_config('camera.capture.seek_time', "00:00:00.5")
DEVICE_SERIAL = get_config('camera.device_serial', None)  # 必须从配置读取
ACCESS_TOKEN = get_config('camera.access_token', None)  # 必须从配置读取

# 分辨率配置
CAPTURE_WIDTH = get_config('camera.capture.resolution.width', 640)
CAPTURE_HEIGHT = get_config('camera.capture.resolution.height', 360)
CAPTURE_QUALITY = get_config('camera.capture.quality', 2)

# 高质量抓拍配置（shot / smart-shot）
PHOTO_WIDTH = get_config('camera.capture.photo.resolution.width', None)
PHOTO_HEIGHT = get_config('camera.capture.photo.resolution.height', None)
PHOTO_QUALITY = get_config('camera.capture.photo.quality', 1)

# 默认参数
DEFAULT_NUM_STEPS = get_config('detection.scan.default_steps', 8)
DEFAULT_TOTAL_ANGLE = get_config('detection.scan.default_angle', 180)

# 模型配置
YOLO_MODEL = get_config('detection.model', "yolov8n")
YOLO_CONFIDENCE = get_config('detection.confidence_threshold', 0.5)

# 云台配置
ROTATION_SPEED = get_config('camera.ptz.rotation_speed', 28)
PTZ_SPEED = get_config('camera.ptz.speed', 2)
PTZ_FAST_SPEED = get_config('camera.ptz.speed_fast', 4)
LEFT_LIMIT_STEP_DURATION = get_config('camera.ptz.left_limit_step_duration', 0.5)
TURN_STABILIZE_TIME = get_config('camera.ptz.turn_stabilize_time', 0.5)

# 跟踪配置
TRACK_CHECK_INTERVAL = get_config('detection.track.check_interval', 5.0)
CENTER_THRESHOLD = get_config('detection.track.center_threshold', 0.1)
MAX_CENTER_ADJUST = get_config('detection.track.max_center_adjust', 30)

# 视频流配置
STREAM_FPS = get_config('stream.fps', 30)
STREAM_BUFFER_SIZE = get_config('stream.buffer_size', 3)
STREAM_LOW_LATENCY = get_config('stream.low_latency', False)

# 目标跟踪配置
TRACKER_MAX_AGE = get_config('tracker.max_age', 5)
TRACKER_MIN_HITS = get_config('tracker.min_hits', 3)
DETECTION_INTERVAL = get_config('tracker.detection_interval', 3)
IOU_THRESHOLD = get_config('tracker.iou_threshold', 0.3)

# 手势配置
# auto: 自动判断画面是否镜像并动态纠偏；normal: 不交换左右；swapped: 固定交换左右
HAND_SIDE_MODE = str(get_config('gesture.hand_side_mode', 'auto')).lower()

# 时间配置（秒）
DETECTION_SLEEP_TIME = 0.3
TRACK_SCAN_DELAY = 0.5

# 方向编码
DIR_LEFT_CODE = 2
DIR_RIGHT_CODE = 3
DIR_UP_CODE = 0
DIR_DOWN_CODE = 1

# PTZ 错误码
PTZ_ERROR_CODES = {
    "20003": "设备忙",
    "20006": "达到限位",
    "20007": "设备不支持此操作",
    "20008": "设备已离线",
}
