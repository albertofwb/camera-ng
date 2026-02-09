# Camera-NG 技术文档

本文档聚焦实现细节与工程信息；功能介绍请看 `README.md`。

## 1. 配置

程序启动时会校验关键配置，缺失将直接退出。

### 配置文件搜索顺序

1. `./memory/camera-config.yaml`
2. `~/.openclaw/workspace/memory/camera-config.yaml`
3. `~/clawd/memory/camera-config.yaml`
4. `~/clawd/mooer-camera-locate/references/camera-config.yaml`
5. `~/.config/mooer-camera/config.yaml`
6. `~/.mooer-camera.yaml`

### 最小可用配置

```yaml
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
    photo:
      resolution:
        width: 1920
        height: 1080
      quality: 1
  ptz:
    rotation_speed: 28
```

## 2. 运行依赖

- Python `>=3.12`
- Python 包依赖：`numpy`、`opencv-python`、`PyYAML`、`ultralytics`
- 系统工具：`ffmpeg`（抓拍/录像）、`openclaw`（Telegram 推送）
- 可选工具：`~/.local/bin/xiaoxiao-tts`、`edge-playback`（语音播报与语音消息）
- 模型文件：`yolov8n.pt`、`yolov8n-pose.pt`

## 3. 命令与执行路径

- 包入口：`python3 -m camera_ng`（`camera_ng/__main__.py` -> `camera_ng/main.py`）
- 核心命令：`human`、`track`、`smart-shot`、`shot`、`calibrate`
- 运行保护：单实例锁文件 `/tmp/mooer_camera.lock`

## 4. 架构与模块职责

- `camera_ng/main.py`：CLI 参数解析、流程编排、Smart-Shot 主循环。
- `camera_ng/controller.py`：云台 PTZ 控制、抓拍、扫描策略（普通/快速/智能惯性）。
- `camera_ng/stream.py`：异步 RTSP 拉流，支持 CPU 或 OpenCV CUDA 解码。
- `camera_ng/vision.py`：YOLO 人体检测 + Pose 抬手检测。
- `camera_ng/tracking.py`：SORT 跟踪器、主目标选择、运动记忆。
- `camera_ng/handlers.py`：手势状态机、录像生命周期、Smart-Shot 队列 worker。
- `camera_ng/tts.py`：晓晓 TTS 合成/播放及常用提示音管理。

## 5. 推理与加速策略

- 检测/姿态模型默认 `device='cuda'`，优先在 GPU 推理。
- `-g` 选项主要作用于视频解码链路，优先尝试 OpenCV CUDA 硬解。
- 跟踪侧采用「YOLO 间隔检测 + SORT 连续跟踪」降低算力开销。
- `-quick` 模式提高检测频率并缩短冷却，换取更高响应速度。

## 6. Smart-Shot 关键实现点

- 右手抬起：提交抓拍任务到后台队列，并触发 Telegram 图片/语音发送。
- 左手抬起：切换录像开始/停止。
- 队列策略：队列满时丢弃最旧任务（drop-oldest），避免阻塞主跟踪循环。
- 丢失目标时：自动停止录像并重置手势状态。

## 7. 媒体与输出

- 抓拍：普通模式优先复用实时流帧；高质量模式走 RTSP 原流截图。
- 录像：`ffmpeg` 后台录制，输出 `~/Desktop/capture/<timestamp>.mp4`。
- 语音：优先播放本地缓存提示音，不存在时走实时 TTS 播放。

## 8. 备注

- 若 CUDA 不可用，模型加载或推理会失败，请先检查驱动与运行环境。
- 线上部署建议先用 `python3 -m camera_ng -h` 与 `track -g` 完成基础链路自检。
