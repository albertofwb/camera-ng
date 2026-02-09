# Camera-NG

Mooer 智能摄像头追踪与抓拍系统，围绕「实时找人 -> 云台跟随 -> 手势触发动作 -> 自动消息推送」设计。

技术实现、架构与依赖说明请看 `TECHNICAL.md`。

## 当前已实现能力

- 实时目标跟踪：`track` 模式结合 YOLOv8 + SORT，对人物持续跟随并动态居中。
- 智能扫描找人：支持完整步进扫描、快速扫描、带运动记忆的惯性扫描。
- Smart-Shot 手势联动：右手抬起触发抓拍+Telegram 发送；左手抬起切换录像开关。
- 录像管理：`ffmpeg` 后台录制，输出到 `~/Desktop/capture/<timestamp>.mp4`，丢失目标时自动停录。
- 语音能力：支持本地 TTS 播报（晓晓），并可将语音文件发送到 Telegram。
- 高质量抓拍：支持从 RTSP 原流单帧抓图，支持独立照片分辨率/质量配置。
- 单实例保护：使用 `/tmp/mooer_camera.lock` 防止多进程同时控制同一云台。

## 命令速查

```bash
# 显示帮助
python3 -m camera_ng -h

# 传统扫描找人
python3 -m camera_ng human

# 实时跟踪（可加 -g 启用 GPU 解码）
python3 -m camera_ng track -g

# Smart-Shot：跟踪 + 手势触发抓拍/录像
python3 -m camera_ng smart-shot -g

# 快速高灵敏模式（track / smart-shot 都支持）
python3 -m camera_ng smart-shot -g -quick

# 找人 -> 居中 -> 抓拍并发送
python3 -m camera_ng shot

# 临时指定云台转速（度/秒）
python3 -m camera_ng track --speed 30

# 调用外部校准脚本
python3 -m camera_ng calibrate
```

## 快速开始

1. 准备配置文件（`camera.rtsp_url` / `camera.device_serial` / `camera.access_token` 为必填）。
2. 运行 `python3 -m camera_ng track -g` 验证实时跟踪。
3. 运行 `python3 -m camera_ng smart-shot -g` 体验手势联动抓拍与录像。

配置模板、搜索路径、依赖安装、模块架构请参考 `TECHNICAL.md`。
