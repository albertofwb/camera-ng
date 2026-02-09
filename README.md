# 🦞 Camera-NG (Next Generation)

基于 **RTX 3060 Ti** 硬件加速的 Mooer 智能摄像头追踪与抓拍系统。

## 🏎️ 核心特性
- **GPU 驱动**：利用 CUDA 12.1 加速 YOLOv8 推理，识别帧率稳定在 **60 FPS**。
- **CPU 释放**：相比原版，CPU 负载降低约 80%+。
- **模块化架构**：代码解耦为 `vision`, `tracking`, `stream`, `controller` 等核心模块。
- **全自动快递**：集成 OpenClaw 命令行，抓拍后自动推送照片至 Telegram。

## 🚀 快速开始

### 1. 环境准备
确保已安装 `uv` 并处于项目根目录：
```bash
cd ~/camera-ng
# 激活环境
source .venv/bin/activate
```

### 2. 常用命令

#### 🏎️ 实时 GPU 追踪 (最推荐)
开启实时目标跟踪模式，模型完全运行在显卡上：
```bash
python3 -m camera_ng track -g
```

#### 🙋 Smart-Shot（右手抬起抓拍）
基于 track 实时模式，检测到抬手后自动抓拍并发送；右手抬起时额外发送中文语音问候：
```bash
python3 -m camera_ng smart-shot -g
```

#### 📸 智能居中抓拍
自动寻找人物，居中对齐，稳定 2 秒后抓拍并发送至手机：
```bash
python3 -m camera_ng shot
```

#### 🔍 传统扫描找人
仅执行水平+垂直（3层限制）扫描：
```bash
python3 -m camera_ng human
```

## 📈 性能优化记录 (2026-02-09)
- **GPU 飞跃**：将 YOLO 推理迁移至 CUDA，追踪帧率从 ~10 FPS 飙升至 **60 FPS**。
- **异步重构**：实现异步拉流与零延迟帧处理，彻底消除单线程阻塞。
- **CPU 减负**：通过 30 FPS 限帧与休眠机制，总 CPU 占用率从 **55% 降至 4% 左右**（降幅达 90%+）。

## 📁 目录结构
- `camera_ng/`：核心 Python 包。
- `test_gpu.py`：GPU 环境验证脚本。
- `pyproject.toml`：项目依赖管理。

---
*Powered by Mooer & Albert 💕*
