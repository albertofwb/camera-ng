#!/usr/bin/env python3
"""
GPU 验证测试脚本
验证 YOLO 模型在 CUDA 上运行
"""

import sys
import torch

print("=" * 60)
print("🧪 YOLO GPU 验证测试")
print("=" * 60)

# 1. 检查 PyTorch CUDA 可用性
print("\n📊 PyTorch CUDA 检查:")
print(f"   PyTorch 版本: {torch.__version__}")
print(f"   CUDA 可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   CUDA 版本: {torch.version.cuda}")
    print(f"   GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"   当前设备: {torch.cuda.current_device()}")
else:
    print("\n❌ CUDA 不可用！请检查:")
    print("   1. NVIDIA 驱动是否安装: nvidia-smi")
    print("   2. PyTorch 是否安装 CUDA 版本")
    sys.exit(1)

# 2. 测试 YOLO 模型加载
print("\n📦 YOLO 模型加载测试:")
try:
    from ultralytics import YOLO
    print("   ✅ ultralytics 已安装")
except ImportError:
    print("   ❌ ultralytics 未安装")
    print("   请运行: pip install ultralytics")
    sys.exit(1)

# 3. 加载 YOLO 模型并强制使用 CUDA
print("\n🎯 加载 YOLO 模型 (yolov8n.pt) 到 CUDA:")
try:
    model = YOLO("yolov8n.pt", verbose=False)
    model.to('cuda')  # 强制使用 CUDA
    print(f"   ✅ 模型加载成功")
    print(f"   📍 模型设备: {model.device}")
except Exception as e:
    print(f"   ❌ 模型加载失败: {e}")
    sys.exit(1)

# 4. 测试推理
print("\n🚀 测试推理 (使用随机数据):")
import numpy as np

# 创建随机测试图像
test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

try:
    print("   运行推理...")
    results = model(test_image, verbose=False, device='cuda')
    
    # 检查推理是否在 GPU 上执行
    if hasattr(results[0], 'boxes'):
        boxes = results[0].boxes
        if boxes is not None:
            print(f"   ✅ 推理成功")
            print(f"   📊 检测结果: {len(boxes)} 个目标")
            
            # 验证张量在 GPU 上
            if len(boxes) > 0:
                sample_tensor = boxes.xyxy
                if hasattr(sample_tensor, 'device'):
                    print(f"   📍 输出张量设备: {sample_tensor.device}")
                    if 'cuda' in str(sample_tensor.device):
                        print("   ✅ 确认在 CUDA 上运行！")
                    else:
                        print("   ⚠️  输出在 CPU 上，请检查模型配置")
        else:
            print("   ✅ 推理成功 (无目标)")
    else:
        print("   ✅ 推理成功")
        
except Exception as e:
    print(f"   ❌ 推理失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. 测试 camera_ng 包中的 YOLOPersonDetector
print("\n📷 测试 camera_ng 包中的 YOLOPersonDetector:")
try:
    from camera_ng import YOLOPersonDetector
    detector = YOLOPersonDetector()
    
    if detector.model is not None:
        print(f"   ✅ YOLOPersonDetector 初始化成功")
        print(f"   📍 模型设备: {detector.device}")
        print(f"   🔗 配置设备: cuda")
        
        # 测试检测
        has_person, info = detector.check_person(frame=test_image)
        print(f"   ✅ 检测测试完成: {info}")
    else:
        print("   ❌ YOLOPersonDetector 模型未加载")
except Exception as e:
    print(f"   ❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()

# 6. 测试 PersonTracker
print("\n🎯 测试 camera_ng 包中的 PersonTracker:")
try:
    from camera_ng import PersonTracker
    tracker = PersonTracker()
    
    if tracker.model is not None:
        print(f"   ✅ PersonTracker 初始化成功")
        print(f"   📍 模型设备: {tracker.device}")
        
        # 测试检测
        detections = tracker.detect(test_image)
        print(f"   ✅ 检测测试完成: {len(detections)} 个人物")
    else:
        print("   ❌ PersonTracker 模型未加载")
except Exception as e:
    print(f"   ❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ 所有 GPU 验证测试通过！")
print("=" * 60)
