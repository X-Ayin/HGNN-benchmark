#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全面的GPU和PyTorch环境验证脚本
"""

import sys
import platform
import time

print("=" * 70)
print("🔍 系统和环境信息")
print("=" * 70)
print(f"Python版本: {sys.version}")
print(f"操作系统: {platform.platform()}")
print()

# 检查PyTorch
print("=" * 70)
print("📦 PyTorch检查")
print("=" * 70)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    print(f"✓ PyTorch已安装")
    print(f"  版本: {torch.__version__}")
except ImportError as e:
    print(f"✗ PyTorch未安装: {e}")
    sys.exit(1)

# 检查NumPy
print()
try:
    import numpy as np
    print(f"✓ NumPy已安装")
    print(f"  版本: {np.__version__}")
except ImportError:
    print("✗ NumPy未安装")

# 检查CUDA可用性
print()
print("=" * 70)
print("🎮 GPU/CUDA检查")
print("=" * 70)

cuda_available = torch.cuda.is_available()
print(f"CUDA可用: {cuda_available}")

if cuda_available:
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.current_device()}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"cuDNN版本: {torch.backends.cudnn.version()}")
    print(f"cuDNN可用: {torch.backends.cudnn.enabled}")
    
    # 详细的GPU内存信息
    print()
    print("GPU内存信息:")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}")
        print(f"    计算能力: {props.major}.{props.minor}")
        print(f"    总内存: {props.total_memory / 1024**3:.2f} GB")
        
    # 获取当前GPU使用情况
    print()
    print("当前GPU使用情况:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  GPU {i}: 已分配 {allocated:.3f} GB / 已预留 {reserved:.3f} GB")
else:
    print("✗ 未检测到GPU/CUDA")
    print("  请检查:")
    print("  1. 是否安装了支持CUDA的PyTorch版本")
    print("  2. 是否安装了NVIDIA GPU驱动")

# 测试1: 基本张量操作
print()
print("=" * 70)
print("⚡ 测试1: 基本张量操作")
print("=" * 70)

try:
    # CPU张量
    x_cpu = torch.randn(1000, 1000)
    y_cpu = torch.randn(1000, 1000)
    z_cpu = torch.matmul(x_cpu, y_cpu)
    print("✓ CPU张量运算: 成功")
    
    if cuda_available:
        # GPU张量
        x_gpu = x_cpu.to('cuda')
        y_gpu = y_cpu.to('cuda')
        z_gpu = torch.matmul(x_gpu, y_gpu)
        print("✓ GPU张量运算: 成功")
        
        # 数据传输测试
        transferred = z_gpu.to('cpu')
        print("✓ GPU->CPU数据传输: 成功")
except Exception as e:
    print(f"✗ 张量操作失败: {e}")

# 测试2: 性能对比
print()
print("=" * 70)
print("⏱️  测试2: CPU vs GPU 性能对比")
print("=" * 70)

try:
    size = 1000
    iterations = 5
    
    # CPU性能测试
    x_cpu = torch.randn(size, size)
    y_cpu = torch.randn(size, size)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        _ = torch.matmul(x_cpu, y_cpu)
    cpu_time = time.time() - start
    
    print(f"CPU: {size}x{size}矩阵乘法 x{iterations} = {cpu_time:.3f}秒")
    
    if cuda_available:
        x_gpu = x_cpu.to('cuda')
        y_gpu = y_cpu.to('cuda')
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            _ = torch.matmul(x_gpu, y_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        print(f"GPU: {size}x{size}矩阵乘法 x{iterations} = {gpu_time:.3f}秒")
        print(f"加速比: {cpu_time/gpu_time:.2f}x")
except Exception as e:
    print(f"✗ 性能测试失败: {e}")

# 测试3: 神经网络模型
print()
print("=" * 70)
print("🧠 测试3: 神经网络模型")
print("=" * 70)

try:
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(100, 50)
            self.fc2 = nn.Linear(50, 10)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
    
    model = SimpleNet()
    print(f"✓ 模型创建成功")
    
    # 模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数数: {total_params}")
    print(f"  可训练参数: {trainable_params}")
    
    if cuda_available:
        model = model.to('cuda')
        print(f"✓ 模型移至GPU")
    
    # 测试前向传播
    input_data = torch.randn(32, 100)
    if cuda_available:
        input_data = input_data.to('cuda')
    
    output = model(input_data)
    print(f"✓ 前向传播成功")
    print(f"  输入形状: {input_data.shape}")
    print(f"  输出形状: {output.shape}")
    
except Exception as e:
    print(f"✗ 模型测试失败: {e}")

# 测试4: 训练循环
print()
print("=" * 70)
print("🎯 测试4: 简单训练循环")
print("=" * 70)

try:
    # 创建模型和优化器
    model = SimpleNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    if cuda_available:
        model = model.to('cuda')
    
    # 创建虚拟数据
    X = torch.randn(100, 100)
    y = torch.randint(0, 10, (100,))
    
    if cuda_available:
        X = X.to('cuda')
        y = y.to('cuda')
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32)
    
    # 训练1个epoch
    model.train()
    torch.cuda.synchronize()
    start = time.time()
    
    total_loss = 0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if cuda_available:
        torch.cuda.synchronize()
    training_time = time.time() - start
    
    print(f"✓ 训练循环成功")
    print(f"  平均损失: {total_loss / len(dataloader):.4f}")
    print(f"  训练时间: {training_time:.3f}秒")
    
except Exception as e:
    print(f"✗ 训练循环失败: {e}")

# 测试5: 内存管理
print()
print("=" * 70)
print("💾 测试5: 内存管理")
print("=" * 70)

try:
    if cuda_available:
        print("GPU内存清理前:")
        print(f"  已分配: {torch.cuda.memory_allocated() / 1024**3:.3f} GB")
        print(f"  已预留: {torch.cuda.memory_reserved() / 1024**3:.3f} GB")
        
        torch.cuda.empty_cache()
        
        print("GPU内存清理后:")
        print(f"  已分配: {torch.cuda.memory_allocated() / 1024**3:.3f} GB")
        print(f"  已预留: {torch.cuda.memory_reserved() / 1024**3:.3f} GB")
        print("✓ 内存管理测试成功")
    else:
        print("✓ CPU模式运行")
except Exception as e:
    print(f"✗ 内存管理测试失败: {e}")

print()
print("=" * 70)
print("✅ 全面检查完成!")
print("=" * 70)
print()
print("总结:")
if cuda_available:
    print("✓ GPU环境完全就绪，可以开始进行深度学习项目")
else:
    print("⚠️  GPU不可用，将使用CPU运行（速度较慢）")
