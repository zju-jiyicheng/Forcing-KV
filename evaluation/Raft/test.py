import torch
import numpy as np
import os
import sys
sys.path.append('core')
import argparse
from raft import RAFT

def check_flow_stability(model, frame1, frame2, device, n_tests=3):
    """
    检查光流计算的稳定性
    对同一对图像计算多次光流，比较结果是否一致
    
    Args:
        model: RAFT模型
        frame1: 第一帧 [1, 3, H, W]
        frame2: 第二帧 [1, 3, H, W]
        device: 设备
        n_tests: 测试次数
    
    Returns:
        is_stable: bool, 是否稳定
        max_diff: float, 最大差异
    """
    model.eval()
    model.to(device)
    
    flows = []
    
    with torch.no_grad():
        for i in range(n_tests):
            # 确保输入一致
            f1 = frame1.to(device).float()
            f2 = frame2.to(device).float()
            
            # 计算光流
            flow_low, flow_up = model(f1, f2, iters=20, test_mode=True)
            
            # 保存到CPU
            flows.append(flow_up.cpu())
    
    # 比较所有结果是否相同
    all_same = True
    max_diff = 0
    
    for i in range(n_tests):
        for j in range(i+1, n_tests):
            diff = torch.abs(flows[i] - flows[j]).max().item()
            max_diff = max(max_diff, diff)
            
            if not torch.allclose(flows[i], flows[j], rtol=1e-5, atol=1e-5):
                all_same = False
    
    return all_same, max_diff

def test_flow_stability(model, device):
    """
    测试光流稳定性
    """
    print("="*50)
    print("光流稳定性测试")
    print("="*50)
    
    # 创建简单的测试图像（移动的方块）
    H, W = 128, 128
    
    # 第一帧：左上角的方块
    frame1 = torch.zeros(1, 3, H, W)
    frame1[0, 0, 10:30, 10:30] = 1.0
    
    # 第二帧：向右下移动的方块
    frame2 = torch.zeros(1, 3, H, W)
    frame2[0, 0, 20:40, 20:40] = 1.0
    
    print("测试1: 使用简单合成图像...")
    is_stable, max_diff = check_flow_stability(model, frame1, frame2, device, n_tests=3)
    
    if is_stable:
        print(f"✅ 光流计算稳定! 最大差异: {max_diff:.2e}")
    else:
        print(f"❌ 光流计算不稳定! 最大差异: {max_diff:.2e}")
    
    # 如果有真实视频，也可以测试真实视频
    print("\n测试2: 使用随机噪声图像...")
    frame1 = torch.randn(1, 3, H, W)
    frame2 = torch.randn(1, 3, H, W)
    
    is_stable, max_diff = check_flow_stability(model, frame1, frame2, device, n_tests=3)
    
    if is_stable:
        print(f"✅ 光流计算稳定! 最大差异: {max_diff:.2e}")
    else:
        print(f"❌ 光流计算不稳定! 最大差异: {max_diff:.2e}")
    
    return is_stable

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint", required=True)
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    
    # 加载模型
    model = RAFT(args)
    
    if args.model:
        checkpoint = torch.load(args.model, map_location=device)
        print(f"模型加载成功: {args.model}")
    
    model.to(device)
    model.eval()
    
    # 测试稳定性
    test_flow_stability(model, device)