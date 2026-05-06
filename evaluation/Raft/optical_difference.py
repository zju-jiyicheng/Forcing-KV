import torch 
import os
import cv2
import glob
import numpy as np
import argparse
import sys
sys.path.append('core') #添加模型组件的路径，可能要改
import math
import numpy as np

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from tqdm import tqdm

import torchvision.io as io
import pandas as pd
import json

def save_flow_line_plot(df, output_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_width = 14 if len(df) > 10 else 10
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    for video_name, row in df.iterrows():
        values = row.dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        ax.plot(range(len(values)), values, linewidth=1.4, label=video_name)

    ax.set_xlabel('Frame pair index')
    ax.set_ylabel('Mean optical-flow magnitude')
    ax.set_title('Optical Flow Magnitudes')
    ax.grid(True, alpha=0.25)

    if len(df) <= 10:
        ax.legend(fontsize=7, loc='best')
        fig.tight_layout()
    else:
        ax.legend(fontsize=6, loc='upper left', bbox_to_anchor=(1.02, 1.0))
        fig.tight_layout(rect=[0, 0, 0.78, 1])

    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def load_single_video(video_path):
    frame_list = []

    video, audio, info = io.read_video(
    filename=video_path,
    start_pts=0,           # 起始时间
    end_pts=None,          # 结束时间
    pts_unit='sec'         # 时间单位
    )

    video = video.float()

    for i in range(video.shape[0]):
        frame_list.append(video[i,:,:,:].permute(2,0,1).unsqueeze(0))

    return frame_list

def process_single_video(video_list, video_name, model, device):
    """
    处理单个视频，返回每帧光流的平均模长序列
    
    Args:
        video_list: 视频帧列表
        video_name: 视频名称
        model: RAFT模型
        device: 计算设备
    
    Returns:
        flow_magnitudes: 每帧光流的平均模长列表
    """
    frames = video_list
    flow_magnitudes = []  # 存储每帧光流的平均模长
    flow_max = []
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for i in tqdm(range(len(frames)-1), desc=f"Processing {video_name}"):
            # 计算光流
            flow_low, flow_up = model(frames[i].to(device), frames[i+1].to(device), iters=20, test_mode=True)
            
            # 获取光流场 [2, H, W]
            flow = flow_up[0]
            dx = flow[0]
            dy = flow[1]
            
            # 计算模长
            magnitude = torch.sqrt(dx**2 + dy**2)
            
            # 计算平均模长
            mean_magnitude = magnitude.mean().item()
            max_magnitude = magnitude.max().item()
            flow_magnitudes.append(mean_magnitude)
            flow_max.append(max_magnitude)
    
    return flow_magnitudes, flow_max

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--device', default='cuda:0', help="device for inference, e.g. cuda:0 or cpu")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--k', default=1, help="number for topk")
    parser.add_argument('--output_path', default=None, help="path for result")
    parser.add_argument('--result_mean', default='result_mean.jsonl', help="name for result")
    parser.add_argument('--video_results', default='optical_flow_magnitudes.jsonl', help="name for video jsonl")
    parser.add_argument('--plot', action='store_true', help="save optical-flow magnitude line plot")
    args = parser.parse_args()
    device = args.device
    
    if args.output_path:
        result_path = args.output_path
    else:
        result_path = os.path.join(args.path, 'raft')

    os.makedirs(result_path, exist_ok=True)

    # 加载模型
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.module
    
    # 视频路径
    video_path = args.path
    video_names = [file for file in os.listdir(video_path) if file.endswith(".mp4")]
    
    # 存储所有视频的光流平均模长序列
    flow_sequences = {}
    max_length = 0  # 记录最长的序列长度
    
    # 处理每个视频
    for video in video_names:
        print(f"\nProcessing video: {video}")
        # 加载视频帧
        video_frames = load_single_video(os.path.join(video_path, video))
        
        # 计算光流平均模长序列
        flow_magnitudes, flow_max = process_single_video(video_frames, video, model, device)
        
        # 存储序列
        flow_sequences[video] = [flow_magnitudes, flow_max]
        
        # 更新最大长度
        max_length = max(max_length, len(flow_magnitudes))
        
        print(f"Video {video}: {len(flow_magnitudes)} frames of optical flow")
    
    # 创建DataFrame，列为帧索引，行为视频名称
    # 首先准备数据字典
    k = int(args.k)
    data_dict = {}
    for video_name, zip_sq in flow_sequences.items():
        # 为每个视频创建一个行，填充序列值
        # 如果序列长度不足max_length，用NaN填充
        sequence = zip_sq[0]
        padded_sequence = sequence + [np.nan] * (max_length - len(sequence))
        data_dict[video_name] = padded_sequence

    data_list = []
    i = 0
    metrics_mean = 0
    for video_name, zip_sq in flow_sequences.items():
        i = i+1
        metrics_dict = {}
        sequence = zip_sq[0]
        padded_sequence = sequence + [np.nan] * (max_length - len(sequence))
        mean = np.mean(padded_sequence)
        top_k_largest = sorted(padded_sequence, reverse=True)[:k]
        k_mean = np.mean(top_k_largest)
        metrics = k_mean / mean
        metrics_dict['index'] = str(i)
        metrics_dict['metrics'] = metrics
        metrics_dict['name'] = video_name
        metrics_dict['frames'] = padded_sequence
        metrics_mean += metrics 
        data_list.append(metrics_dict)

    result_mean = {'index':'1', 'total_metrics':(metrics_mean / i)}
    
    # 创建DataFrame，转置使得行为视频名，列为帧索引
    df = pd.DataFrame(data_dict).T
    
    # 设置列名
    df.columns = [f'frame_{i}' for i in range(max_length)]
    df.index.name = 'video_name'
    
    # 保存到CSV
    output_path = os.path.join(result_path, 'optical_flow_magnitudes.csv')
    df.to_csv(output_path)

    if args.plot:
        plot_path = os.path.join(result_path, 'optical_flow_magnitudes.png')
        save_flow_line_plot(df, plot_path)
        print(f"Plot saved to: {plot_path}")
    
    print(f"\nResults saved to: {output_path}")
    print(f"Total videos processed: {len(flow_sequences)}")
    print(f"Maximum sequence length: {max_length}")
    print(f"DataFrame shape: {df.shape}")
    
    # 可选：打印前几行查看结果
    """
    print("\nFirst few rows of the results:")
    print(df.head())
    """

    jsonl_path = os.path.join(result_path, args.video_results)

    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for data in data_list:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    print("数据已保存到 optical_flow_magnitudes.jsonl")

    result_path = os.path.join(result_path, args.result_mean)
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(result_mean, ensure_ascii=False) + '\n')

    print("均值已保存到 result.jsonl")
