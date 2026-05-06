#!/bin/bash

MODEL_PATH="models/raft-things.pth"
GPU_ID="5"
DEVICE="cuda:${GPU_ID}"
TopK=39 # 30s = 40 chunks

video_path=(
    "/ycji/code/Forcing-KV/videos_new/vbench/forcingkv_{0.5}_longlive_30s_ar4_sink1_s1_t1_d1_patch6_0.33"
)   

for path in "${video_path[@]}"; do
    echo "Using device: $DEVICE"

    # 检查模型文件是否存在
    if [ ! -f "$MODEL_PATH" ]; then
        echo "错误: 模型文件不存在: $MODEL_PATH"
        exit 1
    fi

    # 检查视频路径是否存在
    if [ ! -d "$path" ]; then
        echo "错误: 视频路径不存在: $path"
        exit 1
    fi

    python optical_difference.py --model="$MODEL_PATH" --path="$path" --device="$DEVICE" --k $TopK
done
