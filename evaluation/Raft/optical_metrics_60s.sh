#!/bin/bash

MODEL_PATH="models/raft-things.pth"
GPU_ID="6"
DEVICE="cuda:${GPU_ID}"
TopK=79 # 60s = 80 chunks

video_path=(
    "/ycji/code/Forcing-KV/videos_new/vbench/longlive_60s_sink3_attn10"
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

    python optical_difference.py --model="$MODEL_PATH" --path="$path" --device="$DEVICE" --k $TopK # By default, k=1
done
