#!/bin/bash

MODEL_PATH="models/raft-things.pth"
GPU_ID="1"
DEVICE="cuda:${GPU_ID}"

video_path=(
    "/nfs/ycji_temp/code/DummyForcing/videos/test/dummy_longlive"
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

    python optical_difference.py --model="$MODEL_PATH" --path="$path" --device="$DEVICE" --k 39 #k可以不写默认1，有output result_mean video_results项选择存储文件的路径和结果名字
done
