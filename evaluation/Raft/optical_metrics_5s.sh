#!/bin/bash

MODEL_PATH="models/raft-things.pth"
GPU_ID="1"
DEVICE="cuda:${GPU_ID}"
TopK=6 # 5s = 7 chunks

video_path=(
    "/ycji/code/Forcing-KV/videos_new/vbench/self_forcing_5s"
    "/ycji/code/Forcing-KV/videos_new/vbench/self_forcing_5s_teacache0.2"
    "/ycji/code/Forcing-KV/videos_new/vbench/dummy_self_forcing_5s_ctx1"
    "/ycji/code/Forcing-KV/videos_new/vbench/dummy_self_forcing_5s_ctx6"
    "/ycji/code/Forcing-KV/videos_new/vbench/forcingkv_self_forcing_5s_ar1_sink1_spatial1_temporal1_dynamic1_patch3_0.33"
    #
    "/ycji/code/Forcing-KV/videos_new/vbench/longlive_attn12_5s"
    "/ycji/code/Forcing-KV/videos_new/vbench/longlive_attn21_5s"
    "/ycji/code/Forcing-KV/videos_new/vbench/longlive_attn12_teacache0.2_5s"
    "/ycji/code/Forcing-KV/videos_new/vbench/dummy_longlive_5s"
    "/ycji/code/Forcing-KV/videos_new/vbench/dummy_longlive_ctx2_5s"
    "/ycji/code/Forcing-KV/videos_new/vbench/forcingkv_longlive_5s_ar2_sink1_s1_t1_d1_patch3_0.33"
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
