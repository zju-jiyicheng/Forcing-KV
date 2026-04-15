#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIFT_SCRIPT="${SCRIPT_DIR}/vbench_drift.sh"

JOBS=(
    "/ycji/code/Forcing-KV/videos_new/vbench/longlive_60s_attn12|1|38558"
    "/ycji/code/Forcing-KV/videos_new/vbench/longlive_60s_attn21|1|38558"
    "/ycji/code/Forcing-KV/videos_new/vbench/longlive_60s_teacache0.2|1|38558"
    "/ycji/code/Forcing-KV/videos_new/vbench/dummy_longlive_60s_ctx1|1|38558"
    "/ycji/code/Forcing-KV/videos_new/vbench/dummy_longlive_60s_ctx2|1|38558"
    "/ycji/code/Forcing-KV/videos_new/vbench/forcingkv_longlive_60s_ar4_sink1_s1_t1_d1_patch3_0.33|1|38558"
)

job_index=1
job_total=${#JOBS[@]}
for job in "${JOBS[@]}"; do
    IFS='|' read -r videos_path cuda_visible_devices master_port_base <<<"$job"
    echo "Batch job ${job_index}/${job_total}:"
    echo "  VIDEOS_PATH=${videos_path}"
    echo "  CUDA_VISIBLE_DEVICES=${cuda_visible_devices}"
    echo "  MASTER_PORT_BASE=${master_port_base}"
    bash "$DRIFT_SCRIPT" "$videos_path" "$cuda_visible_devices" "$master_port_base"
    job_index=$((job_index + 1))
done
