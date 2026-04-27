#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIFT_SCRIPT="${SCRIPT_DIR}/vbench_drift_total.sh"

JOBS=(
    "/ycji/code/Forcing-KV/videos_new/vbench/forcingkv_{modify}_{naive_sink3_0.8}_self_forcing_30s_ar1_sink1_s1_t1_d1_patch6_0.33_fp8|7|38518"
    "/ycji/code/Forcing-KV/videos_new/vbench/forcingkv_{naive_sink3_0.8}_longlive_30s_ar4_sink1_s1_t1_d1_patch6_0.33_fp8|7|38518"
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
