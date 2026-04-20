#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIFT_SCRIPT="${SCRIPT_DIR}/vbench_drift_total.sh"

JOBS=(
    "/ycji/code/Forcing-KV/videos_new/vbench/forcingkv_{1+3.1}_{naive_sink3_0.8}_self_forcing_30s_ar1_sink1_s1_t1_d1_patch3_0.33|4|38588"
    "/ycji/code/Forcing-KV/videos_new/vbench/self_forcing_30s_sink0|4|38588"
    "/ycji/code/Forcing-KV/videos_new/vbench/self_forcing_30s_sink0_teacache0.2|4|38588"
    "/ycji/code/Forcing-KV/videos_new/vbench/dummy_self_forcing_30s_ctx1|4|38588"
    "/ycji/code/Forcing-KV/videos_new/vbench/dummy_self_forcing_30s_ctx6|4|38588"
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
