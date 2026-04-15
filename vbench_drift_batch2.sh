#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIFT_SCRIPT="${SCRIPT_DIR}/vbench_drift.sh"

JOBS=(
    "/ycji/code/Forcing-KV/videos_new/vbench/dummy_longlive_30s_ctx1|0|38457"
    "/ycji/code/Forcing-KV/videos_new/vbench/dummy_longlive_30s_ctx2|0|38459"
    "/ycji/code/Forcing-KV/videos_new/vbench/longlive_30s_attn12|0|38460"
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

