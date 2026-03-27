#!/usr/bin/env bash
set -euo pipefail

ROOT="/nfs/ycji_temp/code/DummyForcing"
CFG="$ROOT/configs/forcingkv_longlive_inference.yaml"
BACKUP="$ROOT/configs/forcingkv_longlive_inference.yaml.bak_ar_ctx_sweep_v2"
LOG_DIR="$ROOT/experiment_logs/forcingkv_longlive_ar_ctx_sweep_v2"
SUMMARY="$LOG_DIR/summary.tsv"

mkdir -p "$LOG_DIR"
cp "$CFG" "$BACKUP"

restore_cfg() {
  if [[ -f "$BACKUP" ]]; then
    cp "$BACKUP" "$CFG"
    rm -f "$BACKUP"
  fi
}
trap restore_cfg EXIT

echo -e "ar_start\tlocal_context_length\toutput_folder\tvideo_count\tfps_count\tfps_avg\tstatus\tlog_file" > "$SUMMARY"

combos=(
  "2 2"
  "2 1"
  "2 0.5"
  "4 2"
  "4 1"
  "4 0.5"
)

cd "$ROOT"

for combo in "${combos[@]}"; do
  read -r ar_start local_ctx <<< "$combo"
  local_ctx_tag="${local_ctx/./p}"
  out_folder="videos/inference/forcingkv_longlive_v2/ar${ar_start}_context_${local_ctx_tag}"
  log_file="$LOG_DIR/run_ar${ar_start}_ctx${local_ctx_tag}.log"

  sed -i -E "s/^  ar_start:.*/  ar_start: ${ar_start}/" "$CFG"
  sed -i -E "s/^  local_context_length:.*/  local_context_length: ${local_ctx}/" "$CFG"
  sed -i -E "s#^output_folder:.*#output_folder: ${out_folder}#" "$CFG"

  echo "===== START ar_start=${ar_start}, local_context_length=${local_ctx} =====" | tee "$log_file"
  set +e
  stdbuf -oL -eL bash -lc "source /home/ycji/miniconda3/etc/profile.d/conda.sh && \
    conda activate dummyforcing && \
    export PYTHONUNBUFFERED=1 && \
    cd \"$ROOT\" && \
    sh inference.sh" 2>&1 | tee -a "$log_file"
  run_rc=${PIPESTATUS[0]}
  set -e
  echo "===== END ar_start=${ar_start}, local_context_length=${local_ctx}, rc=${run_rc} =====" | tee -a "$log_file"

  fps_values=$(grep -E "FPS:" "$log_file" | awk '{print $(NF-1)}' || true)
  fps_count=$(echo "$fps_values" | awk 'NF{c++} END{print c+0}')
  fps_avg=$(echo "$fps_values" | awk 'NF{s+=$1; c++} END{if(c>0) printf "%.4f", s/c; else printf "NaN"}')

  if [[ -d "$ROOT/$out_folder" ]]; then
    video_count=$(find "$ROOT/$out_folder" -maxdepth 1 -type f -name "*.mp4" | wc -l | tr -d ' ')
  else
    video_count=0
  fi
  if [[ "$run_rc" -eq 0 ]]; then
    status="ok"
  else
    status="failed(${run_rc})"
  fi

  echo -e "${ar_start}\t${local_ctx}\t${out_folder}\t${video_count}\t${fps_count}\t${fps_avg}\t${status}\t${log_file}" | tee -a "$SUMMARY"
done

printf "\nSaved summary: %s\n" "$SUMMARY"
