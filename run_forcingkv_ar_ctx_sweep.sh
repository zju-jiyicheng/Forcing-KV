#!/usr/bin/env bash
set -euo pipefail

ROOT="/nfs/ycji_temp/code/DummyForcing"
CFG="$ROOT/configs/forcingkv_longlive_inference.yaml"
BACKUP="$ROOT/configs/forcingkv_longlive_inference.yaml.bak_ar_ctx_sweep"
LOG_DIR="$ROOT/experiment_logs/forcingkv_longlive_ar_ctx_sweep"
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

echo -e "ar_start\tlocal_context_length\toutput_folder\tvideo_count\tfps_count\tfps_avg\tlog_file" > "$SUMMARY"

combos=(
  "0 4"
  "0 3"
  "0 2"
  "0 1"
  "7 4"
  "7 3"
  "7 2"
  "7 1"
)

cd "$ROOT"

for combo in "${combos[@]}"; do
  read -r ar_start local_ctx <<< "$combo"
  out_folder="videos/inference/forcingkv_longlive/ar${ar_start}_context${local_ctx}"
  log_file="$LOG_DIR/run_ar${ar_start}_ctx${local_ctx}.log"

  sed -i -E "s/^  ar_start:.*/  ar_start: ${ar_start}/" "$CFG"
  sed -i -E "s/^  local_context_length:.*/  local_context_length: ${local_ctx}/" "$CFG"
  sed -i -E "s#^output_folder:.*#output_folder: ${out_folder}#" "$CFG"

  echo "===== START ar_start=${ar_start}, local_context_length=${local_ctx} =====" | tee "$log_file"
  stdbuf -oL -eL bash -lc "source /home/ycji/miniconda3/etc/profile.d/conda.sh && \
    conda activate dummyforcing && \
    export PYTHONUNBUFFERED=1 && \
    cd \"$ROOT\" && \
    sh inference.sh" 2>&1 | tee -a "$log_file"
  echo "===== END ar_start=${ar_start}, local_context_length=${local_ctx} =====" | tee -a "$log_file"

  fps_values=$(grep -E "FPS:" "$log_file" | awk '{print $(NF-1)}' || true)
  fps_count=$(echo "$fps_values" | awk 'NF{c++} END{print c+0}')
  fps_avg=$(echo "$fps_values" | awk 'NF{s+=$1; c++} END{if(c>0) printf "%.4f", s/c; else printf "NaN"}')

  video_count=$(find "$ROOT/$out_folder" -maxdepth 1 -type f -name "*.mp4" | wc -l | tr -d ' ')
  echo -e "${ar_start}\t${local_ctx}\t${out_folder}\t${video_count}\t${fps_count}\t${fps_avg}\t${log_file}" | tee -a "$SUMMARY"
done

printf "\nSaved summary: %s\n" "$SUMMARY"
