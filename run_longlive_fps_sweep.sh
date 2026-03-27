#!/usr/bin/env bash
set -euo pipefail

ROOT="/nfs/ycji_temp/code/DummyForcing"
CFG="$ROOT/configs/longlive_inference.yaml"
BACKUP="$ROOT/configs/longlive_inference.yaml.bak_fps_sweep"
LOG_DIR="$ROOT/experiment_logs/longlive_fps_sweep"
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

echo -e "local_attn_size\tsink_size\toutput_folder\tfps_count\tfps_avg\tlog_file" > "$SUMMARY"

combos=(
  "12 3"
  "9 3"
  "6 3"
  "3 3"
  "12 0"
  "12 1"
  "9 0"
)

cd "$ROOT"

for combo in "${combos[@]}"; do
  read -r local_attn sink_size <<< "$combo"
  out_folder="videos/inference/longlive_sink_attn/attn${local_attn}_sink${sink_size}"
  log_file="$LOG_DIR/run_attn${local_attn}_sink${sink_size}.log"

  sed -i -E "s/^  local_attn_size:.*/  local_attn_size: ${local_attn}/" "$CFG"
  sed -i -E "s/^  sink_size:.*/  sink_size: ${sink_size}/" "$CFG"
  sed -i -E "s#^output_folder:.*#output_folder: ${out_folder}#" "$CFG"

  echo "===== START local_attn_size=${local_attn}, sink_size=${sink_size} =====" | tee "$log_file"
  # Match user's manual workflow: conda activate env, then run sh inference.sh.
  # Use unbuffered mode so runtime prints (e.g., Load model / FPS) flush to log in real time.
  stdbuf -oL -eL bash -lc "source /home/ycji/miniconda3/etc/profile.d/conda.sh && \
    conda activate dummyforcing && \
    export PYTHONUNBUFFERED=1 && \
    cd \"$ROOT\" && \
    sh inference.sh" 2>&1 | tee -a "$log_file"
  echo "===== END local_attn_size=${local_attn}, sink_size=${sink_size} =====" | tee -a "$log_file"

  fps_values=$(grep -E "FPS:" "$log_file" | awk '{print $(NF-1)}' || true)
  fps_count=$(echo "$fps_values" | awk 'NF{c++} END{print c+0}')
  fps_avg=$(echo "$fps_values" | awk 'NF{s+=$1; c++} END{if(c>0) printf "%.4f", s/c; else printf "NaN"}')

  echo -e "${local_attn}\t${sink_size}\t${out_folder}\t${fps_count}\t${fps_avg}\t${log_file}" | tee -a "$SUMMARY"

done

printf "\nSaved summary: %s\n" "$SUMMARY"
