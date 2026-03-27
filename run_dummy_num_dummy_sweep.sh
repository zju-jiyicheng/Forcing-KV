#!/usr/bin/env bash
set -euo pipefail

ROOT="/nfs/ycji_temp/code/DummyForcing"
CFG_LONG="$ROOT/configs/dummy_longlive_inference.yaml"
CFG_SELF="$ROOT/configs/dummy_self_forcing_inference.yaml"
BKP_LONG="$ROOT/configs/dummy_longlive_inference.yaml.bak_num_dummy_sweep"
BKP_SELF="$ROOT/configs/dummy_self_forcing_inference.yaml.bak_num_dummy_sweep"
LOG_DIR="$ROOT/experiment_logs/dummy_num_dummy_sweep"
SUMMARY="$LOG_DIR/summary.tsv"

mkdir -p "$LOG_DIR"
cp "$CFG_LONG" "$BKP_LONG"
cp "$CFG_SELF" "$BKP_SELF"

restore_cfg() {
  if [[ -f "$BKP_LONG" ]]; then
    cp "$BKP_LONG" "$CFG_LONG"
    rm -f "$BKP_LONG"
  fi
  if [[ -f "$BKP_SELF" ]]; then
    cp "$BKP_SELF" "$CFG_SELF"
    rm -f "$BKP_SELF"
  fi
}
trap restore_cfg EXIT

echo -e "experiment\tconfig\tnum_dummy\toutput_folder\tfps_count\tfps_avg\tlog_file" > "$SUMMARY"

num_dummys=(180 90 270 324)

run_one() {
  local exp_name="$1"
  local cfg="$2"
  local base_out="$3"

  for nd in "${num_dummys[@]}"; do
    local out_folder="${base_out}/dummy${nd}"
    local log_file="$LOG_DIR/${exp_name}_dummy${nd}.log"

    sed -i -E "s/^  num_dummy:.*/  num_dummy: ${nd}/" "$cfg"
    sed -i -E "s#^output_folder:.*#output_folder: ${out_folder}#" "$cfg"

    echo "===== START ${exp_name} num_dummy=${nd} =====" | tee "$log_file"
    stdbuf -oL -eL bash -lc "source /home/ycji/miniconda3/etc/profile.d/conda.sh && \
      conda activate dummyforcing && \
      export PYTHONUNBUFFERED=1 && \
      cd \"$ROOT\" && \
      CUDA_VISIBLE_DEVICES=1 python inference.py --config_path \"$cfg\"" 2>&1 | tee -a "$log_file"
    echo "===== END ${exp_name} num_dummy=${nd} =====" | tee -a "$log_file"

    local fps_values fps_count fps_avg
    fps_values=$(grep -E "FPS:" "$log_file" | awk '{print $(NF-1)}' || true)
    fps_count=$(echo "$fps_values" | awk 'NF{c++} END{print c+0}')
    fps_avg=$(echo "$fps_values" | awk 'NF{s+=$1; c++} END{if(c>0) printf "%.4f", s/c; else printf "NaN"}')

    echo -e "${exp_name}\t${cfg}\t${nd}\t${out_folder}\t${fps_count}\t${fps_avg}\t${log_file}" | tee -a "$SUMMARY"
  done
}

cd "$ROOT"
run_one "dummy_longlive" "$CFG_LONG" "videos/inference/dummy_longlive"
run_one "dummy_self_forcing" "$CFG_SELF" "videos/inference/dummy_self_forcing"

printf "\nSaved summary: %s\n" "$SUMMARY"
