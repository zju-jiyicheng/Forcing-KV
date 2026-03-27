#!/usr/bin/env bash
set -euo pipefail

ROOT="/nfs/ycji_temp/code/DummyForcing"
DUMMY_CFG="$ROOT/configs/dummy_self_forcing_inference.yaml"
FK_CFG="$ROOT/configs/forcingkv_self_forcing_inference.yaml"

DUMMY_BAK="$DUMMY_CFG.bak_req_20260310"
FK_BAK="$FK_CFG.bak_req_20260310"

LOG_DIR="$ROOT/experiment_logs/requested_batch_20260310"
SUMMARY_CSV="$LOG_DIR/fps_summary.csv"

mkdir -p "$LOG_DIR"
cp "$DUMMY_CFG" "$DUMMY_BAK"
cp "$FK_CFG" "$FK_BAK"

restore_cfgs() {
  [[ -f "$DUMMY_BAK" ]] && cp "$DUMMY_BAK" "$DUMMY_CFG" && rm -f "$DUMMY_BAK"
  [[ -f "$FK_BAK" ]] && cp "$FK_BAK" "$FK_CFG" && rm -f "$FK_BAK"
}
trap restore_cfgs EXIT

echo "group,config,local_context_length,ar_start,momentum,kv_merge_enabled,output_folder,fps_count,fps_avg,status,log_file" > "$SUMMARY_CSV"

run_one() {
  local group="$1"
  local cfg="$2"
  local local_ctx="$3"
  local ar_start="$4"
  local momentum="$5"
  local kv_merge="$6"
  local out_folder="$7"
  local tag="$8"

  local log_file="$LOG_DIR/${tag}.log"

  echo "===== START ${group} (${tag}) =====" | tee "$log_file"
  set +e
  stdbuf -oL -eL bash -lc "source /home/ycji/miniconda3/etc/profile.d/conda.sh && \
    conda activate dummyforcing && \
    export PYTHONUNBUFFERED=1 && \
    cd \"$ROOT\" && \
    CUDA_VISIBLE_DEVICES=2 python inference.py --config_path \"$cfg\"" 2>&1 | tee -a "$log_file"
  local run_rc=${PIPESTATUS[0]}
  set -e
  echo "===== END ${group} (${tag}), rc=${run_rc} =====" | tee -a "$log_file"

  local fps_values fps_count fps_avg status
  fps_values=$(grep -E "FPS:" "$log_file" | awk '{print $(NF-1)}' || true)
  fps_count=$(echo "$fps_values" | awk 'NF{c++} END{print c+0}')
  fps_avg=$(echo "$fps_values" | awk 'NF{s+=$1; c++} END{if(c>0) printf "%.4f", s/c; else printf "NaN"}')

  if [[ "$run_rc" -eq 0 ]]; then
    status="ok"
  else
    status="failed(${run_rc})"
  fi

  echo "${group},${cfg},${local_ctx},${ar_start},${momentum},${kv_merge},${out_folder},${fps_count},${fps_avg},${status},${log_file}" | tee -a "$SUMMARY_CSV"
}

cd "$ROOT"

# (1) DummyForcing + Self-Forcing, local_context_length=2
out_dummy="videos/inference/dummy_self_forcing_ctx2/dummy180"
sed -i -E "s/^  local_context_length:.*/  local_context_length: 2/" "$DUMMY_CFG"
sed -i -E "s#^output_folder:.*#output_folder: ${out_dummy}#" "$DUMMY_CFG"
run_one "dummy_self_forcing" "$DUMMY_CFG" "2" "NA" "NA" "NA" "$out_dummy" "g1_dummy_ctx2"

# (2) ForcingKV + Self-Forcing, kv_merge_enabled=true, (ar_start, momentum) in {(4,0.9),(4,0.8),(2,0.9),(2,0.8)}
combos=(
  "4 0.9"
  "4 0.8"
  "2 0.9"
  "2 0.8"
)

for combo in "${combos[@]}"; do
  read -r ar m <<< "$combo"
  mtag="${m/./p}"
  out_fk="videos/inference/forcingkv_self_forcing_merge/ar${ar}_m${mtag}"

  sed -i -E "s/^  ar_start:.*/  ar_start: ${ar}/" "$FK_CFG"
  sed -i -E "s/^  kv_merge_enabled:.*/  kv_merge_enabled: true/" "$FK_CFG"
  sed -i -E "s/^  momentum:.*/  momentum: ${m}/" "$FK_CFG"
  sed -i -E "s#^output_folder:.*#output_folder: ${out_fk}#" "$FK_CFG"

  run_one "forcingkv_self_forcing" "$FK_CFG" "1" "$ar" "$m" "true" "$out_fk" "g2_ar${ar}_m${mtag}"
done

echo "Saved summary CSV: $SUMMARY_CSV"
