#!/usr/bin/env bash
set -euo pipefail

ROOT="/nfs/ycji_temp/code/DummyForcing"
SELF_CFG="$ROOT/configs/forcingkv_self_forcing_inference.yaml"
DUMMY_CFG="$ROOT/configs/dummy_longlive_inference.yaml"
LONG_CFG="$ROOT/configs/forcingkv_longlive_inference.yaml"

SELF_BAK="$SELF_CFG.bak_req_batch_20260309"
DUMMY_BAK="$DUMMY_CFG.bak_req_batch_20260309"
LONG_BAK="$LONG_CFG.bak_req_batch_20260309"

LOG_DIR="$ROOT/experiment_logs/requested_batch_20260309"
SUMMARY="$LOG_DIR/summary.tsv"

mkdir -p "$LOG_DIR"
cp "$SELF_CFG" "$SELF_BAK"
cp "$DUMMY_CFG" "$DUMMY_BAK"
cp "$LONG_CFG" "$LONG_BAK"

restore_cfgs() {
  [[ -f "$SELF_BAK" ]] && cp "$SELF_BAK" "$SELF_CFG" && rm -f "$SELF_BAK"
  [[ -f "$DUMMY_BAK" ]] && cp "$DUMMY_BAK" "$DUMMY_CFG" && rm -f "$DUMMY_BAK"
  [[ -f "$LONG_BAK" ]] && cp "$LONG_BAK" "$LONG_CFG" && rm -f "$LONG_BAK"
}
trap restore_cfgs EXIT

echo -e "group\tconfig\tar_start\tlocal_context_length\tkv_merge_enabled\tmomentum\toutput_folder\tvideo_count\tfps_count\tfps_avg\tstatus\tlog_file" > "$SUMMARY"

run_one() {
  local group="$1"
  local cfg="$2"
  local ar_start="$3"
  local local_ctx="$4"
  local kv_merge="$5"
  local momentum="$6"
  local out_folder="$7"
  local tag="$8"

  local log_file="$LOG_DIR/${tag}.log"
  echo "===== START ${group} (${tag}) =====" | tee "$log_file"

  set +e
  stdbuf -oL -eL bash -lc "source /home/ycji/miniconda3/etc/profile.d/conda.sh && \
    conda activate dummyforcing && \
    export PYTHONUNBUFFERED=1 && \
    cd \"$ROOT\" && \
    CUDA_VISIBLE_DEVICES=1 python inference.py --config_path \"$cfg\"" 2>&1 | tee -a "$log_file"
  local run_rc=${PIPESTATUS[0]}
  set -e

  echo "===== END ${group} (${tag}), rc=${run_rc} =====" | tee -a "$log_file"

  local fps_values fps_count fps_avg video_count status
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

  echo -e "${group}\t${cfg}\t${ar_start}\t${local_ctx}\t${kv_merge}\t${momentum}\t${out_folder}\t${video_count}\t${fps_count}\t${fps_avg}\t${status}\t${log_file}" | tee -a "$SUMMARY"
}

cd "$ROOT"

# Group 1: ForcingKV (no merge) on self-forcing.
combos_self=(
  "2 1"
  "2 2"
  "2 3"
  "4 1"
  "4 2"
  "4 3"
)
for combo in "${combos_self[@]}"; do
  read -r ar local_ctx <<< "$combo"
  out_folder="videos/inference/forcingkv_self_forcing_sweep_nomerge/ar${ar}_ctx${local_ctx}"
  sed -i -E "s/^  ar_start:.*/  ar_start: ${ar}/" "$SELF_CFG"
  sed -i -E "s/^  local_context_length:.*/  local_context_length: ${local_ctx}/" "$SELF_CFG"
  sed -i -E "s/^  kv_merge_enabled:.*/  kv_merge_enabled: false/" "$SELF_CFG"
  sed -i -E "s#^output_folder:.*#output_folder: ${out_folder}#" "$SELF_CFG"
  run_one "forcingkv_self_nomerg" "$SELF_CFG" "$ar" "$local_ctx" "false" "NA" "$out_folder" "g1_ar${ar}_ctx${local_ctx}"
done

# Group 2: DummyForcing on longlive with local_context_length=2.
out_folder_dummy="videos/inference/dummy_longlive/dummy180_ctx2"
sed -i -E "s/^  local_context_length:.*/  local_context_length: 2/" "$DUMMY_CFG"
sed -i -E "s#^output_folder:.*#output_folder: ${out_folder_dummy}#" "$DUMMY_CFG"
run_one "dummy_longlive_ctx2" "$DUMMY_CFG" "2" "2" "NA" "NA" "$out_folder_dummy" "g2_dummy_ctx2"

# Group 3: ForcingKV (merge enabled) on longlive with momentum sweep.
momenta=(0.9 0.5 0.99)
for m in "${momenta[@]}"; do
  mtag="${m/./p}"
  out_folder="videos/inference/forcingkv_longlive_merge_sweep/m${mtag}"
  ar=$(awk '/^  ar_start:/{print $2}' "$LONG_CFG" | head -n1)
  local_ctx=$(awk '/^  local_context_length:/{print $2}' "$LONG_CFG" | head -n1)
  sed -i -E "s/^  kv_merge_enabled:.*/  kv_merge_enabled: true/" "$LONG_CFG"
  sed -i -E "s/^  momentum:.*/  momentum: ${m}/" "$LONG_CFG"
  sed -i -E "s#^output_folder:.*#output_folder: ${out_folder}#" "$LONG_CFG"
  run_one "forcingkv_longlive_merge" "$LONG_CFG" "${ar}" "${local_ctx}" "true" "${m}" "$out_folder" "g3_m${mtag}"
done

printf "\nSaved summary: %s\n" "$SUMMARY"
