#!/usr/bin/env bash
set -euo pipefail

ROOT="/nfs/ycji_temp/code/DummyForcing"
CFG="$ROOT/configs/forcingkv_longlive_inference.yaml"
BKP="$CFG.bak_similarity_disabled_threshold_sweep_20260312"
LOG_DIR="$ROOT/experiment_logs/forcingkv_similarity_disabled_threshold_sweep_20260312"
SUMMARY="$LOG_DIR/summary.csv"

mkdir -p "$LOG_DIR"
cp "$CFG" "$BKP"

restore_cfg() {
  if [[ -f "$BKP" ]]; then
    cp "$BKP" "$CFG"
    rm -f "$BKP"
  fi
}
trap restore_cfg EXIT

echo "similarity_prune_enabled,similarity_threshold,output_folder,fps_count,fps_avg,status,log_file" > "$SUMMARY"

thresholds=(0.9 0.5 0.3 0.1)

cd "$ROOT"

for th in "${thresholds[@]}"; do
  th_tag="${th/./p}"
  out_folder="videos/inference/forcingkv_longlive_similarity_disabled/th${th_tag}"
  log_file="$LOG_DIR/run_th${th_tag}.log"

  sed -i -E "s/^  similarity_prune_enabled:.*/  similarity_prune_enabled: false/" "$CFG"
  sed -i -E "s/^  similarity_threshold:.*/  similarity_threshold: ${th}/" "$CFG"
  sed -i -E "s#^output_folder:.*#output_folder: ${out_folder}#" "$CFG"

  echo "===== START similarity_prune_enabled=false, similarity_threshold=${th} =====" | tee "$log_file"
  set +e
  stdbuf -oL -eL bash -lc "source /home/ycji/miniconda3/etc/profile.d/conda.sh && \
    conda activate dummyforcing && \
    export PYTHONUNBUFFERED=1 && \
    cd \"$ROOT\" && \
    sh inference.sh" 2>&1 | tee -a "$log_file"
  run_rc=${PIPESTATUS[0]}
  set -e
  echo "===== END similarity_prune_enabled=false, similarity_threshold=${th}, rc=${run_rc} =====" | tee -a "$log_file"

  fps_values=$(grep -E "FPS:" "$log_file" | awk '{print $(NF-1)}' || true)
  fps_count=$(echo "$fps_values" | awk 'NF{c++} END{print c+0}')
  fps_avg=$(echo "$fps_values" | awk 'NF{s+=$1; c++} END{if(c>0) printf "%.4f", s/c; else printf "NaN"}')

  if [[ "$run_rc" -eq 0 ]]; then
    status="ok"
  else
    status="failed(${run_rc})"
  fi

  echo "false,${th},${out_folder},${fps_count},${fps_avg},${status},${log_file}" | tee -a "$SUMMARY"
done

printf "\nSaved summary: %s\n" "$SUMMARY"
