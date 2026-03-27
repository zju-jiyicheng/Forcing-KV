#!/usr/bin/env bash
set -euo pipefail

ROOT="/nfs/ycji_temp/code/DummyForcing"
CFG="$ROOT/configs/forcingkv_longlive_inference.yaml"
SUMMARY_DIR="$ROOT/experiment_logs/forcingkv_similarity_temporalctx_sweep_20260313"
SUMMARY="$SUMMARY_DIR/summary.csv"

mkdir -p "$SUMMARY_DIR"

BACKUP="${CFG}.bak_codex_temporalctx_$(date +%s)"
cp "$CFG" "$BACKUP"
restore_cfg() {
  cp "$BACKUP" "$CFG"
  rm -f "$BACKUP"
}
trap restore_cfg EXIT

if [[ ! -f "$SUMMARY" ]]; then
  echo "temporal_context_length,similarity_threshold,similarity_prune_enabled,kv_merge_enabled,ar_start,num_output_frames,data_path,output_folder,fps_count,fps_avg,status,log_file" > "$SUMMARY"
fi

run_one() {
  local ctx="$1"
  local th="$2"
  local ctx_tag="ctx${ctx}"
  local th_tag="th${th//./p}"
  local out_folder="videos/inference/forcingkv_longlive_similarity_temporalctx/${ctx_tag}_${th_tag}"
  local log_file="$SUMMARY_DIR/run_${ctx_tag}_${th_tag}.log"

  python - "$CFG" "$ctx" "$th" "$out_folder" <<'PY'
import sys
from pathlib import Path
cfg_path = Path(sys.argv[1])
ctx = sys.argv[2]
th = sys.argv[3]
out_folder = sys.argv[4]
lines = cfg_path.read_text().splitlines()

repls = {
    "ar_start": f"  ar_start: {ctx and 4}",
    "temporal_context_length": f"  temporal_context_length: {ctx}",
    "kv_merge_enabled": "  kv_merge_enabled: false",
    "similarity_prune_enabled": "  similarity_prune_enabled: true",
    "similarity_threshold": f"  similarity_threshold: {th}",
    "data_path": "data_path: prompts/example_prompts.txt",
    "num_output_frames": "num_output_frames: 240",
    "output_folder": f"output_folder: {out_folder}",
}

for i, line in enumerate(lines):
    s = line.strip()
    if s.startswith("ar_start:"):
        lines[i] = repls["ar_start"]
    elif s.startswith("temporal_context_length:"):
        lines[i] = repls["temporal_context_length"]
    elif s.startswith("kv_merge_enabled:"):
        lines[i] = repls["kv_merge_enabled"]
    elif s.startswith("similarity_prune_enabled:"):
        lines[i] = repls["similarity_prune_enabled"]
    elif s.startswith("similarity_threshold:"):
        lines[i] = repls["similarity_threshold"]
    elif s.startswith("data_path:"):
        lines[i] = repls["data_path"]
    elif s.startswith("num_output_frames:"):
        lines[i] = repls["num_output_frames"]
    elif s.startswith("output_folder:"):
        lines[i] = repls["output_folder"]

cfg_path.write_text("\n".join(lines) + "\n")
PY

  echo "===== START ctx=${ctx}, th=${th} ====="
  set +e
  (cd "$ROOT" && source /home/ycji/miniconda3/etc/profile.d/conda.sh && conda activate dummyforcing && bash inference.sh) 2>&1 | tee "$log_file"
  rc=${PIPESTATUS[0]}
  set -e
  echo "===== END ctx=${ctx}, th=${th}, rc=${rc} ====="

  local fps_count fps_avg status
  if [[ $rc -eq 0 ]]; then
    fps_count=$(grep -oE 'FPS: [0-9]+(\.[0-9]+)?' "$log_file" | wc -l | tr -d ' ')
    if [[ "$fps_count" -gt 0 ]]; then
      fps_avg=$(grep -oE 'FPS: [0-9]+(\.[0-9]+)?' "$log_file" | awk '{sum+=$2; n+=1} END {if(n>0) printf "%.4f", sum/n; else print "NA"}')
    else
      fps_avg="NA"
    fi
    status="ok"
  else
    fps_count=0
    fps_avg="NA"
    status="fail"
  fi

  echo "${ctx},${th},true,false,4,240,prompts/example_prompts.txt,${out_folder},${fps_count},${fps_avg},${status},${log_file}" >> "$SUMMARY"
  echo "saved: ${ctx}, ${th}, fps_avg=${fps_avg}, status=${status}"
}

run_one 1 0.0
run_one 3 0.0
run_one 2 0.0
run_one 1 0.2
run_one 1 0.5
run_one 1 0.8

echo "Saved summary: $SUMMARY"
