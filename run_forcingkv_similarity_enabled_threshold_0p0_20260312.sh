#!/usr/bin/env bash
set -euo pipefail

ROOT="/nfs/ycji_temp/code/DummyForcing"
CFG="$ROOT/configs/forcingkv_longlive_inference.yaml"
SUMMARY_DIR="$ROOT/experiment_logs/forcingkv_similarity_enabled_threshold_sweep_20260312"
SUMMARY="$SUMMARY_DIR/summary.csv"
TH="0.0"
TH_TAG="th0p0"
LOG_FILE="$SUMMARY_DIR/run_${TH_TAG}.log"

mkdir -p "$SUMMARY_DIR"

BACKUP="${CFG}.bak_codex_0p0_$(date +%s)"
cp "$CFG" "$BACKUP"
restore_cfg() {
  cp "$BACKUP" "$CFG"
  rm -f "$BACKUP"
}
trap restore_cfg EXIT

if [[ ! -f "$SUMMARY" ]]; then
  echo "similarity_prune_enabled,similarity_threshold,kv_merge_enabled,data_path,output_folder,fps_count,fps_avg,status,log_file" > "$SUMMARY"
fi

python - "$CFG" "$TH" <<'PY'
import sys
from pathlib import Path
cfg_path = Path(sys.argv[1])
th = sys.argv[2]
text = cfg_path.read_text()
repls = {
    "  kv_merge_enabled: true": "  kv_merge_enabled: false",
    "  kv_merge_enabled: false": "  kv_merge_enabled: false",
    "  similarity_prune_enabled: false": "  similarity_prune_enabled: true",
    "  similarity_prune_enabled: true": "  similarity_prune_enabled: true",
    "  similarity_threshold: 0.2": f"  similarity_threshold: {th}",
    "  similarity_threshold: 0.9": f"  similarity_threshold: {th}",
    "  similarity_threshold: 0.5": f"  similarity_threshold: {th}",
    "  similarity_threshold: 0.3": f"  similarity_threshold: {th}",
    "  similarity_threshold: 0.1": f"  similarity_threshold: {th}",
    "  similarity_threshold: 0.0": f"  similarity_threshold: {th}",
    "data_path: prompts/example_prompts_5.txt": "data_path: prompts/example_prompts.txt",
    "data_path: prompts/example_prompts.txt": "data_path: prompts/example_prompts.txt",
}
for a,b in repls.items():
    text = text.replace(a,b)
text = text.replace(
    "output_folder: videos/inference/forcingkv_longlive_merge_human_head/ar4_m_0.99",
    "output_folder: videos/inference/forcingkv_longlive_similarity_enabled/th0p0",
)
cfg_path.write_text(text)
PY

cd "$ROOT"
source /home/ycji/miniconda3/etc/profile.d/conda.sh
conda activate dummyforcing

echo "===== START similarity_prune_enabled=true, similarity_threshold=${TH}, kv_merge_enabled=false ====="
set +e
bash inference.sh 2>&1 | tee "$LOG_FILE"
rc=${PIPESTATUS[0]}
set -e
echo "===== END similarity_prune_enabled=true, similarity_threshold=${TH}, rc=${rc} ====="

if [[ $rc -eq 0 ]]; then
  fps_count=$(grep -oE 'FPS: [0-9]+(\.[0-9]+)?' "$LOG_FILE" | wc -l | tr -d ' ')
  if [[ "$fps_count" -gt 0 ]]; then
    fps_avg=$(grep -oE 'FPS: [0-9]+(\.[0-9]+)?' "$LOG_FILE" | awk '{sum+=$2; n+=1} END {if(n>0) printf "%.4f", sum/n; else print "NA"}')
  else
    fps_avg="NA"
  fi
  status="ok"
else
  fps_count=0
  fps_avg="NA"
  status="fail"
fi

echo "true,${TH},false,prompts/example_prompts.txt,videos/inference/forcingkv_longlive_similarity_enabled/${TH_TAG},${fps_count},${fps_avg},${status},${LOG_FILE}" >> "$SUMMARY"
echo "Saved summary: $SUMMARY"
