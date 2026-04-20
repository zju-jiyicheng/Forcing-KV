#!/usr/bin/env bash
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh

DEFAULT_VIDEOS_PATH='/ycji/code/Forcing-KV/videos_new/test/longlive_30s_attn12'
DEFAULT_CUDA_VISIBLE_DEVICES='0'
DEFAULT_MASTER_PORT_BASE='38557'

VIDEOS_PATH="${1:-$DEFAULT_VIDEOS_PATH}"
CUDA_VISIBLE_DEVICES_ARG="${2:-$DEFAULT_CUDA_VISIBLE_DEVICES}"
MASTER_PORT_BASE="${3:-$DEFAULT_MASTER_PORT_BASE}"

export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_ARG"

CLIP_SECONDS=5
MIN_DURATION_SECONDS=10
DIMENSIONS=(
    "subject_consistency"
    "background_consistency"
    "aesthetic_quality"
    "imaging_quality"
    "motion_smoothness"
    "dynamic_degree"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORCING_KV_DIR="$SCRIPT_DIR"

cd "$FORCING_KV_DIR"
cd ..
cd ./VBench
VBENCH_DIR="$(pwd)"
cd "$FORCING_KV_DIR"

conda activate vbenchlong

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg not found after activating the vbenchlong conda environment." >&2
    exit 1
fi

if ! command -v ffprobe >/dev/null 2>&1; then
    echo "ffprobe not found after activating the vbenchlong conda environment." >&2
    exit 1
fi

if [[ ! -d "$VIDEOS_PATH" ]]; then
    echo "Input directory does not exist: $VIDEOS_PATH" >&2
    exit 1
fi

echo "Starting total drift evaluation"
echo "  VIDEOS_PATH=$VIDEOS_PATH"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  MASTER_PORT_BASE=$MASTER_PORT_BASE"

OUTPUT_ROOT="${VIDEOS_PATH}/vbench_drift_total"
CLIPS_ROOT="${OUTPUT_ROOT}/clips"
FIRST_CLIPS_DIR="${CLIPS_ROOT}/first5"
LAST_CLIPS_DIR="${CLIPS_ROOT}/last5"
FIRST_EVAL_DIR="${OUTPUT_ROOT}/eval_first5"
LAST_EVAL_DIR="${OUTPUT_ROOT}/eval_last5"
MANIFEST_PATH="${OUTPUT_ROOT}/video_manifest.jsonl"
FIRST_ZIP_PATH="${OUTPUT_ROOT}/first5_vbenchlong_results.zip"
LAST_ZIP_PATH="${OUTPUT_ROOT}/last5_vbenchlong_results.zip"
FIRST_RESULT_JSON_PATH="${FIRST_EVAL_DIR}/result.json"
LAST_RESULT_JSON_PATH="${LAST_EVAL_DIR}/result.json"
SUMMARY_JSON_PATH="${OUTPUT_ROOT}/drift_summary.json"
FIRST_SUBMISSION_DIR="${OUTPUT_ROOT}/submission_first5"
LAST_SUBMISSION_DIR="${OUTPUT_ROOT}/submission_last5"

python - "$OUTPUT_ROOT" <<'PY'
import pathlib
import shutil
import sys

output_root = pathlib.Path(sys.argv[1])
if output_root.exists():
    shutil.rmtree(output_root)
PY

mkdir -p "$FIRST_CLIPS_DIR" "$LAST_CLIPS_DIR" "$FIRST_EVAL_DIR" "$LAST_EVAL_DIR"

echo "Step 1/3: split videos into first ${CLIP_SECONDS}s and last ${CLIP_SECONDS}s clips"
python - "$VIDEOS_PATH" "$FIRST_CLIPS_DIR" "$LAST_CLIPS_DIR" "$MANIFEST_PATH" "$CLIP_SECONDS" "$MIN_DURATION_SECONDS" <<'PY'
import json
import pathlib
import subprocess
import sys

videos_path = pathlib.Path(sys.argv[1])
first_dir = pathlib.Path(sys.argv[2])
last_dir = pathlib.Path(sys.argv[3])
manifest_path = pathlib.Path(sys.argv[4])
clip_seconds = float(sys.argv[5])
min_duration_seconds = float(sys.argv[6])

video_paths = sorted(videos_path.glob("*.mp4"))
if not video_paths:
    raise SystemExit(f"No .mp4 files found directly under {videos_path}")

print(f"Found {len(video_paths)} videos under {videos_path}", flush=True)

with manifest_path.open("w", encoding="utf-8") as manifest_file:
    for idx, video_path in enumerate(video_paths, start=1):
        print(f"  Splitting {idx}/{len(video_paths)}: {video_path.name}", flush=True)

        probe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        probe_result = subprocess.run(
            probe_cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        duration = float(probe_result.stdout.strip())
        if duration < min_duration_seconds:
            raise SystemExit(
                f"Video is shorter than {min_duration_seconds:.0f}s, aborting: "
                f"{video_path} ({duration:.3f}s)"
            )

        stem = video_path.stem
        first_clip_path = first_dir / f"{stem}__first5.mp4"
        last_clip_path = last_dir / f"{stem}__last5.mp4"
        last_start = duration - clip_seconds

        first_cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-t",
            f"{clip_seconds:.3f}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(first_clip_path),
        ]
        subprocess.run(first_cmd, check=True)

        last_cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{last_start:.3f}",
            "-i",
            str(video_path),
            "-t",
            f"{clip_seconds:.3f}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(last_clip_path),
        ]
        subprocess.run(last_cmd, check=True)

        manifest = {
            "video_name": video_path.name,
            "source_video": str(video_path.resolve()),
            "duration_seconds": duration,
            "first_clip": str(first_clip_path.resolve()),
            "last_clip": str(last_clip_path.resolve()),
        }
        manifest_file.write(json.dumps(manifest, ensure_ascii=True) + "\n")

print(f"Saved split manifest to {manifest_path}", flush=True)
PY

run_vbenchlong_eval() {
    local clips_dir="$1"
    local eval_dir="$2"
    local dimension="$3"
    local eval_index="$4"
    local eval_total="$5"
    local clip_label="$6"

    echo "Step 2/3 (${eval_index}/${eval_total}): evaluate ${clip_label} ${dimension}"
    cd "$VBENCH_DIR"
    python vbench2_beta_long/eval_long.py \
        --videos_path "$clips_dir" \
        --output_path "$eval_dir" \
        --dimension "$dimension" \
        --mode long_custom_input \
        --dev_flag
    cd "$FORCING_KV_DIR"
}

eval_step=1
eval_total=$(( ${#DIMENSIONS[@]} * 2 ))
for dimension in "${DIMENSIONS[@]}"; do
    run_vbenchlong_eval "$FIRST_CLIPS_DIR" "$FIRST_EVAL_DIR" "$dimension" "$eval_step" "$eval_total" "first5"
    eval_step=$((eval_step + 1))
done

for dimension in "${DIMENSIONS[@]}"; do
    run_vbenchlong_eval "$LAST_CLIPS_DIR" "$LAST_EVAL_DIR" "$dimension" "$eval_step" "$eval_total" "last5"
    eval_step=$((eval_step + 1))
done

echo "Step 3/3: calculate folder-level total scores and drift summary"
python - "$FIRST_EVAL_DIR" "$FIRST_ZIP_PATH" "$LAST_EVAL_DIR" "$LAST_ZIP_PATH" <<'PY'
import pathlib
import sys
import zipfile

def zip_dir(src_dir: pathlib.Path, zip_path: pathlib.Path):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src_dir))

first_eval_dir = pathlib.Path(sys.argv[1])
first_zip_path = pathlib.Path(sys.argv[2])
last_eval_dir = pathlib.Path(sys.argv[3])
last_zip_path = pathlib.Path(sys.argv[4])

zip_dir(first_eval_dir, first_zip_path)
zip_dir(last_eval_dir, last_zip_path)

print(f"Created {first_zip_path}", flush=True)
print(f"Created {last_zip_path}", flush=True)
PY

echo "  Calculating first5 total score"
cd "$VBENCH_DIR"
python scripts/cal_long_final_score.py \
    --zip_file "$FIRST_ZIP_PATH" \
    --model_name "$FIRST_SUBMISSION_DIR" \
    --output_path "$FIRST_EVAL_DIR"

echo "  Calculating last5 total score"
python scripts/cal_long_final_score.py \
    --zip_file "$LAST_ZIP_PATH" \
    --model_name "$LAST_SUBMISSION_DIR" \
    --output_path "$LAST_EVAL_DIR"
cd "$FORCING_KV_DIR"

python - "$FIRST_RESULT_JSON_PATH" "$LAST_RESULT_JSON_PATH" "$SUMMARY_JSON_PATH" "$VIDEOS_PATH" <<'PY'
import json
import pathlib
import sys
from datetime import datetime

first_result_json_path = pathlib.Path(sys.argv[1])
last_result_json_path = pathlib.Path(sys.argv[2])
summary_json_path = pathlib.Path(sys.argv[3])
videos_path = pathlib.Path(sys.argv[4]).resolve()

if not first_result_json_path.exists():
    raise SystemExit(f"Missing first5 result.json: {first_result_json_path}")
if not last_result_json_path.exists():
    raise SystemExit(f"Missing last5 result.json: {last_result_json_path}")

with first_result_json_path.open("r", encoding="utf-8") as f:
    first_result = json.load(f)
with last_result_json_path.open("r", encoding="utf-8") as f:
    last_result = json.load(f)

score_key = "total & quality_score"
if score_key not in first_result:
    raise SystemExit(f"Missing '{score_key}' in {first_result_json_path}")
if score_key not in last_result:
    raise SystemExit(f"Missing '{score_key}' in {last_result_json_path}")

first_total_score = float(first_result[score_key])
last_total_score = float(last_result[score_key])

summary = {
    "videos_path": str(videos_path),
    "mean_first_total_score": first_total_score,
    "mean_last_total_score": last_total_score,
    "drift_total": first_total_score - last_total_score,
    "drift_formula": "first_5s_total_score - last_5s_total_score",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}

with summary_json_path.open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=True, indent=2)
    f.write("\n")

print(f"Wrote total drift summary to {summary_json_path}", flush=True)
PY

echo "Total drift evaluation finished."
echo "Output root: $OUTPUT_ROOT"
echo "First total result: $FIRST_RESULT_JSON_PATH"
echo "Last total result: $LAST_RESULT_JSON_PATH"
echo "Drift summary: $SUMMARY_JSON_PATH"
