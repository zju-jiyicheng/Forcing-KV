#!/usr/bin/env bash
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh

DEFAULT_VIDEOS_PATH='/ycji/code/Forcing-KV/videos_new/vbench/dummy_longlive_30s_ctx1'
DEFAULT_CUDA_VISIBLE_DEVICES='0'
DEFAULT_MASTER_PORT_BASE='38557'

VIDEOS_PATH="${1:-$DEFAULT_VIDEOS_PATH}"
CUDA_VISIBLE_DEVICES_ARG="${2:-$DEFAULT_CUDA_VISIBLE_DEVICES}"
MASTER_PORT_BASE="${3:-$DEFAULT_MASTER_PORT_BASE}"

export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_ARG"

CLIP_SECONDS=5
MIN_DURATION_SECONDS=10
DIMENSIONS=("imaging_quality" "dynamic_degree")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORCING_KV_DIR="$SCRIPT_DIR"

cd "$FORCING_KV_DIR"
cd ..
cd ./VBench
VBENCH_DIR="$(pwd)"
cd "$FORCING_KV_DIR"

conda activate vbench

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg not found after activating the vbench conda environment." >&2
    exit 1
fi

if ! command -v ffprobe >/dev/null 2>&1; then
    echo "ffprobe not found after activating the vbench conda environment." >&2
    exit 1
fi

if [[ ! -d "$VIDEOS_PATH" ]]; then
    echo "Input directory does not exist: $VIDEOS_PATH" >&2
    exit 1
fi

echo "Starting drift evaluation"
echo "  VIDEOS_PATH=$VIDEOS_PATH"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  MASTER_PORT_BASE=$MASTER_PORT_BASE"

OUTPUT_ROOT="${VIDEOS_PATH}/vbench_drift"
CLIPS_ROOT="${OUTPUT_ROOT}/clips"
FIRST_CLIPS_DIR="${CLIPS_ROOT}/first5"
LAST_CLIPS_DIR="${CLIPS_ROOT}/last5"
FIRST_EVAL_DIR="${OUTPUT_ROOT}/eval_first5"
LAST_EVAL_DIR="${OUTPUT_ROOT}/eval_last5"
MANIFEST_PATH="${OUTPUT_ROOT}/video_manifest.jsonl"
DRIFT_JSONL_PATH="${OUTPUT_ROOT}/drift_metrics.jsonl"
SUMMARY_JSON_PATH="${OUTPUT_ROOT}/drift_summary.json"

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
        print(
            f"  Splitting {idx}/{len(video_paths)}: {video_path.name}",
            flush=True,
        )
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

run_vbench_eval() {
    local clips_dir="$1"
    local eval_dir="$2"
    local dimension="$3"
    local master_port="$4"
    local eval_index="$5"
    local eval_total="$6"
    local clip_label="$7"

    echo "Step 2/3 (${eval_index}/${eval_total}): evaluate ${clip_label} ${dimension}"
    cd "$VBENCH_DIR"
    MASTER_PORT="$master_port" python evaluate.py \
        --videos_path "$clips_dir" \
        --dimension "$dimension" \
        --mode custom_input \
        --output_path "$eval_dir"
    cd "$FORCING_KV_DIR"
}

port="$MASTER_PORT_BASE"
eval_step=1
eval_total=$(( ${#DIMENSIONS[@]} * 2 ))
for dimension in "${DIMENSIONS[@]}"; do
    run_vbench_eval "$FIRST_CLIPS_DIR" "$FIRST_EVAL_DIR" "$dimension" "$port" "$eval_step" "$eval_total" "first5"
    port=$((port + 1))
    eval_step=$((eval_step + 1))
done

for dimension in "${DIMENSIONS[@]}"; do
    run_vbench_eval "$LAST_CLIPS_DIR" "$LAST_EVAL_DIR" "$dimension" "$port" "$eval_step" "$eval_total" "last5"
    port=$((port + 1))
    eval_step=$((eval_step + 1))
done

echo "Step 3/3: aggregate drift metrics"
python - "$MANIFEST_PATH" "$FIRST_EVAL_DIR" "$LAST_EVAL_DIR" "$DRIFT_JSONL_PATH" "$SUMMARY_JSON_PATH" "$VIDEOS_PATH" <<'PY'
import json
import pathlib
import statistics
import sys
from datetime import datetime

manifest_path = pathlib.Path(sys.argv[1])
first_eval_dir = pathlib.Path(sys.argv[2])
last_eval_dir = pathlib.Path(sys.argv[3])
drift_jsonl_path = pathlib.Path(sys.argv[4])
summary_json_path = pathlib.Path(sys.argv[5])
videos_path = pathlib.Path(sys.argv[6]).resolve()


def load_dimension_scores(eval_dir: pathlib.Path, dimension: str):
    result_files = sorted(eval_dir.glob("results_*_eval_results.json"))
    if not result_files:
        raise SystemExit(f"No eval result files found in {eval_dir}")

    matched = []
    for result_file in result_files:
        with result_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if dimension in payload:
            matched.append((result_file, payload[dimension]))

    if len(matched) != 1:
        raise SystemExit(
            f"Expected exactly one result file for {dimension} in {eval_dir}, "
            f"found {len(matched)}"
        )

    _, result_value = matched[0]
    video_entries = result_value[1]
    scores = {}
    for item in video_entries:
        score = item["video_results"]
        if isinstance(score, bool):
            score = int(score)
        else:
            score = float(score)
        scores[str(pathlib.Path(item["video_path"]).resolve())] = score
    return scores


first_imaging = load_dimension_scores(first_eval_dir, "imaging_quality")
first_dynamic = load_dimension_scores(first_eval_dir, "dynamic_degree")
last_imaging = load_dimension_scores(last_eval_dir, "imaging_quality")
last_dynamic = load_dimension_scores(last_eval_dir, "dynamic_degree")

rows = []
with manifest_path.open("r", encoding="utf-8") as manifest_file:
    for line in manifest_file:
        item = json.loads(line)
        first_clip = str(pathlib.Path(item["first_clip"]).resolve())
        last_clip = str(pathlib.Path(item["last_clip"]).resolve())

        if first_clip not in first_imaging:
            raise SystemExit(f"Missing first imaging_quality score for {first_clip}")
        if first_clip not in first_dynamic:
            raise SystemExit(f"Missing first dynamic_degree score for {first_clip}")
        if last_clip not in last_imaging:
            raise SystemExit(f"Missing last imaging_quality score for {last_clip}")
        if last_clip not in last_dynamic:
            raise SystemExit(f"Missing last dynamic_degree score for {last_clip}")

        first_imaging_score = first_imaging[first_clip]
        last_imaging_score = last_imaging[last_clip]
        first_dynamic_score = first_dynamic[first_clip]
        last_dynamic_score = last_dynamic[last_clip]

        row = {
            "record_type": "video",
            "video_name": item["video_name"],
            "source_video": item["source_video"],
            "duration_seconds": round(float(item["duration_seconds"]), 6),
            "first_clip": item["first_clip"],
            "last_clip": item["last_clip"],
            "first_imaging_quality": first_imaging_score,
            "last_imaging_quality": last_imaging_score,
            "drift_quality": first_imaging_score - last_imaging_score,
            "first_dynamic_degree": first_dynamic_score,
            "last_dynamic_degree": last_dynamic_score,
            "drift_dynamic": first_dynamic_score - last_dynamic_score,
        }
        rows.append(row)

if not rows:
    raise SystemExit("No per-video rows were created.")

summary = {
    "record_type": "summary",
    "videos_path": str(videos_path),
    "num_videos": len(rows),
    "mean_first_imaging_quality": statistics.fmean(row["first_imaging_quality"] for row in rows),
    "mean_last_imaging_quality": statistics.fmean(row["last_imaging_quality"] for row in rows),
    "mean_drift_quality": statistics.fmean(row["drift_quality"] for row in rows),
    "mean_first_dynamic_degree": statistics.fmean(row["first_dynamic_degree"] for row in rows),
    "mean_last_dynamic_degree": statistics.fmean(row["last_dynamic_degree"] for row in rows),
    "mean_drift_dynamic": statistics.fmean(row["drift_dynamic"] for row in rows),
    "drift_formula": "first_5s_score - last_5s_score",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}

with drift_jsonl_path.open("w", encoding="utf-8") as out_file:
    for row in rows:
        out_file.write(json.dumps(row, ensure_ascii=True) + "\n")
    out_file.write(json.dumps(summary, ensure_ascii=True) + "\n")

compact_summary = {
    "drift_quality": summary["mean_drift_quality"],
    "drift_dynamic": summary["mean_drift_dynamic"],
}
with summary_json_path.open("w", encoding="utf-8") as out_file:
    json.dump(compact_summary, out_file, ensure_ascii=True)
    out_file.write("\n")

print(f"Wrote per-video drift metrics to {drift_jsonl_path}", flush=True)
print(f"Wrote compact drift summary to {summary_json_path}", flush=True)
PY

echo "Drift evaluation finished."
echo "Output root: $OUTPUT_ROOT"
echo "Manifest: $MANIFEST_PATH"
echo "Drift JSONL: $DRIFT_JSONL_PATH"
echo "Drift summary: $SUMMARY_JSON_PATH"
