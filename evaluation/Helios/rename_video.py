import argparse
import csv
import re
import sys
import uuid
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4"}
FRAME_PATTERN = re.compile(r"(\d+)frame(?:_|$)")
TARGET_NAME_PATTERN = re.compile(r"^(\d+)_(\d+)_ori(\d+)\.mp4$", re.IGNORECASE)
TRAILING_INDEX_PATTERN = re.compile(r"^(.*)-(\d+)$")

# Truncated 30-char prompt prefixes can collide inside helios_720.
# This override was validated manually from the video content.
PROMPT_MATCH_OVERRIDES = {
    ("720", "A dynamic and lively scene in -0.mp4"): "179",
}


def infer_frame_from_path(path: Path) -> int | None:
    for candidate in (path, *path.parents):
        match = FRAME_PATTERN.search(candidate.name)
        if match:
            return int(match.group(1))
    return None


def get_video_files(directory: Path) -> list[Path]:
    def sort_key(item: Path) -> tuple[int, int | str]:
        match = TARGET_NAME_PATTERN.match(item.name)
        if match:
            return (0, int(match.group(1)))
        return (1, item.name)

    return sorted(
        (
            child
            for child in directory.iterdir()
            if child.is_file() and child.suffix.lower() in VIDEO_EXTENSIONS
        ),
        key=sort_key,
    )


def plan_renames(video_files: list[Path], frame: int) -> list[tuple[Path, Path]]:
    renames = []
    for idx, source in enumerate(video_files):
        target = source.with_name(f"{idx}_{frame}_ori{frame}{source.suffix.lower()}")
        renames.append((source, target))
    return renames


def load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def extract_prompt_prefix(path: Path) -> str:
    stem = path.stem
    match = TRAILING_INDEX_PATTERN.match(stem)
    if match:
        return match.group(1)
    return stem


def resolve_prompt_row(
    source: Path,
    csv_rows: list[dict[str, str]],
    frame: int,
) -> dict[str, str]:
    duration = str(frame)
    prompt_prefix = extract_prompt_prefix(source)
    matches = [
        row
        for row in csv_rows
        if row.get("duration") == duration and row.get("prompt", "").startswith(prompt_prefix)
    ]
    if len(matches) == 1:
        return matches[0]

    override_id = PROMPT_MATCH_OVERRIDES.get((duration, source.name))
    if override_id is not None:
        override_matches = [row for row in matches if row.get("id") == override_id]
        if len(override_matches) == 1:
            return override_matches[0]

    if len(matches) == 0:
        raise ValueError(f"no CSV prompt match for {source.name!r} with duration={duration}")

    candidate_ids = ", ".join(row.get("id", "?") for row in matches)
    raise ValueError(f"ambiguous CSV prompt match for {source.name!r}: ids={candidate_ids}")


def plan_prompt_csv_renames(
    video_files: list[Path],
    frame: int,
    csv_rows: list[dict[str, str]],
) -> list[tuple[Path, Path]]:
    renames = []
    used_ids: set[str] = set()
    valid_ids = {row["id"] for row in csv_rows if row.get("duration") == str(frame)}
    for source in video_files:
        existing_match = TARGET_NAME_PATTERN.match(source.name)
        if existing_match and int(existing_match.group(2)) == frame and int(existing_match.group(3)) == frame:
            video_id = existing_match.group(1)
            if video_id not in valid_ids:
                raise ValueError(f"existing file {source.name!r} has id not found in CSV duration={frame}")
        else:
            row = resolve_prompt_row(source, csv_rows, frame)
            video_id = row["id"]
        if video_id in used_ids:
            raise ValueError(f"duplicate CSV id resolved for {source.name!r}: {video_id}")
        used_ids.add(video_id)
        target = source.with_name(f"{video_id}_{frame}_ori{frame}{source.suffix.lower()}")
        renames.append((source, target))
    return renames


def rename_files(renames: list[tuple[Path, Path]], dry_run: bool) -> None:
    temp_pairs: list[tuple[Path, Path]] = []
    for source, _ in renames:
        temp_path = source.with_name(f".rename_tmp_{uuid.uuid4().hex}{source.suffix.lower()}")
        temp_pairs.append((source, temp_path))

    for source, temp_path in temp_pairs:
        print(f"{source.name} -> {temp_path.name}")
        if not dry_run:
            source.rename(temp_path)

    for (_, target), (_, temp_path) in zip(renames, temp_pairs, strict=True):
        print(f"{temp_path.name} -> {target.name}")
        if not dry_run:
            temp_path.rename(target)


def process_directory(directory: Path, frame: int, dry_run: bool, csv_rows: list[dict[str, str]] | None = None) -> None:
    video_files = get_video_files(directory)
    if not video_files:
        print(f"[skip] {directory}: no top-level mp4 files found")
        return

    if csv_rows is None:
        renames = plan_renames(video_files, frame)
        mode = "sequential"
    else:
        renames = plan_prompt_csv_renames(video_files, frame, csv_rows)
        mode = "prompt_csv"

    unchanged = sum(1 for source, target in renames if source.name == target.name)
    print(f"[dir] {directory}")
    print(f"  frame={frame}, videos={len(video_files)}, unchanged={unchanged}, mode={mode}")

    if unchanged == len(renames):
        print("  already in target naming format and order")
        return

    rename_files(renames, dry_run=dry_run)


def resolve_directories(root: Path) -> list[Path]:
    child_dirs = sorted(child for child in root.iterdir() if child.is_dir())
    if child_dirs:
        return child_dirs
    return [root]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rename top-level videos in each immediate child directory to id_frame_oriframe.mp4 format."
    )
    parser.add_argument("root_dir", type=Path, help="Directory containing model/video subdirectories, or a single video dir")
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Frame value used in '<id>_<frame>_ori<frame>.mp4'. If omitted, infer from directory names like '240frame_60s'.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Optional CSV with id,prompt,duration columns. When set, match video filenames against truncated prompt prefixes.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the rename plan without changing files")
    args = parser.parse_args()

    root_dir = args.root_dir.expanduser().resolve()
    if not root_dir.exists():
        print(f"Error: path does not exist: {root_dir}", file=sys.stderr)
        return 1
    if not root_dir.is_dir():
        print(f"Error: path is not a directory: {root_dir}", file=sys.stderr)
        return 1

    directories = resolve_directories(root_dir)
    if not directories:
        print(f"Error: no directories found under {root_dir}", file=sys.stderr)
        return 1

    csv_rows = None
    if args.csv_path is not None:
        csv_path = args.csv_path.expanduser().resolve()
        if not csv_path.exists():
            print(f"Error: CSV path does not exist: {csv_path}", file=sys.stderr)
            return 1
        csv_rows = load_csv_rows(csv_path)

    for directory in directories:
        frame = args.frame if args.frame is not None else infer_frame_from_path(directory)
        if frame is None:
            print(
                f"Error: could not infer frame from {directory}. Pass --frame explicitly.",
                file=sys.stderr,
            )
            return 1
        process_directory(directory, frame=frame, dry_run=args.dry_run, csv_rows=csv_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
