import argparse
import json
from pathlib import Path

import torch


THRESHOLD = 0.5
SKIP_FIRST_K = 1
LAST_K = 4

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a head config from one frame-wise attention directory using forcingkv."
    )
    parser.add_argument(
        "--attn-dir",
        type=Path,
        required=True,
        help="Directory containing frame-wise attention .pt files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to save the predicted head config json.",
    )
    return parser.parse_args()


def compute_attn_score(attn: torch.Tensor, last_k: int, skip_first_k: int) -> float:
    if attn.ndim != 2:
        raise ValueError(f"expected 2D attention, got shape {tuple(attn.shape)}")
    if skip_first_k < 0:
        raise ValueError("skip_first_k must be non-negative")
    if skip_first_k >= attn.shape[1]:
        raise ValueError(
            f"skip_first_k={skip_first_k} must be smaller than key frames={attn.shape[1]}"
        )

    last_k = min(last_k, attn.shape[1])
    valid_attn = attn[:, skip_first_k:]
    valid_total = float(valid_attn.sum().item())
    if valid_total <= 0:
        raise ValueError("attention mass must be positive")

    last_mass = float(attn[:, -last_k:].sum().item())
    return last_mass / valid_total


def load_scores(attn_dir: Path, last_k: int, skip_first_k: int):
    records = {}
    for path in sorted(attn_dir.glob("*.pt")):
        payload = torch.load(path, map_location="cpu")
        layer_idx = int(payload["layer_idx"])
        head_idx = int(payload["head_idx"])
        attn = payload["attn"].float()
        score = compute_attn_score(
            attn=attn,
            last_k=last_k,
            skip_first_k=skip_first_k,
        )
        records[(layer_idx, head_idx)] = score
    return records


def build_head_config(scores):
    layers = sorted({layer_idx for layer_idx, _ in scores})
    heads = sorted({head_idx for _, head_idx in scores})
    num_layers = len(layers)
    num_heads = len(heads)

    output_layers = []
    for layer_idx in layers:
        local_head = []
        temporal_head = []
        for head_idx in heads:
            key = (layer_idx, head_idx)
            if key not in scores:
                raise ValueError(f"missing score for layer={layer_idx}, head={head_idx}")
            if scores[key] >= THRESHOLD:
                local_head.append(head_idx)
            else:
                temporal_head.append(head_idx)
        output_layers.append(
            {
                "layer_idx": layer_idx,
                "local_head": local_head,
                "temporal_head": temporal_head,
            }
        )

    return {
        "format": "forcingkv_offline",
        "profile": "nosink_3",
        "num_layers": num_layers,
        "num_heads": num_heads,
        "layers": output_layers,
    }


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main():
    args = parse_args()
    scores = load_scores(
        attn_dir=args.attn_dir,
        last_k=LAST_K,
        skip_first_k=SKIP_FIRST_K,
    )
    config = build_head_config(scores=scores)
    save_json(args.output_path, config)
    print(f"Saved head config to {args.output_path}")


if __name__ == "__main__":
    main()
