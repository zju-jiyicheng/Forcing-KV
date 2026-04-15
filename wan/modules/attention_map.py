import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import copy
import math
import os
import time


def save_head_attention_map(query, key, ar_step, layer_idx, save_path):
    """
    query: [B, seq_len_q, H, C]
    key:   [B, seq_len_k, H, C]
    Save heatmaps with 10% probability. Y=Query index, X=Key index.
    Fixes:
      (1) Use robust color scaling (log + percentile clip) to avoid "all near 0" look.
      (2) Set figure size according to seq_len_q:seq_len_k ratio so it doesn't look too tall.
    """
    B, seq_len_q, H, C = query.shape
    B2, seq_len_k, H2, C2 = key.shape
    assert B == B2 and H == H2 and C == C2

    os.makedirs(save_path, exist_ok=True)

    # Compute full attention: [B, H, Q, K]
    q = query.transpose(1, 2)  # [B, H, Q, C]
    k = key.transpose(1, 2)    # [B, H, K, C]
    scores = torch.matmul(q, k.transpose(-2, -1)) / (C ** 0.5)  # [B, H, Q, K]
    attn = F.softmax(scores, dim=-1)                             # [B, H, Q, K]

    attn0 = attn[0].detach().float().cpu()  # [H, Q, K]
    Q, K = attn0.shape[1], attn0.shape[2]

    # X-axis ticks every seq_len_q
    tick_step = seq_len_q
    xticks = list(range(0, K + 1, tick_step))

    # figure size proportional to K:Q (width:height)
    base_h = 6.0
    fig_h = base_h
    fig_w = base_h * (K / max(1, Q))  # keep aspect visually consistent with matrix
    fig_w = max(6.0, min(fig_w, 24.0))  # clamp so it doesn't get ridiculous

    # timestamp for unique filenames
    ts = time.strftime("%Y%m%d_%H%M%S")
    ts_ms = int(time.time() * 1000) % 1000  # add ms to reduce collision risk
    stamp = f"{ts}_{ts_ms:03d}"

    # robust color scaling helper
    # use log1p to enhance small values + percentile clip for dynamic range
    def _robust_vmin_vmax(a2d: torch.Tensor):
        flat = a2d.reshape(-1)
        # avoid huge overhead: take a subsample if very large
        if flat.numel() > 2_000_000:
            idx = torch.randint(0, flat.numel(), (2_000_000,))
            flat = flat[idx]
        # log space percentiles
        flat_log = torch.log1p(flat)
        vmin = torch.quantile(flat_log, 0.05).item()
        vmax = torch.quantile(flat_log, 0.995).item()
        # guard: if too close, widen a bit
        if vmax - vmin < 1e-6:
            vmax = vmin + 1e-3
        return vmin, vmax

    for h in range(attn0.shape[0]):
        a = attn0[h]                       # [Q, K]
        a_log = torch.log1p(a)             # enhance contrast for small probs
        vmin, vmax = _robust_vmin_vmax(a)  # compute on log space

        plt.figure(figsize=(fig_w, fig_h))
        plt.imshow(
            a_log.numpy(),
            aspect="auto",
            interpolation="nearest",
            cmap="plasma",   # purple <-> yellow
            vmin=vmin,
            vmax=vmax,
        )
        plt.xlabel("Key index")
        plt.ylabel("Query index")
        plt.title(f"AR_step={ar_step}_layer={layer_idx}_head={h}")

        plt.xticks(xticks, [str(x) for x in xticks], rotation=0)
        for x in xticks:
            plt.axvline(x=x, linewidth=0.8, alpha=0.6)

        plt.colorbar()
        plt.tight_layout()
        out = os.path.join(save_path, f"ar{ar_step:04d}_layer{layer_idx:02d}head{h:02d}_{stamp}.png")
        plt.savefig(out, dpi=200)
        plt.close()

        print(f"Saved attention map: {out}")


def save_head_attention_map_v2(query, key, ar_step, layer_idx, head_indices, save_path):
    """
    query: [B, seq_len_q, H, C]
    key:   [B, seq_len_k, H, C]
    head_indices: iterable of selected heads to save as .pt files
    """
    B, seq_len_q, H, C = query.shape
    B2, seq_len_k, H2, C2 = key.shape
    assert B == B2 and H == H2 and C == C2

    if isinstance(head_indices, int):
        head_indices = [head_indices]

    selected_heads = sorted({int(h) for h in head_indices})
    if not selected_heads:
        return

    for h in selected_heads:
        if h < 0 or h >= H:
            raise ValueError(f"head index {h} is out of range for total heads {H}")

    os.makedirs(save_path, exist_ok=True)

    q = query[:, :, selected_heads].transpose(1, 2)  # [B, H_sel, Q, C]
    k = key[:, :, selected_heads].transpose(1, 2)    # [B, H_sel, K, C]
    scores = torch.matmul(q, k.transpose(-2, -1)) / (C ** 0.5)
    attn = F.softmax(scores, dim=-1)
    attn0 = attn[0].detach().float().cpu()

    ts = time.strftime("%Y%m%d_%H%M%S")
    ts_ms = int(time.time() * 1000) % 1000
    stamp = f"{ts}_{ts_ms:03d}"

    for local_idx, head_idx in enumerate(selected_heads):
        head_attn = attn0[local_idx].clone()
        payload = {
            "attn": head_attn,
            "ar_step": int(ar_step),
            "layer_idx": int(layer_idx),
            "head_idx": int(head_idx),
            "query_len": int(seq_len_q),
            "key_len": int(seq_len_k),
            "timestamp": stamp,
        }
        out = os.path.join(
            save_path,
            f"ar{ar_step:04d}_layer{layer_idx:02d}head{head_idx:02d}_{stamp}.pt",
        )
        torch.save(payload, out)
        print(f"Saved attention pt: {out}")
