import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import copy
import math
import os
import time



def online_head_classification(query, key, ar_start):
    B, L, head, C = query.shape
    HW = L//3 if ar_start ==1 else L
    num_sampled_rows = HW // 3
    sampled_rows = torch.randint(low=0, high=L, size=(num_sampled_rows,))
    sampled_q = query[:, sampled_rows]
    sampled_q = sampled_q.transpose(1, 2) # B, head, L, C
    key = key.transpose(1, 2) # B, head, 3L, C
    sampled_qk_scores = torch.matmul(sampled_q, key.transpose(-2, -1)) / (C ** 0.5)
    sampled_attn_weights = F.softmax(sampled_qk_scores, dim=-1) # B, head, L, 3L
    last_chunk_agg = sampled_attn_weights[:, :, :, -L:].sum(dim=-1).mean(dim=-1)
    mid_chunk_agg = sampled_attn_weights[:, :, :, HW:-L].sum(dim=-1).mean(dim=-1)
    first_chunk_agg = sampled_attn_weights[:, :, :, :HW].sum(dim=-1).mean(dim=-1)
    total_chunk_agg = torch.stack([first_chunk_agg, mid_chunk_agg, last_chunk_agg]) # 3, B, head
    return total_chunk_agg


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
        payload = {
            "attn": attn0[local_idx].contiguous(),
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



def dynamic_head_programming(probs, num_dummy=180):
    """
    probs: [num_layer, num_head, 3] tensor
    num_dummy: target number of elements in group C
    layer_threshold: layers below this use weight 1.0
    weight_after: weight multiplier for layers >= threshold
    Returns: three dicts {layer_idx: [head_indices]} for groups A, B, C
    """
    num_layer, num_head, _ = probs.shape
    p0_flat = probs[:, :, 0].reshape(-1)
    p1_flat = probs[:, :, 1].reshape(-1)
    p0_norm = p0_flat  / p0_flat.sum()
    p1_norm = p1_flat  / p1_flat.sum()
    cost = torch.maximum(p0_norm, p1_norm)
    sorted_indices = torch.argsort(cost)
    c_indices_flat = sorted_indices[:num_dummy]
    assignment = torch.zeros(num_layer * num_head, dtype=torch.long)
    assignment[c_indices_flat] = 2
    remaining_mask = assignment != 2
    remaining_indices = torch.nonzero(remaining_mask, as_tuple=True)[0]

    for idx in remaining_indices:
        if p0_norm[idx] < p1_norm[idx]:
            assignment[idx] = 1
        else:
            assignment[idx] = 0

    assignment = assignment.reshape(num_layer, num_head)
    group_a = {}
    group_b = {}
    group_c = {}
    for layer_idx in range(num_layer):
        group_a[layer_idx] = (assignment[layer_idx] == 0).nonzero(as_tuple=True)[0].tolist()
        group_b[layer_idx] = (assignment[layer_idx] == 1).nonzero(as_tuple=True)[0].tolist()
        group_c[layer_idx] = (assignment[layer_idx] == 2).nonzero(as_tuple=True)[0].tolist()
    return group_a, group_b, group_c



def heterogeneous_memory_allocation(global_kv_cache, num_dummy=180):
    global_frame_attn_score = torch.stack([layer_info['frame_attn_score'][:,0] for layer_info in global_kv_cache]).transpose(1,2)
    global_group_first, global_group_mid, global_group_last = dynamic_head_programming(global_frame_attn_score, num_dummy)
    for layer_idx in range(len(global_kv_cache)):
        group_first, group_mid, group_last = global_group_first[layer_idx], global_group_mid[layer_idx], global_group_last[layer_idx]
        cur_cache = global_kv_cache[layer_idx]
        HW = cur_cache['sink_k'].shape[1]
        cur_cache['sink_k'] =  torch.cat([cur_cache['sink_k'][:, :, group_first], cur_cache['local_k'][:,-HW:,group_last]], dim=2).contiguous().clone()
        cur_cache['sink_v'] =  torch.cat([cur_cache['sink_v'][:, :, group_first], cur_cache['local_v'][:,-HW:,group_last]], dim=2).contiguous().clone()
        cur_cache['local_k'] = cur_cache['local_k'][:, :, group_mid].contiguous().clone()
        cur_cache['local_v'] = cur_cache['local_v'][:, :, group_mid].contiguous().clone()
        cur_cache['headgroup_first'] = group_first
        cur_cache['headgroup_mid'] = group_mid
        cur_cache['headgroup_last'] = group_last
