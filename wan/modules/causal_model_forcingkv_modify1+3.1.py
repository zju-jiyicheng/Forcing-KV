# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
import json
import numpy as np
import os
from wan.modules.attention import attention
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import torch.distributed as dist
from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, log_gpu_memory
from wan.modules.rope_triton import rope_apply_triton
from wan.modules.extract_head_triton import extract_heads_triton
from utils.debug_option import DEBUG
# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
# flex_attention = torch.compile(
#     flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


def _allocate_fixed_cache(source_tensor, keep_tokens):
    if keep_tokens <= 0:
        return source_tensor[:, :0].contiguous().clone()

    cache_tensor = torch.empty(
        source_tensor.shape[0],
        keep_tokens,
        source_tensor.shape[2],
        source_tensor.shape[3],
        device=source_tensor.device,
        dtype=source_tensor.dtype,
    )
    cache_tensor.zero_()

    tail_tensor = source_tensor[:, -keep_tokens:].contiguous()
    tail_len = tail_tensor.shape[1]
    if tail_len > 0:
        cache_tensor[:, keep_tokens - tail_len:].copy_(tail_tensor)
    return cache_tensor


def _allocate_ring_cache(source_tensor, keep_tokens):
    if keep_tokens <= 0:
        return source_tensor[:, :0].contiguous().clone(), 0, 0

    cache_tensor = torch.empty(
        source_tensor.shape[0],
        keep_tokens,
        source_tensor.shape[2],
        source_tensor.shape[3],
        device=source_tensor.device,
        dtype=source_tensor.dtype,
    )
    cache_tensor.zero_()

    tail_tensor = source_tensor[:, -keep_tokens:].contiguous()
    tail_len = tail_tensor.shape[1]
    if tail_len == 0:
        return cache_tensor, 0, 0

    if tail_len >= keep_tokens:
        cache_tensor.copy_(tail_tensor[:, -keep_tokens:])
        return cache_tensor, 0, keep_tokens

    cache_tensor[:, :tail_len].copy_(tail_tensor)
    return cache_tensor, tail_len, tail_len


def _overwrite_fixed_cache_(cache_tensor, new_tensor, keep_tokens, cache_name):
    if keep_tokens <= 0:
        return cache_tensor[:, :0]
    if cache_tensor.shape[1] != keep_tokens:
        raise ValueError(
            f"{cache_name} expected preallocated length {keep_tokens}, got {cache_tensor.shape[1]}"
        )
    if new_tensor.shape[1] < keep_tokens:
        raise ValueError(
            f"{cache_name} requires keep_tokens <= new_tensor length, got keep_tokens={keep_tokens}, "
            f"new_tensor.shape[1]={new_tensor.shape[1]}"
        )

    cache_tensor.copy_(new_tensor[:, -keep_tokens:].contiguous())
    return cache_tensor


def _ring_append_(cache_tensor, new_tensor, keep_tokens, cache_name, write_ptr, valid_tokens):
    if keep_tokens <= 0:
        return 0, 0
    if cache_tensor.shape[1] != keep_tokens:
        raise ValueError(
            f"{cache_name} expected preallocated length {keep_tokens}, got {cache_tensor.shape[1]}"
        )

    append_len = new_tensor.shape[1]
    if append_len == 0:
        return int(write_ptr), int(valid_tokens)

    if append_len >= keep_tokens:
        cache_tensor.copy_(new_tensor[:, -keep_tokens:].contiguous())
        return 0, keep_tokens

    write_ptr = int(write_ptr)
    valid_tokens = int(valid_tokens)
    first_write = min(keep_tokens - write_ptr, append_len)
    cache_tensor[:, write_ptr:write_ptr + first_write].copy_(new_tensor[:, :first_write].contiguous())
    remaining = append_len - first_write
    if remaining > 0:
        cache_tensor[:, :remaining].copy_(new_tensor[:, first_write:first_write + remaining].contiguous())

    new_write_ptr = (write_ptr + append_len) % keep_tokens
    new_valid_tokens = min(keep_tokens, valid_tokens + append_len)
    return new_write_ptr, new_valid_tokens


def _materialize_ring_view(cache_tensor, write_ptr, valid_tokens, cache_name):
    capacity = cache_tensor.shape[1]
    valid_tokens = int(valid_tokens)
    write_ptr = int(write_ptr)

    if capacity == 0 or valid_tokens == 0:
        return cache_tensor[:, :0]
    if valid_tokens > capacity:
        raise ValueError(
            f"{cache_name} valid_tokens must be <= capacity, got valid_tokens={valid_tokens}, capacity={capacity}"
        )
    if not (0 <= write_ptr < capacity):
        raise ValueError(
            f"{cache_name} write_ptr must be in [0, {capacity}), got {write_ptr}"
        )

    if valid_tokens < capacity:
        return cache_tensor[:, :valid_tokens]
    if write_ptr == 0:
        return cache_tensor
    return torch.cat([cache_tensor[:, write_ptr:], cache_tensor[:, :write_ptr]], dim=1).contiguous()


def _fill_cache_from_chunk_indices_(cache_tensor, candidate_tensor, keep_chunk_indices, chunk_tokens, cache_name):
    if cache_tensor.shape[1] == 0:
        return cache_tensor[:, :0]
    if chunk_tokens <= 0:
        raise ValueError(f"{cache_name} requires a positive chunk size, got {chunk_tokens}")
    if keep_chunk_indices is None:
        raise ValueError(f"{cache_name} requires precomputed keep_chunk_indices")

    keep_chunk_indices = keep_chunk_indices.to(device=candidate_tensor.device, dtype=torch.long)
    selected_chunks = candidate_tensor.index_select(dim=1, index=keep_chunk_indices)
    expected_tokens = selected_chunks.shape[1] * chunk_tokens
    if expected_tokens != cache_tensor.shape[1]:
        raise ValueError(
            f"{cache_name} expected {cache_tensor.shape[1]} tokens from selected chunks, got {expected_tokens}"
        )

    selected_tokens = selected_chunks.reshape(
        candidate_tensor.shape[0],
        expected_tokens,
        candidate_tensor.shape[3],
        candidate_tensor.shape[4],
    ).contiguous()
    cache_tensor.copy_(selected_tokens)
    return cache_tensor


class CausalWanSelfAttention(nn.Module):
    shared_dynamic_patch_score = None
    shared_dynamic_chunk_indices = None

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.patch_dynamic_score = None
        # Support list/tuple local_attn_size by converting to list first (handles OmegaConf ListConfig)
        if not isinstance(local_attn_size, int) and hasattr(local_attn_size, "__iter__"):
            values = list(local_attn_size)
        else:
            values = [int(local_attn_size)]
        non_neg_vals = [int(v) for v in values if int(v) != -1]
        max_local = max(non_neg_vals) if len(non_neg_vals) > 0 else -1
        self.max_attention_size = 32760 if max_local == -1 else max_local * 1560
        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def dynamic_compression(
        self,
        layer_idx,
        old_temporal_k,
        old_temporal_v,
        new_temporal_k,
        new_temporal_v,
        num_frame_patch,
        dynamic_keep_tokens,
        grid_h,
        grid_w,
        dynamic_cache_k,
        dynamic_cache_v,
    ):
        retention_ratio = max(0.0, min(1.0, float(getattr(self.args, "sim_retention_ratio", 0.5))))
        num_frame_patch = int(num_frame_patch)
        if num_frame_patch <= 0:
            raise ValueError(f"`num_frame_patch` must be a positive integer, got {num_frame_patch}")

        frame_tokens = grid_h * grid_w
        if frame_tokens % num_frame_patch != 0:
            raise ValueError(
                f"Frame token count {frame_tokens} must be divisible by num_frame_patch {num_frame_patch}"
            )
        if old_temporal_k.shape[1] < frame_tokens or new_temporal_k.shape[1] < 3 * frame_tokens:
            CausalWanSelfAttention.shared_dynamic_patch_score = None
            CausalWanSelfAttention.shared_dynamic_chunk_indices = None
            return 0

        temporal_keep_frames = old_temporal_k.shape[1] // frame_tokens
        if old_temporal_k.shape[1] % frame_tokens != 0:
            raise ValueError(
                f"old_temporal_k length {old_temporal_k.shape[1]} must be divisible by frame_tokens {frame_tokens}"
            )
        if temporal_keep_frames < 1 or temporal_keep_frames > 3:
            raise ValueError(
                f"Dynamic temporal mode currently supports temporal_context_length in [1, 3], got {temporal_keep_frames}"
            )

        chunk_tokens = frame_tokens // num_frame_patch
        total_candidate_chunks = 3 * num_frame_patch
        if dynamic_keep_tokens == 0:
            keep_chunk_count = 0
        else:
            if dynamic_keep_tokens % chunk_tokens != 0:
                raise ValueError(
                    f"dynamic_keep_tokens={dynamic_keep_tokens} must be divisible by chunk_tokens={chunk_tokens}"
                )
            keep_chunk_count = dynamic_keep_tokens // chunk_tokens
        if keep_chunk_count > total_candidate_chunks:
            raise ValueError(
                f"Requested {keep_chunk_count} dynamic chunks, but only {total_candidate_chunks} candidate chunks exist"
            )
        ratio_keep_chunk_count = int(round(total_candidate_chunks * retention_ratio))
        ratio_keep_chunk_count = max(0, min(total_candidate_chunks, ratio_keep_chunk_count))
        if ratio_keep_chunk_count != keep_chunk_count:
            raise ValueError(
                "Dynamic cache capacity and sim_retention_ratio are inconsistent: "
                f"capacity implies {keep_chunk_count} kept chunks, but sim_retention_ratio={retention_ratio} "
                f"implies {ratio_keep_chunk_count} kept chunks out of {total_candidate_chunks}"
            )

        old_frames_k = old_temporal_k.reshape(
            old_temporal_k.shape[0], temporal_keep_frames, frame_tokens, old_temporal_k.shape[2], old_temporal_k.shape[3]
        )
        old_frames_v = old_temporal_v.reshape(
            old_temporal_v.shape[0], temporal_keep_frames, frame_tokens, old_temporal_v.shape[2], old_temporal_v.shape[3]
        )
        new_frames_k = new_temporal_k.reshape(
            new_temporal_k.shape[0], 3, frame_tokens, new_temporal_k.shape[2], new_temporal_k.shape[3]
        )
        new_frames_v = new_temporal_v.reshape(
            new_temporal_v.shape[0], 3, frame_tokens, new_temporal_v.shape[2], new_temporal_v.shape[3]
        )

        carried_new_frames = 3 - temporal_keep_frames
        candidate_frames_k = torch.cat([old_frames_k, new_frames_k[:, :carried_new_frames]], dim=1).contiguous()
        candidate_frames_v = torch.cat([old_frames_v, new_frames_v[:, :carried_new_frames]], dim=1).contiguous()
        boundary_frame_k = new_frames_k[:, carried_new_frames:carried_new_frames + 1].contiguous()

        if layer_idx == 1:
            chain_frames = torch.cat([candidate_frames_k[:1], boundary_frame_k[:1]], dim=1)
            current_chunks = chain_frames[:, :3].reshape(
                1, 3, num_frame_patch, chunk_tokens, chain_frames.shape[3], chain_frames.shape[4]
            ).reshape(1, 3, num_frame_patch, chunk_tokens, -1)
            next_chunks = chain_frames[:, 1:].reshape(
                1, 3, num_frame_patch, chunk_tokens, chain_frames.shape[3], chain_frames.shape[4]
            ).reshape(1, 3, num_frame_patch, chunk_tokens, -1)
            token_scores = F.cosine_similarity(current_chunks, next_chunks, dim=-1)
            patch_scores = token_scores.mean(dim=3)[0]

            if keep_chunk_count > 0:
                flat_scores = patch_scores.reshape(-1)
                keep_indices = torch.topk(flat_scores, k=keep_chunk_count, largest=False, dim=-1).indices
                keep_indices = torch.sort(keep_indices).values
            else:
                keep_indices = torch.empty((0,), device=patch_scores.device, dtype=torch.long)

            CausalWanSelfAttention.shared_dynamic_patch_score = patch_scores.detach()
            CausalWanSelfAttention.shared_dynamic_chunk_indices = keep_indices.detach()
            self.patch_dynamic_score = patch_scores.detach()
        else:
            keep_indices = CausalWanSelfAttention.shared_dynamic_chunk_indices
            if keep_indices is None:
                return 0

        candidate_chunks_k = candidate_frames_k.reshape(
            candidate_frames_k.shape[0], 3, num_frame_patch, chunk_tokens, candidate_frames_k.shape[3], candidate_frames_k.shape[4]
        ).reshape(candidate_frames_k.shape[0], total_candidate_chunks, chunk_tokens, candidate_frames_k.shape[3], candidate_frames_k.shape[4])
        candidate_chunks_v = candidate_frames_v.reshape(
            candidate_frames_v.shape[0], 3, num_frame_patch, chunk_tokens, candidate_frames_v.shape[3], candidate_frames_v.shape[4]
        ).reshape(candidate_frames_v.shape[0], total_candidate_chunks, chunk_tokens, candidate_frames_v.shape[3], candidate_frames_v.shape[4])

        if keep_chunk_count == 0:
            return 0

        _fill_cache_from_chunk_indices_(
            dynamic_cache_k,
            candidate_chunks_k,
            keep_indices,
            chunk_tokens,
            "group_dynamic_temporal_k",
        )
        _fill_cache_from_chunk_indices_(
            dynamic_cache_v,
            candidate_chunks_v,
            keep_indices,
            chunk_tokens,
            "group_dynamic_temporal_v",
        )
        return keep_chunk_count * chunk_tokens

    
    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        sink_recache_after_switch=False,
        layer_idx = -1,
        update_kv_cache = False,
        is_recache = False,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        HW = grid_sizes[0,1]*grid_sizes[0,2]
        q, k, v = qkv_fn(x)
        frame_seqlen = math.prod(grid_sizes[0][1:]).item()
        current_start_frame = current_start // frame_seqlen
        x_rope = rope_apply_triton(torch.cat([q,k],dim=0), grid_sizes, freqs, current_start_frame)
        roped_query, roped_key = torch.chunk(x_rope, chunks=2, dim=0)
        cur_AR_step = current_start_frame // 3
        framecache_step = self.args.ar_start

        spatial_context_len_hw = max(0, int(getattr(self.args, "spatial_context_length", 1)))
        temporal_context_len_hw = max(0, int(getattr(self.args, "temporal_context_length", 3)))
        dynamic_context_len_hw = max(0, int(getattr(self.args, "dynamic_context_length", 3)))
        dynamic_retention_ratio = max(0.0, min(1.0, float(getattr(self.args, "sim_retention_ratio", 0.5))))
        sink_tokens = max(0, int(self.sink_size)) * int(HW)
        spatial_keep_tokens = int(HW) * spatial_context_len_hw
        temporal_keep_tokens = int(HW) * temporal_context_len_hw
        dynamic_keep_tokens = int(HW) * dynamic_context_len_hw
        history_keep_tokens = max(spatial_keep_tokens, temporal_keep_tokens)

        switch_applied = bool(kv_cache.get("cache_switched", False))
        groups_ready = (
            "headgroup_last" in kv_cache
            and "headgroup_mid" in kv_cache
            and "group_sink_spatial_k" in kv_cache
            and "group_sink_spatial_v" in kv_cache
            and "group_spatial_k" in kv_cache
            and "group_spatial_v" in kv_cache
            and "group_sink_temporal_k" in kv_cache
            and "group_sink_temporal_v" in kv_cache
            and "group_temporal_k" in kv_cache
            and "group_temporal_v" in kv_cache
        )
        use_grouped_kv = switch_applied and groups_ready and (layer_idx != 0) and (not is_recache)

        if not use_grouped_kv:
            k0 = torch.cat([kv_cache["sink_k"],kv_cache["local_k"], roped_key], dim=1)
            v0 = torch.cat([kv_cache["sink_v"],kv_cache["local_v"], v],         dim=1)
            x = attention(roped_query, k0, v0)
            if update_kv_cache:
                if not switch_applied:
                    if sink_tokens > 0:
                        kv_cache['sink_k'] = k0[:, :sink_tokens].contiguous()
                        kv_cache['sink_v'] = v0[:, :sink_tokens].contiguous()
                        post_sink_k = k0[:, sink_tokens:].contiguous()
                        post_sink_v = v0[:, sink_tokens:].contiguous()
                    else:
                        kv_cache['sink_k'] = k0[:, :0].contiguous()
                        kv_cache['sink_v'] = v0[:, :0].contiguous()
                        post_sink_k = k0
                        post_sink_v = v0
                else:
                    post_sink_k = k0[:, sink_tokens:].contiguous() if sink_tokens > 0 else k0
                    post_sink_v = v0[:, sink_tokens:].contiguous() if sink_tokens > 0 else v0
                if history_keep_tokens > 0:
                    kv_cache['local_k'] = post_sink_k[:, -history_keep_tokens:].contiguous()
                    kv_cache['local_v'] = post_sink_v[:, -history_keep_tokens:].contiguous()
                else:
                    kv_cache['local_k'] = post_sink_k[:, :0].contiguous()
                    kv_cache['local_v'] = post_sink_v[:, :0].contiguous()
                kv_cache['frame_tokens'] = int(HW)
        else:
            spatial_heads = kv_cache['headgroup_last']
            temporal_heads = kv_cache['headgroup_mid']
            spatial_write_ptr = int(kv_cache.get("group_spatial_write_ptr", 0))
            spatial_valid_tokens = int(kv_cache.get("group_spatial_valid_tokens", kv_cache["group_spatial_k"].shape[1]))
            temporal_write_ptr = int(kv_cache.get("group_temporal_write_ptr", 0))
            temporal_valid_tokens = int(kv_cache.get("group_temporal_valid_tokens", kv_cache["group_temporal_k"].shape[1]))

            spatial_cache_k = _materialize_ring_view(
                kv_cache["group_spatial_k"],
                spatial_write_ptr,
                spatial_valid_tokens,
                "group_spatial_k",
            )
            spatial_cache_v = _materialize_ring_view(
                kv_cache["group_spatial_v"],
                spatial_write_ptr,
                spatial_valid_tokens,
                "group_spatial_v",
            )
            temporal_cache_k = _materialize_ring_view(
                kv_cache["group_temporal_k"],
                temporal_write_ptr,
                temporal_valid_tokens,
                "group_temporal_k",
            )
            temporal_cache_v = _materialize_ring_view(
                kv_cache["group_temporal_v"],
                temporal_write_ptr,
                temporal_valid_tokens,
                "group_temporal_v",
            )

            q1, cur_spatial_k, cur_spatial_v, q2, cur_temporal_k, cur_temporal_v = extract_heads_triton(
                roped_query, roped_key, v, temporal_heads, spatial_heads
            )
            k1 = torch.cat([kv_cache["group_sink_spatial_k"], spatial_cache_k, cur_spatial_k], dim=1).contiguous()
            v1 = torch.cat([kv_cache["group_sink_spatial_v"], spatial_cache_v, cur_spatial_v], dim=1).contiguous()
            # k2 = torch.cat([kv_cache["group_sink_temporal_k"], kv_cache["group_temporal_k"], k2], dim=1).contiguous()
            # v2 = torch.cat([kv_cache["group_sink_temporal_v"], kv_cache["group_temporal_v"], v2], dim=1).contiguous()
            dynamic_valid_tokens = int(kv_cache.get("group_dynamic_temporal_valid_tokens", kv_cache["group_dynamic_temporal_k"].shape[1]))
            dynamic_k = kv_cache["group_dynamic_temporal_k"][:, :dynamic_valid_tokens]
            dynamic_v = kv_cache["group_dynamic_temporal_v"][:, :dynamic_valid_tokens]
            k2 = torch.cat([kv_cache["group_sink_temporal_k"], dynamic_k, temporal_cache_k, cur_temporal_k], dim=1).contiguous()
            v2 = torch.cat([kv_cache["group_sink_temporal_v"], dynamic_v, temporal_cache_v, cur_temporal_v], dim=1).contiguous()
            x1 = attention(q1, k1, v1)
            x2 = attention(q2, k2, v2)
            x = torch.empty_like(roped_query)
            x[:, :, spatial_heads, :] = x1
            x[:, :, temporal_heads, :] = x2
            if update_kv_cache:
                spatial_write_ptr, spatial_valid_tokens = _ring_append_(
                    kv_cache["group_spatial_k"],
                    cur_spatial_k,
                    spatial_keep_tokens,
                    "group_spatial_k",
                    spatial_write_ptr,
                    spatial_valid_tokens,
                )
                kv_cache["group_spatial_write_ptr"] = spatial_write_ptr
                kv_cache["group_spatial_valid_tokens"] = spatial_valid_tokens
                _ring_append_(
                    kv_cache["group_spatial_v"],
                    cur_spatial_v,
                    spatial_keep_tokens,
                    "group_spatial_v",
                    spatial_write_ptr,
                    spatial_valid_tokens,
                )

                # Temporal Heads
                if not self.args.dynamic_temporal_enabled:
                    temporal_write_ptr, temporal_valid_tokens = _ring_append_(
                        kv_cache["group_temporal_k"],
                        cur_temporal_k,
                        temporal_keep_tokens,
                        "group_temporal_k",
                        temporal_write_ptr,
                        temporal_valid_tokens,
                    )
                    kv_cache["group_temporal_write_ptr"] = temporal_write_ptr
                    kv_cache["group_temporal_valid_tokens"] = temporal_valid_tokens
                    _ring_append_(
                        kv_cache["group_temporal_v"],
                        cur_temporal_v,
                        temporal_keep_tokens,
                        "group_temporal_v",
                        temporal_write_ptr,
                        temporal_valid_tokens,
                    )
                    kv_cache["group_dynamic_temporal_valid_tokens"] = 0
                else:
                    if temporal_keep_tokens <= 0 or temporal_keep_tokens > s or (temporal_keep_tokens % int(HW)) != 0:
                        raise ValueError(
                            "Dynamic temporal mode currently requires temporal_context_length to be 1-3 whole frames"
                        )

                    num_frame_patch = getattr(self.args, "num_frame_patch", None)
                    
                    old_temporal_k = temporal_cache_k
                    old_temporal_v = temporal_cache_v
                    new_temporal_k = cur_temporal_k.contiguous()
                    new_temporal_v = cur_temporal_v.contiguous()

                    dynamic_valid_tokens = self.dynamic_compression(
                        layer_idx,
                        old_temporal_k,
                        old_temporal_v,
                        new_temporal_k,
                        new_temporal_v,
                        num_frame_patch,
                        dynamic_keep_tokens,
                        int(grid_sizes[0, 1]),
                        int(grid_sizes[0, 2]),
                        kv_cache["group_dynamic_temporal_k"],
                        kv_cache["group_dynamic_temporal_v"],
                    )

                    temporal_write_ptr, temporal_valid_tokens = _ring_append_(
                        kv_cache["group_temporal_k"],
                        new_temporal_k,
                        temporal_keep_tokens,
                        "group_temporal_k",
                        temporal_write_ptr,
                        temporal_valid_tokens,
                    )
                    kv_cache["group_temporal_write_ptr"] = temporal_write_ptr
                    kv_cache["group_temporal_valid_tokens"] = temporal_valid_tokens
                    _ring_append_(
                        kv_cache["group_temporal_v"],
                        new_temporal_v,
                        temporal_keep_tokens,
                        "group_temporal_v",
                        temporal_write_ptr,
                        temporal_valid_tokens,
                    )

                    kv_cache["group_dynamic_temporal_valid_tokens"] = int(dynamic_valid_tokens)

                kv_cache['frame_tokens'] = int(HW)

        x = x.flatten(2)
        x = self.o(x)
        return x





class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        sink_recache_after_switch=False,
        layer_idx = -1,
        update_cache = False,
        is_recache = False
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start, sink_recache_after_switch, layer_idx, update_cache, is_recache = is_recache)


        # with amp.autocast(dtype=torch.float32):
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False
        # params for Teacache technique
        self.coefficients = [2.39676752e+03, -1.31110545e+03, 2.01331979e+02, -8.29855975e+00, 1.37887774e-01]
        self.accumulated_rel_l1_distance = 0
        self.previous_e0 = None
        self.previous_residual = None
        self._offline_head_groups = None

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def _load_offline_head_groups(self):
        if self._offline_head_groups is not None:
            return self._offline_head_groups

        head_file = str(getattr(self.args, "offline_head_file", "")).strip()
        if not head_file:
            raise ValueError("ForcingKV requires `offline_head_file` in config.")
        if not os.path.isabs(head_file):
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            head_file = os.path.join(repo_root, head_file)
        if not os.path.exists(head_file):
            raise FileNotFoundError(f"ForcingKV offline head file not found: {head_file}")

        with open(head_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        layers = payload.get("layers", payload) if isinstance(payload, dict) else payload
        if isinstance(layers, list):
            layer_items = enumerate(layers)
        elif isinstance(layers, dict):
            layer_items = []
            for key, item in layers.items():
                try:
                    layer_items.append((int(key), item))
                except Exception:
                    continue
        else:
            raise ValueError(f"Unsupported offline head file format: {type(layers)}")

        groups = {}
        for default_idx, item in layer_items:
            if not isinstance(item, dict):
                continue
            layer_idx = int(item.get("layer_idx", default_idx))
            spatial = sorted({
                int(h) for h in item.get("local_head", item.get("local_heads", []))
                if 0 <= int(h) < self.num_heads
            })
            temporal = sorted({
                int(h) for h in item.get("temporal_head", item.get("temporal_heads", []))
                if 0 <= int(h) < self.num_heads
            })
            if sorted(spatial + temporal) != list(range(self.num_heads)):
                raise ValueError(
                    f"Offline head groups must cover all heads exactly once at layer {layer_idx}, "
                    f"got {sorted(spatial + temporal)}"
                )
            groups[layer_idx] = {"spatial_head": spatial, "temporal_head": temporal}

        for layer_idx in range(self.num_layers):
            groups.setdefault(layer_idx, {"spatial_head": [], "temporal_head": list(range(self.num_heads))})

        self._offline_head_groups = groups
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"[ForcingKV] loaded offline head groups from: {head_file}")
        return self._offline_head_groups

    def _apply_offline_head_allocation(self, kv_cache):
        if len(kv_cache) == 0:
            return
        if all(bool(cur_cache.get("cache_switched", False)) for cur_cache in kv_cache):
            return

        groups = self._load_offline_head_groups()
        spatial_keep_tokens = max(0, int(getattr(self.args, "spatial_context_length", 1)))
        temporal_keep_tokens = max(0, int(getattr(self.args, "temporal_context_length", 3)))
        dynamic_keep_tokens = max(0, int(getattr(self.args, "dynamic_context_length", 0)))

        for layer_idx, cur_cache in enumerate(kv_cache):
            cur_cache["cache_switched"] = True
            if layer_idx == 0:
                continue

            layer_group = groups[layer_idx]
            spatial_heads = list(layer_group["spatial_head"])
            temporal_heads = list(layer_group["temporal_head"])
            frame_tokens = int(cur_cache.get("frame_tokens", 1560))
            spatial_keep = spatial_keep_tokens * frame_tokens
            temporal_keep = temporal_keep_tokens * frame_tokens

            full_sink_k = cur_cache["sink_k"]
            full_sink_v = cur_cache["sink_v"]
            full_local_k = cur_cache["local_k"]
            full_local_v = cur_cache["local_v"]

            cur_cache["headgroup_first"] = []
            cur_cache["headgroup_last"] = spatial_heads
            cur_cache["headgroup_mid"] = temporal_heads

            cur_cache["group_sink_spatial_k"] = full_sink_k[:, :, spatial_heads, :].contiguous().clone()
            cur_cache["group_sink_spatial_v"] = full_sink_v[:, :, spatial_heads, :].contiguous().clone()
            cur_cache["group_sink_temporal_k"] = full_sink_k[:, :, temporal_heads, :].contiguous().clone()
            cur_cache["group_sink_temporal_v"] = full_sink_v[:, :, temporal_heads, :].contiguous().clone()

            # Dynamic Cache
            dynamic_keep = dynamic_keep_tokens * frame_tokens
            cur_cache["group_dynamic_temporal_k"] = _allocate_fixed_cache(
                full_local_k[:, :0, temporal_heads, :],
                dynamic_keep,
            )
            cur_cache["group_dynamic_temporal_v"] = _allocate_fixed_cache(
                full_local_v[:, :0, temporal_heads, :],
                dynamic_keep,
            )
            cur_cache["group_dynamic_temporal_valid_tokens"] = 0

            group_spatial_k, group_spatial_write_ptr, group_spatial_valid_tokens = _allocate_ring_cache(
                full_local_k[:, :, spatial_heads, :],
                spatial_keep,
            )
            group_spatial_v, _, _ = _allocate_ring_cache(
                full_local_v[:, :, spatial_heads, :],
                spatial_keep,
            )
            cur_cache["group_spatial_k"] = group_spatial_k
            cur_cache["group_spatial_v"] = group_spatial_v
            cur_cache["group_spatial_write_ptr"] = group_spatial_write_ptr
            cur_cache["group_spatial_valid_tokens"] = group_spatial_valid_tokens

            group_temporal_k, group_temporal_write_ptr, group_temporal_valid_tokens = _allocate_ring_cache(
                full_local_k[:, :, temporal_heads, :],
                temporal_keep,
            )
            group_temporal_v, _, _ = _allocate_ring_cache(
                full_local_v[:, :, temporal_heads, :],
                temporal_keep,
            )
            cur_cache["group_temporal_k"] = group_temporal_k
            cur_cache["group_temporal_v"] = group_temporal_v
            cur_cache["group_temporal_write_ptr"] = group_temporal_write_ptr
            cur_cache["group_temporal_valid_tokens"] = group_temporal_valid_tokens

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
            pass

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        # # debug
        # DEBUG = False
        # if DEBUG:
        #     num_frames = 9
        #     frame_seqlen = 256

        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )

        # attention for clean context frames
        for start in frame_indices:
            context_ends[start:start + attention_block_size] = start + attention_block_size

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            noise_context_ends[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            # first design the mask for clean frames
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # then design the mask for noisy frames
            # noisy frames will attend to all clean preceeding clean frames + itself
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if DEBUG:
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                               padded_length, KV_LEN=total_length + padded_length, device=device)
            import cv2
            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            pass

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    def _apply_cache_updates(self, kv_cache, cache_update_infos):
        """
        Applies cache updates collected from multiple blocks.
        Args:
            kv_cache: List of cache dictionaries for each block
            cache_update_infos: List of (block_index, cache_update_info) tuples
        """
        for block_index, (current_end, local_end_index, update_info) in cache_update_infos:
            if update_info is not None:
                cache = kv_cache[block_index]
                
                if update_info["action"] == "roll_and_insert":
                    # Apply rolling update
                    sink_tokens = update_info["sink_tokens"]
                    num_rolled_tokens = update_info["num_rolled_tokens"]
                    num_evicted_tokens = update_info["num_evicted_tokens"]
                    local_start_index = update_info["local_start_index"]
                    local_end_index = update_info["local_end_index"]
                    write_start_index = update_info.get("write_start_index", local_start_index)
                    write_end_index = update_info.get("write_end_index", local_end_index)
                    new_k = update_info["new_k"]
                    new_v = update_info["new_v"]
                    
                    # Perform the rolling operation
                    cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    
                    # Insert new key/value
                    if write_end_index > write_start_index and new_k.shape[1] == (write_end_index - write_start_index):
                        cache["k"][:, write_start_index:write_end_index] = new_k
                        cache["v"][:, write_start_index:write_end_index] = new_v
                    
                elif update_info["action"] == "direct_insert":
                    # Direct insert
                    local_start_index = update_info["local_start_index"]
                    local_end_index = update_info["local_end_index"]
                    write_start_index = update_info.get("write_start_index", local_start_index)
                    write_end_index = update_info.get("write_end_index", local_end_index)
                    new_k = update_info["new_k"]
                    new_v = update_info["new_v"]
                    
                    # Insert new key/value
                    if write_end_index > write_start_index and new_k.shape[1] == (write_end_index - write_start_index):
                        cache["k"][:, write_start_index:write_end_index] = new_k
                        cache["v"][:, write_start_index:write_end_index] = new_v
            
            # Update indices: do not roll back pointers during recomputation
            is_recompute = False if update_info is None else update_info.get("is_recompute", False)
            if not is_recompute:
                kv_cache[block_index]["global_end_index"].fill_(current_end)
                kv_cache[block_index]["local_end_index"].fill_(local_end_index)

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        updating_cache=False,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
        sink_recache_after_switch=False,
        is_recache=False,
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        
        # print(f"x.device: {x[0].device}, t.device: {t.device}, context.device: {context.device}, seq_len: {seq_len}")

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        # print("patch embedding done")
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)


        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
            sink_recache_after_switch=sink_recache_after_switch
        )

        # Add teacache code here
        teacache_enabled = self.args.teacache_enabled
        modulated_inp = e0
        if t[0,0]==1000 or t[0,0] == 0:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            rescale_func = np.poly1d(self.coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp - self.previous_e0).abs().mean() / self.previous_e0.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance <  self.args.teacache_threshold:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_e0 = modulated_inp.clone()

        if not should_calc and teacache_enabled:
            x += self.previous_residual
            # print(f'Teacache activated at time {t[0,0]}, AR step {current_start//4680}')
        else:
            ori_x = x.clone()
            update_cache = t[0,0] == 0
            AR_step = current_start // (grid_sizes[0,1] * grid_sizes[0,2]*3)
            for block_index, block in enumerate(self.blocks):
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "update_cache": update_cache,
                        "layer_idx": block_index,
                        "is_recache": is_recache,
                    }
                )
                x = block(x, **kwargs)
            self.previous_residual = x - ori_x

        if t[0,0] == 0 and AR_step == self.args.ar_start and not is_recache:
            self._apply_offline_head_allocation(kv_cache)
        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)


    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
