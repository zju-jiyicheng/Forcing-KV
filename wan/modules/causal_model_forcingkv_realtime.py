import functools
import json
import os
from wan.modules.attention import attention
import math
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
import torch
import math
import torch.distributed as dist
import time
import copy
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")

from wan.modules.dummyforcing import save_head_attention_map, save_head_attention_map_v2


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
    return tail_len, tail_len


def _ring_append_(cache_tensor, new_tensor, keep_tokens, cache_name, write_ptr, valid_tokens):
    if keep_tokens <= 0:
        return 0, 0
    if cache_tensor.shape[1] != keep_tokens:
        raise ValueError(f"{cache_name} expected preallocated length {keep_tokens}, got {cache_tensor.shape[1]}")

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


def _ring_replace_tail_(cache_tensor, new_tensor, keep_tokens, cache_name, write_ptr, valid_tokens):
    if keep_tokens <= 0:
        return int(write_ptr), int(valid_tokens)
    if cache_tensor.shape[1] != keep_tokens:
        raise ValueError(f"{cache_name} expected preallocated length {keep_tokens}, got {cache_tensor.shape[1]}")

    replace_len = min(int(valid_tokens), keep_tokens, new_tensor.shape[1])
    if replace_len == 0:
        return int(write_ptr), int(valid_tokens)
    if new_tensor.shape[1] >= keep_tokens and int(valid_tokens) >= keep_tokens:
        cache_tensor.copy_(new_tensor[:, -keep_tokens:].contiguous())
        return 0, keep_tokens

    write_ptr = int(write_ptr)
    start = (write_ptr - replace_len) % keep_tokens
    replacement = new_tensor[:, -replace_len:].contiguous()
    first_write = min(keep_tokens - start, replace_len)
    cache_tensor[:, start:start + first_write].copy_(replacement[:, :first_write])
    remaining = replace_len - first_write
    if remaining > 0:
        cache_tensor[:, :remaining].copy_(replacement[:, first_write:first_write + remaining])
    return write_ptr, int(valid_tokens)


def _materialize_ring_view(cache_tensor, write_ptr, valid_tokens, cache_name):
    capacity = cache_tensor.shape[1]
    valid_tokens = int(valid_tokens)
    write_ptr = int(write_ptr)

    if capacity == 0 or valid_tokens == 0:
        return cache_tensor[:, :0]
    if valid_tokens > capacity:
        raise ValueError(f"{cache_name} valid_tokens must be <= capacity, got valid_tokens={valid_tokens}, capacity={capacity}")
    if not (0 <= write_ptr < capacity):
        raise ValueError(f"{cache_name} write_ptr must be in [0, {capacity}), got {write_ptr}")

    if valid_tokens < capacity:
        return cache_tensor[:, :valid_tokens]
    if write_ptr == 0:
        return cache_tensor
    return torch.cat([cache_tensor[:, write_ptr:], cache_tensor[:, :write_ptr]], dim=1).contiguous()

def rope_params_riflex(max_seq_len, dim, theta=10000, k=0, L_test=None):
    assert dim % 2 == 0
    omega = 1.0 / torch.pow(theta,
                            torch.arange(0, dim, 2).to(torch.float64).div(dim))
    if k is not None:
        print("Doing riflex w/ ltest", L_test)
        omega[k-1] = 0.9 * 2 * torch.pi / L_test
    freqs = torch.outer(
        torch.arange(max_seq_len),
        omega)
                        
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

@functools.lru_cache(maxsize=32)
def get_sdpa_mask(
    device: str, 
    num_frames: int = 21,
    frame_seqlen: int = 1560, 
    num_frame_per_block: int = 1, 
    local_attn_size: int = -1,
    dtype: torch.dtype = torch.bool
):
    """
    Create an attention mask tensor for torch.nn.functional.scaled_dot_product_attention
    
    Args:
        device: Device to create the mask on
        num_frames: Number of frames
        frame_seqlen: Sequence length per frame
        num_frame_per_block: Number of frames per block
        local_attn_size: Local attention window size (-1 for global)
        dtype: Data type for the mask (torch.bool for masking, torch.float for additive)
    
    Returns:
        torch.Tensor: Attention mask of shape (seq_len, seq_len)
                     - True/1.0 for allowed attention
                     - False/-inf for masked attention
    """
    print("Generating SDPA attention mask")
    total_length = num_frames * frame_seqlen

    # Right padding to get to a multiple of 128
    padded_length = math.ceil(total_length / 128) * 128 - total_length
    full_length = total_length + padded_length

    # Create the ends array (same logic as original)
    ends = torch.zeros(full_length, device=device, dtype=torch.long)
    
    frame_indices = torch.arange(
        start=0,
        end=total_length,
        step=frame_seqlen * num_frame_per_block,
        device=device
    )

    for tmp in frame_indices:
        end_idx = min(tmp + frame_seqlen * num_frame_per_block, full_length)
        ends[tmp:end_idx] = end_idx

    # Create q_idx and kv_idx coordinate matrices
    q_indices = torch.arange(full_length, device=device).unsqueeze(1)  # Shape: (seq_len, 1)
    kv_indices = torch.arange(full_length, device=device).unsqueeze(0)  # Shape: (1, seq_len)

    # Apply the attention logic
    if local_attn_size == -1:
        # Global attention within blocks + diagonal
        mask = (kv_indices < ends[q_indices]) | (q_indices == kv_indices)
    else:
        # Local attention within blocks + diagonal
        local_window_start = ends[q_indices] - local_attn_size * frame_seqlen
        mask = ((kv_indices < ends[q_indices]) & 
                (kv_indices >= local_window_start)) | (q_indices == kv_indices)

    if dtype == torch.bool:
        return mask
    elif dtype == torch.float32 or dtype == torch.float16:
        # Convert to additive mask (0.0 for attend, -inf for mask)
        return mask.float() * 0.0 + (~mask).float() * float('-inf')
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

@functools.lru_cache(maxsize=32)
def get_block_mask(
    device: str , num_frames: int = 21,
    frame_seqlen: int = 1560, num_frame_per_block=3, local_attn_size=-1
):
    print("Generating block mask")
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
    block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                    KV_LEN=total_length + padded_length, _compile=False, device=device)
    return block_mask

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


class CausalWanSelfAttention(nn.Module):

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
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560
        self.fused_projections = False

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
    
    @torch.no_grad()
    def fuse_projections(self):
        # if not self.is_cross_attention:
        if self.fused_projections:
            return
        concatenated_weights = torch.cat([self.q.weight.data, self.k.weight.data, self.v.weight.data])
        concatenated_bias = torch.cat([self.q.bias.data, self.k.bias.data, self.v.bias.data])
        out_features, in_features = concatenated_weights.shape
        with torch.device("meta"):
            self.to_qkv = torch.nn.Linear(in_features, out_features, bias=True)
        self.to_qkv.load_state_dict(
            {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
        )
        self.fused_projections = True

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
        blk_idx=0,
        is_recache=False,
        update_cache=False,
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
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        # @torch.compile(dynamic=True, mode="max-autotune-no-cudagraphs")
        def qkv_fn(x):
            if self.fused_projections:
                # print("Using fused projections")
                q, k, v = self.to_qkv(x).chunk(3, dim=-1)
                q = self.norm_q(q).view(b, s, n, d)
                k = self.norm_k(k).view(b, s, n, d)
                v = v.view(b, s, n, d)
            else:
                q = self.norm_q(self.q(x)).view(b, s, n, d)
                k = self.norm_k(self.k(x)).view(b, s, n, d)
                v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None or block_mask is not None:
            # if it is teacher forcing training?
            # is_tf = (s == seq_lens[0].item() * 2)
            is_tf = False
            if is_tf:
                print("Teacher forcing training")
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)

            else:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                local_end_index = roped_key.shape[1]
                kv_cache["k"][:, :local_end_index] = roped_key
                kv_cache["v"][:, :local_end_index] = v

                kv_cache["global_end_index"] = local_end_index
                kv_cache["local_end_index"] = local_end_index
                kv_cache["frame_tokens"] = int(grid_sizes[0, 1].item() * grid_sizes[0, 2].item())

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )
                # print("shape of padded_roped_query", padded_roped_query.shape)
                # print("shape of padded_roped_key", padded_roped_key.shape)

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )


                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1).contiguous(),
                    key=padded_roped_key.transpose(2, 1).contiguous(),
                    value=padded_v.transpose(2, 1).contiguous(),
                    block_mask=block_mask,
                    kernel_options={
                        "BLOCKS_ARE_CONTIGUOUS": True,
                    }
                    
                )[:, :, :-padded_length].transpose(2, 1)
        else:
            frame_seqlen = int(grid_sizes[0, 1].item() * grid_sizes[0, 2].item())
            current_start_frame = current_start // frame_seqlen
            cur_AR_step = current_start_frame // 3

            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            group_ready = bool(kv_cache.get("cache_switched", False)) and not is_recache and all(
                key in kv_cache
                for key in (
                    "headgroup_static",
                    "headgroup_temporal",
                    "group_sink_static_k",
                    "group_sink_static_v",
                    "group_static_prev_k",
                    "group_static_prev_v",
                    "group_sink_temporal_k",
                    "group_sink_temporal_v",
                    "group_temporal_k",
                    "group_temporal_v",
                )
            )
            if group_ready:
                x = roped_query.new_empty(roped_query.shape)
                static_heads = list(kv_cache["headgroup_static"])
                temporal_heads = list(kv_cache["headgroup_temporal"])
                previous_global_end = int(kv_cache.get("global_end_index", current_start))
                append_cache = update_cache and current_end > previous_global_end
                refresh_cache = update_cache and current_end == previous_global_end

                if static_heads:
                    cur_static_k = roped_key[:, :, static_heads, :].contiguous()
                    cur_static_v = v[:, :, static_heads, :].contiguous()
                    static_prev_k = _materialize_ring_view(
                        kv_cache["group_static_prev_k"],
                        kv_cache.get("group_static_prev_write_ptr", 0),
                        kv_cache.get("group_static_prev_valid_tokens", 0),
                        "group_static_prev_k",
                    )
                    static_prev_v = _materialize_ring_view(
                        kv_cache["group_static_prev_v"],
                        kv_cache.get("group_static_prev_write_ptr", 0),
                        kv_cache.get("group_static_prev_valid_tokens", 0),
                        "group_static_prev_v",
                    )
                    k_static = torch.cat(
                        [kv_cache["group_sink_static_k"], static_prev_k, cur_static_k],
                        dim=1,
                    ).contiguous()
                    v_static = torch.cat(
                        [kv_cache["group_sink_static_v"], static_prev_v, cur_static_v],
                        dim=1,
                    ).contiguous()
                    x[:, :, static_heads, :] = attention(
                        roped_query[:, :, static_heads, :].contiguous(),
                        k_static,
                        v_static,
                    )
                    if append_cache or refresh_cache:
                        static_keep = kv_cache["group_static_prev_k"].shape[1]
                        static_write_ptr = kv_cache.get("group_static_prev_write_ptr", 0)
                        static_valid_tokens = kv_cache.get("group_static_prev_valid_tokens", 0)
                        ring_update = _ring_append_ if append_cache else _ring_replace_tail_
                        write_ptr, valid_tokens = ring_update(
                            kv_cache["group_static_prev_k"],
                            cur_static_k,
                            static_keep,
                            "group_static_prev_k",
                            static_write_ptr,
                            static_valid_tokens,
                        )
                        ring_update(
                            kv_cache["group_static_prev_v"],
                            cur_static_v,
                            static_keep,
                            "group_static_prev_v",
                            static_write_ptr,
                            static_valid_tokens,
                        )
                        kv_cache["group_static_prev_write_ptr"] = write_ptr
                        kv_cache["group_static_prev_valid_tokens"] = valid_tokens

                if temporal_heads:
                    cur_temporal_k = roped_key[:, :, temporal_heads, :].contiguous()
                    cur_temporal_v = v[:, :, temporal_heads, :].contiguous()
                    temporal_cache_k = _materialize_ring_view(
                        kv_cache["group_temporal_k"],
                        kv_cache.get("group_temporal_write_ptr", 0),
                        kv_cache.get("group_temporal_valid_tokens", 0),
                        "group_temporal_k",
                    )
                    temporal_cache_v = _materialize_ring_view(
                        kv_cache["group_temporal_v"],
                        kv_cache.get("group_temporal_write_ptr", 0),
                        kv_cache.get("group_temporal_valid_tokens", 0),
                        "group_temporal_v",
                    )
                    k_temporal = torch.cat(
                        [kv_cache["group_sink_temporal_k"], temporal_cache_k, cur_temporal_k],
                        dim=1,
                    ).contiguous()
                    v_temporal = torch.cat(
                        [kv_cache["group_sink_temporal_v"], temporal_cache_v, cur_temporal_v],
                        dim=1,
                    ).contiguous()
                    x[:, :, temporal_heads, :] = attention(
                        roped_query[:, :, temporal_heads, :].contiguous(),
                        k_temporal,
                        v_temporal,
                    )
                    if append_cache or refresh_cache:
                        temporal_keep = kv_cache["group_temporal_k"].shape[1]
                        temporal_write_ptr = kv_cache.get("group_temporal_write_ptr", 0)
                        temporal_valid_tokens = kv_cache.get("group_temporal_valid_tokens", 0)
                        ring_update = _ring_append_ if append_cache else _ring_replace_tail_
                        write_ptr, valid_tokens = ring_update(
                            kv_cache["group_temporal_k"],
                            cur_temporal_k,
                            temporal_keep,
                            "group_temporal_k",
                            temporal_write_ptr,
                            temporal_valid_tokens,
                        )
                        ring_update(
                            kv_cache["group_temporal_v"],
                            cur_temporal_v,
                            temporal_keep,
                            "group_temporal_v",
                            temporal_write_ptr,
                            temporal_valid_tokens,
                        )
                        kv_cache["group_temporal_write_ptr"] = write_ptr
                        kv_cache["group_temporal_valid_tokens"] = valid_tokens

                if update_cache:
                    kv_cache["global_end_index"] = current_end
                    kv_cache["local_end_index"] = int(kv_cache.get("local_end_index", 0)) + max(0, current_end - previous_global_end)
                    kv_cache["frame_tokens"] = frame_seqlen
            else:
                # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
                kv_cache_size = kv_cache["k"].shape[1]
                num_new_tokens = roped_query.shape[1]
                global_end_index = int(kv_cache["global_end_index"])
                local_end_index = int(kv_cache["local_end_index"])
                if self.local_attn_size != -1 and (current_end > global_end_index) and (
                        num_new_tokens + local_end_index > kv_cache_size):
                    # Calculate the number of new tokens added in this step
                    # Shift existing cache content left to discard oldest tokens
                    # Clone the source slice to avoid overlapping memory error
                    num_evicted_tokens = num_new_tokens + local_end_index - kv_cache_size
                    num_rolled_tokens = local_end_index - num_evicted_tokens - sink_tokens
                    kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    # Insert the new keys/values at the end
                    local_end_index = local_end_index + current_end - global_end_index - num_evicted_tokens
                    local_start_index = local_end_index - num_new_tokens
                    kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                    kv_cache["v"][:, local_start_index:local_end_index] = v
                else:
                    # Assign new keys/values directly up to current_end
                    local_end_index = local_end_index + current_end - global_end_index
                    local_start_index = local_end_index - num_new_tokens
                    kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                    kv_cache["v"][:, local_start_index:local_end_index] = v

                # # MODIFIED
                # ar_steps_print = {4}
                # layer_print = {
                #         0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                #         10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                #         20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                #         30, 31, 32, 33, 34, 35, 36, 37, 38, 39
                #     }
                # if cur_AR_step in ar_steps_print and blk_idx in layer_print:
                #     save_head_attention_map(roped_query, kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index], cur_AR_step, blk_idx, "/ycji/code/Forcing-KV/visualize/realtime_a4")

                x = attention(
                    roped_query,
                    kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index],
                    kv_cache["v"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
                )
                kv_cache["global_end_index"] = current_end
                kv_cache["local_end_index"] = local_end_index
                kv_cache["frame_tokens"] = frame_seqlen

        # output
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
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
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
        blk_idx=0,
        is_recache=False,
        update_cache=False,
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
            freqs, block_mask, kv_cache, current_start, cache_start, blk_idx, is_recache, update_cache)

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
            # rope_params_riflex(1024, d - 4 * (d // 6), ),
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
        self._offline_head_groups = None

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def _load_offline_head_groups(self):
        if self._offline_head_groups is not None:
            return self._offline_head_groups

        head_file = str(getattr(self.args, "offline_head_file", "")).strip()
        if not head_file:
            raise ValueError("ForcingKV realtime requires `offline_head_file` in config.")
        if not os.path.isabs(head_file):
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            head_file = os.path.join(repo_root, head_file)
        if not os.path.exists(head_file):
            raise FileNotFoundError(f"ForcingKV realtime offline head file not found: {head_file}")

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

        def read_heads(item, names):
            heads = []
            for name in names:
                if name in item:
                    heads.extend(item.get(name) or [])
            return sorted({int(h) for h in heads if 0 <= int(h) < self.num_heads})

        groups = {}
        for default_idx, item in layer_items:
            if not isinstance(item, dict):
                continue
            layer_idx = int(item.get("layer_idx", default_idx))
            static_heads = read_heads(
                item,
                ("static_head", "static_heads", "local_head", "local_heads", "spatial_head", "spatial_heads"),
            )
            dynamic_heads = read_heads(item, ("dynamic_head", "dynamic_heads", "temporal_head", "temporal_heads"))
            if sorted(static_heads + dynamic_heads) != list(range(self.num_heads)):
                raise ValueError(
                    f"Offline head groups must cover all heads exactly once at layer {layer_idx}, "
                    f"got {sorted(static_heads + dynamic_heads)}"
                )
            groups[layer_idx] = {"static_head": static_heads, "dynamic_head": dynamic_heads}

        for layer_idx in range(self.num_layers):
            groups.setdefault(layer_idx, {"static_head": [], "dynamic_head": list(range(self.num_heads))})

        self._offline_head_groups = groups
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"[ForcingKV-Realtime] loaded offline head groups from: {head_file}")
        return self._offline_head_groups

    def _apply_offline_head_allocation(self, kv_cache):
        if not kv_cache:
            return
        if all(bool(cur_cache.get("cache_switched", False)) for cur_cache in kv_cache):
            return

        groups = self._load_offline_head_groups()
        temporal_keep_frames = max(0, int(getattr(self.args, "temporal_context_length", 3)))

        for layer_idx, cur_cache in enumerate(kv_cache):
            cur_cache["cache_switched"] = True

            layer_group = groups[layer_idx]
            static_heads = list(layer_group["static_head"])
            temporal_heads = list(layer_group["dynamic_head"])
            frame_tokens = int(cur_cache.get("frame_tokens", 1560))
            local_end_index = int(cur_cache.get("local_end_index", 0))
            sink_tokens = int(self.blocks[layer_idx].self_attn.sink_size) * frame_tokens
            sink_tokens = min(sink_tokens, local_end_index)

            full_k = cur_cache["k"][:, :local_end_index]
            full_v = cur_cache["v"][:, :local_end_index]
            full_sink_k = full_k[:, :sink_tokens]
            full_sink_v = full_v[:, :sink_tokens]
            full_local_k = full_k[:, sink_tokens:]
            full_local_v = full_v[:, sink_tokens:]

            cur_cache["headgroup_static"] = static_heads
            cur_cache["headgroup_temporal"] = temporal_heads

            cur_cache["group_sink_static_k"] = full_sink_k[:, :, static_heads, :].contiguous().clone()
            cur_cache["group_sink_static_v"] = full_sink_v[:, :, static_heads, :].contiguous().clone()
            cur_cache["group_sink_temporal_k"] = full_sink_k[:, :, temporal_heads, :].contiguous().clone()
            cur_cache["group_sink_temporal_v"] = full_sink_v[:, :, temporal_heads, :].contiguous().clone()

            static_keep = frame_tokens
            group_static_prev_k, static_write_ptr, static_valid_tokens = _allocate_ring_cache(
                full_local_k[:, :, static_heads, :],
                static_keep,
            )
            group_static_prev_v, _, _ = _allocate_ring_cache(
                full_local_v[:, :, static_heads, :],
                static_keep,
            )
            cur_cache["group_static_prev_k"] = group_static_prev_k
            cur_cache["group_static_prev_v"] = group_static_prev_v
            cur_cache["group_static_prev_write_ptr"] = static_write_ptr
            cur_cache["group_static_prev_valid_tokens"] = static_valid_tokens

            temporal_keep = temporal_keep_frames * frame_tokens
            group_temporal_k, temporal_write_ptr, temporal_valid_tokens = _allocate_ring_cache(
                full_local_k[:, :, temporal_heads, :],
                temporal_keep,
            )
            group_temporal_v, _, _ = _allocate_ring_cache(
                full_local_v[:, :, temporal_heads, :],
                temporal_keep,
            )
            cur_cache["group_temporal_k"] = group_temporal_k
            cur_cache["group_temporal_v"] = group_temporal_v
            cur_cache["group_temporal_write_ptr"] = temporal_write_ptr
            cur_cache["group_temporal_valid_tokens"] = temporal_valid_tokens

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        block_mask = get_block_mask(str(device), num_frames, frame_seqlen, num_frame_per_block, local_attn_size)
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
        # debug
        DEBUG = False
        if DEBUG:
            num_frames = 9
            frame_seqlen = 256

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
            print(block_mask)
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

        # if not dist.is_initialized() or dist.get_rank() == 0:
            # print(
            #     f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            # print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
        sink_recache_after_switch: bool = False,
        is_recache: bool = False,
        updating_cache: bool = False,
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
        if current_start == 0 and kv_cache:
            try:
                cache_is_empty = int(kv_cache[0].get("global_end_index", 0)) == 0
            except Exception:
                cache_is_empty = False
            if cache_is_empty:
                for cur_cache in kv_cache:
                    cur_cache["cache_switched"] = False

        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
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
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
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
            block_mask=self.block_mask
        )
        # print("Block mask in forward : ", self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        update_cache = int(t.flatten()[0].item()) == 0
        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "is_recache": is_recache,
                        "update_cache": update_cache,
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "is_recache": is_recache,
                        "update_cache": update_cache,
                    }
                )
                x = block(x, blk_idx=block_index, **kwargs)

        frame_tokens = int(grid_sizes[0, 1].item() * grid_sizes[0, 2].item())
        block_tokens = frame_tokens * int(self.num_frame_per_block)
        ar_step = current_start // block_tokens if block_tokens > 0 else 0
        if (
            kv_cache
            and hasattr(self, "args")
            and int(t.flatten()[0].item()) == 0
            and ar_step == int(getattr(self.args, "ar_start", -1))
            and not is_recache
        ):
            self._apply_offline_head_allocation(kv_cache)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

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

        # Construct blockwise causal attn mask
        if self.block_mask is None:
            if clean_x is not None:
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block
                    )
            else:
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
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

        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
            ])

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]

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
        result = self._forward_inference(*args, **kwargs)
        # if kwargs.get('kv_cache', None) is not None:
        # else:
        #     result = self._forward_train(*args, **kwargs)

        return result

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
