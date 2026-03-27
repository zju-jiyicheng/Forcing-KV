# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
import numpy as np
import json
import os
import random
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

        # For forcingkv, temporal_context_length is in HW units.
        # Example: temporal_context_length=3 is equivalent to old local_context_length=1.
        temporal_context_len_hw_raw = getattr(self.args, "temporal_context_length", 3)
        temporal_context_len_hw = int(temporal_context_len_hw_raw)
        if float(temporal_context_len_hw_raw) != float(temporal_context_len_hw):
            raise ValueError(
                f"forcingkv.temporal_context_length must be an integer (HW units), got {temporal_context_len_hw_raw}."
            )
        temporal_keep_tokens = max(0, int(HW) * temporal_context_len_hw)
        kv_merge_enabled = bool(getattr(self.args, "kv_merge_enabled", False))
        similarity_prune_enabled = bool(getattr(self.args, "similarity_prune_enabled", False))
        # Reuse similarity_threshold as a keep-ratio in [0, 1]:
        # keep the lowest-similarity tokens from old context by this proportion.
        similarity_keep_ratio = float(getattr(self.args, "similarity_threshold", 0.8))
        similarity_keep_ratio = max(0.0, min(1.0, similarity_keep_ratio))
        if kv_merge_enabled and similarity_prune_enabled:
            raise ValueError("kv_merge_enabled and similarity_prune_enabled are mutually exclusive.")
        momentum = float(getattr(self.args, "momentum", 0.8))
        momentum = max(0.0, min(1.0, momentum))
        merge_tokens = 3 * HW
        sink_tokens = max(0, int(self.sink_size)) * int(HW)
        current_tokens = roped_key.shape[1]
        full_window_tokens = int(self.max_attention_size)
        similarity_base_tokens = min(temporal_keep_tokens, max(0, full_window_tokens - sink_tokens - current_tokens))
        similarity_candidate_tokens = max(0, full_window_tokens - sink_tokens - current_tokens - similarity_base_tokens)
        similarity_history_tokens = similarity_base_tokens + similarity_candidate_tokens
        temporal_history_tokens = similarity_history_tokens if similarity_prune_enabled else temporal_keep_tokens

        def _keep_recent_temporal_tokens(tensor):
            if temporal_history_tokens <= 0:
                return tensor[:, :0].contiguous()
            return tensor[:, -temporal_history_tokens:].contiguous()

        def _keep_recent_local_tokens(tensor):
            local_keep_tokens = int(0*HW)
            if local_keep_tokens <= 0:
                return tensor[:, :0].contiguous()
            return tensor[:, -local_keep_tokens:].contiguous()

        def _fix_len(tensor, target_tokens):
            if tensor.shape[1] == target_tokens:
                return tensor.contiguous()
            if tensor.shape[1] > target_tokens:
                return tensor[:, -target_tokens:].contiguous()
            pad_tokens = target_tokens - tensor.shape[1]
            if tensor.shape[1] > 0:
                pad = tensor[:, :1].expand(-1, pad_tokens, -1, -1)
            else:
                pad = tensor.new_zeros((tensor.shape[0], pad_tokens, tensor.shape[2], tensor.shape[3]))
            return torch.cat([pad, tensor], dim=1).contiguous()

        def _similarity_prune_temporal_context(full_k, full_v, base_tokens, candidate_tokens, keep_ratio):
            # Keep the most recent base_tokens as base context. The remaining
            # candidate window is split into base_tokens-sized chunks, and each
            # chunk is independently compared against the base context. For every
            # chunk we keep the lowest-similarity token subset according to
            # keep_ratio.
            if base_tokens <= 0:
                return full_k[:, :0].contiguous(), full_v[:, :0].contiguous()
            if full_k.shape[1] == 0:
                return full_k, full_v
            chunk_tokens = int(base_tokens)
            if chunk_tokens <= 0:
                return full_k[:, :0].contiguous(), full_v[:, :0].contiguous()

            base_k = full_k[:, -base_tokens:].contiguous()
            base_v = full_v[:, -base_tokens:].contiguous()
            if candidate_tokens <= 0:
                return base_k, base_v

            candidate_window = full_k[:, :-base_tokens].contiguous()
            candidate_window_v = full_v[:, :-base_tokens].contiguous()
            if candidate_window.shape[1] == 0:
                return base_k, base_v

            usable_candidate_tokens = min(candidate_tokens, candidate_window.shape[1])
            if usable_candidate_tokens <= 0:
                return base_k, base_v
            candidate_window = candidate_window[:, -usable_candidate_tokens:].contiguous()
            candidate_window_v = candidate_window_v[:, -usable_candidate_tokens:].contiguous()

            num_chunks = candidate_window.shape[1] // chunk_tokens
            if num_chunks <= 0:
                return base_k, base_v

            usable_candidate_tokens = num_chunks * chunk_tokens
            candidate_window = candidate_window[:, -usable_candidate_tokens:].contiguous()
            candidate_window_v = candidate_window_v[:, -usable_candidate_tokens:].contiguous()

            kept_k_chunks = []
            kept_v_chunks = []
            base_k_cmp = base_k.float()
            keep_count_per_chunk = int(round(chunk_tokens * keep_ratio))
            keep_count_per_chunk = max(0, min(chunk_tokens, keep_count_per_chunk))
            for chunk_idx in range(num_chunks):
                start = chunk_idx * chunk_tokens
                end = start + chunk_tokens
                chunk_k = candidate_window[:, start:end].contiguous()
                chunk_v = candidate_window_v[:, start:end].contiguous()
                if keep_count_per_chunk <= 0:
                    continue
                if keep_count_per_chunk >= chunk_tokens:
                    kept_k_chunks.append(chunk_k)
                    kept_v_chunks.append(chunk_v)
                    continue

                sim = torch.nn.functional.cosine_similarity(
                    chunk_k.float(),
                    base_k_cmp,
                    dim=-1,
                )  # [B, chunk_tokens, H]
                sim_token = sim.mean(dim=(0, 2))  # [chunk_tokens]
                keep_idx = torch.topk(sim_token, k=keep_count_per_chunk, largest=False).indices
                keep_idx, _ = torch.sort(keep_idx)
                kept_k_chunks.append(chunk_k[:, keep_idx].contiguous())
                kept_v_chunks.append(chunk_v[:, keep_idx].contiguous())

            if len(kept_k_chunks) > 0:
                pruned_older_k = torch.cat(kept_k_chunks, dim=1).contiguous()
                pruned_older_v = torch.cat(kept_v_chunks, dim=1).contiguous()
            else:
                pruned_older_k = full_k[:, :0].contiguous()
                pruned_older_v = full_v[:, :0].contiguous()

            out_k = torch.cat([pruned_older_k, base_k], dim=1).contiguous()
            out_v = torch.cat([pruned_older_v, base_v], dim=1).contiguous()
            return out_k, out_v

        offline_groups_ready = "headgroup_local" in kv_cache and "headgroup_temporal" in kv_cache
        total_layers = getattr(self, "total_layers", None)
        bypass_edge_layers = bool(getattr(self.args, "bypass_edge_layers_full_kv", True))
        is_edge_layer = bool(total_layers is not None and layer_idx in (0, total_layers - 1))
        
        use_full_kv = (
            (cur_AR_step <= framecache_step)
            or is_recache
            or (not offline_groups_ready)
            or (bypass_edge_layers and is_edge_layer)
        )

        if not use_full_kv:
            local_heads = list(kv_cache["headgroup_local"])
            temporal_heads = list(kv_cache["headgroup_temporal"])
            # extract_heads_triton returns (group2, group1) in this call pattern:
            # q1/k1/v1 -> second arg (temporal_heads), q2/k2/v2 -> first arg (local_heads).
            q_temporal, k_temporal_new, v_temporal_new, q_local, k_local_new, v_local_new = extract_heads_triton(
                roped_query, roped_key, v, local_heads, temporal_heads
            )

            # Two groups share the full sink. Local keeps the most recent HW tokens;
            # temporal keeps recent tokens controlled by temporal_context_length (in HW units).
            local_k = torch.cat(
                [
                    kv_cache["group_sink_local_k"],
                    kv_cache["group_local_k"],
                    k_local_new,
                ],
                dim=1,
            ).contiguous()
            local_v = torch.cat(
                [
                    kv_cache["group_sink_local_v"],
                    kv_cache["group_local_v"],
                    v_local_new,
                ],
                dim=1,
            ).contiguous()
            temporal_context_k = kv_cache["group_temporal_k"]
            temporal_context_v = kv_cache["group_temporal_v"]
            if kv_merge_enabled and "temporal_merge_k" in kv_cache and "temporal_merge_v" in kv_cache:
                temporal_context_k = _fix_len(kv_cache["temporal_merge_k"], merge_tokens)
                temporal_context_v = _fix_len(kv_cache["temporal_merge_v"], merge_tokens)
            elif similarity_prune_enabled:
                temporal_context_k, temporal_context_v = _similarity_prune_temporal_context(
                    temporal_context_k,
                    temporal_context_v,
                    base_tokens=similarity_base_tokens,
                    candidate_tokens=similarity_candidate_tokens,
                    keep_ratio=similarity_keep_ratio,
                )
            temporal_k = torch.cat(
                [
                    kv_cache["group_sink_temporal_k"],
                    temporal_context_k,
                    k_temporal_new,
                ],
                dim=1,
            ).contiguous()
            temporal_v = torch.cat(
                [
                    kv_cache["group_sink_temporal_v"],
                    temporal_context_v,
                    v_temporal_new,
                ],
                dim=1,
            ).contiguous()

            x_local = attention(q_local, local_k, local_v)
            x_temporal = attention(q_temporal, temporal_k, temporal_v)
            x = torch.empty_like(roped_query)
            x[:, :, local_heads, :] = x_local
            x[:, :, temporal_heads, :] = x_temporal

            if update_kv_cache:
                # Keep a full-head recent cache for warmup / edge-layer full-KV paths.
                local_k_all = torch.cat([kv_cache["local_k"], roped_key], dim=1).contiguous()
                local_v_all = torch.cat([kv_cache["local_v"], v], dim=1).contiguous()
                kv_cache["local_k"] = _keep_recent_temporal_tokens(local_k_all)
                kv_cache["local_v"] = _keep_recent_temporal_tokens(local_v_all)

                # Update grouped caches (this is the hot path for offline forcingkv).
                group_local_k_all = torch.cat([kv_cache["group_local_k"], k_local_new], dim=1).contiguous()
                group_local_v_all = torch.cat([kv_cache["group_local_v"], v_local_new], dim=1).contiguous()
                kv_cache["group_local_k"] = _keep_recent_local_tokens(group_local_k_all)
                kv_cache["group_local_v"] = _keep_recent_local_tokens(group_local_v_all)

                group_temporal_k_all = torch.cat([kv_cache["group_temporal_k"], k_temporal_new], dim=1).contiguous()
                group_temporal_v_all = torch.cat([kv_cache["group_temporal_v"], v_temporal_new], dim=1).contiguous()
                kv_cache["group_temporal_k"] = _keep_recent_temporal_tokens(group_temporal_k_all)
                kv_cache["group_temporal_v"] = _keep_recent_temporal_tokens(group_temporal_v_all)
                kv_cache["frame_tokens"] = int(HW)

                if kv_merge_enabled:
                    prev_merge_k = kv_cache.get("temporal_merge_k", kv_cache["group_temporal_k"])
                    prev_merge_v = kv_cache.get("temporal_merge_v", kv_cache["group_temporal_v"])
                    prev_merge_k = _fix_len(prev_merge_k, merge_tokens)
                    prev_merge_v = _fix_len(prev_merge_v, merge_tokens)
                    curr_merge_k = _fix_len(k_temporal_new, merge_tokens)
                    curr_merge_v = _fix_len(v_temporal_new, merge_tokens)
                    kv_cache["temporal_merge_k"] = (
                        momentum * curr_merge_k + (1.0 - momentum) * prev_merge_k
                    ).contiguous()
                    kv_cache["temporal_merge_v"] = (
                        momentum * curr_merge_v + (1.0 - momentum) * prev_merge_v
                    ).contiguous()
        else:
            k0 = torch.cat([kv_cache["sink_k"], kv_cache["local_k"], roped_key], dim=1)
            v0 = torch.cat([kv_cache["sink_v"], kv_cache["local_v"], v], dim=1)
            x = attention(roped_query, k0, v0)
            if update_kv_cache:
                if sink_tokens > 0:
                    kv_cache["sink_k"] = k0[:, :sink_tokens].contiguous()
                    kv_cache["sink_v"] = v0[:, :sink_tokens].contiguous()
                    post_sink_k = k0[:, sink_tokens:].contiguous()
                    post_sink_v = v0[:, sink_tokens:].contiguous()
                else:
                    kv_cache["sink_k"] = k0[:, :0].contiguous()
                    kv_cache["sink_v"] = v0[:, :0].contiguous()
                    post_sink_k = k0
                    post_sink_v = v0
                kv_cache["local_k"] = _keep_recent_temporal_tokens(post_sink_k)
                kv_cache["local_v"] = _keep_recent_temporal_tokens(post_sink_v)
                kv_cache["frame_tokens"] = int(HW)

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
        # Let attention layers know total depth for layer-specific routing rules.
        for block in self.blocks:
            block.self_attn.total_layers = num_layers

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

    @staticmethod
    def _repo_root():
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    @staticmethod
    def _sanitize_head_ids(head_ids, num_heads):
        if head_ids is None:
            return []
        out = []
        seen = set()
        for head_id in head_ids:
            try:
                idx = int(head_id)
            except Exception:
                continue
            if idx < 0 or idx >= num_heads or idx in seen:
                continue
            seen.add(idx)
            out.append(idx)
        return out

    def _normalize_layer_head_groups(
        self,
        layer_idx,
        local_heads,
        temporal_heads,
        explicit_first=None,
        explicit_last=None,
    ):
        local = self._sanitize_head_ids(local_heads, self.num_heads)
        temporal = [h for h in self._sanitize_head_ids(temporal_heads, self.num_heads) if h not in local]

        missing = [h for h in range(self.num_heads) if h not in local and h not in temporal]
        temporal.extend(missing)

        if len(local) == 0 and len(temporal) > 0:
            local.append(temporal.pop())
        if len(temporal) == 0 and len(local) > 0:
            temporal.append(local.pop())

        return {
            "local_head": local,
            "temporal_head": temporal,
        }

    def _build_random_offline_head_groups(self):
        seed = int(getattr(self.args, "offline_head_seed", 20260309))
        local_ratio = float(getattr(self.args, "offline_local_ratio", 0.5))
        local_ratio = max(0.1, min(0.9, local_ratio))

        rng = random.Random(seed)
        groups = {}
        for layer_idx in range(self.num_layers):
            perm = list(range(self.num_heads))
            rng.shuffle(perm)
            local_count = int(round(self.num_heads * local_ratio))
            local_count = max(1, min(self.num_heads - 1, local_count))
            local_heads = perm[:local_count]
            temporal_heads = perm[local_count:]
            groups[layer_idx] = self._normalize_layer_head_groups(
                layer_idx=layer_idx,
                local_heads=local_heads,
                temporal_heads=temporal_heads,
            )
        return groups

    def _resolve_offline_head_file(self):
        config_path = str(getattr(self.args, "offline_head_file", "")).strip()
        if len(config_path) == 0:
            config_path = os.path.join("configs", "forcingkv_heads_longlive_random.json")
        if os.path.isabs(config_path):
            return config_path
        return os.path.abspath(os.path.join(self._repo_root(), config_path))

    def _load_offline_head_groups(self):
        if self._offline_head_groups is not None:
            return self._offline_head_groups

        head_file = self._resolve_offline_head_file()
        groups = None

        if os.path.exists(head_file):
            with open(head_file, "r", encoding="utf-8") as f:
                payload = json.load(f)

            layers = payload.get("layers", payload) if isinstance(payload, dict) else payload
            layer_map = {}
            if isinstance(layers, list):
                for i, item in enumerate(layers):
                    if not isinstance(item, dict):
                        continue
                    layer_idx = int(item.get("layer_idx", i))
                    layer_map[layer_idx] = item
            elif isinstance(layers, dict):
                for k, item in layers.items():
                    if not isinstance(item, dict):
                        continue
                    try:
                        layer_map[int(k)] = item
                    except Exception:
                        continue

            groups = {}
            for layer_idx in range(self.num_layers):
                entry = layer_map.get(layer_idx, {})
                groups[layer_idx] = self._normalize_layer_head_groups(
                    layer_idx=layer_idx,
                    local_heads=entry.get("local_head", entry.get("local_heads", [])),
                    temporal_heads=entry.get("temporal_head", entry.get("temporal_heads", [])),
                )
        else:
            groups = self._build_random_offline_head_groups()
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"[ForcingKV] offline head file not found, fallback to random in-memory groups: {head_file}")

        self._offline_head_groups = groups
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"[ForcingKV] loaded offline head groups from: {head_file}")
        return self._offline_head_groups

    def _apply_offline_head_allocation(self, kv_cache):
        if len(kv_cache) == 0:
            return
        if "headgroup_local" in kv_cache[0] and "headgroup_temporal" in kv_cache[0]:
            return

        kv_merge_enabled = bool(getattr(self.args, "kv_merge_enabled", False))
        groups = self._load_offline_head_groups()
        for layer_idx, cur_cache in enumerate(kv_cache):
            layer_group = groups[layer_idx]
            local_heads = list(layer_group["local_head"])
            temporal_heads = list(layer_group["temporal_head"])
            cur_cache["headgroup_local"] = local_heads
            cur_cache["headgroup_temporal"] = temporal_heads

            # Build grouped caches once at switch time so the hot path avoids repeated
            # full-head advanced indexing.
            cur_cache["group_sink_local_k"] = cur_cache["sink_k"][:, :, local_heads, :].contiguous().clone()
            cur_cache["group_sink_local_v"] = cur_cache["sink_v"][:, :, local_heads, :].contiguous().clone()
            cur_cache["group_sink_temporal_k"] = cur_cache["sink_k"][:, :, temporal_heads, :].contiguous().clone()
            cur_cache["group_sink_temporal_v"] = cur_cache["sink_v"][:, :, temporal_heads, :].contiguous().clone()

            frame_tokens = int(cur_cache.get("frame_tokens", 1560))
            cur_cache["group_local_k"] = cur_cache["local_k"][:, -frame_tokens:, local_heads, :].contiguous().clone()
            cur_cache["group_local_v"] = cur_cache["local_v"][:, -frame_tokens:, local_heads, :].contiguous().clone()
            cur_cache["group_temporal_k"] = cur_cache["local_k"][:, :, temporal_heads, :].contiguous().clone()
            cur_cache["group_temporal_v"] = cur_cache["local_v"][:, :, temporal_heads, :].contiguous().clone()

            if kv_merge_enabled:
                merge_tokens = 3 * frame_tokens
                temporal_merge_k = cur_cache["group_temporal_k"][:, -merge_tokens:, :, :].contiguous()
                temporal_merge_v = cur_cache["group_temporal_v"][:, -merge_tokens:, :, :].contiguous()
                if temporal_merge_k.shape[1] < merge_tokens:
                    pad_tokens = merge_tokens - temporal_merge_k.shape[1]
                    if temporal_merge_k.shape[1] > 0:
                        pad_k = temporal_merge_k[:, :1].expand(-1, pad_tokens, -1, -1)
                        pad_v = temporal_merge_v[:, :1].expand(-1, pad_tokens, -1, -1)
                    else:
                        bsz = cur_cache["sink_k"].shape[0]
                        num_temporal = len(temporal_heads)
                        dim = cur_cache["sink_k"].shape[-1]
                        pad_k = cur_cache["sink_k"].new_zeros((bsz, pad_tokens, num_temporal, dim))
                        pad_v = cur_cache["sink_v"].new_zeros((bsz, pad_tokens, num_temporal, dim))
                    temporal_merge_k = torch.cat([pad_k, temporal_merge_k], dim=1).contiguous()
                    temporal_merge_v = torch.cat([pad_v, temporal_merge_v], dim=1).contiguous()
                cur_cache["temporal_merge_k"] = temporal_merge_k.clone()
                cur_cache["temporal_merge_v"] = temporal_merge_v.clone()

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
        AR_step = current_start // (grid_sizes[0,1] * grid_sizes[0,2] * 3)
        # Switch to offline groups at the first denoising step of the target AR block.
        if AR_step == self.args.ar_start and not is_recache:
            self._apply_offline_head_allocation(kv_cache)

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
