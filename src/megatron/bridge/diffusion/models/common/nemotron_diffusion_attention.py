# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NemotronDiffusionAttention for sbd_block_diff diffusion LM training with YARN RoPE."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention
from transformers import ROPE_INIT_FUNCTIONS

from megatron.bridge.diffusion.common.dllm import compute_block_mask
from megatron.bridge.diffusion.common.ctc import compute_ctc_block_mask


# ---------------------------------------------------------------------------
# Compiled flex_attention kernel
# ---------------------------------------------------------------------------


@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs", dynamic=False)
def fused_flex_attention(q, k, v, score_mod=None, block_mask=None, return_lse=False):
    """Thin compiled wrapper around flex_attention."""
    return flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask, return_lse=return_lse)


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------


def rotate_half(x):
    """Rotate the last half of the hidden dimension for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply rotary position embeddings to query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match query heads for GQA."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _get_llama_4_attn_scale(position_ids: torch.Tensor, beta: float, max_position_embeddings: int) -> torch.Tensor:
    scaling = 1 + beta * torch.log(1 + torch.floor(position_ids / max_position_embeddings))
    return scaling.unsqueeze(-1)


# ---------------------------------------------------------------------------
# YARN-aware Rotary Embedding (supports default + yarn rope_type)
# ---------------------------------------------------------------------------


class Ministral3RotaryEmbedding(nn.Module):
    """RoPE with YARN support, driven by HF ``rope_parameters`` config."""

    inv_freq: torch.Tensor

    def __init__(self, config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        self.rope_type = config.rope_parameters["rope_type"]
        rope_init_fn = self._compute_default_rope_parameters
        if self.rope_type != "default":
            rp = getattr(config, "rope_parameters", {})
            if not hasattr(config, "rope_theta") and "rope_theta" in rp:
                config.rope_theta = rp["rope_theta"]
            if not hasattr(config, "rope_scaling"):
                config.rope_scaling = {k: v for k, v in rp.items() if k != "rope_type"}
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = inv_freq

    @staticmethod
    def _compute_default_rope_parameters(config=None, device=None, seq_len=None):
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, 1.0

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# NemotronDiffusionAttention  (sbd_block_diff only)
# ---------------------------------------------------------------------------


class NemotronDiffusionAttention(MegatronModule):
    """NemotronDiffusionAttention for semi-block-diffusion (sbd_block_diff) training.

    The sequence is laid out as ``[xt | x0]``.  For standard sbd_block_diff,
    both halves have equal length.  For CTC paradigm, xt is 2x the length of x0
    (total = 3 * x0_len).  RoPE is applied independently to each half.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
        softmax_scale: float = None,
        cp_comm_type: str = None,
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(config=config)
        self.config = config

        assert config.context_parallel_size == 1, "Context parallelism is only supported by TEDotProductAttention!"

        self.layer_number = max(1, layer_number)

        projection_size = config.kv_channels * config.num_attention_heads

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp"])
        else:
            assert hasattr(pg_collection, "tp"), "NemotronDiffusionAttention pg_collection must have tp process group"

        world_size = pg_collection.tp.size()
        self.hidden_size_per_partition = divide(projection_size, world_size)
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = divide(config.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(config.num_query_groups, world_size)

        if softmax_scale is None:
            self.softmax_scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head)
        else:
            self.softmax_scale = softmax_scale

        if config.apply_query_key_layer_scaling:
            self.softmax_scale /= self.layer_number

        self.attention_dropout = torch.nn.Dropout(
            config.attention_dropout if attention_dropout is None else attention_dropout
        )

        # RoPE setup (always required)
        hf_text_config = config.hf_config.text_config
        hf_text_config.max_position_embeddings = config.seq_length
        self.rope_embedding_module = Ministral3RotaryEmbedding(hf_text_config)

        # Llama-4 style query scaling (optional)
        self.beta = None
        self.max_position_embeddings = None
        if getattr(config, "apply_llama4_style_query_key_layer_scaling", False):
            self.beta = hf_text_config.rope_parameters["llama_4_scaling_beta"]
            self.max_position_embeddings = hf_text_config.rope_parameters["original_max_position_embeddings"]
            if (
                hasattr(config, "yarn_rotary_scaling_factor")
                and config.yarn_rotary_scaling_factor != hf_text_config.rope_parameters["factor"]
            ):
                hf_text_config.rope_parameters["factor"] = config.yarn_rotary_scaling_factor

        # Pre-compute the attention block mask
        self.dlm_paradigm = getattr(config, "dlm_paradigm", "sbd_block_diff")
        if self.dlm_paradigm == "ctc":
            self.xt_len = config.seq_length * 2  # 2x expansion
            self.x0_len = config.seq_length
            self.mask = compute_ctc_block_mask(
                xt_block_size=getattr(config, "block_size", 128),
                x0_block_size=getattr(config, "ctc_target_block_size", 64),
                x0_len=config.seq_length,
            )
        else:
            self.xt_len = None
            self.x0_len = None
            self.mask = compute_block_mask(
                block_size=getattr(config, "block_size", 16),
                max_seq_length=config.seq_length,
            )

        import torch._dynamo.config as dcfg

        dcfg.cache_size_limit = 512

        # Inference state
        self._inference_mode = False
        self._inference_causal = True
        self._cache_enabled = False
        self._kv_cache_k = None
        self._kv_cache_v = None
        self._kv_cache_seq_len = 0

    def set_inference_mode(self, enabled: bool):
        """Enable or disable inference mode. Clears cache on disable."""
        self._inference_mode = enabled
        if not enabled:
            self.clear_kv_cache()

    def set_inference_params(self, causal: bool, cache_enabled: bool):
        self._inference_causal = causal
        self._cache_enabled = cache_enabled

    def clear_kv_cache(self):
        self._kv_cache_k = None
        self._kv_cache_v = None
        self._kv_cache_seq_len = 0

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor = None,
        attn_mask_type: AttnMaskType = None,
        attention_bias: Tensor = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        assert packed_seq_params is None, "Packed sequence is not supported by NemotronDiffusionAttention."

        if self._inference_mode:
            return self._inference_forward(query, key, value)

        # [sq, b, np, hn] -> [b, np, sq, hn]
        query = query.transpose(0, 1).transpose(1, 2)
        key = key.transpose(0, 1).transpose(1, 2)
        value = value.transpose(0, 1).transpose(1, 2)

        if self.dlm_paradigm == "ctc":
            # Asymmetric split: xt_len (2*x0_len) + x0_len
            xt_len = self.xt_len
            x0_len = self.x0_len

            q_xt = query[:, :, :xt_len]
            q_x0 = query[:, :, xt_len:]
            k_xt = key[:, :, :xt_len]
            k_x0 = key[:, :, xt_len:]

            # RoPE: xt gets positions 0..xt_len-1, x0 gets positions 0..x0_len-1
            pos_xt = torch.arange(xt_len, device=query.device).unsqueeze(0)
            pos_x0 = torch.arange(x0_len, device=query.device).unsqueeze(0)

            cos_xt, sin_xt = self.rope_embedding_module(q_xt, pos_xt)
            cos_x0, sin_x0 = self.rope_embedding_module(q_x0, pos_x0)

            q_xt, k_xt = apply_rotary_pos_emb(q_xt, k_xt, cos_xt, sin_xt)
            q_x0, k_x0 = apply_rotary_pos_emb(q_x0, k_x0, cos_x0, sin_x0)

            query = torch.cat([q_xt, q_x0], dim=2)
            key = torch.cat([k_xt, k_x0], dim=2)
        else:
            # Symmetric split: equal halves
            half_seq_len = query.shape[2] // 2
            position_ids = torch.arange(half_seq_len, device=query.device).unsqueeze(0)
            cos, sin = self.rope_embedding_module(query, position_ids)

            q1, q2 = query.chunk(2, dim=2)
            k1, k2 = key.chunk(2, dim=2)
            q1, k1 = apply_rotary_pos_emb(q1, k1, cos, sin)
            q2, k2 = apply_rotary_pos_emb(q2, k2, cos, sin)
            query = torch.cat([q1, q2], dim=2)
            key = torch.cat([k1, k2], dim=2)

        # Llama-4 attention scaling
        if self.beta is not None:
            cache_position = torch.arange(query.shape[2], device=query.device)
            query = query * _get_llama_4_attn_scale(cache_position, self.beta, self.max_position_embeddings).to(
                query.dtype
            )

        # GQA: expand KV heads
        n_rep = self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        key = repeat_kv(key, n_rep)
        value = repeat_kv(value, n_rep)

        # NemotronDiffusionAttention with pre-computed block mask
        context = fused_flex_attention(query, key, value, block_mask=self.mask)

        # Dropout
        if not self.config.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                context = self.attention_dropout(context)
        else:
            context = self.attention_dropout(context)

        # [b, np, sq, hn] -> [sq, b, hp]
        context = context.transpose(1, 2).transpose(0, 1)
        new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        context = context.contiguous().view(*new_context_shape)

        return context

    def _inference_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tensor:
        """SDPA-based forward for inference with KV cache support."""
        sq = query.shape[0]

        # Transpose to [b, np, s, hn]
        query = query.transpose(0, 1).transpose(1, 2)
        key = key.transpose(0, 1).transpose(1, 2)
        value = value.transpose(0, 1).transpose(1, 2)

        # Position IDs: new tokens start after the cached tokens
        offset = self._kv_cache_seq_len
        q_position_ids = torch.arange(offset, offset + sq, device=query.device).unsqueeze(0)
        k_position_ids = torch.arange(offset, offset + sq, device=key.device).unsqueeze(0)

        cos, sin = self.rope_embedding_module(query, q_position_ids)
        cos_k, sin_k = self.rope_embedding_module(key, k_position_ids)

        # Apply RoPE to new Q and K
        cos_q = cos.unsqueeze(1)
        sin_q = sin.unsqueeze(1)
        cos_k = cos_k.unsqueeze(1)
        sin_k = sin_k.unsqueeze(1)
        query = (query * cos_q) + (rotate_half(query) * sin_q)
        key = (key * cos_k) + (rotate_half(key) * sin_k)

        # Llama-4 attention scaling on query
        if self.beta is not None:
            scale = _get_llama_4_attn_scale(q_position_ids.squeeze(0), self.beta, self.max_position_embeddings).to(
                query.dtype
            )
            query = query * scale

        # Concatenate with KV cache
        if self._kv_cache_k is not None:
            full_key = torch.cat([self._kv_cache_k, key], dim=2)
            full_value = torch.cat([self._kv_cache_v, value], dim=2)
        else:
            full_key = key
            full_value = value

        # Update cache if enabled
        if self._cache_enabled:
            self._kv_cache_k = full_key.detach()
            self._kv_cache_v = full_value.detach()
            self._kv_cache_seq_len = full_key.shape[2]

        # GQA: repeat KV heads to match query heads
        n_rep = self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        full_key_expanded = repeat_kv(full_key, n_rep)
        full_value_expanded = repeat_kv(full_value, n_rep)

        sk = full_key_expanded.shape[2]

        # Build attention mask for SDPA
        if not self._inference_causal:
            attn_mask = None
            is_causal = False
        elif sq == sk:
            attn_mask = None
            is_causal = True
        else:
            q_pos = torch.arange(offset, offset + sq, device=query.device)
            k_pos = torch.arange(sk, device=query.device)
            mask = q_pos[:, None] >= k_pos[None, :]
            attn_mask = torch.zeros(sq, sk, dtype=query.dtype, device=query.device)
            attn_mask.masked_fill_(~mask, float("-inf"))
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            is_causal = False

        context = F.scaled_dot_product_attention(
            query,
            full_key_expanded,
            full_value_expanded,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=self.softmax_scale,
        )

        # Reshape back to Megatron layout: [sq, b, hp]
        context = context.transpose(1, 2).transpose(0, 1)
        new_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        context = context.contiguous().view(*new_shape)
        return context
