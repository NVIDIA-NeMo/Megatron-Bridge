# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""FlexAttention for sbd_block_diff diffusion LM training with YARN RoPE."""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention
from transformers import ROPE_INIT_FUNCTIONS
from transformers.utils.generic import maybe_autocast

from megatron.bridge.diffusion.common.dllm import compute_block_mask
from megatron.core import tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide


# ---------------------------------------------------------------------------
# Compiled flex_attention kernel
# ---------------------------------------------------------------------------

@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs", dynamic=False)
def fused_flex_attention(q, k, v, score_mod=None, block_mask=None, return_lse=False):
    return flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask, return_lse=return_lse)


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
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
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# FlexDotProductAttention  (sbd_block_diff only)
# ---------------------------------------------------------------------------

class FlexDotProductAttention(MegatronModule):
    """FlexAttention for semi-block-diffusion (sbd_block_diff) training.

    The sequence is doubled to ``[xt | x0]`` where xt are noised tokens and x0
    are clean tokens.  RoPE is applied independently to each half.  Llama-4
    style query-key layer scaling is applied when configured.
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

        assert config.context_parallel_size == 1, (
            "Context parallelism is only supported by TEDotProductAttention!"
        )

        self.layer_number = max(1, layer_number)

        projection_size = config.kv_channels * config.num_attention_heads

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp'])
        else:
            assert hasattr(pg_collection, 'tp'), (
                "FlexDotProductAttention pg_collection must have tp process group"
            )

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

        # Llama-4 style query-key layer scaling + YARN RoPE
        self.beta = None
        self.max_position_embeddings = None
        if getattr(config, 'apply_llama4_style_query_key_layer_scaling', False):
            hf_text_config = config.hf_config.text_config
            self.beta = hf_text_config.rope_parameters["llama_4_scaling_beta"]
            self.max_position_embeddings = hf_text_config.rope_parameters["original_max_position_embeddings"]
            hf_text_config.max_position_embeddings = config.seq_length
            if hasattr(config, "yarn_rotary_scaling_factor") and config.yarn_rotary_scaling_factor != hf_text_config.rope_parameters["factor"]:
                hf_text_config.rope_parameters["factor"] = config.yarn_rotary_scaling_factor
            self.rope_embedding_module = Ministral3RotaryEmbedding(hf_text_config)

        # Pre-compute the sbd_block_diff block mask
        self.mask = compute_block_mask(
            block_size=getattr(config, 'block_size', 16),
            max_seq_length=config.seq_length,
        )

        import torch._dynamo.config as dcfg
        dcfg.cache_size_limit = 512

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
        assert packed_seq_params is None, (
            "Packed sequence is not supported by FlexDotProductAttention."
        )

        # Position ids for each half of the doubled sequence
        half_seq_len = query.shape[0] // 2
        position_ids = torch.arange(half_seq_len, device=query.device).unsqueeze(0)
        cos, sin = self.rope_embedding_module(query, position_ids)

        # [sq, b, np, hn] -> [b, np, sq, hn]
        query = query.transpose(0, 1).transpose(1, 2)
        key = key.transpose(0, 1).transpose(1, 2)
        value = value.transpose(0, 1).transpose(1, 2)

        # Apply RoPE independently to each half (xt and x0)
        q1, q2 = query.chunk(2, dim=2)
        k1, k2 = key.chunk(2, dim=2)
        q1, k1 = apply_rotary_pos_emb(q1, k1, cos, sin)
        q2, k2 = apply_rotary_pos_emb(q2, k2, cos, sin)
        query = torch.cat([q1, q2], dim=2)
        key = torch.cat([k1, k2], dim=2)

        # Llama-4 attention scaling
        cache_position = torch.arange(query.shape[2], device=query.device)
        query = query * _get_llama_4_attn_scale(
            cache_position, self.beta, self.max_position_embeddings
        ).to(query.dtype)

        # GQA: expand KV heads
        n_rep = self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        key = repeat_kv(key, n_rep)
        value = repeat_kv(value, n_rep)

        # FlexAttention with pre-computed block mask
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
