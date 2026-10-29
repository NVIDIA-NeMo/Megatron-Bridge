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

"""MiMo-V2-Flash modeling building blocks.

Houses the custom modules used by ``MiMoV2FlashModelProvider``:
- ``MiMoV2FlashRotaryEmbedding``: dual-base RoPE (local for SWA, global for full).
- ``MiMoV2FlashSelfAttention``: per-layer KV head switching and asymmetric V head dim.
- ``MiMoV2FlashTEDotProductAttention``: per-layer SWA window and learnable softmax
  for SWA layers (vanilla for full).
- ``MiMoV2FlashMTPSelfAttention`` / ``MiMoV2FlashMTPTEDotProductAttention``: MTP variants
  (all MTP layers behave like SWA layers).
- ``mimo_v2_flash_layer_spec``: GPT layer spec builder that injects the custom modules.
"""

import copy
from functools import lru_cache
from typing import List, Optional, Tuple

import torch
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import ModuleSpec, TransformerConfig
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import build_module
from torch import Tensor

from megatron.bridge.utils.import_utils import safe_import_from


TEDotProductAttention, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TEDotProductAttention")
SplitAlongDim, _ = safe_import_from("megatron.core.extensions.transformer_engine", "SplitAlongDim")


def _is_local_attn_layer(
    layer_number: int,
    hybrid_attention_pattern: List[int],
) -> bool:
    return hybrid_attention_pattern[layer_number - 1] == 1


class MiMoV2FlashRotaryEmbedding(RotaryEmbedding):
    """Dual-base rotary embeddings for MiMo-V2-Flash.
    This is the same pattern as Gemma3RotaryEmbedding.
    """

    def __init__(
        self,
        rotary_base: int = 5_000_000,
        rotary_base_local: int = 10_000,
        **kwargs,
    ):
        # Initialize global
        super().__init__(rotary_base=rotary_base, **kwargs)

        # Initialize local
        self.rope_local = RotaryEmbedding(rotary_base=rotary_base_local, **kwargs)

    def forward(
        self,
        max_seq_len: int,
        offset: int = 0,
        packed_seq: bool = False,
        cp_group: torch.distributed.ProcessGroup | None = None,
    ) -> torch.Tensor:
        """Get both local and global rope embeddings stacked [local, global]."""
        if cp_group is not None:
            rope_global = super().forward(max_seq_len, offset, packed_seq, cp_group)
            rope_local = self.rope_local.forward(max_seq_len, offset, packed_seq, cp_group)
            return torch.stack([rope_local, rope_global], dim=0)
        return self._forward_cached(max_seq_len, offset, packed_seq)

    @lru_cache(maxsize=32)
    def _forward_cached(
        self,
        max_seq_len: int,
        offset: int = 0,
        packed_seq: bool = False,
    ) -> torch.Tensor:
        """Cached forward for hashable parameters."""
        rope_global = super().forward(max_seq_len, offset, packed_seq, None)
        rope_local = self.rope_local.forward(max_seq_len, offset, packed_seq, None)
        return torch.stack([rope_local, rope_global], dim=0)


class MiMoV2FlashSelfAttention(SelfAttention):
    """MiMo-V2-Flash self attention.

    Customizations over standard SelfAttention:
    - Per-layer KV head count: SWA layers use swa_num_query_groups, full layers use full_attn_num_query_groups
    - Asymmetric V head dim: Q/K use qk_channels=192, V uses v_head_dim=128
    - Dual RoPE: local rope for SWA layers, global rope for full layers
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        *args,
        **kwargs,
    ):
        config = copy.deepcopy(config)
        if _is_local_attn_layer(layer_number, config.hybrid_attention_pattern):
            config.num_query_groups = config.swa_num_query_groups
        else:
            config.num_query_groups = config.full_attn_num_query_groups
        super().__init__(config, submodules, layer_number, *args, **kwargs)

        # --- Asymmetric V head dim fixup ---
        v_head_dim = config.v_head_dim
        qk_channels = config.kv_channels

        self.val_hidden_size = v_head_dim

        self.query_projection_size = qk_channels * config.num_attention_heads
        self.key_projection_size = qk_channels * config.num_query_groups
        self.value_projection_size = v_head_dim * config.num_query_groups
        self.linear_qkv_out_dim = self.query_projection_size + self.key_projection_size + self.value_projection_size
        self.linear_qkv = build_module(
            submodules.linear_qkv,
            config.hidden_size,
            self.linear_qkv_out_dim,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=config.add_bias_linear or config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="qkv",
            tp_group=self.pg_collection.tp,
        )

        self.linear_proj = build_module(
            submodules.linear_proj,
            v_head_dim * config.num_attention_heads,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="proj",
            tp_group=self.pg_collection.tp,
        )

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None, **kwargs):
        """Split fused QKV with asymmetric V head dim."""
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        qk_ch = self.hidden_size_per_attention_head
        v_ch = self.config.v_head_dim

        # [sq, b, hp] -> [sq, b, ng, (heads_per_group*qk_ch + qk_ch + v_ch)]
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (self.num_attention_heads_per_partition // self.num_query_groups_per_partition) * qk_ch + qk_ch + v_ch,
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        split_arg_list = [
            (self.num_attention_heads_per_partition // self.num_query_groups_per_partition) * qk_ch,  # Q
            qk_ch,  # K
            v_ch,  # V
        ]

        if SplitAlongDim is not None:
            (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
        else:
            (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)

        # [sq, b, ng, heads_per_group * qk_ch] -> [sq, b, np, qk_ch]
        query = query.reshape(query.size(0), query.size(1), -1, qk_ch)

        return query, key, value

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tuple[Tensor, Tensor]] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Switch to either local or global rope embedding before forward"""
        assert isinstance(rotary_pos_emb, torch.Tensor) and rotary_pos_emb.ndim >= 1 and rotary_pos_emb.size(0) == 2
        assert rotary_pos_cos is None and rotary_pos_sin is None

        if _is_local_attn_layer(self.layer_number, self.config.hybrid_attention_pattern):
            final_rotary_pos_emb = rotary_pos_emb[0]
        else:
            final_rotary_pos_emb = rotary_pos_emb[1]
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            key_value_states=key_value_states,
            inference_context=inference_context,
            rotary_pos_emb=final_rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            inference_params=inference_params,
        )


class MiMoV2FlashTEDotProductAttention(TEDotProductAttention):
    """MiMoV2Flash core attention.

    Switches between global and local sliding window attention
    based on the layer_number and pre-defined layer pattern.
    SWA layers use a learnable softmax (attention-sink bias);
    full-attention layers use vanilla softmax.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        **kwargs,
    ):
        config = copy.deepcopy(config)
        if _is_local_attn_layer(layer_number, config.hybrid_attention_pattern):
            config.window_size = (config.window_size - 1, 0)
            config.softmax_type = "learnable"
        else:
            config.window_size = None
            config.softmax_type = "vanilla"
        self._attention_value_scale = getattr(config, "attention_value_scale", None)
        # Pass k_channels/v_channels to TE so it knows about asymmetric V head dim
        kwargs["k_channels"] = config.kv_channels
        kwargs["v_channels"] = config.v_head_dim

        super().__init__(
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            attention_dropout=attention_dropout,
            **kwargs,
        )

    def forward(self, query, key, value, attention_mask, attn_mask_type, **kwargs):
        if self._attention_value_scale is not None:
            value = value * self._attention_value_scale
        return super().forward(query, key, value, attention_mask, attn_mask_type, **kwargs)


class MiMoV2FlashMTPSelfAttention(MiMoV2FlashSelfAttention):
    """Overrides attention module for MTP"""

    def __init__(self, config, submodules, layer_number, *args, **kwargs):
        config = copy.deepcopy(config)
        config.hybrid_attention_pattern = [1] * config.mtp_num_layers
        super().__init__(config, submodules, layer_number, *args, **kwargs)


class MiMoV2FlashMTPTEDotProductAttention(MiMoV2FlashTEDotProductAttention):
    """Overrides core attention for MTP"""

    def __init__(self, config, layer_number, *args, **kwargs):
        config = copy.deepcopy(config)
        config.hybrid_attention_pattern = [1] * config.mtp_num_layers
        super().__init__(config, layer_number, *args, **kwargs)


def mimo_v2_flash_layer_spec(config) -> ModuleSpec:
    """Layer spec for MiMo-V2-Flash with custom hybrid attention modules.

    Builds the block spec (handles MoE/dense split) then injects custom
    self-attention and core-attention modules into every layer spec.
    """
    spec = get_gpt_decoder_block_spec(config, use_transformer_engine=True)
    for layer_spec in spec.layer_specs:
        layer_spec.submodules.self_attention.module = MiMoV2FlashSelfAttention
        layer_spec.submodules.self_attention.submodules.core_attention = MiMoV2FlashTEDotProductAttention
    return spec
