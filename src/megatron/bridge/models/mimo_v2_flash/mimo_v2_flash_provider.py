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

"""MiMo-V2-Flash Model Provider with dual-base RoPE.

MiMo-V2-Flash requires a custom provider because:
1. Dual rope bases: full attention uses theta=5M, SWA uses theta=10K
2. Extra fields needed for hybrid attention config and per-layer KV heads

The hybrid attention pattern (full vs SWA per layer) and per-layer KV head
switching are handled by storing config on the provider. Future MCore support
for per-layer attention switching will allow these to be wired more directly.
"""
import copy
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, List, Optional, Union, Tuple

import torch
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.transformer import ModuleSpec, TransformerConfig
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from torch import Tensor

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.utils.import_utils import safe_import_from

TEDotProductAttention, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TEDotProductAttention")
TELayerNormColumnParallelLinear, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TELayerNormColumnParallelLinear")
TERowParallelLinear, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TERowParallelLinear")


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
        # Initialize global (full attention) rope
        super().__init__(rotary_base=rotary_base, **kwargs)

        # Initialize local (SWA) rope
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



def _is_local_attn_layer(
    layer_number: int,
    hybrid_attention_pattern: List[int],
) -> bool:
    # MCore layer_number starts at 1, pattern is 0-indexed
    return hybrid_attention_pattern[layer_number - 1] == 1


class MiMoV2FlashSelfAttention(SelfAttention):
    """MoMoV2Flash self attention.

    Uses local rope embedding for local layers,
    global rope embedding for global layers.
    Per-layer KV head count: SWA layers use swa_num_query_groups, full layers use full_attn_num_query_groups.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules,
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
        # Overwrite config.window_size based on layer_number
        config = copy.deepcopy(config)
        if _is_local_attn_layer(layer_number, config.hybrid_attention_pattern):
            config.window_size = (config.window_size - 1, 0)
            config.softmax_type = "learnable" if config.add_swa_attention_sink_bias else "vanilla"
        else:
            config.window_size = None
            config.softmax_type = "learnable" if config.add_full_attention_sink_bias else "vanilla"
        self._attention_value_scale = getattr(config, "attention_value_scale", None)
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
            # MiMoV2Flash uses an attention value scale factor
            value = value * self._attention_value_scale
        return super().forward(query, key, value, attention_mask, attn_mask_type, **kwargs)



@dataclass
class MiMoV2FlashModelProvider(GPTModelProvider):
    """Configuration and provider for MiMo-V2-Flash models.

    Extends GPTModelProvider with MiMo-V2-Flash-specific fields that need
    to persist in run_config.yaml and be accessible to custom modules.

    The hybrid attention pattern, per-layer KV heads, and dual RoPE bases
    are stored here. The ``provide()`` override replaces the standard RoPE
    with a dual-base version (same pattern as Gemma3ModelProvider).
    """

    # Hybrid attention: 0=full, 1=SWA, one entry per layer
    hybrid_attention_pattern: Optional[List[int]] = None
    window_size: Union[int, tuple, None] = 128

    # Dual rope bases: (local/SWA theta, global/full theta)
    rotary_base: Union[int, float, tuple] = (10_000, 5_000_000)

    # Per-layer KV heads (full attention vs SWA layers)
    full_attn_num_query_groups: int = 4
    swa_num_query_groups: int = 8

    # Asymmetric V head dimension (Q/K use kv_channels=192, V uses v_head_dim=128)
    v_head_dim: int = 128

    # Attention sink bias
    add_swa_attention_sink_bias: bool = True
    add_full_attention_sink_bias: bool = False

    # Attention value scale
    attention_value_scale: Optional[float] = None

    # Architecture defaults that differ from GPTModelProvider
    normalization: str = "RMSNorm"
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    position_embedding_type: str = "rope"
    share_embeddings_and_output_weights: bool = False

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        """Configure and instantiate a Megatron Core GPT model for MiMo-V2-Flash.

        Replaces the model's RoPE with a dual-base version that computes both
        local (SWA) and global (full attention) embeddings. This follows the
        same pattern as Gemma3ModelProvider.
        """
        # Resolve dual rope base — temporarily set the local base for MCore init
        if isinstance(self.rotary_base, tuple):
            rotary_base_local, rotary_base_global = self.rotary_base
        else:
            rotary_base_local = 10_000
            rotary_base_global = self.rotary_base

        self.rotary_base = rotary_base_local
        model = super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        self.rotary_base = (rotary_base_local, rotary_base_global)

        # Patch layer specs to use custom hybrid attention modules.
        # get_gpt_decoder_block_spec already handled MoE/dense split
        for layer_spec in model.decoder.submodules.layer_specs:
            layer_spec.submodules.self_attention.module = MiMoV2FlashSelfAttention
            layer_spec.submodules.self_attention.submodules.core_attention = MiMoV2FlashTEDotProductAttention

        # Replace model's RoPE with dual-base version
        model.rotary_pos_emb = MiMoV2FlashRotaryEmbedding(
            kv_channels=self.kv_channels,
            rotary_percent=self.rotary_percent,
            rotary_interleaved=self.rotary_interleaved,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            rotary_base=rotary_base_global,
            rope_scaling=False,
            use_cpu_initialization=self.use_cpu_initialization,
            rotary_base_local=rotary_base_local,
        )

        return model
