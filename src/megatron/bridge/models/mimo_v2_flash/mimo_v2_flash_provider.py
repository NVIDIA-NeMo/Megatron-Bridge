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

"""MiMo-V2-Flash Model Provider with dual-base RoPE.

The hybrid attention pattern (full vs SWA per layer) and per-layer KV head
switching are handled by storing config on the provider.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.transformer import ModuleSpec

from megatron.bridge.models import gpt_provider
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.mimo_v2_flash.modeling_mimo_v2_flash import (
    MiMoV2FlashMTPSelfAttention,
    MiMoV2FlashMTPTEDotProductAttention,
    MiMoV2FlashRotaryEmbedding,
    mimo_v2_flash_layer_spec,
)


@dataclass
class MiMoV2FlashModelProvider(GPTModelProvider):
    """Configuration and provider for MiMo-V2-Flash models.

    Extends GPTModelProvider with MiMo-V2-Flash-specific fields that need
    to persist in run_config.yaml and be accessible to custom modules.

    The hybrid attention pattern, per-layer KV heads, and dual RoPE bases
    are stored here. The ``provide()`` override replaces the standard RoPE
    with a dual-base version (same pattern as Gemma3ModelProvider).
    """

    transformer_layer_spec: Union[ModuleSpec, Callable[["MiMoV2FlashModelProvider"], ModuleSpec]] = field(
        default_factory=lambda: mimo_v2_flash_layer_spec
    )

    # Hybrid attention: 0=full, 1=SWA, one entry per layer
    hybrid_attention_pattern: Optional[List[int]] = None
    window_size: Union[int, tuple, None] = 128

    # Dual rope bases: (local/SWA theta, global/full theta)
    rotary_base: Union[int, float, tuple] = (10_000, 5_000_000)

    # Per-layer KV heads (full attention vs SWA layers)
    full_attn_num_query_groups: int = 4
    swa_num_query_groups: int = 8

    # Asymmetric V head dimension
    v_head_dim: int = 128

    # Attention value scale
    attention_value_scale: Optional[float] = None

    # Architecture defaults that differ from GPTModelProvider
    normalization: str = "RMSNorm"
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    position_embedding_type: str = "rope"
    share_embeddings_and_output_weights: bool = False

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        """Configure and instantiate a Megatron Core GPT model for MiMo-V2-Flash."""
        min_kv_groups = min(self.full_attn_num_query_groups, self.swa_num_query_groups)
        assert self.tensor_model_parallel_size <= min_kv_groups, "MiMo-V2-Flash requires TP size <= num query groups"
        assert self.context_parallel_size <= 1, "MiMo-V2-Flash does not support context parallelism yet."
        if isinstance(self.rotary_base, (tuple, list)):
            rotary_base_local, rotary_base_global = self.rotary_base
        else:
            rotary_base_local = 10_000
            rotary_base_global = self.rotary_base

        # MTP spec patch
        def mimov2flash_mtp_block_spec(config, vp_stage=None):
            if not getattr(config, "mtp_num_layers", None):
                return None
            dense_spec = get_gpt_layer_with_transformer_engine_spec(
                num_experts=None,
                moe_grouped_gemm=False,
                qk_layernorm=False,
                multi_latent_attention=False,
            )
            dense_spec.submodules.self_attention.module = MiMoV2FlashMTPSelfAttention
            dense_spec.submodules.self_attention.submodules.core_attention = MiMoV2FlashMTPTEDotProductAttention
            return get_gpt_mtp_block_spec(config, dense_spec, use_transformer_engine=True, vp_stage=vp_stage)

        original_mtp_block_spec = gpt_provider.mtp_block_spec
        gpt_provider.mtp_block_spec = mimov2flash_mtp_block_spec
        self.rotary_base = rotary_base_local
        try:
            model = super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        finally:
            gpt_provider.mtp_block_spec = original_mtp_block_spec
        self.rotary_base = (rotary_base_local, rotary_base_global)

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
