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

"""GPT configuration and layer-spec wiring for Dual Chunk Attention."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, TypeAlias

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules

from megatron.bridge.models.gpt.gpt_builder import GPTModelConfig
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.transformer.dca import (
    DualChunkAttention,
    DualChunkSelfAttention,
    DualChunkTransformerConfig,
    validate_dual_chunk_transformer_config,
)


GPTConfig: TypeAlias = GPTModelConfig | GPTModelProvider


def _transformer_config(config: GPTConfig) -> DualChunkTransformerConfig | GPTModelProvider:
    return config.transformer if isinstance(config, GPTModelConfig) else config


def validate_dual_chunk_gpt_config(config: GPTConfig) -> None:
    """Validate GPT-level settings required by DCA.

    Args:
        config: Modern GPT model config or legacy GPT model provider.

    Raises:
        ValueError: If the GPT configuration selects an unsupported mode.
    """
    transformer_config = _transformer_config(config)
    validate_dual_chunk_transformer_config(transformer_config)

    if config.position_embedding_type not in {"rope", "yarn"}:
        raise ValueError("DCA requires position_embedding_type to be 'rope' or 'yarn'.")
    if config.position_embedding_type == "yarn":
        required_yarn_fields = (
            "yarn_rotary_scaling_factor",
            "yarn_original_max_position_embeddings",
            "yarn_beta_fast",
            "yarn_beta_slow",
            "yarn_correction_range_round_to_int",
        )
        missing_yarn_fields = [
            name for name in required_yarn_fields if getattr(transformer_config, name, None) is None
        ]
        if missing_yarn_fields:
            raise ValueError(f"DCA with YARN requires: {', '.join(missing_yarn_fields)}.")
    if config.attention_backend == AttnBackend.local:
        raise ValueError("DCA is not compatible with attention_backend=local.")
    if getattr(config, "use_transformer_engine_full_layer_spec", False):
        raise ValueError("DCA does not support use_transformer_engine_full_layer_spec yet.")
    if getattr(config, "restore_modelopt_state", False):
        raise ValueError("DCA does not support restore_modelopt_state yet.")
    if getattr(config, "mtp_enabled", False):
        raise ValueError("DCA does not support MTP yet.")


def get_dca_gpt_layer_spec(
    config: GPTConfig,
    vp_stage: int | None = None,
) -> TransformerBlockSubmodules:
    """Build a GPT transformer block spec using DCA in every decoder layer.

    Args:
        config: Modern GPT model config or legacy GPT model provider.
        vp_stage: Optional virtual pipeline stage used for layer slicing.

    Returns:
        Transformer block submodules with DCA self-attention installed.
    """
    validate_dual_chunk_gpt_config(config)
    transformer_config = _transformer_config(config)
    use_transformer_engine = transformer_config.transformer_impl == "transformer_engine"
    block_spec = deepcopy(
        get_gpt_decoder_block_spec(
            transformer_config,
            use_transformer_engine=use_transformer_engine,
            normalization=transformer_config.normalization,
            qk_l2_norm=transformer_config.qk_l2_norm,
            vp_stage=vp_stage,
        )
    )

    for layer_spec in block_spec.layer_specs:
        self_attention_spec = layer_spec.submodules.self_attention
        self_attention_spec.module = DualChunkSelfAttention
        self_attention_spec.params = dict(self_attention_spec.params or {})
        self_attention_spec.params["attn_mask_type"] = AttnMaskType.causal
        self_attention_spec.submodules.core_attention = ModuleSpec(
            module=DualChunkAttention,
            params={
                "dca_chunk_size": transformer_config.dca_chunk_size,
                "dca_local_size": transformer_config.dca_local_size,
            },
        )

    return block_spec


@dataclass(kw_only=True)
class DualChunkGPTModelConfig(GPTModelConfig):
    """Modern GPT builder configuration with DCA enabled by construction."""

    transformer: DualChunkTransformerConfig
    transformer_layer_spec: ModuleSpec | Callable[[GPTModelConfig], ModuleSpec] | None = get_dca_gpt_layer_spec

    def finalize(self) -> None:
        """Finalize the GPT configuration and validate DCA integration."""
        super().finalize()
        validate_dual_chunk_gpt_config(self)


@dataclass(kw_only=True)
class DualChunkGPTModelProvider(GPTModelProvider):
    """Legacy GPT provider with explicit DCA parameters and layer spec."""

    dca_chunk_size: int
    dca_local_size: int
    transformer_layer_spec: ModuleSpec | Callable[[GPTModelProvider], ModuleSpec] = get_dca_gpt_layer_spec

    def finalize(self) -> None:
        """Finalize the provider and validate DCA integration."""
        super().finalize()
        validate_dual_chunk_gpt_config(self)


__all__ = [
    "DualChunkGPTModelConfig",
    "DualChunkGPTModelProvider",
    "get_dca_gpt_layer_spec",
    "validate_dual_chunk_gpt_config",
]
