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

"""Megatron Bridge for EXAONE 4.0 (LG AI Research).

EXAONE 4.0 architecture overview:
- Pure Post-LayerNorm (no Pre-LN / input_layernorm)
- QK RMSNorm on attention query and key projections
- GQA with 32 heads / 8 KV heads
- SwiGLU activation
- RoPE with llama3-style scaling
- Tied word embeddings (embed_tokens == lm_head)

Key differences from standard GPT-style model bridges:
- No input_layernorm or pre_feedforward_layernorm weights
- Has post_attention_layernorm (after self-attention output)
- Has post_feedforward_layernorm (after MLP output, EXAONE-specific)
- Post-LN mapping follows Gemma2 pattern: *.post_layernorm.weight

References:
- HuggingFace: LGAI-EXAONE/EXAONE-4.0-1.2B
- Gemma2 bridge: Post-LN via TERowParallelLinearLayerNorm pattern
- EXAONE bridge: QK layernorm mapping pattern
"""

import logging
from typing import Any

import torch
import torch.nn.functional as F
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import PretrainedConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.exaone.exaone4.exaone4_provider import exaone4_layer_spec
from megatron.bridge.models.gpt_provider import GPTModelProvider


logger = logging.getLogger(__name__)

# Register custom EXAONE modules for AutoMapping weight distribution
# TERowParallelLinearLayerNorm is a row-parallel linear with post-layernorm
# (same pattern as Gemma2 for Post-LN architectures)
AutoMapping.register_module_type("TERowParallelLinearLayerNorm", "row")


@MegatronModelBridge.register_bridge(
    source="Exaone4ForCausalLM",  # HF architecture string (auto_map / trust_remote_code)
    target=GPTModel,
    model_type="exaone4",
)
class Exaone4Bridge(MegatronModelBridge):
    """
    Megatron Bridge for EXAONE 4.0 Causal LM.

    Supports bidirectional conversion between HuggingFace EXAONE 4.0 checkpoints
    and Megatron-Core GPTModel format.

    Architecture notes:
    - EXAONE 4.0 uses pure Post-LayerNorm (no input_layernorm).
    - Post-LN is implemented via custom layer spec with TERowParallelLinearLayerNorm,
      following the same pattern established by Gemma2 bridge.
    - QK RMSNorm is mapped using EXAONE attention norm parameters.
    - 1.2B model uses full attention only (no sliding window / hybrid attention).
    - 32B model introduces hybrid attention (LLLG pattern) — future extension.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained(
        ...     "LGAI-EXAONE/EXAONE-4.0-1.2B",
        ...     trust_remote_code=True,
        ... )
        >>> model_config = bridge.get_model_config()
    """

    USE_MODEL_CONFIG_FOR_CONVERSION = True

    # Megatron Core applies llama3 RoPE scaling with these fixed values; only the factor is configurable.
    _MCORE_LLAMA3_ROPE_SCALING = {
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    }

    def hf_config_to_model_config_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Convert a Hugging Face EXAONE 4.0 config to builder config kwargs.

        Sets EXAONE-specific options including Post-LN, QK norm, and RoPE scaling.
        Size-dependent fields are populated by the shared HF config mapping.
        """
        config_kwargs = super().hf_config_to_model_config_kwargs(hf_config)
        config_kwargs.update(
            normalization="RMSNorm",
            activation_func=F.silu,
            gated_linear_unit=True,
            position_embedding_type="rope",
            add_bias_linear=False,
            add_qkv_bias=False,
            qk_layernorm=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            transformer_layer_spec=exaone4_layer_spec,
            autocast_dtype=torch.bfloat16,
            masked_softmax_fusion=True,
            rope_scaling=False,
            rope_scaling_factor=1.0,
        )

        # RoPE scaling for EXAONE 4.0 (llama3-style)
        hf_rope_scaling = getattr(hf_config, "rope_scaling", None)
        if hf_rope_scaling is not None and hf_rope_scaling.get("rope_type") == "llama3":
            config_kwargs.update(rope_scaling=True, rope_scaling_factor=hf_rope_scaling.get("factor", 16.0))
            unsupported = {
                key: hf_rope_scaling[key]
                for key, value in self._MCORE_LLAMA3_ROPE_SCALING.items()
                if key in hf_rope_scaling and hf_rope_scaling[key] != value
            }
            if unsupported:
                logger.warning(
                    "EXAONE 4.0 llama3 RoPE scaling values %s differ from the fixed values Megatron Core applies (%s).",
                    unsupported,
                    self._MCORE_LLAMA3_ROPE_SCALING,
                )

        return config_kwargs

    def hf_config_to_provider_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Adapt the canonical builder mapping to the deprecated provider path."""
        return self.hf_config_to_model_config_kwargs(hf_config)

    @classmethod
    def megatron_to_hf_config(cls, provider: GPTModelProvider) -> dict:
        """Convert Megatron GPTModelProvider config to HuggingFace config dict.

        Args:
            provider: GPTModelProvider with EXAONE configuration

        Returns:
            Dictionary of HuggingFace Exaone4Config parameters
        """
        hf_config = super(Exaone4Bridge, cls).megatron_to_hf_config(provider)

        # EXAONE-specific config fields
        hf_config["model_type"] = "exaone4"
        hf_config["tie_word_embeddings"] = provider.share_embeddings_and_output_weights

        # RoPE scaling (read from provider, no hard-coded constants)
        if provider.rope_scaling:
            hf_config["rope_scaling"] = {
                "rope_type": "llama3",
                "factor": provider.rope_scaling_factor,
                # Megatron Core's fixed llama3 values, unless a caller set them explicitly.
                **{
                    key: getattr(provider, f"rope_scaling_{key}", value)
                    for key, value in cls._MCORE_LLAMA3_ROPE_SCALING.items()
                },
            }

        return hf_config

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Return MegatronMappingRegistry containing parameter mappings.

        EXAONE 4.0 weight mapping combines patterns from:
        - Llama: Basic GPT structure (embed, QKV, GatedMLP, final_layernorm)
        - EXAONE: QK layernorm (q_norm → q_layernorm, k_norm → k_layernorm)
        - Gemma2: Post-LN (post_*_layernorm → *.post_layernorm.weight)

        Key difference: No input_layernorm or pre_feedforward_layernorm mappings
        because EXAONE uses pure Post-LN (not Pre-LN or sandwich norm).
        """

        # =====================================================================
        # 1:1 Parameter Mappings (Megatron → HF)
        # =====================================================================
        param_mappings = {
            # Embedding & output
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            # NOTE: No lm_head.weight mapping — tie_word_embeddings=true reuses embed_tokens
            "decoder.final_layernorm.weight": "model.norm.weight",
            # Attention output projection
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            # QK RMSNorm
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",
            # Post-LN: post-attention layernorm (Gemma2 pattern)
            "decoder.layers.*.self_attention.linear_proj.post_layernorm.weight": (
                "model.layers.*.post_attention_layernorm.weight"
            ),
            # Post-LN: post-feedforward layernorm (Gemma2 pattern)
            "decoder.layers.*.mlp.linear_fc2.post_layernorm.weight": (
                "model.layers.*.post_feedforward_layernorm.weight"
            ),
            # MLP down projection
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
        }

        mapping_list = []
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # =====================================================================
        # Composite Mappings (require concatenation/splitting)
        # =====================================================================
        mapping_list.extend(
            [
                # QKV: Merge separate Q, K, V projections into single QKV matrix
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),
                # Gated MLP: Merge gate and up projections into single FC1 matrix
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
