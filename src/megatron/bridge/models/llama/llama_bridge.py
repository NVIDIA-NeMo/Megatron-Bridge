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

import logging
from typing import TYPE_CHECKING

from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import LlamaForCausalLM

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.llama.llama_provider import apply_rope_scaling


if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


# =============================================================================
# Rope Scaling Helper
# =============================================================================


@MegatronModelBridge.register_bridge(source=LlamaForCausalLM, target=GPTModel)
class LlamaBridge(MegatronModelBridge):
    """
    Megatron Bridge for Llama Causal LM.

    As a user you would not use this bridge directly, but through `AutoBridge`.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        >>> provider = bridge.to_megatron_provider()
    """

    # Llama inherits CONFIG_MAPPING and ACTIVATION_MAPPING from MegatronModelBridge base class
    # No additional model-specific mappings needed for Llama

    # Llama-specific defaults for Megatron provider
    # Note: Fields in CONFIG_MAPPING are dynamically converted, not set here
    MEGATRON_DEFAULTS = {
        "normalization": "RMSNorm",
        "gated_linear_unit": True,
        "position_embedding_type": "rope",
        "hidden_dropout": 0.0,
        "bias_activation_fusion": True,
        "masked_softmax_fusion": True,
        "persist_layer_norm": True,
        "bias_dropout_fusion": True,
        "apply_rope_fusion": True,
        "rotary_percent": 1.0,
    }

    # Llama-specific defaults for HF config
    # Note: Fields in CONFIG_MAPPING are dynamically converted, not set here
    HF_DEFAULTS = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
    }

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GPTModelProvider:
        """Convert HuggingFace Llama config to Megatron GPTModelProvider.

        Uses base class implementation for common conversion, then adds
        RoPE scaling config for Llama 3.1/3.2 models.

        Args:
            hf_pretrained: HuggingFace PreTrainedCausalLM containing the Llama config

        Returns:
            GPTModelProvider configured for Llama architecture
        """
        # Use base class for common conversion
        provider = super().provider_bridge(hf_pretrained)

        # Store RoPE scaling config on provider for Llama 3.1/3.2 models
        hf_config = hf_pretrained.config
        rope_scaling = getattr(hf_config, "rope_scaling", None)
        if rope_scaling is not None and rope_scaling.get("rope_type") == "llama3":
            provider._rope_scaling = {
                "factor": rope_scaling.get("factor", 8.0),
                "low_freq_factor": rope_scaling.get("low_freq_factor", 1.0),
                "high_freq_factor": rope_scaling.get("high_freq_factor", 4.0),
                "original_max_position_embeddings": rope_scaling.get("original_max_position_embeddings", 8192),
            }

        return provider

    @staticmethod
    def apply_rope_scaling_to_model(model: GPTModel, provider: GPTModelProvider) -> None:
        """Apply RoPE scaling to a model if the provider has rope_scaling config.

        This should be called after model creation for Llama 3.1/3.2 models that
        require RoPE scaling for extended context lengths.

        Args:
            model: The GPTModel instance to apply RoPE scaling to
            provider: The GPTModelProvider that may contain _rope_scaling config
        """
        rope_scaling = getattr(provider, "_rope_scaling", None)
        if rope_scaling is None:
            return

        if hasattr(model, "rotary_pos_emb") and model.rotary_pos_emb is not None:
            model.rotary_pos_emb.inv_freq = apply_rope_scaling(
                model.rotary_pos_emb.inv_freq,
                factor=rope_scaling["factor"],
                low_freq_factor=rope_scaling["low_freq_factor"],
                high_freq_factor=rope_scaling["high_freq_factor"],
                old_context_len=rope_scaling["original_max_position_embeddings"],
            )

    @classmethod
    def megatron_to_hf_config(cls, provider: GPTModelProvider) -> dict:
        """Convert Megatron GPTModelProvider config to HuggingFace Llama config dict.

        Uses base class implementation, then adds RoPE scaling for Llama 3.1/3.2.

        Args:
            provider: GPTModelProvider with Llama configuration

        Returns:
            Dictionary of HuggingFace LlamaConfig parameters
        """
        # Use base class for common conversion (provider_to_hf_config + HF_DEFAULTS)
        hf_config = super(LlamaBridge, cls).megatron_to_hf_config(provider)

        # Handle RoPE scaling for Llama 3.1 models
        rope_scaling = getattr(provider, "_rope_scaling", None)
        if rope_scaling is not None:
            hf_config["rope_scaling"] = {
                "rope_type": "llama3",
                **rope_scaling,
            }

        return hf_config

    def mapping_registry(self) -> MegatronMappingRegistry:
        # Return MegatronMappingRegistry containing parameter mappings from Megatron to HF format
        # First create simple 1:1 parameter mappings using a dictionary for readability

        # Dictionary maps Megatron parameter names -> HF parameter names
        # Supports wildcard (*) patterns for layer-specific parameters
        param_mappings = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",  # te implementation
            "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",  # local implementation
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",  # te implementation
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",  # local implementation
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
        }

        mapping_list = []
        # Convert each dictionary entry to AutoMapping(megatron_param, hf_param)
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Add special mappings that require parameter concatenation/transformation
        mapping_list.extend(
            [
                # QKV: Combine separate Q, K, V matrices into single QKV matrix
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),
                # Gated MLP: Combine gate and up projection matrices into single FC1 matrix
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
