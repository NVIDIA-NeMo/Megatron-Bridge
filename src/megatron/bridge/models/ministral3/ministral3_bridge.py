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

"""
Megatron Bridge for Ministral 3 Vision-Language Models.

This module provides the bridge implementation for converting between HuggingFace
Ministral-3 models and Megatron-Core format.

Supported models:
- Ministral-3-3B-Base-2512
- Ministral-3-3B-Instruct-2512
- Ministral-3-3B-Reasoning-2512
- Ministral-3-8B-Base-2512
- Ministral-3-8B-Instruct-2512
- Ministral-3-8B-Reasoning-2512
- Ministral-3-14B-Base-2512
- Ministral-3-14B-Instruct-2512
- Ministral-3-14B-Reasoning-2512

Reference: https://huggingface.co/mistralai/Ministral-3-3B-Base-2512
"""

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.ministral3.ministral3_provider import Ministral3ModelProvider


# Import HuggingFace model classes with fallback for older transformers versions
try:
    from transformers import Mistral3ForConditionalGeneration

    HAS_MISTRAL3 = True
except ImportError:
    Mistral3ForConditionalGeneration = None
    HAS_MISTRAL3 = False


class Ministral3Bridge(MegatronModelBridge):
    """
    Megatron Bridge for Ministral 3 Vision-Language Models.

    This bridge handles conversion between HuggingFace Mistral3ForConditionalGeneration
    and Megatron-Core Ministral3Model format for vision-language models.

    The weight mappings handle:
    - Vision model weights (vision encoder)
    - Language model weights
    - Multimodal projector weights
    - Special token embeddings

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("mistralai/Ministral-3-3B-Base-2512")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> Ministral3ModelProvider:
        """
        Create a Ministral3ModelProvider from a HuggingFace pretrained VL model.

        Args:
            hf_pretrained: HuggingFace pretrained VLM model

        Returns:
            Ministral3ModelProvider configured with the HF model's parameters
        """
        hf_config = hf_pretrained.config

        # Ministral 3 has separate text_config and vision_config
        text_config = getattr(hf_config, "text_config", hf_config)
        provider = Ministral3ModelProvider(
            hidden_size=text_config.hidden_size,
            ffn_hidden_size=text_config.intermediate_size,
            num_layers=text_config.num_hidden_layers,
            share_embeddings_and_output_weights=getattr(text_config, "tie_word_embeddings", False),
            rotary_base=text_config.rope_parameters["rope_theta"],
            vocab_size=text_config.vocab_size,
            hf_config=hf_config,
        )

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """
        Return MegatronMappingRegistry containing parameter mappings for VL models.

        HuggingFace weight structure:
        - language_model.model.embed_tokens.weight
        - language_model.model.layers.{i}.input_layernorm.weight
        - language_model.model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
        - language_model.model.layers.{i}.post_attention_layernorm.weight
        - language_model.model.layers.{i}.mlp.{gate,up,down}_proj.weight
        - language_model.model.norm.weight
        - language_model.lm_head.weight
        - vision_tower.** (patch_conv, ln_pre, transformer layers)
        - multi_modal_projector.{norm,linear}.weight

        Returns:
            MegatronMappingRegistry with all parameter mappings
        """
        # Language model direct mappings
        # Maps: Megatron param name -> HuggingFace param name
        param_mappings = {
            # Embeddings and output layers
            "language_model.embedding.word_embeddings.weight": "language_model.model.embed_tokens.weight",
            "language_model.output_layer.weight": "language_model.lm_head.weight",
            "language_model.decoder.final_layernorm.weight": "language_model.model.norm.weight",
            # Layer normalization for attention and MLP
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "language_model.model.layers.*.input_layernorm.weight",
            "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "language_model.model.layers.*.post_attention_layernorm.weight",
            # Attention output projection
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": "language_model.model.layers.*.self_attn.o_proj.weight",
            # MLP output projection
            "language_model.decoder.layers.*.mlp.linear_fc2.weight": "language_model.model.layers.*.mlp.down_proj.weight",
        }

        mapping_list = []
        # Convert each dictionary entry to AutoMapping(megatron_param, hf_param)
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Add special mappings that require parameter transformation
        mapping_list.extend(
            [
                # Vision tower weights are replicated directly
                # Includes: patch_conv, ln_pre, transformer.layers.*.attention.*, transformer.layers.*.feed_forward.*
                ReplicatedMapping(
                    megatron_param="vision_tower.**",
                    hf_param="vision_tower.**",
                ),
                # Multimodal projector weights (norm.weight, linear.weight)
                ReplicatedMapping(
                    megatron_param="multi_modal_projector.**",
                    hf_param="multi_modal_projector.**",
                ),
                # QKV: Combine separate Q, K, V matrices into single QKV matrix
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    q="language_model.model.layers.*.self_attn.q_proj.weight",
                    k="language_model.model.layers.*.self_attn.k_proj.weight",
                    v="language_model.model.layers.*.self_attn.v_proj.weight",
                ),
                # Gated MLP: Combine gate and up projection matrices into single FC1 matrix
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.linear_fc1.weight",
                    gate="language_model.model.layers.*.mlp.gate_proj.weight",
                    up="language_model.model.layers.*.mlp.up_proj.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)


# Register the bridge if Mistral3ForConditionalGeneration is available
if HAS_MISTRAL3 and Mistral3ForConditionalGeneration is not None:
    # Import Ministral3Model for target registration
    from megatron.bridge.models.ministral3.modeling_ministral3 import Ministral3Model

    # Dynamically register the bridge with Ministral3Model as target
    try:
        Ministral3Bridge = MegatronModelBridge.register_bridge(
            source=Mistral3ForConditionalGeneration, target=Ministral3Model
        )(Ministral3Bridge)
    except Exception:
        # If registration fails, the bridge will still work manually
        pass
