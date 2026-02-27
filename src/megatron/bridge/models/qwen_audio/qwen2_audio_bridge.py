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
Megatron Bridge for Qwen2-Audio Models.

This module provides the bridge implementation for converting between HuggingFace
Qwen2-Audio models and Megatron-Core format.

Supported models:
- Qwen2-Audio-7B
- Qwen2-Audio-7B-Instruct

Reference: https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct
"""

from transformers import Qwen2AudioForConditionalGeneration

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.qwen_audio.modeling_qwen2_audio import Qwen2AudioModel
from megatron.bridge.models.qwen_audio.qwen2_audio_provider import Qwen2AudioModelProvider


@MegatronModelBridge.register_bridge(
    source=Qwen2AudioForConditionalGeneration,
    target=Qwen2AudioModel,
    provider=Qwen2AudioModelProvider,
    model_type="qwen2_audio",
)
class Qwen2AudioBridge(MegatronModelBridge):
    """
    Megatron Bridge for Qwen2-Audio Models.

    This bridge handles conversion between HuggingFace Qwen2AudioForConditionalGeneration
    and Megatron-Core Qwen2AudioModel format for audio-language models.

    The weight mappings handle:
    - Audio encoder weights (audio_tower)
    - Language model weights
    - Multimodal projector weights

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> Qwen2AudioModelProvider:
        """
        Create a Qwen2AudioModelProvider from a HuggingFace pretrained model.

        Args:
            hf_pretrained: HuggingFace pretrained model

        Returns:
            Qwen2AudioModelProvider configured with the HF model's parameters
        """
        hf_config = hf_pretrained.config

        # Qwen2-Audio has separate text_config and audio_config
        text_config = getattr(hf_config, "text_config", hf_config)

        # Use base class helper for common config conversion
        provider_kwargs = self.hf_config_to_provider_kwargs(text_config)
        provider = Qwen2AudioModelProvider(**provider_kwargs)

        # Qwen2-specific settings
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_qkv_bias = True
        provider.add_bias_linear = False
        provider.hidden_dropout = 0.0

        # Audio-specific settings
        provider.hf_config = hf_config
        provider.audio_token_id = getattr(hf_config, "audio_token_index", 151646)
        provider.bos_token_id = getattr(hf_config, "bos_token_id", 151643)
        provider.eos_token_id = getattr(hf_config, "eos_token_id", 151645)
        provider.pad_token_id = getattr(hf_config, "pad_token_id", 151643)

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """
        Return MegatronMappingRegistry containing parameter mappings for audio-language models.

        HuggingFace weight structure:
        - language_model.model.embed_tokens.weight
        - language_model.model.layers.{i}.input_layernorm.weight
        - language_model.model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
        - language_model.model.layers.{i}.post_attention_layernorm.weight
        - language_model.model.layers.{i}.mlp.{gate,up,down}_proj.weight
        - language_model.model.norm.weight
        - language_model.lm_head.weight
        - audio_tower.** (conv1, conv2, embed_positions, layers, layer_norm, avg_pooler)
        - multi_modal_projector.linear.weight

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
                # Audio tower weights are replicated directly
                # Includes: conv1, conv2, embed_positions, layers.*.self_attn.*, layers.*.fc1, layers.*.fc2, layer_norm, avg_pooler
                ReplicatedMapping(
                    megatron_param="audio_tower.**",
                    hf_param="audio_tower.**",
                ),
                # Multimodal projector weights (linear layer)
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
                # QKV bias: Combine separate Q, K, V biases into single QKV bias (Qwen2 specific)
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.bias",
                    q="language_model.model.layers.*.self_attn.q_proj.bias",
                    k="language_model.model.layers.*.self_attn.k_proj.bias",
                    v="language_model.model.layers.*.self_attn.v_proj.bias",
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
