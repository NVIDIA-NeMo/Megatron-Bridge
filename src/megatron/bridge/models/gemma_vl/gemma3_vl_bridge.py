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

import math

import torch
from transformers import Gemma3ForConditionalGeneration

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.gemma_vl.gemma3_vl_provider import Gemma3VLModelProvider
from megatron.bridge.models.gemma_vl.modeling_gemma3_vl import Gemma3VLModel
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM


@MegatronModelBridge.register_bridge(
    source=Gemma3ForConditionalGeneration,
    target=Gemma3VLModel,
    provider=Gemma3VLModelProvider,
    model_type="gemma3_vl",
)
class Gemma3VLBridge(MegatronModelBridge):
    """
    Megatron Bridge for Gemma3 VL.

    This bridge handles the conversion between HuggingFace Gemma3ForConditionalGeneration
    and Megatron-Core Gemma3VLModel formats, including weight mappings and
    configuration translation for vision-language models.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("google/gemma-3-4b-it")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> Gemma3VLModelProvider:
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config
        vision_config = hf_config.vision_config

        # Use base class helper for common config conversion
        provider_kwargs = self.hf_config_to_provider_kwargs(text_config)
        provider = Gemma3VLModelProvider(**provider_kwargs)

        # Gemma3-specific features not in CONFIG_MAPPING
        provider.window_size = text_config.sliding_window
        provider.rotary_base = (text_config.rope_local_base_freq, text_config.rope_theta)
        provider.softmax_scale = 1.0 / math.sqrt(text_config.query_pre_attn_scalar)
        provider.rope_scaling_factor = text_config.rope_scaling["factor"] if text_config.rope_scaling else 1.0

        # Override dtype and vocab settings to match baseline
        provider.bf16 = True
        provider.params_dtype = torch.bfloat16
        provider.autocast_dtype = torch.bfloat16
        provider.make_vocab_size_divisible_by = 128

        # Vision configuration
        provider.vision_config = vision_config
        provider.mm_tokens_per_image = hf_config.mm_tokens_per_image

        # VL-specific token IDs
        provider.bos_token_id = getattr(hf_config, "bos_token_id", 0)
        provider.eos_token_id = getattr(hf_config, "eos_token_id", 1)
        provider.vision_start_token_id = getattr(hf_config, "vision_start_token_id", 255999)
        provider.vision_end_token_id = getattr(hf_config, "vision_end_token_id", 256000)
        provider.image_token_id = getattr(hf_config, "image_token_id", 262144)

        # Vision projector configuration
        provider.vision_projector_config.input_size = vision_config.hidden_size
        provider.vision_projector_config.hidden_size = text_config.hidden_size

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        # Return MegatronMappingRegistry containing parameter mappings from Megatron to HF format
        # First create simple 1:1 parameter mappings using a dictionary for readability

        # Dictionary maps Megatron parameter names -> HF parameter names
        # Supports wildcard (*) patterns for layer-specific parameters
        param_mappings = {
            "language_model.model.embed_tokens.weight": "language_model.embedding.word_embeddings.weight",
            "language_model.model.layers.*.input_layernorm.weight": "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "language_model.model.layers.*.self_attn.q_norm.weight": "language_model.decoder.layers.*.self_attention.q_layernorm.weight",
            "language_model.model.layers.*.self_attn.k_norm.weight": "language_model.decoder.layers.*.self_attention.k_layernorm.weight",
            "language_model.model.layers.*.self_attn.o_proj.weight": "language_model.decoder.layers.*.self_attention.linear_proj.weight",
            "language_model.model.layers.*.post_attention_layernorm.weight": (
                "language_model.decoder.layers.*.self_attention.linear_proj.post_layernorm.weight"
            ),
            "language_model.model.layers.*.pre_feedforward_layernorm.weight": "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "language_model.model.layers.*.mlp.down_proj.weight": "language_model.decoder.layers.*.mlp.linear_fc2.weight",
            "language_model.model.layers.*.post_feedforward_layernorm.weight": (
                "language_model.decoder.layers.*.mlp.linear_fc2.post_layernorm.weight"
            ),
            "language_model.model.norm.weight": "language_model.decoder.final_layernorm.weight",
            # Vision projector
            "multi_modal_projector.mm_soft_emb_norm.weight": "multi_modal_projector.mm_soft_embed_norm.weight",
        }

        mapping_list = []
        # Convert each dictionary entry to AutoMapping(megatron_param, hf_param)
        for hf_param, megatron_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Add special mappings that require parameter concatenation/transformation
        mapping_list.extend(
            [
                ReplicatedMapping(
                    megatron_param="vision_tower.**",
                    hf_param="vision_tower.**",
                ),
                AutoMapping(
                    megatron_param="multi_modal_projector.proj.weight",
                    hf_param="multi_modal_projector.mm_input_projection_weight",
                    permute_dims=(1, 0),
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
