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


@MegatronModelBridge.register_bridge(source=Gemma3ForConditionalGeneration, target=Gemma3VLModel)
class Gemma3VLBridge(MegatronModelBridge):
    """
    Megatron Bridge for Gemma3 VL.
    """

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> Gemma3VLModelProvider:
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config
        vision_config = hf_config.vision_config

        provider = Gemma3VLModelProvider(
            # Text configuration
            init_method_std=text_config.initializer_range,
            hidden_size=text_config.hidden_size,
            ffn_hidden_size=text_config.intermediate_size,
            kv_channels=text_config.head_dim,
            seq_length=text_config.max_position_embeddings,
            num_attention_heads=text_config.num_attention_heads,
            num_layers=text_config.num_hidden_layers,
            num_query_groups=text_config.num_key_value_heads,
            window_size=text_config.sliding_window,
            rotary_base=(text_config.rope_local_base_freq, text_config.rope_theta),
            layernorm_epsilon=text_config.rms_norm_eps,
            vocab_size=text_config.vocab_size,
            softmax_scale=1.0 / math.sqrt(text_config.query_pre_attn_scalar),
            rope_scaling_factor=text_config.rope_scaling["factor"] if text_config.rope_scaling else 1.0,
            # Vision configuration
            vision_config=vision_config,
            mm_tokens_per_image=hf_config.mm_tokens_per_image,
            # VL-specific token IDs
            bos_token_id=getattr(hf_config, "bos_token_id", 0),
            eos_token_id=getattr(hf_config, "eos_token_id", 1),
            vision_start_token_id=getattr(hf_config, "vision_start_token_id", 255999),
            vision_end_token_id=getattr(hf_config, "vision_end_token_id", 256000),
            image_token_id=getattr(hf_config, "image_token_id", 151655),
            # Precision configuration
            fp16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16),
            bf16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16),
            params_dtype=self.dtype_from_hf(hf_config, default=torch.float32),
        )

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
