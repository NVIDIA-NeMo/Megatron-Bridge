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

import torch
from transformers import Glm4vMoeForConditionalGeneration

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.glm_vl.glm_45v_provider import GLM45VModelProvider
from megatron.bridge.models.glm_vl.modeling_glm_45v import GLM45VModel
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM


@MegatronModelBridge.register_bridge(source=Glm4vMoeForConditionalGeneration, target=GLM45VModel)
class GLM45VBridge(MegatronModelBridge):
    """
    Megatron Bridge for GLM 4.5 Vision-Language (VL) Models.
    """

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> GLM45VModelProvider:
        hf_config = hf_pretrained.config

        # GLM 4.5 has separate text_config and vision_config
        text_config = getattr(hf_config, "text_config", hf_config)
        moe_layer_freq = [0] * text_config.first_k_dense_replace + [1] * (
            text_config.num_hidden_layers - text_config.first_k_dense_replace
        )
        provider = GLM45VModelProvider(
            add_qkv_bias=text_config.attention_bias,
            kv_channels=text_config.head_dim,
            hidden_size=text_config.hidden_size,
            rotary_base=text_config.rope_theta,
            rotary_percent=text_config.partial_rotary_factor,
            init_method_std=text_config.initializer_range,
            ffn_hidden_size=text_config.intermediate_size,
            seq_length=text_config.max_position_embeddings,
            moe_ffn_hidden_size=text_config.moe_intermediate_size,
            # norm topk prob
            num_attention_heads=text_config.num_attention_heads,
            # n group, topk group
            num_moe_experts=text_config.n_routed_experts,
            # n shared expert
            moe_shared_expert_intermediate_size=text_config.moe_intermediate_size,
            moe_router_topk_scaling_factor=text_config.routed_scaling_factor,
            moe_router_topk=text_config.num_experts_per_tok,
            moe_layer_freq=moe_layer_freq,
            num_layers=text_config.num_hidden_layers,
            num_query_groups=text_config.num_key_value_heads,
            layernorm_epsilon=text_config.rms_norm_eps,
            mtp_num_layers=0,  # No MTP for VL models
            qk_layernorm=text_config.use_qk_norm,
            vocab_size=text_config.vocab_size,
            fp16=(self.dtype_from_hf(text_config, default=torch.float32) == torch.float16),
            bf16=(self.dtype_from_hf(text_config, default=torch.float32) == torch.bfloat16),
            params_dtype=self.dtype_from_hf(text_config, default=torch.float32),
            vision_config=hf_config.vision_config,
            # VL-specific token IDs
            eos_token_id=getattr(text_config, "eos_token_id", 151329),
            image_start_token_id=getattr(text_config, "image_start_token_id", 151339),
            image_end_token_id=getattr(text_config, "image_end_token_id", 151340),
            video_start_token_id=getattr(text_config, "video_start_token_id", 151341),
            video_end_token_id=getattr(text_config, "video_end_token_id", 151342),
            image_token_id=getattr(text_config, "image_token_id", 151363),
            video_token_id=getattr(text_config, "video_token_id", 151364),
        )
        return provider

    @classmethod
    def get_hf_tokenizer_kwargs(cls) -> dict:
        """Return HuggingFace tokenizer kwargs specific to GLM 4.5V models.

        GLM 4.5V requires use_fast=True to properly load the tokenizer.
        """
        return {"use_fast": True}

    def mapping_registry(self) -> MegatronMappingRegistry:
        # Return MegatronMappingRegistry containing parameter mappings from Megatron to HF format
        # First create simple 1:1 parameter mappings using a dictionary for readability

        # Dictionary maps Megatron parameter names -> HF parameter names
        # Supports wildcard (*) patterns for layer-specific parameters
        mapping_list = []

        param_mappings = {
            # Embed
            "language_model.embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
            # LM Head
            "language_model.decoder.final_layernorm.weight": "model.language_model.norm.weight",
            "language_model.output_layer.weight": "lm_head.weight",
        }

        layer_specific_mappings = {
            # Attention
            "language_model.decoder.layers.*.input_layernorm.weight": "model.language_model.layers.*.input_layernorm.weight",
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": "model.language_model.layers.*.self_attn.o_proj.weight",
            # Reference: https://github.com/NVIDIA/NeMo/blob/50cceb9c90ea1f440d1e14074fa13bd45f60a1c4/nemo/collections/llm/gpt/model/deepseek.py#L637-L650
            #  In GLM, HF weight `model.language_model.layers.*.post_attention_layernorm.weight` is mapped to the following mcore weights depending on the layer type:
            #  (a) `language_model.decoder.layers.*.pre_mlp_layernorm.weight`, if the layer is MoE
            #  (b) `language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight`, if the layer is dense
            "language_model.decoder.layers.*.pre_mlp_layernorm.weight": "model.language_model.layers.*.post_attention_layernorm.weight",
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.language_model.layers.*.input_layernorm.weight",
            "language_model.decoder.layers.*.self_attention.q_layernorm.weight": "model.language_model.layers.*.self_attn.q_norm.weight",
            "language_model.decoder.layers.*.self_attention.k_layernorm.weight": "model.language_model.layers.*.self_attn.k_norm.weight",
            # MLP
            "language_model.decoder.layers.*.mlp.linear_fc2.weight": "model.language_model.layers.*.mlp.down_proj.weight",
            "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.language_model.layers.*.post_attention_layernorm.weight",
            "language_model.decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "model.language_model.layers.*.mlp.shared_experts.down_proj.weight",
            "language_model.decoder.layers.*.mlp.shared_experts.router.weight": "model.language_model.layers.*.mlp.shared_experts.gate.weight",
            "language_model.decoder.layers.*.mlp.experts.linear_fc2.weight*": "model.language_model.layers.*.mlp.experts.*.down_proj.weight",
            "language_model.decoder.layers.*.mlp.router.weight": "model.language_model.layers.*.mlp.gate.weight",
            "language_model.decoder.layers.*.mlp.router.expert_bias": "model.language_model.layers.*.mlp.gate.e_score_correction_bias",
        }

        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        for megatron_param, hf_param in layer_specific_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Add special mappings that require parameter concatenation/transformation
        mapping_list.extend(
            [
                ReplicatedMapping(
                    megatron_param="visual.**",
                    hf_param="model.visual.**",
                ),
                # QKV: Combine separate Q, K, V matrices into single QKV matrix
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.language_model.layers.*.self_attn.q_proj.weight",
                    k="model.language_model.layers.*.self_attn.k_proj.weight",
                    v="model.language_model.layers.*.self_attn.v_proj.weight",
                ),
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.bias",
                    q="model.language_model.layers.*.self_attn.q_proj.bias",
                    k="model.language_model.layers.*.self_attn.k_proj.bias",
                    v="model.language_model.layers.*.self_attn.v_proj.bias",
                ),
                # Gated MLP: Combine gate and up projection matrices into single FC1 matrix
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.language_model.layers.*.mlp.gate_proj.weight",
                    up="model.language_model.layers.*.mlp.up_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="model.language_model.layers.*.mlp.shared_experts.gate_proj.weight",
                    up="model.language_model.layers.*.mlp.shared_experts.up_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="model.language_model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.language_model.layers.*.mlp.experts.*.up_proj.weight",
                ),
            ]
        )
        return MegatronMappingRegistry(*mapping_list)
