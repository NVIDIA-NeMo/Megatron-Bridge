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
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    ConcatenatedQKVMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.sarvam.sarvam_provider import SarvamMoEModelProvider


@MegatronModelBridge.register_bridge(source="SarvamMoEForCausalLM", target=GPTModel)
class SarvamMoEBridge(MegatronModelBridge):
    """
    Megatron Hub Bridge for Sarvam MoE Causal LM.

    This bridge handles the conversion between HuggingFace SarvamMoEForCausalLM
    and Megatron-Core GPTModel formats. Sarvam MoE models use mixture of experts
    architecture with QKV layernorm.
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> SarvamMoEModelProvider:
        hf_config = hf_pretrained.config

        provider = SarvamMoEModelProvider(
            num_layers=hf_config.num_hidden_layers,
            hidden_size=hf_config.hidden_size,
            ffn_hidden_size=hf_config.intermediate_size,
            moe_ffn_hidden_size=hf_config.moe_intermediate_size,  # Maps to moe_intermediate_size in HF
            num_attention_heads=hf_config.num_attention_heads,
            kv_channels=hf_config.head_dim,
            num_query_groups=hf_config.num_key_value_heads,
            num_moe_experts=hf_config.num_experts,
            moe_router_topk=hf_config.num_experts_per_tok,  # Maps to num_experts_per_tok in HF
            moe_shared_expert_intermediate_size=hf_config.num_shared_experts * hf_config.moe_intermediate_size,
            moe_router_enable_expert_bias=hf_config.moe_router_enable_expert_bias,
            moe_layer_freq=[0] * hf_config.first_k_dense_replace + [1] * (hf_config.num_hidden_layers - hf_config.first_k_dense_replace),
            vocab_size=hf_config.vocab_size,
            seq_length=hf_config.max_position_embeddings,
            generation_config=hf_pretrained.generation_config,
            rotary_base=hf_config.rope_theta,
            fp16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16),
            bf16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16),
            params_dtype=self.dtype_from_hf(hf_config, default=torch.float32),
        )

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
       # Return MegatronMappingRegistry containing parameter mappings from Megatron to HF format
        # First create simple 1:1 parameter mappings using a dictionary for readability

        # Dictionary maps Megatron parameter names -> HF parameter names
        param_mappings = {
            # Embed
            "embedding.word_embeddings.weight": "model.word_embeddings.weight",

            # Attention
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.gate.expert_bias",
            "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",

            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.attention.query_layernorm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.attention.key_layernorm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.attention.dense.weight",
            
            # Dense MLP
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.mlp.experts.linear_fc2.weight*": "model.layers.*.mlp.experts.*.down_proj.weight",
            "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "model.layers.*.mlp.shared_experts.down_proj.weight",
            
            "final_layernorm.weight": "final_layernorm.weight",

            # LM Head
            "decoder.final_layernorm.weight": "model.norm.weight",
            "output_layer.weight": "lm_head.weight",
        }

        mapping_list = []
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(hf_param=hf_param, megatron_param=megatron_param))

        mapping_list.extend(
            [
                ConcatenatedQKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    hf_param="model.layers.*.attention.query_key_value.weight",
                ),
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="model.layers.*.mlp.shared_experts.gate_proj.weight",
                    up="model.layers.*.mlp.shared_experts.up_proj.weight",
                ),
                
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
