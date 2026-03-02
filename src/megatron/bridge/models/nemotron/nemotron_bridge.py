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
from transformers import NemotronForCausalLM

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    QKVMapping,
)
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


def squared_relu(x):
    """Squared ReLU activation function."""
    return torch.pow(torch.nn.functional.relu(x), 2)


@MegatronModelBridge.register_bridge(
    source=NemotronForCausalLM,
    target=GPTModel,
    provider=GPTModelProvider,
    model_type="nemotron",
)
class NemotronBridge(MegatronModelBridge):
    """
    Megatron Bridge for Nemotron Causal LM.

    As a user you would not use this bridge directly, but through `AutoBridge`.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("nvidia/Nemotron-4-340B-Instruct")
        >>> provider = bridge.to_megatron_provider()
    """

    CONFIG_MAPPING = MegatronModelBridge.CONFIG_MAPPING + [
        # Nemotron uses norm_eps instead of rms_norm_eps
        ("norm_eps", "layernorm_epsilon"),
    ]

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GPTModelProvider:
        """Convert HuggingFace Nemotron config to GPTModelProvider."""
        # Use base class for common config conversion
        provider = super().provider_bridge(hf_pretrained)

        provider.normalization = "LayerNorm"
        provider.activation_func = squared_relu
        provider.position_embedding_type = "rope"
        provider.add_bias_linear = False
        provider.hidden_dropout = 0.0
        provider.attention_dropout = 0.0
        provider.masked_softmax_fusion = True
        provider.persist_layer_norm = True
        provider.bias_dropout_add_fusion = False
        provider.layernorm_zero_centered_gamma = True
        provider.cross_entropy_loss_fusion = True
        provider.apply_rope_fusion = True
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        # Return MegatronMappingRegistry containing parameter mappings from Megatron to HF format
        # First create simple 1:1 parameter mappings using a dictionary for readability

        # Dictionary maps Megatron parameter names -> HF parameter names
        # Supports wildcard (*) patterns for layer-specific parameters
        param_mappings = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "decoder.final_layernorm.bias": "model.norm.bias",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_bias": "model.layers.*.input_layernorm.bias",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_bias": "model.layers.*.post_attention_layernorm.bias",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.mlp.linear_fc1.weight": "model.layers.*.mlp.up_proj.weight",
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
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
