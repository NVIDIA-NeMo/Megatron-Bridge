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

import torch
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import MixtralForCausalLM

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mixtral.mixtral_provider import MixtralModelProvider


G_LOGGER = logging.getLogger(__name__)


@MegatronModelBridge.register_bridge(source=MixtralForCausalLM, target=GPTModel)
class MixtralBridge(MegatronModelBridge):
    """
    Megatron Bridge for Mixtral MoE Models.

    This bridge handles the conversion between HuggingFace MixtralForCausalLM
    and Megatron-Core GPTModel formats. Mixtral models use a sparse mixture of
    experts (MoE) architecture with 8 experts and top-2 routing.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("mistralai/Mixtral-8x7B-v0.1")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> MixtralModelProvider:
        """Convert HuggingFace config to Megatron provider."""
        hf_config = hf_pretrained.config

        provider = MixtralModelProvider(
            add_qkv_bias=getattr(hf_config, "attention_bias", False),
            hidden_size=hf_config.hidden_size,
            init_method_std=hf_config.initializer_range,
            ffn_hidden_size=hf_config.intermediate_size,
            moe_ffn_hidden_size=hf_config.intermediate_size,
            seq_length=hf_config.max_position_embeddings,
            kv_channels=hf_config.hidden_size // hf_config.num_attention_heads,
            num_attention_heads=hf_config.num_attention_heads,
            num_moe_experts=hf_config.num_local_experts,
            moe_router_topk=hf_config.num_experts_per_tok,
            num_layers=hf_config.num_hidden_layers,
            num_query_groups=hf_config.num_key_value_heads,
            layernorm_epsilon=hf_config.rms_norm_eps,
            rotary_base=hf_config.rope_theta,
            moe_aux_loss_coeff=hf_config.router_aux_loss_coef,
            vocab_size=hf_config.vocab_size,
            share_embeddings_and_output_weights=getattr(hf_config, "tie_word_embeddings", False),
            fp16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16),
            bf16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16),
            params_dtype=self.dtype_from_hf(hf_config, default=torch.float32),
            generation_config=hf_pretrained.generation_config,
        )

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Define parameter mappings between HuggingFace and Megatron formats."""
        mapping_list = []

        # Direct parameter mappings
        param_mappings = {
            # Embeddings and output
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            # Attention layer norms
            "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            # MLP layer norms
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            # MoE router
            "decoder.layers.*.mlp.router.weight": "model.layers.*.block_sparse_moe.gate.weight",
            # MoE expert weights
            "decoder.layers.*.mlp.experts.linear_fc2.weight*": "model.layers.*.block_sparse_moe.experts.*.w2.weight",
        }

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
                # Gated MLP for experts: Combine w1 (gate) and w3 (up) projection matrices
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="model.layers.*.block_sparse_moe.experts.*.w1.weight",
                    up="model.layers.*.block_sparse_moe.experts.*.w3.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
