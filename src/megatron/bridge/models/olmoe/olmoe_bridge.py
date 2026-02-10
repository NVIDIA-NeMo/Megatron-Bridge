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
from transformers import OlmoeForCausalLM

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.olmoe.olmoe_provider import olmoe_layer_spec


logger = logging.getLogger(__name__)


@MegatronModelBridge.register_bridge(source=OlmoeForCausalLM, target=GPTModel, model_type="olmoe")
class OlMoEBridge(MegatronModelBridge):
    """
    Megatron Bridge for OlMoE Models.

    This bridge handles the conversion between HuggingFace OlMoEForCausalLM
    and Megatron-Core GPTModel formats. OlMoE models use mixture of experts
    architecture with QK layernorm.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("allenai/OLMoE-1B-7B-0125")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GPTModelProvider:
        """Convert HuggingFace OlMoE config to Megatron GPTModelProvider.

        Uses base class implementation for common conversion, then sets
        OlMoE-specific config. OlMoE uses QK layernorm and mixture of experts.

        Args:
            hf_pretrained: HuggingFace PreTrainedCausalLM containing the OlMoE config

        Returns:
            GPTModelProvider configured for OlMoE architecture
        """
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        # OlMoE uses custom layer spec with OLMoESelfAttention for QK layernorm
        provider.transformer_layer_spec = olmoe_layer_spec

        # Set kv_channels (head_dim) - OLMoE HF config doesn't have head_dim, so calculate it
        provider.kv_channels = getattr(hf_config, "head_dim", None) or (
            hf_config.hidden_size // hf_config.num_attention_heads
        )

        # OlMoE-specific architecture settings
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.position_embedding_type = "rope"
        provider.add_bias_linear = False
        provider.hidden_dropout = 0.0
        provider.share_embeddings_and_output_weights = False
        provider.qk_layernorm = True
        provider.persist_layer_norm = True
        provider.autocast_dtype = torch.bfloat16

        # MoE-specific settings
        provider.moe_ffn_hidden_size = hf_config.intermediate_size
        provider.moe_aux_loss_coeff = hf_config.router_aux_loss_coef
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_router_load_balancing_type = "seq_aux_loss"
        provider.moe_router_pre_softmax = True
        provider.moe_grouped_gemm = True
        provider.moe_router_score_function = "softmax"
        provider.moe_permute_fusion = True
        provider.moe_router_dtype = "fp32"

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        mapping_list = []

        param_mappings = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            # Attention
            "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",
            # MLP
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.mlp.experts.linear_fc2.weight*": "model.layers.*.mlp.experts.*.down_proj.weight",
            "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
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
                # Gated MLP: Combine gate and up projection matrices into single FC1 matrix
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
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
