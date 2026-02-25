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
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import Qwen3NextForCausalLM

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    GDNConv1dMapping,
    GDNLinearMapping,
    QKVMapping,
    ReplicatedMapping,
    RMSNorm2ZeroCenteredRMSNormMapping,
)


@MegatronModelBridge.register_bridge(source=Qwen3NextForCausalLM, target=GPTModel, model_type="qwen3_next")
class Qwen3NextBridge(MegatronModelBridge):
    """
    Megatron Bridge for Qwen3-Next Causal LM.

    This bridge handles the conversion between HuggingFace Qwen3NextForCausalLM
    and Megatron-Core GPTModel formats. Qwen3-Next uses a hybrid architecture
    combining gated delta net linear attention with standard softmax attention,
    mixture of experts with shared experts, and zero-centered RMSNorm.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained):
        """Convert HuggingFace Qwen3-Next config to GPTModelProvider."""
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        # Standard GPT settings (shared with Qwen3 MoE)
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.position_embedding_type = "rope"
        provider.add_bias_linear = False
        provider.add_qkv_bias = False
        provider.hidden_dropout = 0.0
        provider.qk_layernorm = True
        provider.autocast_dtype = torch.bfloat16

        # MoE settings
        provider.moe_grouped_gemm = True
        provider.moe_router_load_balancing_type = "global_aux_loss"
        provider.moe_aux_loss_coeff = 1e-3
        provider.moe_router_pre_softmax = False
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_permute_fusion = True
        provider.moe_shared_expert_gate = True
        provider.moe_router_dtype = "fp32"
        provider.moe_shared_expert_intermediate_size = hf_config.shared_expert_intermediate_size

        # Qwen3-Next: zero-centered RMSNorm and gated attention
        provider.layernorm_zero_centered_gamma = True
        provider.attention_output_gate = True

        # Qwen3-Next: hybrid gated delta net + standard attention
        provider.transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec
        provider.experimental_attention_variant = "gated_delta_net"
        provider.linear_attention_freq = hf_config.full_attention_interval
        provider.linear_conv_kernel_dim = hf_config.linear_conv_kernel_dim
        provider.linear_key_head_dim = hf_config.linear_key_head_dim
        provider.linear_value_head_dim = hf_config.linear_value_head_dim
        provider.linear_num_key_heads = hf_config.linear_num_key_heads
        provider.linear_num_value_heads = hf_config.linear_num_value_heads

        # Heterogeneous checkpointing for mixed attention layers
        provider.hetereogenous_dist_checkpoint = True

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        # Return MegatronMappingRegistry containing parameter mappings from Megatron to HF format
        # First create simple 1:1 parameter mappings using a dictionary for readability

        # Dictionary maps Megatron parameter names -> HF parameter names
        # Supports wildcard (*) patterns for layer-specific parameters
        param_mappings = {
            # Embedding and output
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            # MoE
            "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            # Standard attention
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            # Linear attention
            "decoder.layers.*.self_attention.in_proj.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.out_proj.weight": "model.layers.*.linear_attn.out_proj.weight",
            "decoder.layers.*.self_attention.A_log": "model.layers.*.linear_attn.A_log",
            "decoder.layers.*.self_attention.dt_bias": "model.layers.*.linear_attn.dt_bias",
            # MTP projection and norms
            "mtp.layers.0.eh_proj.weight": "mtp.fc.weight",
            "mtp.layers.0.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
            "mtp.layers.0.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
            "mtp.layers.0.final_layernorm.weight": "mtp.norm.weight",
            # MTP MoE
            "mtp.layers.0.transformer_layer.mlp.router.weight": "mtp.layers.0.mlp.gate.weight",
            "mtp.layers.0.transformer_layer.pre_mlp_layernorm.weight": "mtp.layers.0.post_attention_layernorm.weight",
            # MTP standard attention
            "mtp.layers.0.transformer_layer.self_attention.linear_qkv.layer_norm_weight": "mtp.layers.0.input_layernorm.weight",
            "mtp.layers.0.transformer_layer.self_attention.q_layernorm.weight": "mtp.layers.0.self_attn.q_norm.weight",
            "mtp.layers.0.transformer_layer.self_attention.k_layernorm.weight": "mtp.layers.0.self_attn.k_norm.weight",
            "mtp.layers.0.transformer_layer.self_attention.linear_proj.weight": "mtp.layers.0.self_attn.o_proj.weight",
        }

        mapping_list = []
        # Convert each dictionary entry to AutoMapping(megatron_param, hf_param)
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))
        AutoMapping.register_module_type("SharedExpertMLP", "column")
        AutoMapping.register_module_type("GatedDeltaNet", "column")

        # Add special mappings that require parameter concatenation/transformation
        mapping_list.extend(
            [
                # QKV: Combine separate Q, K, V matrices into single QKV matrix
                # Note: Qwen3 MoE does NOT have bias in QKV projections
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),
                QKVMapping(
                    megatron_param="mtp.layers.*.transformer_layer.self_attention.linear_qkv.weight",
                    q="mtp.layers.*.self_attn.q_proj.weight",
                    k="mtp.layers.*.self_attn.k_proj.weight",
                    v="mtp.layers.*.self_attn.v_proj.weight",
                ),
                # GDNLinear: Combine separate QKVZ_proj and BA_proj into single in_proj for GDN
                # Note: Qwen3-Next does NOT have bias in the input linear projections
                GDNConv1dMapping(
                    megatron_param="decoder.layers.*.self_attention.conv1d.weight",
                    hf_param="model.layers.*.linear_attn.conv1d.weight",
                ),
                GDNLinearMapping(
                    megatron_param="decoder.layers.*.self_attention.in_proj.weight",
                    qkvz="model.layers.*.linear_attn.in_proj_qkvz.weight",
                    ba="model.layers.*.linear_attn.in_proj_ba.weight",
                ),
                # Gated MLP of experts
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                    hf_param="model.layers.*.mlp.experts.*.down_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="mtp.layers.*.transformer_layer.mlp.experts.linear_fc1.weight*",
                    gate="mtp.layers.*.mlp.experts.*.gate_proj.weight",
                    up="mtp.layers.*.mlp.experts.*.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="mtp.layers.*.transformer_layer.mlp.experts.linear_fc2.weight*",
                    hf_param="mtp.layers.*.mlp.experts.*.down_proj.weight",
                ),
                # Gated MLP of shared expert
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="model.layers.*.mlp.shared_expert.gate_proj.weight",
                    up="model.layers.*.mlp.shared_expert.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc2.weight",
                    hf_param="model.layers.*.mlp.shared_expert.down_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="mtp.layers.*.transformer_layer.mlp.shared_experts.linear_fc1.weight",
                    gate="mtp.layers.*.mlp.shared_expert.gate_proj.weight",
                    up="mtp.layers.*.mlp.shared_expert.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="mtp.layers.*.transformer_layer.mlp.shared_experts.linear_fc2.weight",
                    hf_param="mtp.layers.*.mlp.shared_expert.down_proj.weight",
                ),
                # Shared expert gate
                ReplicatedMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.gate_weight",
                    hf_param="model.layers.*.mlp.shared_expert_gate.weight",
                ),
                ReplicatedMapping(
                    megatron_param="mtp.layers.0.transformer_layer.mlp.shared_experts.gate_weight",
                    hf_param="mtp.layers.0.mlp.shared_expert_gate.weight",
                ),
                # Qwen3-Next implements the output norm as a standard RMSNorm while initializing weight to ones,
                # while other norms are regular zero-centered RMSNorms.
                # To correctly load the output norm weight, we need to subtract 1 from it.
                RMSNorm2ZeroCenteredRMSNormMapping(
                    "decoder.layers.*.self_attention.out_norm.weight",
                    "model.layers.*.linear_attn.norm.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
