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
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.qwen.qwen_provider import Qwen3NextModelProvider


@MegatronModelBridge.register_bridge(source=Qwen3NextForCausalLM, target=GPTModel)
class Qwen3NextBridge(MegatronModelBridge):
    """
    Megatron Hub Bridge for Qwen3 MoE Causal LM.

    This bridge handles the conversion between HuggingFace Qwen3MoeForCausalLM
    and Megatron-Core GPTModel formats. Qwen3 MoE models use mixture of experts
    architecture with QK layernorm.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen3-235B-A22B")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Qwen3NextModelProvider:
        hf_config = hf_pretrained.config

        provider = Qwen3NextModelProvider(
            num_layers=hf_config.num_hidden_layers,
            hidden_size=hf_config.hidden_size,
            ffn_hidden_size=hf_config.intermediate_size,
            moe_ffn_hidden_size=hf_config.moe_intermediate_size,  # Maps to moe_intermediate_size in HF
            num_attention_heads=hf_config.num_attention_heads,
            num_query_groups=hf_config.num_key_value_heads,
            num_moe_experts=hf_config.num_experts,
            moe_router_topk=hf_config.num_experts_per_tok,  # Maps to num_experts_per_tok in HF
            init_method_std=hf_config.initializer_range,
            layernorm_epsilon=hf_config.rms_norm_eps,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(hf_config.vocab_size),
            rotary_base=hf_config.rope_theta,
            share_embeddings_and_output_weights=getattr(hf_config, "tie_word_embeddings", False),
            vocab_size=hf_config.vocab_size,
            seq_length=hf_config.max_position_embeddings,
            fp16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16),
            bf16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16),
            params_dtype=self.dtype_from_hf(hf_config, default=torch.float32),
            qk_layernorm=True,  # Qwen3 MoE uses QK layernorm
            moe_grouped_gemm=True,
            kv_channels=hf_config.head_dim,
            # New for Qwen3-Next
            layernorm_zero_centered_gamma=True,
            attention_output_gate=True,
            experimental_attention_variant="gated_delta_net",
            linear_attention_freq=hf_config.full_attention_interval,
            rotary_percent=hf_config.partial_rotary_factor,
            moe_shared_expert_intermediate_size=hf_config.shared_expert_intermediate_size,
            moe_shared_expert_gate=True,
            linear_conv_kernel_dim=hf_config.linear_conv_kernel_dim,
            linear_key_head_dim=hf_config.linear_key_head_dim,
            linear_value_head_dim=hf_config.linear_value_head_dim,
            linear_num_key_heads=hf_config.linear_num_key_heads,
            linear_num_value_heads=hf_config.linear_num_value_heads,
            mtp_num_layers=None,  # Set to 1 if need MTP
        )

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
