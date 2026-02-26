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

from megatron.core.activations import squared_relu
from megatron.core.models.mamba import MambaModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    MambaConv1dMapping,
    MambaInProjMapping,
    QKVMapping,
    RowParallelMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mamba.mamba_provider import MambaModelProvider


logger = logging.getLogger(__name__)


@MegatronModelBridge.register_bridge(
    source="NemotronHForCausalLM",
    target=MambaModel,
    provider=MambaModelProvider,
    model_type="nemotron_h",
)
class NemotronHBridge(MegatronModelBridge):
    """
    Megatron Bridge for Nemotron-H Causal LM.

    This bridge handles the conversion between HuggingFace NemotronHForCausalLM
    and Megatron-Core MambaModel formats, including weight mappings and
    configuration translation.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("nvidia/Nemotron-H-8B-Base-8K", trust_remote_code=True)
        >>> provider = bridge.to_megatron_provider()
    """

    # Extend CONFIG_MAPPING with Nemotron-H/Mamba-specific fields
    CONFIG_MAPPING = MegatronModelBridge.CONFIG_MAPPING + [
        # Mamba-specific fields
        ("mamba_head_dim", "mamba_head_dim"),
        ("mamba_num_heads", "mamba_num_heads"),
        ("n_groups", "mamba_num_groups"),
        ("ssm_state_size", "mamba_state_dim"),
        ("hybrid_override_pattern", "hybrid_override_pattern"),
        ("residual_in_fp32", "fp32_residual_connection"),
        ("use_bias", "add_bias_linear"),
        ("layer_norm_epsilon", "layernorm_epsilon"),
        # MoE-specific fields (already in base but with different HF names)
        ("moe_shared_expert_intermediate_size", "moe_shared_expert_intermediate_size"),
    ]

    # Additional files to copy during HF export (reasoning parser utilities)
    ADDITIONAL_FILE_PATTERNS = ["*reasoning_parser.py"]

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> MambaModelProvider:
        """Convert HuggingFace Nemotron-H config to MambaModelProvider."""
        # Use base class for common config conversion
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        # Nemotron-H specific defaults
        provider.activation_func = squared_relu
        provider.masked_softmax_fusion = True
        provider.apply_query_key_layer_scaling = False
        provider.persist_layer_norm = True
        provider.attention_softmax_in_fp32 = False
        provider.first_last_layers_bf16 = True
        provider.is_hybrid_model = True

        # MoE-specific defaults (only if MoE is enabled)
        if hasattr(hf_config, "n_routed_experts") and hf_config.n_routed_experts > 0:
            provider.moe_aux_loss_coeff = 0.0001
            provider.moe_router_score_function = "sigmoid"
            provider.moe_router_enable_expert_bias = True
            provider.moe_router_load_balancing_type = "seq_aux_loss"
            provider.moe_router_dtype = "fp32"
            provider.moe_grouped_gemm = True
            provider.moe_token_dispatcher_type = "alltoall"
            provider.moe_permute_fusion = True
            provider.moe_shared_expert_overlap = True

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        # Return MegatronMappingRegistry containing parameter mappings from Megatron to HF format
        # First create simple 1:1 parameter mappings using a dictionary for readability

        # Dictionary maps Megatron parameter names -> HF parameter names
        # Supports wildcard (*) patterns for layer-specific parameters
        param_mappings = {
            "decoder.layers.*.mlp.linear_fc1.weight": "backbone.layers.*.mixer.up_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "backbone.layers.*.mixer.down_proj.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "backbone.layers.*.mixer.o_proj.weight",
            "decoder.final_norm.weight": "backbone.norm_f.weight",
            # Fused TE layer norm weights (when using TELayerNormColumnParallelLinear)
            # if the megatron key does not exist for a given layer it will be ignored,
            # so only one of these will be used per layer
            "decoder.layers.*.mixer.in_proj.layer_norm_weight": "backbone.layers.*.norm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "backbone.layers.*.norm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "backbone.layers.*.norm.weight",
            # Separate Norm layer weights (when using Norm for quantization)
            # These are used when quantization spec uses Norm instead of TENorm
            "decoder.layers.*.norm.weight": "backbone.layers.*.norm.weight",
            "decoder.layers.*.pre_mlp_layernorm.weight": "backbone.layers.*.norm.weight",
            "decoder.layers.*.input_layernorm.weight": "backbone.layers.*.norm.weight",
            # TODO (@maanug): need to find a way to prune the vocab padding from the vocab dimension for these params
            "embedding.word_embeddings.weight": "backbone.embeddings.weight",
            "output_layer.weight": "lm_head.weight",
            # MoE layers
            "decoder.layers.*.mlp.router.weight": "backbone.layers.*.mixer.gate.weight",
            "decoder.layers.*.mlp.router.expert_bias": "backbone.layers.*.mixer.gate.e_score_correction_bias",
            "decoder.layers.*.mlp.experts.linear_fc1.weight*": "backbone.layers.*.mixer.experts.*.up_proj.weight",
            "decoder.layers.*.mlp.experts.linear_fc2.weight*": "backbone.layers.*.mixer.experts.*.down_proj.weight",
            "decoder.layers.*.mlp.shared_experts.linear_fc1.weight": "backbone.layers.*.mixer.shared_experts.up_proj.weight",
            "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "backbone.layers.*.mixer.shared_experts.down_proj.weight",
        }

        mapping_list = []
        # Convert each dictionary entry to AutoMapping(megatron_param, hf_param)
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Handling Mamba Mixer submodules separately for more clarity
        # Special Handling for InProj and Conv1d due to specific TP logic
        for mixer_sub_module in ["A_log", "D", "dt_bias", "norm.weight"]:
            mapping_list.extend(
                [
                    ColumnParallelMapping(
                        megatron_param=rf"decoder.layers.*.mixer.{mixer_sub_module}",
                        hf_param=rf"backbone.layers.*.mixer.{mixer_sub_module}",
                    ),
                ]
            )
        mapping_list.extend(
            [
                RowParallelMapping(
                    megatron_param="decoder.layers.*.mixer.out_proj.weight",
                    hf_param="backbone.layers.*.mixer.out_proj.weight",
                ),
            ]
        )
        mapping_list.extend(
            [
                MambaInProjMapping(
                    megatron_param="decoder.layers.*.mixer.in_proj.weight",
                    hf_param="backbone.layers.*.mixer.in_proj.weight",
                ),
            ]
        )
        for conv1d_sub_module in ["weight", "bias"]:
            mapping_list.extend(
                [
                    MambaConv1dMapping(
                        megatron_param=rf"decoder.layers.*.mixer.conv1d.{conv1d_sub_module}",
                        hf_param=rf"backbone.layers.*.mixer.conv1d.{conv1d_sub_module}",
                    ),
                ]
            )
        # Add special mappings that require parameter concatenation/transformation, pruning, etc.
        mapping_list.extend(
            [
                # QKV: Combine separate Q, K, V matrices into single QKV matrix
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="backbone.layers.*.mixer.q_proj.weight",
                    k="backbone.layers.*.mixer.k_proj.weight",
                    v="backbone.layers.*.mixer.v_proj.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
