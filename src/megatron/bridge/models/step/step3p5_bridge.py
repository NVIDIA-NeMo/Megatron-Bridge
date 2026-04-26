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
from transformers import AutoConfig, AutoModelForCausalLM

from megatron.bridge.models.step.configuration_step3p5 import Step3p5Config
from megatron.bridge.models.step.modeling_step3p5 import MockStep3p5ForCausalLM
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


# Register the Step3.5 config and model classes with transformers Auto classes.
# This allows AutoConfig.from_pretrained and AutoModelForCausalLM to resolve "step3p5"
# without requiring hub access (works in offline CI environments).
AutoConfig.register("step3p5", Step3p5Config, exist_ok=True)
AutoModelForCausalLM.register(Step3p5Config, MockStep3p5ForCausalLM, exist_ok=True)


@MegatronModelBridge.register_bridge(source="Step3p5ForCausalLM", target=GPTModel, model_type="step3p5")
class Step3p5Bridge(MegatronModelBridge):
    """
    Megatron Bridge for Step3.5 Causal LM.

    This bridge handles the conversion between HuggingFace Step3p5ForCausalLM
    and Megatron-Core GPTModel formats. Step3.5 models use mixture of experts
    architecture with QK layernorm.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("stepfun-ai/Step-3.5-Flash")
        >>> provider = bridge.to_megatron_provider()
    """

    @classmethod
    def megatron_to_hf_config(cls, provider) -> dict:
        """Convert Megatron provider config to HuggingFace Step3p5Config dict."""
        hf_config = super().megatron_to_hf_config(provider)
        # hf_config["decoder_sparse_step"] = 1  # All layers are MoE in Step3.5
        # hf_config["norm_topk_prob"] = False  # Step3.5 does not normalize top-k probs
        return hf_config

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GPTModelProvider:
        """Convert HuggingFace Step3.5 config to GPTModelProvider."""
        provider = super().provider_bridge(hf_pretrained)

        # Step3p5Config uses non-standard field names that don't match CONFIG_MAPPING.
        # num_key_value_heads -> num_attention_groups, num_experts -> moe_num_experts, topk -> moe_top_k
        hf_config = hf_pretrained.config
        provider.num_query_groups = hf_config.num_attention_groups
        provider.num_moe_experts = hf_config.moe_num_experts
        provider.moe_router_topk = hf_config.moe_top_k

        if isinstance(hf_config.rope_theta, list):
            # Per-layer RoPE theta values.
            provider.rotary_base_per_layer = hf_config.rope_theta
            # override the global rotary_base
            provider.rotary_base = max(hf_config.rope_theta)
        # sliding window attention
        if hf_config.sliding_window is not None:
            provider.window_size = (hf_config.sliding_window, 0)
        if hf_config.layer_types is not None:
            provider.window_attn_skip_freq = [t == "full_attention" for t in hf_config.layer_types]
        # Per-head scalar gate in attention module
        provider.use_head_wise_attn_gate = hf_config.use_head_wise_attn_gate
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.add_qkv_bias = False  # Step3.5 does NOT have QKV bias
        provider.hidden_dropout = 0.0
        provider.qk_layernorm = True  # Step3.5 uses QK layernorm
        provider.autocast_dtype = torch.bfloat16

        provider.moe_grouped_gemm = True
        provider.moe_router_load_balancing_type = "aux_loss"
        provider.moe_aux_loss_coeff = 1e-3
        provider.moe_router_pre_softmax = False
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_permute_fusion = True

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
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
        }

        mapping_list = []
        # Convert each dictionary entry to AutoMapping(megatron_param, hf_param)
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

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
                # Expert mappings for TEGroupedMLP
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                    hf_param="model.layers.*.mlp.experts.*.down_proj.weight",
                ),
                # Expert mappings for SequentialMLP (used by quantization)
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
                    gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",
                    hf_param="model.layers.*.mlp.experts.*.down_proj.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
