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
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.models.hybrid.hybrid_model import HybridModel
from transformers import Qwen3MoeForCausalLM

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.qwen.qwen_hybrid import (
    QwenHybridModelProvider,
    configure_qwen_hybrid_layers,
    qwen_logical_layer_count,
    qwen_physical_layer_indices,
)


@MegatronModelBridge.register_bridge(
    source=Qwen3MoeForCausalLM,
    target=HybridModel,
    provider=QwenHybridModelProvider,
    model_type="qwen3_moe",
)
class Qwen3MoEBridge(MegatronModelBridge):
    """
    Megatron Bridge for Qwen3 MoE Causal LM.

    This bridge handles the conversion between HuggingFace Qwen3MoeForCausalLM
    and Megatron-Core HybridModel formats. Qwen3 MoE models use mixture of experts
    architecture with QK layernorm.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen3-235B-A22B")
        >>> provider = bridge.to_megatron_provider()
    """

    @classmethod
    def megatron_to_hf_config(cls, provider) -> dict:
        """Convert Megatron provider config to HuggingFace Qwen3MoeConfig dict."""
        hf_config = super().megatron_to_hf_config(provider)
        logical_layer_count = qwen_logical_layer_count(provider.hybrid_layer_pattern)
        if logical_layer_count is not None:
            hf_config["num_hidden_layers"] = logical_layer_count
        hf_config["decoder_sparse_step"] = 1  # All layers are MoE in Qwen3 MoE
        hf_config["norm_topk_prob"] = not provider.moe_router_pre_softmax
        return hf_config

    def provider_bridge(self, hf_pretrained):
        """Convert a Hugging Face Qwen3 MoE config to HybridModelProvider."""
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.add_qkv_bias = False  # Qwen3 MoE does NOT have QKV bias
        provider.hidden_dropout = 0.0
        provider.qk_layernorm = True  # Qwen3 MoE uses QK layernorm
        provider.autocast_dtype = torch.bfloat16

        provider.moe_grouped_gemm = True
        provider.moe_router_load_balancing_type = "aux_loss"
        provider.moe_aux_loss_coeff = 1e-3
        provider.moe_router_pre_softmax = not hf_pretrained.config.norm_topk_prob
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_permute_fusion = True
        provider.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", True)

        configure_qwen_hybrid_layers(
            provider,
            num_logical_layers=hf_config.num_hidden_layers,
            mlp_symbols=Symbols.MOE,
            mtp_mlp_symbol=Symbols.MOE,
        )

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        # Return MegatronMappingRegistry containing parameter mappings from Megatron to HF format
        # First create simple 1:1 parameter mappings using a dictionary for readability

        param_mappings = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_norm.weight": "model.norm.weight",
        }

        mapping_list = [AutoMapping(megatron_param=k, hf_param=v) for k, v in param_mappings.items()]

        for logical_layer_idx in range(self.hf_config.num_hidden_layers):
            attention_layer_idx, moe_layer_idx = qwen_physical_layer_indices(logical_layer_idx)
            hf_layer = f"model.layers.{logical_layer_idx}"
            attention_layer = f"decoder.layers.{attention_layer_idx}.self_attention"
            moe_layer = f"decoder.layers.{moe_layer_idx}"
            mapping_list.extend(
                [
                    AutoMapping(
                        megatron_param=f"{attention_layer}.linear_qkv.layer_norm_weight",
                        hf_param=f"{hf_layer}.input_layernorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"{attention_layer}.q_layernorm.weight",
                        hf_param=f"{hf_layer}.self_attn.q_norm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"{attention_layer}.k_layernorm.weight",
                        hf_param=f"{hf_layer}.self_attn.k_norm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"{attention_layer}.linear_proj.weight",
                        hf_param=f"{hf_layer}.self_attn.o_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"{moe_layer}.pre_mlp_layernorm.weight",
                        hf_param=f"{hf_layer}.post_attention_layernorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"{moe_layer}.mlp.router.weight",
                        hf_param=f"{hf_layer}.mlp.gate.weight",
                    ),
                    QKVMapping(
                        megatron_param=f"{attention_layer}.linear_qkv.weight",
                        q=f"{hf_layer}.self_attn.q_proj.weight",
                        k=f"{hf_layer}.self_attn.k_proj.weight",
                        v=f"{hf_layer}.self_attn.v_proj.weight",
                    ),
                    GatedMLPMapping(
                        megatron_param=f"{moe_layer}.mlp.experts.linear_fc1.weight*",
                        gate=f"{hf_layer}.mlp.experts.*.gate_proj.weight",
                        up=f"{hf_layer}.mlp.experts.*.up_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"{moe_layer}.mlp.experts.linear_fc2.weight*",
                        hf_param=f"{hf_layer}.mlp.experts.*.down_proj.weight",
                    ),
                    GatedMLPMapping(
                        megatron_param=f"{moe_layer}.mlp.experts.local_experts.*.linear_fc1.weight",
                        gate=f"{hf_layer}.mlp.experts.*.gate_proj.weight",
                        up=f"{hf_layer}.mlp.experts.*.up_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"{moe_layer}.mlp.experts.local_experts.*.linear_fc2.weight",
                        hf_param=f"{hf_layer}.mlp.experts.*.down_proj.weight",
                    ),
                ]
            )

        return MegatronMappingRegistry(*mapping_list)
