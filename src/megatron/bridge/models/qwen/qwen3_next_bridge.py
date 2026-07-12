# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from megatron.bridge.models.conversion.transformers_compat import full_attention_interval_from_hf
from megatron.bridge.models.qwen.qwen_hybrid import (
    QwenHybridModelProvider,
    configure_qwen_hybrid_layers,
    qwen_attention_symbols,
    qwen_logical_layer_count,
    qwen_physical_layer_indices,
)


@MegatronModelBridge.register_bridge(
    source=Qwen3NextForCausalLM,
    target=HybridModel,
    provider=QwenHybridModelProvider,
    model_type="qwen3_next",
)
class Qwen3NextBridge(MegatronModelBridge):
    """Megatron Bridge for Qwen3-Next Causal LM."""

    @classmethod
    def megatron_to_hf_config(cls, provider) -> dict:
        """Convert a Hybrid Qwen3-Next provider to a Hugging Face config dictionary."""
        hf_config = super().megatron_to_hf_config(provider)
        logical_layer_count = qwen_logical_layer_count(provider.hybrid_layer_pattern)
        if logical_layer_count is not None:
            hf_config["num_hidden_layers"] = logical_layer_count
        return hf_config

    def provider_bridge(self, hf_pretrained):
        """Convert a Hugging Face Qwen3-Next config to HybridModelProvider."""
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.position_embedding_type = "rope"
        provider.add_bias_linear = False
        provider.add_qkv_bias = False
        provider.hidden_dropout = 0.0
        provider.qk_layernorm = True
        provider.autocast_dtype = torch.bfloat16
        provider.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)

        provider.moe_grouped_gemm = True
        provider.moe_router_load_balancing_type = "global_aux_loss"
        provider.moe_aux_loss_coeff = 1e-3
        provider.moe_router_pre_softmax = False
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_permute_fusion = True
        provider.moe_shared_expert_gate = True
        provider.moe_router_dtype = "fp32"
        provider.moe_shared_expert_intermediate_size = hf_config.shared_expert_intermediate_size

        provider.layernorm_zero_centered_gamma = True
        provider.attention_output_gate = True
        provider.experimental_attention_variant = "gated_delta_net"
        provider.linear_attention_freq = full_attention_interval_from_hf(hf_config)
        provider.linear_conv_kernel_dim = hf_config.linear_conv_kernel_dim
        provider.linear_key_head_dim = hf_config.linear_key_head_dim
        provider.linear_value_head_dim = hf_config.linear_value_head_dim
        provider.linear_num_key_heads = hf_config.linear_num_key_heads
        provider.linear_num_value_heads = hf_config.linear_num_value_heads
        provider.hetereogenous_dist_checkpoint = True

        configure_qwen_hybrid_layers(
            provider,
            num_logical_layers=hf_config.num_hidden_layers,
            mlp_symbols=Symbols.MOE,
            linear_attention_freq=provider.linear_attention_freq,
            mtp_mlp_symbol=Symbols.MOE,
        )

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Return explicit logical-to-physical parameter mappings for Qwen3-Next."""
        mapping_list = [
            AutoMapping("embedding.word_embeddings.weight", "model.embed_tokens.weight"),
            AutoMapping("output_layer.weight", "lm_head.weight"),
            AutoMapping("decoder.final_norm.weight", "model.norm.weight"),
        ]

        AutoMapping.register_module_type("SharedExpertMLP", "column")
        AutoMapping.register_module_type("GatedDeltaNet", "column")

        num_layers = self.hf_config.num_hidden_layers
        attention_symbols = qwen_attention_symbols(
            num_layers,
            full_attention_interval_from_hf(self.hf_config),
        )
        for logical_layer_idx, attention_symbol in enumerate(attention_symbols):
            attention_layer_idx, moe_layer_idx = qwen_physical_layer_indices(logical_layer_idx)
            hf_layer = f"model.layers.{logical_layer_idx}"
            attention_layer = f"decoder.layers.{attention_layer_idx}.self_attention"
            moe_layer = f"decoder.layers.{moe_layer_idx}"

            if attention_symbol == Symbols.ATTENTION:
                mapping_list.extend(
                    [
                        AutoMapping(
                            f"{attention_layer}.linear_qkv.layer_norm_weight",
                            f"{hf_layer}.input_layernorm.weight",
                        ),
                        AutoMapping(
                            f"{attention_layer}.q_layernorm.weight",
                            f"{hf_layer}.self_attn.q_norm.weight",
                        ),
                        AutoMapping(
                            f"{attention_layer}.k_layernorm.weight",
                            f"{hf_layer}.self_attn.k_norm.weight",
                        ),
                        AutoMapping(
                            f"{attention_layer}.linear_proj.weight",
                            f"{hf_layer}.self_attn.o_proj.weight",
                        ),
                        QKVMapping(
                            megatron_param=f"{attention_layer}.linear_qkv.weight",
                            q=f"{hf_layer}.self_attn.q_proj.weight",
                            k=f"{hf_layer}.self_attn.k_proj.weight",
                            v=f"{hf_layer}.self_attn.v_proj.weight",
                        ),
                    ]
                )
            else:
                mapping_list.extend(
                    [
                        AutoMapping(
                            f"{attention_layer}.in_proj.layer_norm_weight",
                            f"{hf_layer}.input_layernorm.weight",
                        ),
                        AutoMapping(f"{attention_layer}.out_proj.weight", f"{hf_layer}.linear_attn.out_proj.weight"),
                        AutoMapping(f"{attention_layer}.A_log", f"{hf_layer}.linear_attn.A_log"),
                        AutoMapping(f"{attention_layer}.dt_bias", f"{hf_layer}.linear_attn.dt_bias"),
                        GDNConv1dMapping(
                            megatron_param=f"{attention_layer}.conv1d.weight",
                            hf_param=f"{hf_layer}.linear_attn.conv1d.weight",
                        ),
                        GDNLinearMapping(
                            megatron_param=f"{attention_layer}.in_proj.weight",
                            qkvz=f"{hf_layer}.linear_attn.in_proj_qkvz.weight",
                            ba=f"{hf_layer}.linear_attn.in_proj_ba.weight",
                        ),
                        RMSNorm2ZeroCenteredRMSNormMapping(
                            f"{attention_layer}.out_norm.weight",
                            f"{hf_layer}.linear_attn.norm.weight",
                        ),
                    ]
                )

            mapping_list.extend(self._moe_layer_mappings(moe_layer, hf_layer))

        mapping_list.extend(self._mtp_mappings())
        return MegatronMappingRegistry(*mapping_list)

    @staticmethod
    def _moe_layer_mappings(megatron_layer: str, hf_layer: str) -> list:
        """Build mappings for one Qwen3-Next MoE physical layer."""
        return [
            AutoMapping(f"{megatron_layer}.mlp.router.weight", f"{hf_layer}.mlp.gate.weight"),
            AutoMapping(
                f"{megatron_layer}.pre_mlp_layernorm.weight",
                f"{hf_layer}.post_attention_layernorm.weight",
            ),
            GatedMLPMapping(
                megatron_param=f"{megatron_layer}.mlp.experts.linear_fc1.weight*",
                gate=f"{hf_layer}.mlp.experts.*.gate_proj.weight",
                up=f"{hf_layer}.mlp.experts.*.up_proj.weight",
            ),
            AutoMapping(
                f"{megatron_layer}.mlp.experts.linear_fc2.weight*",
                f"{hf_layer}.mlp.experts.*.down_proj.weight",
            ),
            GatedMLPMapping(
                megatron_param=f"{megatron_layer}.mlp.experts.local_experts.*.linear_fc1.weight",
                gate=f"{hf_layer}.mlp.experts.*.gate_proj.weight",
                up=f"{hf_layer}.mlp.experts.*.up_proj.weight",
            ),
            AutoMapping(
                f"{megatron_layer}.mlp.experts.local_experts.*.linear_fc2.weight",
                f"{hf_layer}.mlp.experts.*.down_proj.weight",
            ),
            GatedMLPMapping(
                megatron_param=f"{megatron_layer}.mlp.shared_experts.linear_fc1.weight",
                gate=f"{hf_layer}.mlp.shared_expert.gate_proj.weight",
                up=f"{hf_layer}.mlp.shared_expert.up_proj.weight",
            ),
            AutoMapping(
                f"{megatron_layer}.mlp.shared_experts.linear_fc2.weight",
                f"{hf_layer}.mlp.shared_expert.down_proj.weight",
            ),
            ReplicatedMapping(
                f"{megatron_layer}.mlp.shared_experts.gate_weight",
                f"{hf_layer}.mlp.shared_expert_gate.weight",
            ),
        ]

    @staticmethod
    def _mtp_mappings() -> list:
        """Build mappings for the optional two-layer Hybrid MTP block."""
        attention_layer = "mtp.layers.0.mtp_model_layer.layers.0.self_attention"
        moe_layer = "mtp.layers.0.mtp_model_layer.layers.1"
        mappings = [
            AutoMapping("mtp.layers.0.eh_proj.weight", "mtp.fc.weight"),
            AutoMapping("mtp.layers.0.enorm.weight", "mtp.pre_fc_norm_embedding.weight"),
            AutoMapping("mtp.layers.0.hnorm.weight", "mtp.pre_fc_norm_hidden.weight"),
            AutoMapping("mtp.layers.0.final_layernorm.weight", "mtp.norm.weight"),
            AutoMapping(f"{attention_layer}.linear_qkv.layer_norm_weight", "mtp.layers.0.input_layernorm.weight"),
            AutoMapping(f"{attention_layer}.q_layernorm.weight", "mtp.layers.0.self_attn.q_norm.weight"),
            AutoMapping(f"{attention_layer}.k_layernorm.weight", "mtp.layers.0.self_attn.k_norm.weight"),
            AutoMapping(f"{attention_layer}.linear_proj.weight", "mtp.layers.0.self_attn.o_proj.weight"),
            QKVMapping(
                megatron_param="mtp.layers.*.mtp_model_layer.layers.0.self_attention.linear_qkv.weight",
                q="mtp.layers.*.self_attn.q_proj.weight",
                k="mtp.layers.*.self_attn.k_proj.weight",
                v="mtp.layers.*.self_attn.v_proj.weight",
            ),
        ]
        mappings.extend(Qwen3NextBridge._moe_layer_mappings(moe_layer, "mtp.layers.0"))
        return mappings
