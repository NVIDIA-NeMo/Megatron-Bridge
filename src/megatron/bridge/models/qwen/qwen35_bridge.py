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
from transformers import Qwen3_5ForCausalLM, Qwen3_5MoeForCausalLM

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    FusedExpertMapping,
    FusedGatedExpertMapping,
    GatedMLPMapping,
    GDNConv1dMapping,
    GDNLinearMappingSeparate,
    QKVMapping,
    ReplicatedMapping,
    RMSNorm2ZeroCenteredRMSNormMapping,
)
from megatron.bridge.models.conversion.transformers_compat import full_attention_interval_from_hf
from megatron.bridge.models.conversion.utils import moe_experts_stored_packed
from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.models.qwen.qwen_hybrid import (
    QwenHybridModelProvider,
    configure_qwen_hybrid_layers,
    qwen_attention_symbols,
    qwen_logical_layer_count,
    qwen_physical_layer_indices,
)


def _apply_qwen35_common_config(provider: HybridModelProvider, text_config) -> None:
    """Apply configuration shared by dense and MoE Qwen3.5 language models."""
    provider.normalization = "RMSNorm"
    provider.gated_linear_unit = True
    provider.add_qkv_bias = getattr(text_config, "attention_bias", False)
    provider.add_bias_linear = False
    provider.qk_layernorm = True
    provider.hidden_dropout = 0.0

    provider.layernorm_zero_centered_gamma = True
    provider.attention_output_gate = True
    provider.experimental_attention_variant = "gated_delta_net"
    provider.linear_attention_freq = full_attention_interval_from_hf(text_config)
    provider.linear_num_value_heads = getattr(text_config, "linear_num_value_heads", 32)
    provider.rotary_percent = getattr(text_config, "rope_parameters", {}).get("partial_rotary_factor", 0.25)
    provider.linear_conv_kernel_dim = getattr(text_config, "linear_conv_kernel_dim", 4)
    provider.linear_key_head_dim = getattr(text_config, "linear_key_head_dim", 128)
    provider.linear_value_head_dim = getattr(text_config, "linear_value_head_dim", 128)
    provider.linear_num_key_heads = getattr(text_config, "linear_num_key_heads", 16)

    if provider.mtp_num_layers:
        provider.mtp_loss_scaling_factor = 0.1


def _apply_qwen35_moe_config(provider: HybridModelProvider, text_config) -> None:
    """Apply Qwen3.5 MoE configuration in addition to the common settings."""
    _apply_qwen35_common_config(provider, text_config)
    provider.moe_ffn_hidden_size = getattr(text_config, "moe_intermediate_size", 1024)
    provider.num_moe_experts = getattr(text_config, "num_experts", 512)
    provider.moe_router_topk = getattr(text_config, "num_experts_per_tok", 10)
    provider.moe_shared_expert_intermediate_size = getattr(text_config, "shared_expert_intermediate_size", None)
    provider.moe_shared_expert_gate = True
    provider.moe_grouped_gemm = True
    provider.moe_router_load_balancing_type = "global_aux_loss"
    provider.moe_router_pre_softmax = False
    provider.moe_token_dispatcher_type = "alltoall"
    provider.moe_permute_fusion = True


def _base_lm_mappings(hf_prefix: str, megatron_prefix: str) -> list:
    return [
        AutoMapping(f"{megatron_prefix}embedding.word_embeddings.weight", f"{hf_prefix}embed_tokens.weight"),
        AutoMapping(f"{megatron_prefix}output_layer.weight", "lm_head.weight"),
        AutoMapping(f"{megatron_prefix}decoder.final_norm.weight", f"{hf_prefix}norm.weight"),
    ]


def _qwen35_attention_mappings(
    logical_layer_idx: int,
    attention_layer_idx: int,
    attention_symbol: str,
    *,
    hf_prefix: str,
    megatron_prefix: str,
) -> list:
    hf_layer = f"{hf_prefix}layers.{logical_layer_idx}"
    attention_layer = f"{megatron_prefix}decoder.layers.{attention_layer_idx}.self_attention"
    if attention_symbol == Symbols.ATTENTION:
        return [
            AutoMapping(f"{attention_layer}.linear_qkv.layer_norm_weight", f"{hf_layer}.input_layernorm.weight"),
            AutoMapping(f"{attention_layer}.q_layernorm.weight", f"{hf_layer}.self_attn.q_norm.weight"),
            AutoMapping(f"{attention_layer}.k_layernorm.weight", f"{hf_layer}.self_attn.k_norm.weight"),
            AutoMapping(f"{attention_layer}.linear_proj.weight", f"{hf_layer}.self_attn.o_proj.weight"),
            QKVMapping(
                megatron_param=f"{attention_layer}.linear_qkv.weight",
                q=f"{hf_layer}.self_attn.q_proj.weight",
                k=f"{hf_layer}.self_attn.k_proj.weight",
                v=f"{hf_layer}.self_attn.v_proj.weight",
            ),
        ]

    return [
        AutoMapping(f"{attention_layer}.in_proj.layer_norm_weight", f"{hf_layer}.input_layernorm.weight"),
        AutoMapping(f"{attention_layer}.out_proj.weight", f"{hf_layer}.linear_attn.out_proj.weight"),
        AutoMapping(f"{attention_layer}.A_log", f"{hf_layer}.linear_attn.A_log"),
        AutoMapping(f"{attention_layer}.dt_bias", f"{hf_layer}.linear_attn.dt_bias"),
        GDNConv1dMapping(
            megatron_param=f"{attention_layer}.conv1d.weight",
            hf_param=f"{hf_layer}.linear_attn.conv1d.weight",
        ),
        GDNLinearMappingSeparate(
            megatron_param=f"{attention_layer}.in_proj.weight",
            qkv=f"{hf_layer}.linear_attn.in_proj_qkv.weight",
            z=f"{hf_layer}.linear_attn.in_proj_z.weight",
            b=f"{hf_layer}.linear_attn.in_proj_b.weight",
            a=f"{hf_layer}.linear_attn.in_proj_a.weight",
        ),
        RMSNorm2ZeroCenteredRMSNormMapping(
            f"{attention_layer}.out_norm.weight",
            f"{hf_layer}.linear_attn.norm.weight",
        ),
    ]


def _dense_layer_mappings(
    logical_layer_idx: int,
    mlp_layer_idx: int,
    *,
    hf_prefix: str,
    megatron_prefix: str,
) -> list:
    hf_layer = f"{hf_prefix}layers.{logical_layer_idx}"
    mlp_layer = f"{megatron_prefix}decoder.layers.{mlp_layer_idx}.mlp"
    return [
        AutoMapping(f"{mlp_layer}.linear_fc1.layer_norm_weight", f"{hf_layer}.post_attention_layernorm.weight"),
        AutoMapping(f"{mlp_layer}.linear_fc2.weight", f"{hf_layer}.mlp.down_proj.weight"),
        GatedMLPMapping(
            megatron_param=f"{mlp_layer}.linear_fc1.weight",
            gate=f"{hf_layer}.mlp.gate_proj.weight",
            up=f"{hf_layer}.mlp.up_proj.weight",
        ),
    ]


def _moe_layer_mappings(
    logical_layer_idx: int,
    moe_layer_idx: int,
    *,
    hf_prefix: str,
    megatron_prefix: str,
    experts_packed: bool,
) -> list:
    hf_layer = f"{hf_prefix}layers.{logical_layer_idx}"
    moe_layer = f"{megatron_prefix}decoder.layers.{moe_layer_idx}"
    return [
        AutoMapping(f"{moe_layer}.mlp.router.weight", f"{hf_layer}.mlp.gate.weight"),
        AutoMapping(f"{moe_layer}.pre_mlp_layernorm.weight", f"{hf_layer}.post_attention_layernorm.weight"),
        *_moe_routed_expert_mappings(hf_layer, moe_layer, experts_packed),
        GatedMLPMapping(
            megatron_param=f"{moe_layer}.mlp.shared_experts.linear_fc1.weight",
            gate=f"{hf_layer}.mlp.shared_expert.gate_proj.weight",
            up=f"{hf_layer}.mlp.shared_expert.up_proj.weight",
        ),
        AutoMapping(
            f"{moe_layer}.mlp.shared_experts.linear_fc2.weight",
            f"{hf_layer}.mlp.shared_expert.down_proj.weight",
        ),
        ReplicatedMapping(
            f"{moe_layer}.mlp.shared_experts.gate_weight",
            f"{hf_layer}.mlp.shared_expert_gate.weight",
        ),
    ]


def _moe_routed_expert_mappings(
    hf_layer: str,
    moe_layer: str,
    experts_packed: bool,
    *,
    transpose_on_export: bool = False,
) -> list:
    """Map grouped and sequential routed experts to the checkpoint's storage layout."""
    grouped_fc1 = f"{moe_layer}.mlp.experts.linear_fc1.weight*"
    grouped_fc2 = f"{moe_layer}.mlp.experts.linear_fc2.weight*"
    sequential_fc1 = f"{moe_layer}.mlp.experts.local_experts.*.linear_fc1.weight"
    sequential_fc2 = f"{moe_layer}.mlp.experts.local_experts.*.linear_fc2.weight"
    if experts_packed:
        gate_up = f"{hf_layer}.mlp.experts.gate_up_proj"
        down = f"{hf_layer}.mlp.experts.down_proj"
        return [
            FusedGatedExpertMapping(
                megatron_param=grouped_fc1,
                hf_param=gate_up,
                transpose_on_export=transpose_on_export,
            ),
            FusedExpertMapping(
                megatron_param=grouped_fc2,
                hf_param=down,
                transpose_on_export=transpose_on_export,
            ),
            FusedGatedExpertMapping(
                megatron_param=sequential_fc1,
                hf_param=gate_up,
                transpose_on_export=transpose_on_export,
            ),
            FusedExpertMapping(
                megatron_param=sequential_fc2,
                hf_param=down,
                transpose_on_export=transpose_on_export,
            ),
        ]

    gate = f"{hf_layer}.mlp.experts.*.gate_proj.weight"
    up = f"{hf_layer}.mlp.experts.*.up_proj.weight"
    down = f"{hf_layer}.mlp.experts.*.down_proj.weight"
    return [
        GatedMLPMapping(megatron_param=grouped_fc1, gate=gate, up=up),
        AutoMapping(megatron_param=grouped_fc2, hf_param=down),
        GatedMLPMapping(megatron_param=sequential_fc1, gate=gate, up=up),
        AutoMapping(megatron_param=sequential_fc2, hf_param=down),
    ]


def _mtp_common_mappings(megatron_prefix: str) -> list:
    attention_layer = f"{megatron_prefix}mtp.layers.0.mtp_model_layer.layers.0.self_attention"
    return [
        AutoMapping(f"{megatron_prefix}mtp.layers.0.eh_proj.weight", "mtp.fc.weight"),
        AutoMapping(f"{megatron_prefix}mtp.layers.0.enorm.weight", "mtp.pre_fc_norm_embedding.weight"),
        AutoMapping(f"{megatron_prefix}mtp.layers.0.hnorm.weight", "mtp.pre_fc_norm_hidden.weight"),
        AutoMapping(f"{megatron_prefix}mtp.layers.0.final_layernorm.weight", "mtp.norm.weight"),
        AutoMapping(f"{attention_layer}.linear_qkv.layer_norm_weight", "mtp.layers.0.input_layernorm.weight"),
        AutoMapping(f"{attention_layer}.q_layernorm.weight", "mtp.layers.0.self_attn.q_norm.weight"),
        AutoMapping(f"{attention_layer}.k_layernorm.weight", "mtp.layers.0.self_attn.k_norm.weight"),
        AutoMapping(f"{attention_layer}.linear_proj.weight", "mtp.layers.0.self_attn.o_proj.weight"),
        QKVMapping(
            megatron_param=f"{megatron_prefix}mtp.layers.*.mtp_model_layer.layers.0.self_attention.linear_qkv.weight",
            q="mtp.layers.*.self_attn.q_proj.weight",
            k="mtp.layers.*.self_attn.k_proj.weight",
            v="mtp.layers.*.self_attn.v_proj.weight",
        ),
    ]


class _Qwen35HybridBridgeMixin:
    @classmethod
    def megatron_to_hf_config(cls, provider) -> dict:
        hf_config = super().megatron_to_hf_config(provider)
        logical_layer_count = qwen_logical_layer_count(provider.hybrid_layer_pattern)
        if logical_layer_count is not None:
            hf_config["num_hidden_layers"] = logical_layer_count
        return hf_config


@MegatronModelBridge.register_bridge(
    source=Qwen3_5MoeForCausalLM,
    target=HybridModel,
    provider=QwenHybridModelProvider,
    model_type="qwen3_5_moe_text",
)
class Qwen35MoEBridge(_Qwen35HybridBridgeMixin, MegatronModelBridge):
    """Megatron Bridge for the MoE Qwen3.5 and Qwen3.6 language models."""

    @staticmethod
    def _get_moe_lm_mappings(
        num_layers: int = 1,
        linear_attention_freq: int | list[int] = 1,
        hf_prefix: str = "model.",
        megatron_prefix: str = "",
        experts_packed: bool = False,
    ) -> list:
        mappings = _base_lm_mappings(hf_prefix, megatron_prefix)
        attention_symbols = qwen_attention_symbols(num_layers, linear_attention_freq)
        for logical_layer_idx, attention_symbol in enumerate(attention_symbols):
            attention_layer_idx, moe_layer_idx = qwen_physical_layer_indices(logical_layer_idx)
            mappings.extend(
                _qwen35_attention_mappings(
                    logical_layer_idx,
                    attention_layer_idx,
                    attention_symbol,
                    hf_prefix=hf_prefix,
                    megatron_prefix=megatron_prefix,
                )
            )
            mappings.extend(
                _moe_layer_mappings(
                    logical_layer_idx,
                    moe_layer_idx,
                    hf_prefix=hf_prefix,
                    megatron_prefix=megatron_prefix,
                    experts_packed=experts_packed,
                )
            )
        AutoMapping.register_module_type("SharedExpertMLP", "column")
        AutoMapping.register_module_type("GatedDeltaNet", "column")
        return mappings

    @staticmethod
    def _get_moe_mtp_mappings(megatron_prefix: str = "", mtp_experts_packed: bool = False) -> list:
        mappings = _mtp_common_mappings(megatron_prefix)
        moe_layer = f"{megatron_prefix}mtp.layers.0.mtp_model_layer.layers.1"
        mappings.extend(
            [
                AutoMapping(f"{moe_layer}.mlp.router.weight", "mtp.layers.0.mlp.gate.weight"),
                AutoMapping(
                    f"{moe_layer}.pre_mlp_layernorm.weight",
                    "mtp.layers.0.post_attention_layernorm.weight",
                ),
                GatedMLPMapping(
                    megatron_param=f"{moe_layer}.mlp.shared_experts.linear_fc1.weight",
                    gate="mtp.layers.0.mlp.shared_expert.gate_proj.weight",
                    up="mtp.layers.0.mlp.shared_expert.up_proj.weight",
                ),
                AutoMapping(
                    f"{moe_layer}.mlp.shared_experts.linear_fc2.weight",
                    "mtp.layers.0.mlp.shared_expert.down_proj.weight",
                ),
                ReplicatedMapping(
                    f"{moe_layer}.mlp.shared_experts.gate_weight",
                    "mtp.layers.0.mlp.shared_expert_gate.weight",
                ),
            ]
        )
        if mtp_experts_packed:
            mappings.extend(
                [
                    FusedGatedExpertMapping(
                        megatron_param=f"{moe_layer}.mlp.experts.linear_fc1.weight*",
                        hf_param="mtp.layers.0.mlp.experts.gate_up_proj",
                    ),
                    FusedExpertMapping(
                        megatron_param=f"{moe_layer}.mlp.experts.linear_fc2.weight*",
                        hf_param="mtp.layers.0.mlp.experts.down_proj",
                    ),
                ]
            )
        else:
            mappings.extend(
                [
                    GatedMLPMapping(
                        megatron_param=f"{moe_layer}.mlp.experts.linear_fc1.weight*",
                        gate="mtp.layers.0.mlp.experts.*.gate_proj.weight",
                        up="mtp.layers.0.mlp.experts.*.up_proj.weight",
                    ),
                    AutoMapping(
                        f"{moe_layer}.mlp.experts.linear_fc2.weight*",
                        "mtp.layers.0.mlp.experts.*.down_proj.weight",
                    ),
                ]
            )
        return mappings

    def provider_bridge(self, hf_pretrained):
        """Convert a Hugging Face Qwen3.5 MoE config to HybridModelProvider."""
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config
        _apply_qwen35_moe_config(provider, hf_config)
        provider.position_embedding_type = "rope"
        provider.autocast_dtype = torch.bfloat16
        provider.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)
        provider.bos_token_id = getattr(hf_config, "bos_token_id", 248045)
        provider.eos_token_id = getattr(hf_config, "eos_token_id", 248046)
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
        mtp_experts_packed = False
        hf_pretrained = getattr(self, "hf_pretrained", None)
        if hasattr(hf_pretrained, "state") and hasattr(hf_pretrained.state, "source"):
            hf_keys = set(hf_pretrained.state.source.get_all_keys())
            mtp_experts_packed = "mtp.layers.0.mlp.experts.gate_up_proj" in hf_keys
        experts_packed = moe_experts_stored_packed(hf_pretrained, "model.layers.")

        mappings = self._get_moe_lm_mappings(
            self.hf_config.num_hidden_layers,
            full_attention_interval_from_hf(self.hf_config),
            experts_packed=experts_packed,
        )
        mappings.extend(self._get_moe_mtp_mappings(mtp_experts_packed=mtp_experts_packed))
        return MegatronMappingRegistry(*mappings)


@MegatronModelBridge.register_bridge(
    source=Qwen3_5ForCausalLM,
    target=HybridModel,
    provider=QwenHybridModelProvider,
    model_type="qwen3_5_text",
)
class Qwen35Bridge(_Qwen35HybridBridgeMixin, MegatronModelBridge):
    """Megatron Bridge for dense Qwen3.5 language models."""

    @staticmethod
    def _get_dense_lm_mappings(
        num_layers: int,
        linear_attention_freq: int | list[int],
        hf_prefix: str = "model.",
        megatron_prefix: str = "",
    ) -> list:
        mappings = _base_lm_mappings(hf_prefix, megatron_prefix)
        attention_symbols = qwen_attention_symbols(num_layers, linear_attention_freq)
        for logical_layer_idx, attention_symbol in enumerate(attention_symbols):
            attention_layer_idx, mlp_layer_idx = qwen_physical_layer_indices(logical_layer_idx)
            mappings.extend(
                _qwen35_attention_mappings(
                    logical_layer_idx,
                    attention_layer_idx,
                    attention_symbol,
                    hf_prefix=hf_prefix,
                    megatron_prefix=megatron_prefix,
                )
            )
            mappings.extend(
                _dense_layer_mappings(
                    logical_layer_idx,
                    mlp_layer_idx,
                    hf_prefix=hf_prefix,
                    megatron_prefix=megatron_prefix,
                )
            )
        AutoMapping.register_module_type("GatedDeltaNet", "column")
        return mappings

    @staticmethod
    def _get_dense_mtp_mappings(megatron_prefix: str = "") -> list:
        mappings = _mtp_common_mappings(megatron_prefix)
        mlp_layer = f"{megatron_prefix}mtp.layers.0.mtp_model_layer.layers.1.mlp"
        mappings.extend(
            [
                AutoMapping(
                    f"{mlp_layer}.linear_fc1.layer_norm_weight",
                    "mtp.layers.0.post_attention_layernorm.weight",
                ),
                AutoMapping(f"{mlp_layer}.linear_fc2.weight", "mtp.layers.0.mlp.down_proj.weight"),
                GatedMLPMapping(
                    megatron_param=f"{mlp_layer}.linear_fc1.weight",
                    gate="mtp.layers.0.mlp.gate_proj.weight",
                    up="mtp.layers.0.mlp.up_proj.weight",
                ),
            ]
        )
        return mappings

    def provider_bridge(self, hf_pretrained):
        """Convert a Hugging Face dense Qwen3.5 config to HybridModelProvider."""
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config
        _apply_qwen35_common_config(provider, hf_config)
        provider.position_embedding_type = "rope"
        provider.autocast_dtype = torch.bfloat16
        provider.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)
        provider.bos_token_id = getattr(hf_config, "bos_token_id", 248045)
        provider.eos_token_id = getattr(hf_config, "eos_token_id", 248046)
        provider.hetereogenous_dist_checkpoint = True
        configure_qwen_hybrid_layers(
            provider,
            num_logical_layers=hf_config.num_hidden_layers,
            mlp_symbols=Symbols.MLP,
            linear_attention_freq=provider.linear_attention_freq,
            mtp_mlp_symbol=Symbols.MLP,
        )
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        mappings = self._get_dense_lm_mappings(
            self.hf_config.num_hidden_layers,
            full_attention_interval_from_hf(self.hf_config),
        )
        mappings.extend(self._get_dense_mtp_mappings())
        return MegatronMappingRegistry(*mappings)
