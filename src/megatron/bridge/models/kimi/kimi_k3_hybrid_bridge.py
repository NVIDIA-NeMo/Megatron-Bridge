# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Native Hybrid conversion candidate for Kimi K3."""

from copy import copy
from typing import Any

from transformers import PretrainedConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.kimi.kimi_k3_gpt_bridge import KimiK3GPTBridge
from megatron.bridge.models.kimi.kimi_k3_hybrid_config import (
    KimiK3HybridModelConfig,
    native_kimi_k3_kwargs,
    validate_kimi_k3_hybrid_geometry,
)
from megatron.bridge.models.kimi.kimi_k3_hybrid_provider import KimiK3HybridModelProvider


class KimiK3HybridBridge(KimiK3GPTBridge):
    """Native model and config conversion for the Kimi K3 language backbone."""

    MODEL_CONFIG_CLASS = KimiK3HybridModelConfig
    PROVIDER_CLASS = KimiK3HybridModelProvider

    def hf_config_to_model_config_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Derive native entry geometry from the existing HF architecture mapping."""
        return native_kimi_k3_kwargs(super().hf_config_to_model_config_kwargs(hf_config))

    @classmethod
    def megatron_to_hf_config(cls, provider) -> dict:
        """Recover HF block counts and architecture from the native entry pattern."""
        validate_kimi_k3_hybrid_geometry(provider)
        pattern = provider.hybrid_layer_pattern.replace("|", "")
        if len(pattern) % 2 or any(
            pattern[i] not in "K+" or pattern[i + 1] not in "-E" for i in range(0, len(pattern), 2)
        ):
            raise ValueError("Kimi K3 export requires alternating KDA/MLA and dense/expert entries.")
        blocks = len(pattern) // 2
        mask = [int(entry == "E") for entry in pattern[1::2]]
        first_moe = next((i for i, flag in enumerate(mask) if flag), blocks)
        frequency = next(
            (
                stride
                for stride in range(1, blocks + 1)
                if mask == [int(i >= first_moe and i % stride == 0) for i in range(blocks)]
            ),
            None,
        )
        if frequency is None:
            raise ValueError("This expert-entry pattern cannot be represented by Kimi K3's HF MoE frequency.")
        shared = provider.moe_shared_expert_intermediate_size or 0
        if shared % provider.moe_ffn_hidden_size:
            raise ValueError("Kimi K3 shared-expert width must be divisible by its expert intermediate size.")
        text = super().megatron_to_hf_config(provider)
        text.pop("architectures", None)
        text.pop("model_type", None)
        text.update(
            num_hidden_layers=blocks,
            first_k_dense_replace=first_moe,
            moe_layer_freq=frequency,
            linear_attn_config={
                "kda_layers": [i + 1 for i, entry in enumerate(pattern[::2]) if entry == "K"],
                "num_heads": provider.linear_num_key_heads,
                "head_dim": provider.linear_key_head_dim,
                "short_conv_kernel_size": provider.linear_conv_kernel_dim,
                "gate_lower_bound": provider.kimi_kda_gate_lower_bound,
            },
            attn_res_block_size=provider.kimi_attn_res_block_size,
            routed_expert_hidden_size=provider.moe_latent_size,
            latent_moe_use_norm=True,
            num_shared_experts=shared // provider.moe_ffn_hidden_size,
            num_experts_per_token=provider.moe_router_topk,
            moe_router_activation_func=provider.moe_router_score_function,
            use_grouped_topk=provider.moe_router_num_groups is not None,
            num_expert_group=provider.moe_router_num_groups,
            topk_group=provider.moe_router_group_topk,
        )
        return {
            "architectures": ["KimiK3ForConditionalGeneration"],
            "model_type": "kimi_k3",
            "text_config": text,
        }

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Reuse the GPT weight transforms with explicit split-entry names."""
        source = super().mapping_registry()
        text = getattr(self.hf_config, "text_config", self.hf_config)
        mappings = []
        for mapping in source.mappings:
            if mapping.megatron_param.startswith("decoder.layers.*."):
                for index in range(text.num_hidden_layers):
                    translated = copy(mapping)
                    suffix = mapping.megatron_param.removeprefix("decoder.layers.*.")
                    branch = int(suffix.startswith(("mlp", "pre_mlp")))
                    translated.megatron_param = f"decoder.layers.{2 * index + branch}.{suffix}"
                    if isinstance(mapping.hf_param, dict):
                        translated.hf_param = {
                            k: v.replace("layers.*.", f"layers.{index}.", 1) for k, v in mapping.hf_param.items()
                        }
                    else:
                        translated.hf_param = mapping.hf_param.replace("layers.*.", f"layers.{index}.", 1)
                    mappings.append(translated)
            else:
                translated = copy(mapping)
                translated.megatron_param = translated.megatron_param.replace(
                    "decoder.final_layernorm.", "decoder.final_norm."
                )
                for suffix in ("output_attn_res_norm.weight", "output_attn_res_proj.weight"):
                    if translated.megatron_param.endswith(suffix):
                        translated.megatron_param = f"decoder.{suffix}"
                mappings.append(translated)
        return MegatronMappingRegistry(*mappings)
