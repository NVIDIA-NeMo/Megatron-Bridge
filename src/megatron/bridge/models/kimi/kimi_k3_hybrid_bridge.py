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

from typing import Any

from transformers import PretrainedConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    ReplicatedMapping,
    RowParallelMapping,
)
from megatron.bridge.models.kimi.kimi_k3_gpt_bridge import KimiK3GPTBridge
from megatron.bridge.models.kimi.kimi_k3_hybrid_config import KimiK3HybridModelConfig, native_kimi_k3_kwargs
from megatron.bridge.models.kimi.kimi_k3_hybrid_mapping import KimiK3AttnResQueryMapping, KimiK3QKVMapping
from megatron.bridge.models.kimi.kimi_k3_hybrid_provider import KimiK3HybridModelProvider
from megatron.bridge.utils.activation_map import callable_to_str


class KimiK3HybridBridge(KimiK3GPTBridge):
    """Native model and config conversion for the Kimi K3 language backbone."""

    MODEL_CONFIG_CLASS = KimiK3HybridModelConfig
    PROVIDER_CLASS = KimiK3HybridModelProvider

    @classmethod
    def megatron_to_hf_activation(cls, activation_func) -> str:
        """Preserve K3's HF activation name when exporting native SiTU-GLU."""
        if callable_to_str(activation_func) == "situlu":
            return "situ"
        return super().megatron_to_hf_activation(activation_func)

    def hf_config_to_model_config_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Derive native entry geometry from the existing HF architecture mapping."""
        return native_kimi_k3_kwargs(super().hf_config_to_model_config_kwargs(hf_config))

    @classmethod
    def megatron_to_hf_config(cls, provider) -> dict:
        """Recover HF block counts and architecture from the native entry pattern."""
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
        if provider.attn_res_block_layers % 2:
            raise ValueError("Kimi K3 HF AttnRes blocks must contain complete attention/FFN pairs.")
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
                "gate_lower_bound": provider.kda_lower_bound,
            },
            attn_res_block_size=provider.attn_res_block_layers // 2,
            routed_expert_hidden_size=provider.moe_latent_size,
            latent_moe_use_norm=provider.moe_use_norm_before_up_proj,
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
        """Map HF blocks to their separate attention and feed-forward entries."""
        if self.hf_config is None:
            raise ValueError("Kimi K3 mapping construction requires the Hugging Face config")
        text = getattr(self.hf_config, "text_config", self.hf_config)
        prefix = "language_model.model"
        mappings = [
            AutoMapping("embedding.word_embeddings.weight", f"{prefix}.embed_tokens.weight"),
            AutoMapping("output_layer.weight", "language_model.lm_head.weight"),
            ReplicatedMapping("decoder.final_norm.weight", f"{prefix}.norm.weight"),
            ReplicatedMapping("decoder.final_attn_res.key_norm_weight", f"{prefix}.output_attn_res_norm.weight"),
            KimiK3AttnResQueryMapping("decoder.final_attn_res.pseudo_query", f"{prefix}.output_attn_res_proj.weight"),
        ]
        for index in range(text.num_hidden_layers):
            hf = f"{prefix}.layers.{index}"
            attention = f"decoder.layers.{2 * index}"
            ffn = f"decoder.layers.{2 * index + 1}"
            for native, branch in ((attention, "self_attention"), (ffn, "mlp")):
                mappings.extend(
                    [
                        ReplicatedMapping(f"{native}.attn_res.key_norm_weight", f"{hf}.{branch}_res_norm.weight"),
                        KimiK3AttnResQueryMapping(f"{native}.attn_res.pseudo_query", f"{hf}.{branch}_res_proj.weight"),
                    ]
                )
            mappings.append(
                ReplicatedMapping(f"{attention}.inner_layer.input_layernorm.weight", f"{hf}.input_layernorm.weight")
            )
            native = f"{attention}.inner_layer.self_attention"
            source = f"{hf}.self_attn"
            if index + 1 in text.linear_attn_config["kda_layers"]:
                for key in ("q", "k", "v"):
                    mappings.append(
                        ColumnParallelMapping(f"{native}.{key}_proj.weight", f"{source}.{key}_proj.weight")
                    )
                for fused, split in (("conv1d.weight", "conv1d.weight"),):
                    mappings.append(
                        KimiK3QKVMapping(
                            f"{native}.{fused}", **{key: f"{source}.{key}_{split}" for key in ("q", "k", "v")}
                        )
                    )
                for dest, origin, cls in (
                    ("f_a_proj.weight", "f_a_proj.weight", ReplicatedMapping),
                    ("f_b_proj.weight", "f_b_proj.weight", ColumnParallelMapping),
                    ("beta_proj.weight", "b_proj.weight", ColumnParallelMapping),
                    ("g_proj.weight", "g_proj.weight", ColumnParallelMapping),
                    ("out_norm.weight", "o_norm.weight", ReplicatedMapping),
                    ("out_proj.weight", "o_proj.weight", RowParallelMapping),
                    ("A_log", "A_log", ColumnParallelMapping),
                    ("dt_bias", "dt_bias", ColumnParallelMapping),
                ):
                    mappings.append(cls(f"{native}.{dest}", f"{source}.{origin}"))
            else:
                for dest, origin, cls in (
                    ("linear_q_down_proj.weight", "q_a_proj.weight", ReplicatedMapping),
                    ("linear_q_up_proj.weight", "q_b_proj.weight", ColumnParallelMapping),
                    ("linear_kv_down_proj.weight", "kv_a_proj_with_mqa.weight", ReplicatedMapping),
                    ("linear_kv_up_proj.weight", "kv_b_proj.weight", ColumnParallelMapping),
                    ("linear_gate.weight", "g_proj.weight", ColumnParallelMapping),
                    ("linear_proj.weight", "o_proj.weight", RowParallelMapping),
                    ("linear_q_up_proj.layer_norm_weight", "q_a_layernorm.weight", ReplicatedMapping),
                    ("linear_kv_up_proj.layer_norm_weight", "kv_a_layernorm.weight", ReplicatedMapping),
                    ("q_layernorm.weight", "q_a_layernorm.weight", ReplicatedMapping),
                    ("kv_layernorm.weight", "kv_a_layernorm.weight", ReplicatedMapping),
                ):
                    mappings.append(cls(f"{native}.{dest}", f"{source}.{origin}"))
            inner = f"{ffn}.inner_layer"
            moe = index >= text.first_k_dense_replace and index % text.moe_layer_freq == 0
            if not moe:
                mappings.extend(
                    [
                        ReplicatedMapping(
                            f"{inner}.mlp.linear_fc1.layer_norm_weight", f"{hf}.post_attention_layernorm.weight"
                        ),
                        GatedMLPMapping(
                            f"{inner}.mlp.linear_fc1.weight",
                            gate=f"{hf}.mlp.gate_proj.weight",
                            up=f"{hf}.mlp.up_proj.weight",
                        ),
                        RowParallelMapping(f"{inner}.mlp.linear_fc2.weight", f"{hf}.mlp.down_proj.weight"),
                    ]
                )
                continue
            source = f"{hf}.block_sparse_moe"
            mappings.append(
                ReplicatedMapping(f"{inner}.pre_mlp_layernorm.weight", f"{hf}.post_attention_layernorm.weight")
            )
            for dest, origin in (
                ("router.weight", "gate.weight"),
                ("router.expert_bias", "gate.e_score_correction_bias"),
                ("fc1_latent_proj.weight", "routed_expert_down_proj.weight"),
                ("fc2_latent_proj.weight", "routed_expert_up_proj.weight"),
                ("fc2_norm.weight", "routed_expert_norm.weight"),
                ("fc2_latent_proj.layer_norm_weight", "routed_expert_norm.weight"),
            ):
                mappings.append(ReplicatedMapping(f"{inner}.mlp.{dest}", f"{source}.{origin}"))
            mappings.extend(
                [
                    GatedMLPMapping(
                        f"{inner}.mlp.shared_experts.linear_fc1.weight",
                        gate=f"{source}.shared_experts.gate_proj.weight",
                        up=f"{source}.shared_experts.up_proj.weight",
                    ),
                    RowParallelMapping(
                        f"{inner}.mlp.shared_experts.linear_fc2.weight", f"{source}.shared_experts.down_proj.weight"
                    ),
                    GatedMLPMapping(
                        f"{inner}.mlp.experts.linear_fc1.weight*",
                        gate=f"{source}.experts.*.w1.weight",
                        up=f"{source}.experts.*.w3.weight",
                    ),
                    RowParallelMapping(f"{inner}.mlp.experts.linear_fc2.weight*", f"{source}.experts.*.w2.weight"),
                    GatedMLPMapping(
                        f"{inner}.mlp.experts.local_experts.*.linear_fc1.weight",
                        gate=f"{source}.experts.*.w1.weight",
                        up=f"{source}.experts.*.w3.weight",
                    ),
                    RowParallelMapping(
                        f"{inner}.mlp.experts.local_experts.*.linear_fc2.weight", f"{source}.experts.*.w2.weight"
                    ),
                ]
            )
        return MegatronMappingRegistry(*mappings)
