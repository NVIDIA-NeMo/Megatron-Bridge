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

from copy import deepcopy

import torch
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.experimental_attention_variant.dsa import is_dsa_skip_topk_layer
from torch import nn
from transformers import GlmMoeDsaForCausalLM

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.glm_moe_dsa.glm5_provider import GLM5ModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.hybrid.hybrid_provider import get_default_hybrid_stack_spec


def glm5_hybrid_stack_spec(config: GLM5ModelProvider) -> ModuleSpec:
    """Return the selected Hybrid stack with GLM's MLA query and KV norms."""
    stack_spec = deepcopy(get_default_hybrid_stack_spec(config))
    dsa_submodules = stack_spec.submodules.dsa_layer.submodules
    native_norm = dsa_submodules.input_layernorm
    attention_submodules = dsa_submodules.self_attention.submodules
    attention_submodules.q_layernorm = native_norm
    attention_submodules.kv_layernorm = native_norm
    return stack_spec


def _get_mlp_layer_types(hf_config) -> list[str]:
    """Return and validate the logical GLM MLP layer types."""
    num_layers = hf_config.num_hidden_layers
    layer_types = getattr(hf_config, "mlp_layer_types", None)
    if layer_types is None:
        first_k_dense = getattr(hf_config, "first_k_dense_replace", min(3, num_layers))
        layer_types = ["dense"] * first_k_dense + ["sparse"] * (num_layers - first_k_dense)
    else:
        layer_types = list(layer_types)

    if len(layer_types) != num_layers:
        raise ValueError(f"mlp_layer_types has {len(layer_types)} entries, but num_hidden_layers is {num_layers}.")
    invalid_types = sorted(set(layer_types) - {"dense", "sparse"})
    if invalid_types:
        raise ValueError(f"Unsupported GLM MLP layer types: {invalid_types}. Expected 'dense' or 'sparse'.")
    return layer_types


def _indexer_types_for_schedule(num_layers: int, *, topk_freq: int, skip_topk_offset: int) -> list[str]:
    """Build an HF-style logical IndexShare schedule from MCore parameters."""
    return [
        "shared" if is_dsa_skip_topk_layer(layer_number, skip_topk_offset, topk_freq) else "full"
        for layer_number in range(1, num_layers + 1)
    ]


def _get_index_share_schedule(hf_config) -> tuple[int, int]:
    """Derive and validate the logical IndexShare frequency and offset."""
    num_layers = hf_config.num_hidden_layers
    indexer_types = getattr(hf_config, "indexer_types", None)
    if indexer_types is None:
        raise ValueError("GLM IndexShare conversion requires indexer_types in the Hugging Face config.")
    indexer_types = list(indexer_types)
    if len(indexer_types) != num_layers:
        raise ValueError(f"indexer_types has {len(indexer_types)} entries, but num_hidden_layers is {num_layers}.")
    invalid_types = sorted(set(indexer_types) - {"full", "shared"})
    if invalid_types:
        raise ValueError(f"Unsupported GLM indexer types: {invalid_types}. Expected 'full' or 'shared'.")

    if all(indexer_type == "full" for indexer_type in indexer_types):
        return 1, 0

    configured_freq = getattr(hf_config, "index_topk_freq", None)
    configured_offset = getattr(hf_config, "index_skip_topk_offset", None)
    if configured_freq is not None and configured_offset is not None:
        configured_schedule = _indexer_types_for_schedule(
            num_layers,
            topk_freq=configured_freq,
            skip_topk_offset=configured_offset,
        )
        if configured_schedule != indexer_types:
            raise ValueError(
                "indexer_types does not match index_topk_freq/index_skip_topk_offset: "
                f"expected {configured_schedule}, got {indexer_types}."
            )
        return configured_freq, configured_offset

    candidates = [
        (topk_freq, skip_topk_offset)
        for topk_freq in range(2, num_layers + 1)
        for skip_topk_offset in range(num_layers + 1)
        if _indexer_types_for_schedule(
            num_layers,
            topk_freq=topk_freq,
            skip_topk_offset=skip_topk_offset,
        )
        == indexer_types
    ]
    if configured_freq is not None:
        candidates = [candidate for candidate in candidates if candidate[0] == configured_freq]
    if configured_offset is not None:
        candidates = [candidate for candidate in candidates if candidate[1] == configured_offset]
    if not candidates:
        raise ValueError(
            f"indexer_types cannot be represented by MCore's periodic DSA IndexShare schedule: {indexer_types}."
        )

    # Prefer the widest matching period when a short config ends before the next full indexer.
    return max(candidates, key=lambda candidate: (candidate[0], -candidate[1]))


class _GLM5IndexShareMapping(AutoMapping):
    """Export a live MCore indexer to its source and shared HF layers."""

    def __init__(self, megatron_param: str, hf_param: str, shared_hf_params: tuple[str, ...] = ()):
        super().__init__(megatron_param=megatron_param, hf_param=hf_param)
        self.shared_hf_params = shared_hf_params

    def megatron_to_hf(
        self,
        megatron_weights: torch.Tensor | None,
        megatron_module: nn.Module | None,
    ) -> dict[str, torch.Tensor]:
        """Copy the computing indexer's weight to HF's unused shared modules."""
        result = super().megatron_to_hf(megatron_weights, megatron_module)
        if result:
            source_weight = result[self.hf_param]
            result.update({hf_param: source_weight.clone() for hf_param in self.shared_hf_params})
        return result


@MegatronModelBridge.register_bridge(
    source=GlmMoeDsaForCausalLM,
    target=HybridModel,
    provider=GLM5ModelProvider,
    model_type="glm_moe_dsa",
)
class GLM5Bridge(MegatronModelBridge):
    """
    Megatron Bridge for GLM-5 / GLM-5.1 / GLM-5.2 (MoE + MLA + DSA).

    This bridge handles conversion between HuggingFace GlmMoeDsaForCausalLM
    and Megatron-Core HybridModel formats. The GLM-5 family shares the same
    architecture and configuration shape, so all variants are auto-detected
    through this bridge.

    The architecture uses Multi-Latent Attention (MLA), Dynamic Sparse Attention
    (DSA) indexer layers, and Mixture-of-Experts (MoE).
    Requires transformers>=5.2.0.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("zai-org/GLM-5.1")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GLM5ModelProvider:
        """Convert a Hugging Face GLM-5-family config to a Hybrid MLA provider."""
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.share_embeddings_and_output_weights = False
        provider.qk_layernorm = True
        provider.multi_latent_attention = True
        provider.sequence_parallel = True

        # Disable MTP (Multi-Token Prediction) by default
        # HF config has num_nextn_predict_layers=1
        provider.mtp_num_layers = None

        provider.moe_grouped_gemm = True
        provider.moe_router_pre_softmax = True
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_router_load_balancing_type = "seq_aux_loss"
        provider.moe_shared_expert_overlap = True
        provider.moe_router_score_function = "sigmoid"
        provider.moe_router_enable_expert_bias = True
        provider.moe_router_dtype = "fp32"
        provider.moe_permute_fusion = True

        provider.hidden_dropout = 0.0
        provider.attention_softmax_in_fp32 = False

        provider.make_vocab_size_divisible_by = 1280

        mlp_layer_types = _get_mlp_layer_types(hf_config)
        provider.hybrid_layer_pattern = "".join(
            Symbols.DS_ATTENTION + (Symbols.MLP if layer_type == "dense" else Symbols.MOE)
            for layer_type in mlp_layer_types
        )
        provider.num_layers = len(provider.hybrid_layer_pattern)
        provider.moe_layer_freq = [int(layer_type == Symbols.MOE) for layer_type in provider.hybrid_layer_pattern]
        provider.hybrid_stack_spec = glm5_hybrid_stack_spec

        provider.moe_shared_expert_intermediate_size = hf_config.moe_intermediate_size * hf_config.n_shared_experts

        # GLM5-specific: rotary_base is nested in rope_parameters
        provider.rotary_base = hf_config.rope_parameters["rope_theta"]
        # GLM5 uses default rope (no YaRN scaling)
        provider.rotary_scaling_factor = 1.0
        provider.mscale = 1.0
        provider.mscale_all_dim = 1.0
        # Transformers ignores the legacy interleave fields and applies split-half RoPE.
        provider.rotary_interleaved = False
        provider.dsa_indexer_rope_interleaved = False
        provider.apply_rope_fusion = False

        # DSA indexer params
        provider.experimental_attention_variant = "dsa"
        provider.dsa_indexer_head_dim = hf_config.index_head_dim
        provider.dsa_indexer_n_heads = hf_config.index_n_heads
        provider.dsa_indexer_topk = hf_config.index_topk
        provider.dsa_indexer_loss_coeff = 0.001
        provider.dsa_indexer_use_sparse_loss = True

        logical_freq, logical_offset = _get_index_share_schedule(hf_config)
        provider.dsa_indexer_topk_freq = 2 * logical_freq
        provider.dsa_indexer_skip_topk_offset = 2 * max(logical_offset, 1) - 1

        return provider

    @classmethod
    def megatron_to_hf_config(cls, provider) -> dict:
        """Restore the logical GLM layer count when exporting a Hybrid model."""
        hf_config = super().megatron_to_hf_config(provider)
        hybrid_layer_pattern = getattr(provider, "hybrid_layer_pattern", None)
        if hybrid_layer_pattern:
            main_pattern = hybrid_layer_pattern.split(Symbols.MTP_SEPARATOR)[0]
            hf_config["num_hidden_layers"] = main_pattern.count(Symbols.DS_ATTENTION)
        return hf_config

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Return explicit logical-to-physical GLM parameter mappings."""
        hf_config = self.hf_config
        _get_index_share_schedule(hf_config)
        indexer_types = list(hf_config.indexer_types)
        index_share_groups: dict[int, list[int]] = {}
        source_layer_idx = None
        for layer_idx, indexer_type in enumerate(indexer_types):
            if indexer_type == "full":
                source_layer_idx = layer_idx
                index_share_groups[source_layer_idx] = []
            elif source_layer_idx is None:
                raise ValueError("The first GLM indexer layer must be 'full'.")
            else:
                index_share_groups[source_layer_idx].append(layer_idx)

        mapping_list = [
            AutoMapping(
                megatron_param="embedding.word_embeddings.weight",
                hf_param="model.embed_tokens.weight",
            ),
            AutoMapping(megatron_param="decoder.final_norm.weight", hf_param="model.norm.weight"),
            AutoMapping(megatron_param="output_layer.weight", hf_param="lm_head.weight"),
        ]

        for hf_layer_idx, mlp_layer_type in enumerate(_get_mlp_layer_types(hf_config)):
            dsa_layer_idx = 2 * hf_layer_idx
            ffn_layer_idx = dsa_layer_idx + 1
            hf_layer = f"model.layers.{hf_layer_idx}"
            dsa_layer = f"decoder.layers.{dsa_layer_idx}"
            ffn_layer = f"decoder.layers.{ffn_layer_idx}"

            mapping_list.extend(
                [
                    AutoMapping(
                        megatron_param=f"{dsa_layer}.input_layernorm.weight",
                        hf_param=f"{hf_layer}.input_layernorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"{dsa_layer}.self_attention.linear_proj.weight",
                        hf_param=f"{hf_layer}.self_attn.o_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"{dsa_layer}.self_attention.linear_q_down_proj.weight",
                        hf_param=f"{hf_layer}.self_attn.q_a_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"{dsa_layer}.self_attention.linear_q_up_proj.weight",
                        hf_param=f"{hf_layer}.self_attn.q_b_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"{dsa_layer}.self_attention.q_layernorm.weight",
                        hf_param=f"{hf_layer}.self_attn.q_a_layernorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"{dsa_layer}.self_attention.linear_kv_down_proj.weight",
                        hf_param=f"{hf_layer}.self_attn.kv_a_proj_with_mqa.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"{dsa_layer}.self_attention.linear_kv_up_proj.weight",
                        hf_param=f"{hf_layer}.self_attn.kv_b_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"{dsa_layer}.self_attention.kv_layernorm.weight",
                        hf_param=f"{hf_layer}.self_attn.kv_a_layernorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"{dsa_layer}.self_attention.linear_q_proj.weight",
                        hf_param=f"{hf_layer}.self_attn.q_proj.weight",
                    ),
                ]
            )

            if indexer_types[hf_layer_idx] == "full":
                shared_layer_indices = index_share_groups[hf_layer_idx]
                indexer_mappings = {
                    "linear_wq_b.weight": "wq_b.weight",
                    "linear_wk.weight": "wk.weight",
                    "k_norm.weight": "k_norm.weight",
                    "k_norm.bias": "k_norm.bias",
                    "linear_weights_proj.weight": "weights_proj.weight",
                }
                for megatron_suffix, hf_suffix in indexer_mappings.items():
                    mapping_list.append(
                        _GLM5IndexShareMapping(
                            megatron_param=(f"{dsa_layer}.self_attention.core_attention.indexer.{megatron_suffix}"),
                            hf_param=f"{hf_layer}.self_attn.indexer.{hf_suffix}",
                            shared_hf_params=tuple(
                                f"model.layers.{shared_idx}.self_attn.indexer.{hf_suffix}"
                                for shared_idx in shared_layer_indices
                            ),
                        )
                    )

            if mlp_layer_type == "dense":
                mapping_list.extend(
                    [
                        AutoMapping(
                            megatron_param=f"{ffn_layer}.mlp.linear_fc1.layer_norm_weight",
                            hf_param=f"{hf_layer}.post_attention_layernorm.weight",
                        ),
                        AutoMapping(
                            megatron_param=f"{ffn_layer}.mlp.linear_fc2.weight",
                            hf_param=f"{hf_layer}.mlp.down_proj.weight",
                        ),
                        GatedMLPMapping(
                            megatron_param=f"{ffn_layer}.mlp.linear_fc1.weight",
                            gate=f"{hf_layer}.mlp.gate_proj.weight",
                            up=f"{hf_layer}.mlp.up_proj.weight",
                        ),
                    ]
                )
                continue

            mapping_list.extend(
                [
                    AutoMapping(
                        megatron_param=f"{ffn_layer}.pre_mlp_layernorm.weight",
                        hf_param=f"{hf_layer}.post_attention_layernorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"{ffn_layer}.mlp.router.weight",
                        hf_param=f"{hf_layer}.mlp.gate.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"{ffn_layer}.mlp.router.expert_bias",
                        hf_param=f"{hf_layer}.mlp.gate.e_score_correction_bias",
                    ),
                    AutoMapping(
                        megatron_param=f"{ffn_layer}.mlp.shared_experts.router.weight",
                        hf_param=f"{hf_layer}.mlp.shared_experts.gate.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"{ffn_layer}.mlp.shared_experts.linear_fc2.weight",
                        hf_param=f"{hf_layer}.mlp.shared_experts.down_proj.weight",
                    ),
                    GatedMLPMapping(
                        megatron_param=f"{ffn_layer}.mlp.shared_experts.linear_fc1.weight",
                        gate=f"{hf_layer}.mlp.shared_experts.gate_proj.weight",
                        up=f"{hf_layer}.mlp.shared_experts.up_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"{ffn_layer}.mlp.experts.linear_fc2.weight*",
                        hf_param=f"{hf_layer}.mlp.experts.*.down_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"{ffn_layer}.mlp.experts.local_experts.*.linear_fc2.weight",
                        hf_param=f"{hf_layer}.mlp.experts.*.down_proj.weight",
                    ),
                    GatedMLPMapping(
                        megatron_param=f"{ffn_layer}.mlp.experts.linear_fc1.weight*",
                        gate=f"{hf_layer}.mlp.experts.*.gate_proj.weight",
                        up=f"{hf_layer}.mlp.experts.*.up_proj.weight",
                    ),
                    GatedMLPMapping(
                        megatron_param=f"{ffn_layer}.mlp.experts.local_experts.*.linear_fc1.weight",
                        gate=f"{hf_layer}.mlp.experts.*.gate_proj.weight",
                        up=f"{hf_layer}.mlp.experts.*.up_proj.weight",
                    ),
                ]
            )

        # MTP execution remains disabled, but keep the existing conversion registrations.
        mtp_param_mappings = {
            # Embed
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            # LM Head
            "decoder.final_layernorm.weight": "model.norm.weight",
            "output_layer.weight": "lm_head.weight",
            # Attention layernorm
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
            # Attention output
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            # Post-attention layernorm — MoE layers use pre_mlp_layernorm, dense layers use layer_norm_weight
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            # MLA weights
            "decoder.layers.*.self_attention.linear_q_down_proj.weight": "model.layers.*.self_attn.q_a_proj.weight",
            "decoder.layers.*.self_attention.linear_q_up_proj.weight": "model.layers.*.self_attn.q_b_proj.weight",
            "decoder.layers.*.self_attention.linear_q_up_proj.layer_norm_weight": "model.layers.*.self_attn.q_a_layernorm.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_a_layernorm.weight",
            "decoder.layers.*.self_attention.linear_kv_down_proj.weight": "model.layers.*.self_attn.kv_a_proj_with_mqa.weight",
            "decoder.layers.*.self_attention.linear_kv_up_proj.weight": "model.layers.*.self_attn.kv_b_proj.weight",
            "decoder.layers.*.self_attention.linear_kv_up_proj.layer_norm_weight": "model.layers.*.self_attn.kv_a_layernorm.weight",
            "decoder.layers.*.self_attention.kv_layernorm.weight": "model.layers.*.self_attn.kv_a_layernorm.weight",
            # For non-MLA attention (fallback)
            "decoder.layers.*.self_attention.linear_q_proj.weight": "model.layers.*.self_attn.q_proj.weight",
            # DSA indexer
            "decoder.layers.*.self_attention.core_attention.indexer.linear_wq_b.weight": "model.layers.*.self_attn.indexer.wq_b.weight",
            "decoder.layers.*.self_attention.core_attention.indexer.linear_wk.weight": "model.layers.*.self_attn.indexer.wk.weight",
            "decoder.layers.*.self_attention.core_attention.indexer.k_norm.weight": "model.layers.*.self_attn.indexer.k_norm.weight",
            "decoder.layers.*.self_attention.core_attention.indexer.k_norm.bias": "model.layers.*.self_attn.indexer.k_norm.bias",
            "decoder.layers.*.self_attention.core_attention.indexer.linear_weights_proj.weight": "model.layers.*.self_attn.indexer.weights_proj.weight",
            # Dense MLP
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            # MoE router
            "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.gate.e_score_correction_bias",
            # MoE shared experts
            "decoder.layers.*.mlp.shared_experts.router.weight": "model.layers.*.mlp.shared_experts.gate.weight",
            "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "model.layers.*.mlp.shared_experts.down_proj.weight",
            # MoE expert weights (per-expert format: experts.N.down_proj)
            "decoder.layers.*.mlp.experts.linear_fc2.weight*": "model.layers.*.mlp.experts.*.down_proj.weight",
            "decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight": "model.layers.*.mlp.experts.*.down_proj.weight",
        }

        num_mtp_layers = getattr(hf_config, "num_nextn_predict_layers", 0) or 0
        num_transformer_layers = hf_config.num_hidden_layers
        for mtp_layer in range(num_mtp_layers):
            # MTP specific mappings
            mapping_list.extend(
                [
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.enorm.weight",
                        hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.enorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.hnorm.weight",
                        hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.hnorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.eh_proj.weight",
                        hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.eh_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.final_layernorm.weight",
                        hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.shared_head.norm.weight",
                    ),
                ]
            )

            for layer_prefix in ("transformer_layer", "mtp_model_layer"):
                for megatron_param, hf_param in mtp_param_mappings.items():
                    megatron_param = (
                        megatron_param.replace(".*", f".*.{layer_prefix}", 1)
                        .replace("decoder", "mtp")
                        .replace(".*", f".{mtp_layer}", 1)
                    )
                    hf_param = hf_param.replace("layers.*", f"layers.{mtp_layer + num_transformer_layers}")
                    mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))
                # Special mappings that require parameter concatenation/transformation
                mapping_list.extend(
                    [
                        QKVMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.{layer_prefix}.self_attention.linear_qkv.weight",
                            q=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.q_proj.weight",
                            k=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.k_proj.weight",
                            v=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.v_proj.weight",
                        ),
                        QKVMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.{layer_prefix}.self_attention.linear_qkv.bias",
                            q=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.q_proj.bias",
                            k=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.k_proj.bias",
                            v=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.v_proj.bias",
                        ),
                        GatedMLPMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.{layer_prefix}.mlp.linear_fc1.weight",
                            gate=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.gate_proj.weight",
                            up=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.up_proj.weight",
                        ),
                        GatedMLPMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.{layer_prefix}.mlp.shared_experts.linear_fc1.weight",
                            gate=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.shared_experts.gate_proj.weight",
                            up=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.shared_experts.up_proj.weight",
                        ),
                        GatedMLPMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.{layer_prefix}.mlp.experts.linear_fc1.weight*",
                            gate=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.experts.*.gate_proj.weight",
                            up=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.experts.*.up_proj.weight",
                        ),
                        GatedMLPMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.{layer_prefix}.mlp.experts.local_experts.*.linear_fc1.weight",
                            gate=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.experts.*.gate_proj.weight",
                            up=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.experts.*.up_proj.weight",
                        ),
                    ]
                )

        return MegatronMappingRegistry(*mapping_list)
