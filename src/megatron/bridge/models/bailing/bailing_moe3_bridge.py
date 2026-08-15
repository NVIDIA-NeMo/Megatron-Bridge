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

"""Megatron Bridge support for the public Ling 3.0 Tiny and Flash checkpoints."""

from typing import Any

import torch.nn.functional as F
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer.enums import AttnBackend

from megatron.bridge.models.bailing.bailing_moe3_mappings import (
    _BailingMoe3KDAConv1dMapping,
    _BailingMoe3KDAInProjMapping,
)
from megatron.bridge.models.bailing.bailing_moe3_provider import BailingMoe3HybridProvider
from megatron.bridge.models.bailing.bailing_moe3_spec import bailing_moe3_hybrid_stack_spec
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


_TINY_NUM_LOGICAL_LAYERS = 24
_TINY_GROUP_SIZE = 4
_TINY_NUM_EXPERTS = 128
_TINY_TOPK = 8
_TINY_PATTERN = "K-KEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+E"

_FLASH_NUM_LOGICAL_LAYERS = 42
_FLASH_GROUP_SIZE = 6
_FLASH_NUM_EXPERTS = 512
_FLASH_TOPK = 8
_FLASH_MTP_PATTERN = "+E"


def _hybrid_layer_pattern(hf_config: Any) -> str:
    """Build the main HybridModel pattern from the public HF layer schedule."""
    num_layers = getattr(hf_config, "num_hidden_layers", None)
    group_size = getattr(hf_config, "layer_group_size", None)
    first_dense = getattr(hf_config, "first_k_dense_replace", None)
    if not all(isinstance(value, int) for value in (num_layers, group_size, first_dense)):
        raise ValueError(
            "Ling 3.0 requires integer num_hidden_layers, layer_group_size, and first_k_dense_replace; "
            f"got {num_layers=}, {group_size=}, {first_dense=}."
        )
    if num_layers <= 0 or group_size <= 0 or not 0 <= first_dense <= num_layers:
        raise ValueError(
            "Ling 3.0 requires num_hidden_layers>0, layer_group_size>0, and "
            "0 <= first_k_dense_replace <= num_hidden_layers; "
            f"got {num_layers=}, {group_size=}, {first_dense=}."
        )

    # The public HF implementation places MLA at the end of every complete
    # group, and at every layer in a final incomplete group.
    complete_groups_end = (num_layers // group_size) * group_size
    return "".join(
        ("+" if (logical_layer + 1) % group_size == 0 or logical_layer >= complete_groups_end else "K")
        + ("-" if logical_layer < first_dense else "E")
        for logical_layer in range(num_layers)
    )


def _tiny_hybrid_layer_pattern(hf_config: Any) -> str:
    """Build and validate the 48-position Tiny HybridModel pattern."""
    num_layers = getattr(hf_config, "num_hidden_layers", None)
    group_size = getattr(hf_config, "layer_group_size", None)
    first_dense = getattr(hf_config, "first_k_dense_replace", None)
    if (num_layers, group_size, first_dense) != (_TINY_NUM_LOGICAL_LAYERS, _TINY_GROUP_SIZE, 1):
        raise ValueError(
            "Ling 3.0 Tiny requires num_hidden_layers=24, layer_group_size=4, and "
            "first_k_dense_replace=1; "
            f"got {num_layers=}, {group_size=}, {first_dense=}."
        )

    pattern = _hybrid_layer_pattern(hf_config)
    if pattern != _TINY_PATTERN:
        raise ValueError(f"Unexpected Ling 3.0 Tiny hybrid pattern: {pattern}")
    return pattern


def _flash_hybrid_layer_pattern(hf_config: Any) -> str:
    """Build and validate the 84-position Flash HybridModel pattern."""
    num_layers = getattr(hf_config, "num_hidden_layers", None)
    group_size = getattr(hf_config, "layer_group_size", None)
    first_dense = getattr(hf_config, "first_k_dense_replace", None)
    if (num_layers, group_size, first_dense) != (_FLASH_NUM_LOGICAL_LAYERS, _FLASH_GROUP_SIZE, 2):
        raise ValueError(
            "Ling 3.0 Flash requires num_hidden_layers=42, layer_group_size=6, and "
            "first_k_dense_replace=2; "
            f"got {num_layers=}, {group_size=}, {first_dense=}."
        )
    return _hybrid_layer_pattern(hf_config)


def _validate_common_config(hf_config: Any) -> None:
    """Validate fields shared by the public Ling 3.0 variants."""
    if getattr(hf_config, "gated_attention_proj_granularity_type", None) != "head_wise":
        raise ValueError("Ling 3.0 Bridge requires head-wise MLA output gating.")
    if getattr(hf_config, "use_kda_lora", False) or not getattr(hf_config, "no_kda_lora", False):
        raise ValueError("Ling 3.0 Bridge requires direct KDA projections with no_kda_lora=true.")

    for name in ("use_qkv_bias", "attention_bias", "use_bias", "mlp_bias"):
        if getattr(hf_config, name, False):
            raise ValueError(f"Ling 3.0 Bridge only supports bias-free projections; {name}=true.")

    scoring_func = getattr(hf_config, "scoring_func", getattr(hf_config, "score_function", "sigmoid"))
    if scoring_func != "sigmoid":
        raise ValueError(f"Ling 3.0 Bridge requires sigmoid routing, got {scoring_func!r}.")


def _validate_tiny_config(hf_config: Any) -> None:
    """Reject configurations that are not the public Ling 3.0 Tiny variant."""
    expected_values = {
        "hidden_size": 1536,
        "intermediate_size": 4608,
        "vocab_size": 157184,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "num_experts": _TINY_NUM_EXPERTS,
        "num_experts_per_tok": _TINY_TOPK,
        "moe_intermediate_size": 512,
        "moe_shared_expert_intermediate_size": 512,
        "num_shared_experts": 1,
        "q_lora_rank": 256,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "short_conv_kernel_size": 4,
    }
    for name, expected in expected_values.items():
        actual = getattr(hf_config, name, None)
        if actual != expected:
            raise ValueError(f"Ling 3.0 Tiny requires {name}={expected!r}, got {actual!r}.")

    _tiny_hybrid_layer_pattern(hf_config)
    if getattr(hf_config, "num_nextn_predict_layers", 0) != 0:
        raise ValueError("Ling 3.0 Tiny does not support MTP; expected num_nextn_predict_layers=0.")


def _validate_flash_config(hf_config: Any) -> None:
    """Reject configurations that are not the public Ling 3.0 Flash variant."""
    expected_values = {
        "hidden_size": 2560,
        "intermediate_size": 6144,
        "vocab_size": 157184,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "num_experts": _FLASH_NUM_EXPERTS,
        "num_experts_per_tok": _FLASH_TOPK,
        "moe_intermediate_size": 768,
        "moe_shared_expert_intermediate_size": 768,
        "num_shared_experts": 1,
        "kv_lora_rank": 512,
        "qk_head_dim": 192,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "head_dim": 128,
        "short_conv_kernel_size": 4,
    }
    for name, expected in expected_values.items():
        actual = getattr(hf_config, name, None)
        if actual != expected:
            raise ValueError(f"Ling 3.0 Flash requires {name}={expected!r}, got {actual!r}.")

    _flash_hybrid_layer_pattern(hf_config)
    if getattr(hf_config, "q_lora_rank", None) is not None:
        raise ValueError("Ling 3.0 Flash requires direct-Q MLA with q_lora_rank=null.")
    if getattr(hf_config, "num_nextn_predict_layers", 0) != 1:
        raise ValueError("Ling 3.0 Flash requires exactly one MTP layer.")
    if getattr(hf_config, "mtp_use_kda", False):
        raise ValueError("Ling 3.0 Flash MTP is an MLA layer; expected mtp_use_kda=false.")


def _is_tiny_config(hf_config: Any) -> bool:
    """Return whether the config has the public Tiny architecture signature."""
    return (
        getattr(hf_config, "num_hidden_layers", None),
        getattr(hf_config, "layer_group_size", None),
        getattr(hf_config, "first_k_dense_replace", None),
        getattr(hf_config, "hidden_size", None),
        getattr(hf_config, "num_experts", None),
    ) == (_TINY_NUM_LOGICAL_LAYERS, _TINY_GROUP_SIZE, 1, 1536, _TINY_NUM_EXPERTS)


def _is_flash_config(hf_config: Any) -> bool:
    """Return whether the config has the public Flash architecture signature."""
    return (
        getattr(hf_config, "num_hidden_layers", None),
        getattr(hf_config, "layer_group_size", None),
        getattr(hf_config, "first_k_dense_replace", None),
        getattr(hf_config, "hidden_size", None),
        getattr(hf_config, "num_experts", None),
    ) == (_FLASH_NUM_LOGICAL_LAYERS, _FLASH_GROUP_SIZE, 2, 2560, _FLASH_NUM_EXPERTS)


def _model_variant(hf_config: Any) -> str:
    """Return the validated public Ling 3.0 variant name."""
    if _is_tiny_config(hf_config):
        return "tiny"
    if _is_flash_config(hf_config):
        return "flash"
    raise ValueError(
        "Unsupported BailingMoeV3 config. The Ling 3.0 Bridge currently supports "
        "the public Tiny and Flash variants only."
    )


def _layer_positions(hf_config: Any) -> tuple[tuple[int, int], ...]:
    """Return ``(attention_position, mlp_position)`` for each logical layer."""
    pattern = _hybrid_layer_pattern(hf_config)
    num_layers = getattr(hf_config, "num_hidden_layers")
    if len(pattern) != 2 * num_layers:
        raise ValueError(f"Unexpected Ling 3.0 main pattern length: {len(pattern)}")
    return tuple((2 * logical_layer, 2 * logical_layer + 1) for logical_layer in range(num_layers))


def _append_mtp_mappings(mappings: list[Any], hf_config: Any) -> None:
    """Append Flash's direct-Q MLA plus MoE MTP mappings."""
    num_mtp_layers = int(getattr(hf_config, "num_nextn_predict_layers", 0) or 0)
    num_main_layers = hf_config.num_hidden_layers
    for mtp_layer in range(num_mtp_layers):
        hf_layer = f"model.layers.{num_main_layers + mtp_layer}"
        # Hybrid MTP builds an inner HybridStack, so its physical layer lives
        # below ``mtp_model_layer.layers.0``.  The GPT MTP compatibility alias
        # ``transformer_layer`` does not apply when an MTP pattern is present.
        mg_attention = f"mtp.layers.{mtp_layer}.mtp_model_layer.layers.0"
        mg_mlp = f"mtp.layers.{mtp_layer}.mtp_model_layer.layers.1"
        mappings.extend(
            [
                ReplicatedMapping(
                    f"{mg_attention}.input_layernorm.weight",
                    f"{hf_layer}.input_layernorm.weight",
                ),
                AutoMapping(
                    f"{mg_attention}.self_attention.linear_q_proj.weight",
                    f"{hf_layer}.attention.q_proj.weight",
                ),
                ReplicatedMapping(
                    f"{mg_attention}.self_attention.linear_kv_down_proj.weight",
                    f"{hf_layer}.attention.kv_a_proj_with_mqa.weight",
                ),
                ReplicatedMapping(
                    f"{mg_attention}.self_attention.kv_layernorm.weight",
                    f"{hf_layer}.attention.kv_a_layernorm.weight",
                ),
                AutoMapping(
                    f"{mg_attention}.self_attention.linear_kv_up_proj.weight",
                    f"{hf_layer}.attention.kv_b_proj.weight",
                ),
                AutoMapping(
                    f"{mg_attention}.self_attention.linear_gate.weight",
                    f"{hf_layer}.attention.g_proj.weight",
                ),
                AutoMapping(
                    f"{mg_attention}.self_attention.linear_proj.weight",
                    f"{hf_layer}.attention.dense.weight",
                ),
                ReplicatedMapping(
                    f"{mg_mlp}.pre_mlp_layernorm.weight",
                    f"{hf_layer}.post_attention_layernorm.weight",
                ),
                ReplicatedMapping(
                    f"{mg_mlp}.mlp.router.weight",
                    f"{hf_layer}.mlp.gate.weight",
                ),
                ReplicatedMapping(
                    f"{mg_mlp}.mlp.router.expert_bias",
                    f"{hf_layer}.mlp.gate.expert_bias",
                ),
                GatedMLPMapping(
                    f"{mg_mlp}.mlp.experts.linear_fc1.weight*",
                    gate=f"{hf_layer}.mlp.experts.*.gate_proj.weight",
                    up=f"{hf_layer}.mlp.experts.*.up_proj.weight",
                ),
                AutoMapping(
                    f"{mg_mlp}.mlp.experts.linear_fc2.weight*",
                    f"{hf_layer}.mlp.experts.*.down_proj.weight",
                ),
                GatedMLPMapping(
                    f"{mg_mlp}.mlp.shared_experts.linear_fc1.weight",
                    gate=f"{hf_layer}.mlp.shared_experts.gate_proj.weight",
                    up=f"{hf_layer}.mlp.shared_experts.up_proj.weight",
                ),
                AutoMapping(
                    f"{mg_mlp}.mlp.shared_experts.linear_fc2.weight",
                    f"{hf_layer}.mlp.shared_experts.down_proj.weight",
                ),
            ]
        )

        mappings.extend(
            [
                ReplicatedMapping(
                    f"mtp.layers.{mtp_layer}.enorm.weight",
                    f"{hf_layer}.enorm.weight",
                ),
                ReplicatedMapping(
                    f"mtp.layers.{mtp_layer}.hnorm.weight",
                    f"{hf_layer}.hnorm.weight",
                ),
                AutoMapping(
                    f"mtp.layers.{mtp_layer}.eh_proj.weight",
                    f"{hf_layer}.eh_proj.weight",
                ),
                ReplicatedMapping(
                    f"mtp.layers.{mtp_layer}.final_layernorm.weight",
                    f"{hf_layer}.final_layernorm.weight",
                ),
            ]
        )


@MegatronModelBridge.register_bridge(
    source="BailingMoeV3ForCausalLM",
    target=HybridModel,
    provider=BailingMoe3HybridProvider,
    model_type="bailing_hybrid",
)
class BailingMoeV3Bridge(MegatronModelBridge):
    """Bridge for the public Ling 3.0 Tiny and Flash HybridModel layouts."""

    CONFIG_MAPPING = MegatronModelBridge.CONFIG_MAPPING + [
        ("short_conv_kernel_size", "linear_conv_kernel_dim"),
        ("no_kda_lora", "no_kda_lora"),
        ("kda_safe_gate", "kda_safe_gate"),
        ("kda_lower_bound", "kda_lower_bound"),
    ]

    @staticmethod
    def _validate_config(hf_config: Any) -> None:
        """Validate the source configuration before constructing a provider."""
        _validate_common_config(hf_config)
        if _is_tiny_config(hf_config):
            _validate_tiny_config(hf_config)
        elif _is_flash_config(hf_config):
            _validate_flash_config(hf_config)
        else:
            _model_variant(hf_config)

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> BailingMoe3HybridProvider:
        """Translate a public Ling 3.0 config into a native HybridModel provider."""
        hf_config = hf_pretrained.config
        self._validate_config(hf_config)
        variant = _model_variant(hf_config)
        provider = super().provider_bridge(hf_pretrained)

        pattern = _hybrid_layer_pattern(hf_config)
        provider.hybrid_layer_pattern = pattern
        provider.num_layers = len(pattern)
        provider.hybrid_stack_spec = bailing_moe3_hybrid_stack_spec
        provider.vocab_size = hf_config.vocab_size
        provider.should_pad_vocab = False
        provider.seq_length = hf_config.max_position_embeddings

        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.add_qkv_bias = False
        provider.share_embeddings_and_output_weights = False
        provider.attention_backend = AttnBackend.auto
        provider.position_embedding_type = "rope"
        provider.apply_rope_fusion = False
        # MCore's MLA path owns the Q/K rotary interleaving.  Enabling the
        # generic rotary_interleaved flag is rejected for multi-latent attention.
        provider.rotary_interleaved = False
        provider.rotary_base = hf_config.rope_theta
        provider.rotary_percent = hf_config.partial_rotary_factor
        provider.activation_func = F.silu
        provider.hidden_dropout = 0.0
        provider.attention_dropout = 0.0

        # HybridModelProvider intentionally does not declare all MLA-only fields
        # in the temporary MCore pin.  BailingMoe3HybridProvider declares them so
        # the provider remains serializable for DCP run_config reconstruction.
        provider.multi_latent_attention = True
        # qk_layernorm enables the KV norm as well.  The variant-aware module
        # spec makes Q's norm an IdentityOp for direct-Q Flash.
        provider.qk_layernorm = True
        provider.attention_output_gate = True
        provider.q_lora_rank = getattr(hf_config, "q_lora_rank", None)
        provider.kv_lora_rank = hf_config.kv_lora_rank
        provider.qk_head_dim = hf_config.qk_nope_head_dim
        provider.qk_pos_emb_head_dim = hf_config.qk_rope_head_dim
        provider.v_head_dim = hf_config.v_head_dim
        provider.rope_type = "rope"
        provider.rotary_scaling_factor = 1.0
        provider.original_max_position_embeddings = getattr(hf_config, "original_max_position_embeddings", 4096)
        provider.beta_fast = 32
        provider.beta_slow = 1
        provider.mscale = 1.0
        provider.mscale_all_dim = 0.0
        provider.cache_mla_latents = False
        provider.mla_down_proj_fusion = False

        provider.linear_conv_kernel_dim = hf_config.short_conv_kernel_size
        linear_head_dim = getattr(hf_config, "head_dim", 128)
        linear_num_heads = getattr(hf_config, "num_kv_heads_for_linear_attn", 0) or hf_config.num_attention_heads
        provider.linear_key_head_dim = linear_head_dim
        provider.linear_value_head_dim = linear_head_dim
        provider.linear_num_key_heads = linear_num_heads
        provider.linear_num_value_heads = linear_num_heads
        provider.no_kda_lora = getattr(hf_config, "no_kda_lora", True)
        provider.kda_safe_gate = getattr(hf_config, "kda_safe_gate", True)
        provider.kda_lower_bound = getattr(hf_config, "kda_lower_bound", -5.0)

        provider.num_moe_experts = hf_config.num_experts
        provider.moe_router_topk = hf_config.num_experts_per_tok
        provider.moe_layer_freq = [1 if symbol == "E" else 0 for symbol in pattern]
        provider.moe_ffn_hidden_size = hf_config.moe_intermediate_size
        provider.moe_shared_expert_intermediate_size = hf_config.moe_shared_expert_intermediate_size
        provider.moe_grouped_gemm = True
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_router_score_function = "sigmoid"
        provider.moe_router_pre_softmax = False
        provider.moe_router_topk_scaling_factor = getattr(hf_config, "routed_scaling_factor", 1.0)
        provider.moe_router_num_groups = getattr(hf_config, "n_group", None)
        provider.moe_router_group_topk = getattr(hf_config, "topk_group", None)
        provider.moe_router_enable_expert_bias = True
        provider.moe_router_bias_update_rate = 0.0
        provider.moe_router_dtype = "fp32"
        provider.moe_router_load_balancing_type = "none"
        provider.moe_aux_loss_coeff = 0.0
        provider.moe_shared_expert_overlap = False
        provider.moe_permute_fusion = True

        num_mtp_layers = int(getattr(hf_config, "num_nextn_predict_layers", 0) or 0)
        provider.mtp_num_layers = num_mtp_layers
        provider.mtp_hybrid_override_pattern = _FLASH_MTP_PATTERN if num_mtp_layers else None
        provider.mtp_use_repeated_layer = False
        provider.mtp_loss_scaling_factor = getattr(hf_config, "mtp_loss_scaling_factor", 0.1)
        provider.is_hybrid_model = True

        # Keep the branch visible in the provider setup: it is useful when
        # inspecting serialized configs and prevents direct-Q Flash from being
        # mistaken for a low-rank Tiny checkpoint.  The actual Q module choice
        # is made by bailing_moe3_hybrid_stack_spec(provider).
        if variant == "flash":
            provider.q_lora_rank = None
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Build variant-aware logical-to-physical parameter mappings."""
        hf_config = self.hf_config
        self._validate_config(hf_config)
        pattern = _hybrid_layer_pattern(hf_config)
        layer_positions = _layer_positions(hf_config)
        mappings: list[Any] = [
            AutoMapping("embedding.word_embeddings.weight", "model.word_embeddings.weight"),
            ReplicatedMapping("decoder.final_norm.weight", "model.norm.weight"),
            AutoMapping("output_layer.weight", "lm_head.weight"),
        ]

        for logical_layer, (attention_position, mlp_position) in enumerate(layer_positions):
            hf_layer = f"model.layers.{logical_layer}"
            mg_attention = f"decoder.layers.{attention_position}"
            mg_mlp = f"decoder.layers.{mlp_position}"

            mappings.append(
                ReplicatedMapping(
                    f"{mg_attention}.input_layernorm.weight",
                    f"{hf_layer}.input_layernorm.weight",
                )
            )

            if pattern[2 * logical_layer] == "+":
                if hf_config.q_lora_rank is None:
                    mappings.append(
                        AutoMapping(
                            f"{mg_attention}.self_attention.linear_q_proj.weight",
                            f"{hf_layer}.attention.q_proj.weight",
                        )
                    )
                else:
                    mappings.extend(
                        [
                            ReplicatedMapping(
                                f"{mg_attention}.self_attention.linear_q_down_proj.weight",
                                f"{hf_layer}.attention.q_a_proj.weight",
                            ),
                            ReplicatedMapping(
                                f"{mg_attention}.self_attention.q_layernorm.weight",
                                f"{hf_layer}.attention.q_a_layernorm.weight",
                            ),
                            AutoMapping(
                                f"{mg_attention}.self_attention.linear_q_up_proj.weight",
                                f"{hf_layer}.attention.q_b_proj.weight",
                            ),
                        ]
                    )
                mappings.extend(
                    [
                        ReplicatedMapping(
                            f"{mg_attention}.self_attention.linear_kv_down_proj.weight",
                            f"{hf_layer}.attention.kv_a_proj_with_mqa.weight",
                        ),
                        ReplicatedMapping(
                            f"{mg_attention}.self_attention.kv_layernorm.weight",
                            f"{hf_layer}.attention.kv_a_layernorm.weight",
                        ),
                        AutoMapping(
                            f"{mg_attention}.self_attention.linear_kv_up_proj.weight",
                            f"{hf_layer}.attention.kv_b_proj.weight",
                        ),
                        AutoMapping(
                            f"{mg_attention}.self_attention.linear_gate.weight",
                            f"{hf_layer}.attention.g_proj.weight",
                        ),
                        AutoMapping(
                            f"{mg_attention}.self_attention.linear_proj.weight",
                            f"{hf_layer}.attention.dense.weight",
                        ),
                    ]
                )
            else:
                mappings.extend(
                    [
                        _BailingMoe3KDAInProjMapping(
                            f"{mg_attention}.self_attention.in_proj.weight",
                            q=f"{hf_layer}.attention.q_proj.weight",
                            k=f"{hf_layer}.attention.k_proj.weight",
                            v=f"{hf_layer}.attention.v_proj.weight",
                            f=f"{hf_layer}.attention.f_proj.weight",
                            g=f"{hf_layer}.attention.g_proj.weight",
                        ),
                        _BailingMoe3KDAConv1dMapping(
                            f"{mg_attention}.self_attention.conv1d.weight",
                            q=f"{hf_layer}.attention.q_conv1d.weight",
                            k=f"{hf_layer}.attention.k_conv1d.weight",
                            v=f"{hf_layer}.attention.v_conv1d.weight",
                        ),
                        ColumnParallelMapping(
                            f"{mg_attention}.self_attention.A_log",
                            f"{hf_layer}.attention.A_log",
                        ),
                        ColumnParallelMapping(
                            f"{mg_attention}.self_attention.dt_bias",
                            f"{hf_layer}.attention.dt_bias",
                        ),
                        ReplicatedMapping(
                            f"{mg_attention}.self_attention.out_norm.weight",
                            f"{hf_layer}.attention.o_norm.weight",
                        ),
                        AutoMapping(
                            f"{mg_attention}.self_attention.out_proj.weight",
                            f"{hf_layer}.attention.o_proj.weight",
                        ),
                        AutoMapping(
                            f"{mg_attention}.self_attention.beta_proj.weight",
                            f"{hf_layer}.attention.b_proj.weight",
                        ),
                    ]
                )

            mappings.append(
                ReplicatedMapping(
                    f"{mg_mlp}.pre_mlp_layernorm.weight"
                    if pattern[2 * logical_layer + 1] == "E"
                    else f"{mg_mlp}.mlp.linear_fc1.layer_norm_weight",
                    f"{hf_layer}.post_attention_layernorm.weight",
                )
            )

            if pattern[2 * logical_layer + 1] == "-":
                mappings.extend(
                    [
                        GatedMLPMapping(
                            f"{mg_mlp}.mlp.linear_fc1.weight",
                            gate=f"{hf_layer}.mlp.gate_proj.weight",
                            up=f"{hf_layer}.mlp.up_proj.weight",
                        ),
                        AutoMapping(
                            f"{mg_mlp}.mlp.linear_fc2.weight",
                            f"{hf_layer}.mlp.down_proj.weight",
                        ),
                    ]
                )
            else:
                mappings.extend(
                    [
                        ReplicatedMapping(
                            f"{mg_mlp}.mlp.router.weight",
                            f"{hf_layer}.mlp.gate.weight",
                        ),
                        ReplicatedMapping(
                            f"{mg_mlp}.mlp.router.expert_bias",
                            f"{hf_layer}.mlp.gate.expert_bias",
                        ),
                        GatedMLPMapping(
                            f"{mg_mlp}.mlp.experts.linear_fc1.weight*",
                            gate=f"{hf_layer}.mlp.experts.*.gate_proj.weight",
                            up=f"{hf_layer}.mlp.experts.*.up_proj.weight",
                        ),
                        AutoMapping(
                            f"{mg_mlp}.mlp.experts.linear_fc2.weight*",
                            f"{hf_layer}.mlp.experts.*.down_proj.weight",
                        ),
                        GatedMLPMapping(
                            f"{mg_mlp}.mlp.shared_experts.linear_fc1.weight",
                            gate=f"{hf_layer}.mlp.shared_experts.gate_proj.weight",
                            up=f"{hf_layer}.mlp.shared_experts.up_proj.weight",
                        ),
                        AutoMapping(
                            f"{mg_mlp}.mlp.shared_experts.linear_fc2.weight",
                            f"{hf_layer}.mlp.shared_experts.down_proj.weight",
                        ),
                    ]
                )

        _append_mtp_mappings(mappings, hf_config)
        return MegatronMappingRegistry(*mappings)
