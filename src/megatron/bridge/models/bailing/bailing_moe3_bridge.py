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

from megatron.core.models.hybrid.hybrid_model import HybridModel

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
_FLASH_NUM_LOGICAL_LAYERS = 42
_FLASH_GROUP_SIZE = 6
_FLASH_MTP_PATTERN = "+E"


def _build_hybrid_pattern(*, num_layers: int, group_size: int, first_dense: int) -> str:
    """Build a Ling 3.0 HybridModel pattern from its logical layer schedule."""
    complete_groups_end = (num_layers // group_size) * group_size
    return "".join(
        ("+" if (logical_layer + 1) % group_size == 0 or logical_layer >= complete_groups_end else "K")
        + ("-" if logical_layer < first_dense else "E")
        for logical_layer in range(num_layers)
    )


_TINY_PATTERN = _build_hybrid_pattern(
    num_layers=_TINY_NUM_LOGICAL_LAYERS,
    group_size=_TINY_GROUP_SIZE,
    first_dense=1,
)
_FLASH_PATTERN = _build_hybrid_pattern(
    num_layers=_FLASH_NUM_LOGICAL_LAYERS,
    group_size=_FLASH_GROUP_SIZE,
    first_dense=2,
)


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

    return _build_hybrid_pattern(num_layers=num_layers, group_size=group_size, first_dense=first_dense)


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

    return _hybrid_layer_pattern(hf_config)


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
    """Reject Ling modes that the current spec or parameter mappings cannot represent."""
    if getattr(hf_config, "gated_attention_proj_granularity_type", "head_wise") != "head_wise":
        raise ValueError("Ling 3.0 Bridge requires head-wise MLA output gating.")
    if getattr(hf_config, "use_kda_lora", False) or not getattr(hf_config, "no_kda_lora", True):
        raise ValueError("Ling 3.0 Bridge requires direct KDA projections with no_kda_lora=true.")

    for name in ("use_qkv_bias", "attention_bias", "use_bias", "mlp_bias"):
        if getattr(hf_config, name, False):
            raise ValueError(f"Ling 3.0 Bridge only supports bias-free projections; {name}=true.")

    supported_modes = {
        "tie_word_embeddings": False,
        "use_qk_norm": True,
        "num_shared_experts": 1,
        "moe_router_enable_expert_bias": True,
        "router_dtype": "fp32",
        "scoring_func": "sigmoid",
        "score_function": "sigmoid",
        "norm_topk_prob": True,
        "topk_method": "noaux_tc",
        "partial_rotary_factor": 0.5,
        "rope_interleave": True,
    }
    for name, supported in supported_modes.items():
        if getattr(hf_config, name, None) != supported:
            raise ValueError(f"Ling 3.0 Bridge only supports {name}={supported!r}.")

    # Transformers 5 exposes an unscaled default RoPE config through the
    # ``rope_scaling`` compatibility property even when config.json stores
    # ``rope_scaling: null``. Reject actual scaling modes, not that normalized
    # default representation.
    rope_scaling = getattr(hf_config, "rope_scaling", None)
    if rope_scaling is not None:
        if not isinstance(rope_scaling, dict):
            raise ValueError(f"Ling 3.0 requires default RoPE; got {rope_scaling=}.")
        rope_type = rope_scaling.get("rope_type") or rope_scaling.get("type")
        if rope_type not in (None, "default"):
            raise ValueError(f"Ling 3.0 requires default RoPE; got {rope_type=}.")

    qk_nope_head_dim = getattr(hf_config, "qk_nope_head_dim", None)
    qk_rope_head_dim = getattr(hf_config, "qk_rope_head_dim", None)
    qk_head_dim = getattr(hf_config, "qk_head_dim", None)
    if None not in (qk_nope_head_dim, qk_rope_head_dim, qk_head_dim):
        expected_qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        if qk_head_dim != expected_qk_head_dim:
            raise ValueError(
                "Ling 3.0 requires qk_head_dim=qk_nope_head_dim+qk_rope_head_dim; "
                f"got {qk_head_dim=}, {qk_nope_head_dim=}, {qk_rope_head_dim=}."
            )

    rotary_dim = getattr(hf_config, "rotary_dim", qk_rope_head_dim)
    if qk_rope_head_dim is not None and rotary_dim != qk_rope_head_dim:
        raise ValueError(f"Ling 3.0 requires rotary_dim=qk_rope_head_dim; got {rotary_dim=}, {qk_rope_head_dim=}.")


def _validate_tiny_config(hf_config: Any) -> None:
    """Validate the Tiny topology and its low-rank-Q MLA layout."""
    _tiny_hybrid_layer_pattern(hf_config)
    if getattr(hf_config, "q_lora_rank", None) is None:
        raise ValueError("Ling 3.0 Tiny requires low-rank-Q MLA with q_lora_rank set.")
    if getattr(hf_config, "num_nextn_predict_layers", 0) != 0:
        raise ValueError("Ling 3.0 Tiny does not support MTP; expected num_nextn_predict_layers=0.")


def _validate_flash_config(hf_config: Any) -> None:
    """Validate the Flash topology and its direct-Q MLA/MTP layout."""
    _flash_hybrid_layer_pattern(hf_config)
    if getattr(hf_config, "q_lora_rank", None) is not None:
        raise ValueError("Ling 3.0 Flash requires direct-Q MLA with q_lora_rank=null.")
    if getattr(hf_config, "num_nextn_predict_layers", 0) != 1:
        raise ValueError("Ling 3.0 Flash requires exactly one MTP layer.")
    if getattr(hf_config, "mtp_use_kda", False):
        raise ValueError("Ling 3.0 Flash MTP is an MLA layer; expected mtp_use_kda=false.")


def _is_tiny_config(hf_config: Any) -> bool:
    """Return whether the config has the public Tiny layer topology."""
    return (
        getattr(hf_config, "num_hidden_layers", None),
        getattr(hf_config, "layer_group_size", None),
        getattr(hf_config, "first_k_dense_replace", None),
    ) == (_TINY_NUM_LOGICAL_LAYERS, _TINY_GROUP_SIZE, 1)


def _is_flash_config(hf_config: Any) -> bool:
    """Return whether the config has the public Flash layer topology."""
    return (
        getattr(hf_config, "num_hidden_layers", None),
        getattr(hf_config, "layer_group_size", None),
        getattr(hf_config, "first_k_dense_replace", None),
    ) == (_FLASH_NUM_LOGICAL_LAYERS, _FLASH_GROUP_SIZE, 2)


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
                AutoMapping(
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

    # Ling currently uses the provider path because its variant-aware Hybrid
    # stack spec is resolved from provider configuration at model construction.
    # Do not expose the default GPT builder config as a compatible path.
    MODEL_CONFIG_CLASS = None

    # Ling names its projection-bias fields differently from the common HF
    # conventions, so map those aliases before the generic fields and append
    # the remaining one-to-one family fields afterwards.
    CONFIG_MAPPING = (
        [
            ("use_qkv_bias", "add_qkv_bias"),
            ("use_bias", "add_bias_linear"),
        ]
        + MegatronModelBridge.CONFIG_MAPPING
        + [
            ("short_conv_kernel_size", "linear_conv_kernel_dim"),
            ("kda_safe_gate", "kda_safe_gate"),
            ("kda_lower_bound", "kda_lower_bound"),
            ("router_dtype", "moe_router_dtype"),
            ("moe_shared_expert_intermediate_size", "moe_shared_expert_intermediate_size"),
            ("moe_router_enable_expert_bias", "moe_router_enable_expert_bias"),
            ("mtp_loss_scaling_factor", "mtp_loss_scaling_factor"),
        ]
    )

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
        provider = super().provider_bridge(hf_pretrained)

        pattern = _hybrid_layer_pattern(hf_config)
        provider.hybrid_layer_pattern = pattern
        provider.num_layers = len(pattern)
        provider.hybrid_stack_spec = bailing_moe3_hybrid_stack_spec

        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.position_embedding_type = "rope"
        provider.apply_rope_fusion = False
        # MCore's MLA path owns the Q/K rotary interleaving.  Enabling the
        # generic rotary_interleaved flag is rejected for multi-latent attention.
        provider.rotary_interleaved = False
        # qk_pos_emb_head_dim already identifies the full rotary section of an
        # MLA head. Applying HF's head-relative partial factor again would
        # rotate only half of that section.
        provider.rotary_percent = 1.0
        # Public Ling checkpoints use default RoPE without YaRN scaling. Override
        # MLATransformerConfig's DeepSeek-oriented scaling defaults explicitly.
        provider.rotary_scaling_factor = 1.0
        provider.mscale_all_dim = 1.0
        # The variant-aware module spec makes Q's norm an IdentityOp for
        # direct-Q Flash; use_qk_norm from HF still enables the KV norm.
        provider.attention_output_gate = True
        provider.gated_attention_proj_granularity = "headwise"
        provider.rope_type = "rope"

        linear_head_dim = hf_config.head_dim
        linear_num_heads = hf_config.num_kv_heads_for_linear_attn or hf_config.num_attention_heads
        provider.linear_key_head_dim = linear_head_dim
        provider.linear_value_head_dim = linear_head_dim
        provider.linear_num_key_heads = linear_num_heads
        provider.linear_num_value_heads = linear_num_heads

        provider.moe_layer_freq = [1 if symbol == "E" else 0 for symbol in pattern]
        provider.moe_grouped_gemm = True
        # Ling applies sigmoid before top-k and always normalizes the selected
        # scores. MCore's sigmoid path has the same behavior regardless of
        # moe_router_pre_softmax, so this is not the inverse of norm_topk_prob.
        provider.moe_router_pre_softmax = False
        # ``noaux_tc`` is a responsibility split rather than a matching MCore enum:
        # ``noaux`` disables auxiliary balancing loss, while token-choice routing is
        # represented by top-k/group fields and expert bias mapped from the HF config.
        # _validate_common_config rejects every other topk_method before this point.
        provider.moe_router_load_balancing_type = "none"

        num_mtp_layers = int(provider.mtp_num_layers or 0)
        provider.mtp_hybrid_override_pattern = _FLASH_MTP_PATTERN if num_mtp_layers else None
        provider.is_hybrid_model = True
        return provider

    @staticmethod
    def _validate_export_structure(
        provider: BailingMoe3HybridProvider,
        *,
        variant: str,
        mtp_patterns: tuple[str, ...],
    ) -> None:
        """Reject provider layouts that the public Ling HF format cannot represent."""
        expected_values = {
            "normalization": "RMSNorm",
            "gated_linear_unit": True,
            "add_bias_linear": False,
            "add_qkv_bias": False,
            "share_embeddings_and_output_weights": False,
            "position_embedding_type": "rope",
            "qk_layernorm": True,
            "attention_output_gate": True,
            "gated_attention_proj_granularity": "headwise",
            "rotary_percent": 1.0,
            "rotary_interleaved": False,
            "rope_type": "rope",
            "moe_grouped_gemm": True,
            "moe_router_score_function": "sigmoid",
            "moe_router_dtype": "fp32",
            "moe_router_load_balancing_type": "none",
            "moe_router_enable_expert_bias": True,
            "mtp_use_repeated_layer": False,
        }
        for name, expected in expected_values.items():
            actual = getattr(provider, name)
            if actual != expected:
                raise ValueError(f"Ling 3.0 export requires {name}={expected!r}, got {actual!r}.")

        expected_mtp_pattern = None if variant == "Tiny" else _FLASH_MTP_PATTERN
        expected_mtp_patterns = () if expected_mtp_pattern is None else (expected_mtp_pattern,)
        if provider.mtp_hybrid_override_pattern != expected_mtp_pattern:
            raise ValueError(
                "Ling 3.0 "
                f"{variant} export requires mtp_hybrid_override_pattern={expected_mtp_pattern!r}, "
                f"got {provider.mtp_hybrid_override_pattern!r}."
            )
        if mtp_patterns and mtp_patterns != expected_mtp_patterns:
            raise ValueError(f"Ling 3.0 {variant} export does not support MTP pattern suffixes {mtp_patterns!r}.")

        if variant == "Tiny" and provider.q_lora_rank is None:
            raise ValueError("Ling 3.0 Tiny export requires low-rank-Q MLA with q_lora_rank set.")
        if variant == "Flash" and provider.q_lora_rank is not None:
            raise ValueError("Ling 3.0 Flash export requires direct-Q MLA with q_lora_rank=None.")

        linear_head_dims = (provider.linear_key_head_dim, provider.linear_value_head_dim)
        if linear_head_dims != (provider.kv_channels, provider.kv_channels):
            raise ValueError(
                "Ling 3.0 export requires KDA key/value head dimensions to match kv_channels; "
                f"got {linear_head_dims=}, kv_channels={provider.kv_channels}."
            )
        if provider.linear_num_key_heads != provider.linear_num_value_heads:
            raise ValueError(
                "Ling 3.0 export requires equal KDA key/value head counts; "
                f"got key={provider.linear_num_key_heads}, value={provider.linear_num_value_heads}."
            )
        if not provider.moe_shared_expert_intermediate_size:
            raise ValueError("Ling 3.0 export requires one non-empty shared expert.")

    @classmethod
    def megatron_to_hf_config(cls, provider: BailingMoe3HybridProvider) -> dict[str, Any]:
        """Reconstruct the public logical Ling config from a physical Hybrid pattern."""
        hf_config = super().megatron_to_hf_config(provider)
        if provider.hybrid_layer_pattern is None:
            raise ValueError("Ling 3.0 provider is missing hybrid_layer_pattern.")
        pattern_parts = provider.hybrid_layer_pattern.split("/")
        main_pattern = pattern_parts[0]
        if main_pattern == _TINY_PATTERN:
            variant = "Tiny"
            num_hidden_layers = _TINY_NUM_LOGICAL_LAYERS
            layer_group_size = _TINY_GROUP_SIZE
            first_k_dense_replace = 1
            expected_mtp_layers = 0
        elif main_pattern == _FLASH_PATTERN:
            variant = "Flash"
            num_hidden_layers = _FLASH_NUM_LOGICAL_LAYERS
            layer_group_size = _FLASH_GROUP_SIZE
            first_k_dense_replace = 2
            expected_mtp_layers = 1
        else:
            raise ValueError(f"Unsupported Ling 3.0 Hybrid pattern: {main_pattern!r}.")

        actual_mtp_layers = int(provider.mtp_num_layers or 0)
        if actual_mtp_layers != expected_mtp_layers:
            raise ValueError(
                f"Ling 3.0 {variant} export requires mtp_num_layers={expected_mtp_layers}, got {actual_mtp_layers}."
            )

        cls._validate_export_structure(provider, variant=variant, mtp_patterns=tuple(pattern_parts[1:]))

        num_kv_heads_for_linear_attn = (
            0 if provider.linear_num_key_heads == provider.num_attention_heads else provider.linear_num_key_heads
        )
        hf_config.update(
            num_hidden_layers=num_hidden_layers,
            layer_group_size=layer_group_size,
            first_k_dense_replace=first_k_dense_replace,
            num_shared_experts=1,
            qk_head_dim=provider.qk_head_dim + provider.qk_pos_emb_head_dim,
            num_kv_heads_for_linear_attn=num_kv_heads_for_linear_attn,
            gated_attention_proj_granularity_type="head_wise",
            use_kda_lora=False,
            no_kda_lora=True,
            score_function="sigmoid",
            norm_topk_prob=True,
            topk_method="noaux_tc",
            partial_rotary_factor=0.5,
            rotary_dim=provider.qk_pos_emb_head_dim,
            rope_interleave=True,
            mtp_use_kda=False,
        )
        return hf_config

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
                            AutoMapping(
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
                        AutoMapping(
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
