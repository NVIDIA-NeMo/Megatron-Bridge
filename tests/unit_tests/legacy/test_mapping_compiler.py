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

"""Normalization, ambiguity, and Qwen3 MoE parity tests for legacy maps."""

from collections.abc import Mapping
from typing import Any

import pytest

from megatron.bridge.legacy.mapping_compiler import compile_legacy_mapping_registry
from megatron.bridge.legacy.qwen3_moe import (
    HF_ARCHITECTURE,
    HF_MODEL_ID,
    HF_REVISION,
    MIN_TRANSFORMERS_VERSION,
    Qwen3MoELegacyMapping,
)
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    FusedExpertMapping,
    FusedGatedExpertMapping,
    GatedMLPMapping,
    MegatronParamMapping,
    QKVMapping,
)
from megatron.bridge.models.qwen.qwen3_moe_bridge import Qwen3MoEBridge


def _mapping_by_pattern(
    registry: MegatronMappingRegistry,
    pattern: str,
) -> MegatronParamMapping[Any]:
    matches = [mapping for mapping in registry.mappings if mapping.megatron_param == pattern]
    assert len(matches) == 1, f"Expected exactly one mapping for {pattern!r}, got {matches!r}."
    return matches[0]


def _assert_same_mapping_semantics(
    compiled: MegatronMappingRegistry,
    production: MegatronMappingRegistry,
    pattern: str,
) -> None:
    compiled_mapping = _mapping_by_pattern(compiled, pattern)
    production_mapping = _mapping_by_pattern(production, pattern)

    assert type(compiled_mapping) is type(production_mapping)
    assert compiled_mapping.hf_param == production_mapping.hf_param


def test_dense_declarations_show_all_core_normalization_rules() -> None:
    """Normalize exact, layer, layernorm, QKV, and gated-MLP declarations."""
    registry = compile_legacy_mapping_registry(
        direct_mapping={
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        },
        attention_mapping={
            "self_attention.linear_proj.weight": ("model.layers.{layer_number}.self_attn.o_proj.weight",),
            "self_attention.linear_qkv.weight": (
                "model.layers.{layer_number}.self_attn.v_proj.weight",
                "model.layers.{layer_number}.self_attn.q_proj.weight",
                "model.layers.{layer_number}.self_attn.k_proj.weight",
            ),
        },
        mlp_mapping={
            "pre_mlp_layernorm": ("model.layers.{layer_number}.post_attention_layernorm.weight",),
            "mlp.linear_fc1.weight": (
                "model.layers.{layer_number}.mlp.up_proj.weight",
                "model.layers.{layer_number}.mlp.gate_proj.weight",
            ),
        },
    )

    direct = _mapping_by_pattern(registry, "embedding.word_embeddings.weight")
    projection = _mapping_by_pattern(registry, "decoder.layers.*.self_attention.linear_proj.weight")
    qkv = _mapping_by_pattern(registry, "decoder.layers.*.self_attention.linear_qkv.weight")
    layernorm = _mapping_by_pattern(registry, "decoder.layers.*.pre_mlp_layernorm.weight")
    gated_mlp = _mapping_by_pattern(registry, "decoder.layers.*.mlp.linear_fc1.weight")

    assert isinstance(direct, AutoMapping)
    assert direct.hf_param == "model.embed_tokens.weight"
    assert isinstance(projection, AutoMapping)
    assert projection.hf_param == "model.layers.*.self_attn.o_proj.weight"
    assert isinstance(qkv, QKVMapping)
    assert qkv.hf_param == {
        "q": "model.layers.*.self_attn.q_proj.weight",
        "k": "model.layers.*.self_attn.k_proj.weight",
        "v": "model.layers.*.self_attn.v_proj.weight",
    }
    assert isinstance(layernorm, AutoMapping)
    assert layernorm.hf_param == "model.layers.*.post_attention_layernorm.weight"
    assert isinstance(gated_mlp, GatedMLPMapping)
    assert gated_mlp.hf_param == {
        "gate": "model.layers.*.mlp.gate_proj.weight",
        "up": "model.layers.*.mlp.up_proj.weight",
    }


def test_moe_expert_placeholder_compiles_grouped_and_sequential_patterns() -> None:
    """Expand a single legacy expert declaration to both current MoE layouts."""
    registry = compile_legacy_mapping_registry(
        direct_mapping={},
        attention_mapping={},
        mlp_mapping={
            "mlp.experts.linear_fc1": (
                "model.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.weight",
                "model.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.weight",
            ),
            "mlp.experts.linear_fc2": ("model.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.weight",),
        },
    )

    grouped_fc1 = _mapping_by_pattern(registry, "decoder.layers.*.mlp.experts.linear_fc1.weight*")
    sequential_fc1 = _mapping_by_pattern(
        registry,
        "decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
    )
    grouped_fc2 = _mapping_by_pattern(registry, "decoder.layers.*.mlp.experts.linear_fc2.weight*")
    sequential_fc2 = _mapping_by_pattern(
        registry,
        "decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",
    )

    for mapping in (grouped_fc1, sequential_fc1):
        assert isinstance(mapping, GatedMLPMapping)
        assert mapping.hf_param == {
            "gate": "model.layers.*.mlp.experts.*.gate_proj.weight",
            "up": "model.layers.*.mlp.experts.*.up_proj.weight",
        }
    for mapping in (grouped_fc2, sequential_fc2):
        assert isinstance(mapping, AutoMapping)
        assert mapping.hf_param == "model.layers.*.mlp.experts.*.down_proj.weight"


def test_packed_hf_experts_use_fused_current_primitives() -> None:
    """Compile packed HF expert tensors to fused expert primitives."""
    registry = compile_legacy_mapping_registry(
        direct_mapping={},
        attention_mapping={},
        mlp_mapping={
            "mlp.experts.linear_fc1": ("model.layers.{layer_number}.mlp.experts.gate_up_proj",),
            "mlp.experts.linear_fc2": ("model.layers.{layer_number}.mlp.experts.down_proj",),
        },
    )

    for pattern in (
        "decoder.layers.*.mlp.experts.linear_fc1.weight*",
        "decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
    ):
        assert isinstance(_mapping_by_pattern(registry, pattern), FusedGatedExpertMapping)
    for pattern in (
        "decoder.layers.*.mlp.experts.linear_fc2.weight*",
        "decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",
    ):
        assert isinstance(_mapping_by_pattern(registry, pattern), FusedExpertMapping)


@pytest.mark.parametrize(
    ("category", "key", "targets", "message"),
    (
        (
            "attention",
            "self_attention.linear_qkv.weight",
            (
                "model.layers.{layer_number}.self_attn.q_proj.weight",
                "model.layers.{layer_number}.self_attn.k_proj.weight",
            ),
            "expected one direct target or three Q/K/V targets",
        ),
        (
            "attention",
            "self_attention.linear_qkv.weight",
            (
                "model.layers.{layer_number}.self_attn.q_proj.weight",
                "model.layers.{layer_number}.self_attn.q_proj.bias",
                "model.layers.{layer_number}.self_attn.v_proj.weight",
            ),
            "role 'q' is declared more than once",
        ),
        (
            "mlp",
            "mlp.linear_fc1.weight",
            (
                "model.layers.{layer_number}.mlp.gate_proj.weight",
                "model.layers.{layer_number}.mlp.gate_proj.bias",
            ),
            "role 'gate' is declared more than once",
        ),
        (
            "mlp",
            "mlp.experts.linear_fc1",
            (
                "model.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.weight",
                "model.layers.{layer_number}.mlp.experts.up_proj.weight",
            ),
            "mixes expert-specific and fused HF targets",
        ),
        (
            "mlp",
            "mlp.linear_fc1.weight",
            (
                "model.layers.{layer_number}.mlp.proj_a.weight",
                "model.layers.{layer_number}.mlp.proj_b.weight",
            ),
            "must identify exactly one",
        ),
    ),
)
def test_ambiguous_multi_target_declarations_fail_explicitly(
    category: str,
    key: str,
    targets: tuple[str, ...],
    message: str,
) -> None:
    """Make each unsupported grouping fail instead of guessing semantics."""
    attention_mapping = {key: targets} if category == "attention" else {}
    mlp_mapping = {key: targets} if category == "mlp" else {}

    with pytest.raises(ValueError, match=message):
        compile_legacy_mapping_registry(
            direct_mapping={},
            attention_mapping=attention_mapping,
            mlp_mapping=mlp_mapping,
        )


@pytest.mark.parametrize(
    ("direct_mapping", "attention_mapping", "mlp_mapping", "message"),
    (
        (
            {"decoder.{layer_number}.weight": "model.weight"},
            {},
            {},
            "cannot contain placeholder",
        ),
        (
            {},
            {
                "decoder.layers.*.self_attention.linear_proj.weight": (
                    "model.layers.{layer_number}.self_attn.o_proj.weight",
                )
            },
            {},
            "must be a suffix",
        ),
        (
            {},
            {"self_attention.linear_proj.weight": ("model.layers.{layer_number}.blocks.{block_number}.weight",)},
            {},
            "unsupported placeholder",
        ),
        (
            {},
            {"self_attention.linear_proj.weight": ("model.layers.0.self_attn.o_proj.weight",)},
            {},
            "must contain exactly one",
        ),
        (
            {},
            {"same.weight": ("model.layers.{layer_number}.attention.weight",)},
            {"same.weight": ("model.layers.{layer_number}.mlp.weight",)},
            "duplicate Megatron pattern",
        ),
    ),
)
def test_invalid_normalization_contracts_fail_explicitly(
    direct_mapping: Mapping[str, str],
    attention_mapping: Mapping[str, tuple[str, ...]],
    mlp_mapping: Mapping[str, tuple[str, ...]],
    message: str,
) -> None:
    """Reject placeholders, normalized input, and duplicate normalized output."""
    with pytest.raises(ValueError, match=message):
        compile_legacy_mapping_registry(
            direct_mapping=direct_mapping,
            attention_mapping=attention_mapping,
            mlp_mapping=mlp_mapping,
        )


def test_string_layer_targets_are_rejected_as_wrong_shape() -> None:
    """Do not treat a string as a sequence of one-character targets."""
    with pytest.raises(TypeError, match="sequence of HF targets"):
        compile_legacy_mapping_registry(
            direct_mapping={},
            attention_mapping={"self_attention.linear_proj.weight": "model.layers.{layer_number}.weight"},
            mlp_mapping={},
        )


def test_qwen3_moe_inherits_legacy_qwen2_moe_declarations() -> None:
    """Keep Qwen3 MoE as a declaration-only inheritance migration proof."""
    assert "_DIRECT_MAPPING" not in Qwen3MoELegacyMapping.__dict__
    assert "_ATTENTION_MAPPING" not in Qwen3MoELegacyMapping.__dict__
    assert "_MLP_MAPPING" not in Qwen3MoELegacyMapping.__dict__


def test_qwen3_moe_proof_metadata_is_exactly_pinned() -> None:
    """Pin the migrated proof independently of historical card evidence."""
    assert HF_MODEL_ID == "Qwen/Qwen3-30B-A3B"
    assert HF_REVISION == "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"  # pragma: allowlist secret
    assert HF_ARCHITECTURE == "Qwen3MoeForCausalLM"
    assert MIN_TRANSFORMERS_VERSION == "5.8.1"


def test_qwen3_moe_stale_inherited_entries_are_compiler_coverage_only() -> None:
    """Exclude inherited QKV-bias/shared-expert entries from parity claims."""
    assert "self_attention.linear_qkv.bias" in Qwen3MoELegacyMapping._ATTENTION_MAPPING
    assert {
        "shared_experts.linear_fc1.weight",
        "shared_experts.linear_fc2.weight",
        "shared_experts.gate_weight",
    } < set(Qwen3MoELegacyMapping._MLP_MAPPING)

    compiled_patterns = {mapping.megatron_param for mapping in Qwen3MoELegacyMapping.mapping_registry().mappings}
    production_patterns = {mapping.megatron_param for mapping in Qwen3MoEBridge().mapping_registry().mappings}
    stale_patterns = {
        "decoder.layers.*.self_attention.linear_qkv.bias",
        "decoder.layers.*.shared_experts.linear_fc1.weight",
        "decoder.layers.*.shared_experts.linear_fc2.weight",
        "decoder.layers.*.shared_experts.gate_weight",
    }

    assert stale_patterns <= compiled_patterns
    assert stale_patterns.isdisjoint(production_patterns)


@pytest.mark.parametrize(
    "pattern",
    (
        "decoder.layers.*.self_attention.linear_qkv.weight",
        "decoder.layers.*.mlp.router.weight",
        "decoder.layers.*.mlp.experts.linear_fc1.weight*",
        "decoder.layers.*.mlp.experts.linear_fc2.weight*",
        "decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
        "decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",
    ),
)
def test_qwen3_moe_compiler_matches_selected_production_parameter_semantics(pattern: str) -> None:
    """Prove parity only for actual Qwen3 parameters in the selected set."""
    compiled = Qwen3MoELegacyMapping.mapping_registry()
    production = Qwen3MoEBridge().mapping_registry()

    _assert_same_mapping_semantics(compiled, production, pattern)
