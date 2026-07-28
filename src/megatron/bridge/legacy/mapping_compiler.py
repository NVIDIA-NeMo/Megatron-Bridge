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

"""Compile declarative legacy weight maps into current mapping primitives."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from string import Formatter

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    FusedExpertMapping,
    FusedGatedExpertMapping,
    GatedMLPMapping,
    MegatronParamMapping,
    QKVMapping,
)


_LAYER_FIELD = "layer_number"
_EXPERT_FIELD = "expert_id"
_ALLOWED_FIELDS = frozenset({_LAYER_FIELD, _EXPERT_FIELD})
_GROUPED_EXPERT_RE = re.compile(r"(?P<prefix>.*\.mlp\.experts)\.linear_fc(?P<projection>[12])$")


def compile_legacy_mapping_registry(
    *,
    direct_mapping: Mapping[str, str],
    attention_mapping: Mapping[str, Sequence[str]],
    mlp_mapping: Mapping[str, Sequence[str]],
) -> MegatronMappingRegistry:
    """Compile legacy mapping dictionaries into a current mapping registry.

    Normalization rules:

    - Direct names remain exact and cannot contain legacy placeholders.
    - Attention and MLP keys are layer-local suffixes prefixed with
      ``decoder.layers.*.``.
    - ``{layer_number}`` becomes the layer wildcard ``*``.
    - A bare layer-local ``...layernorm`` key gains its parameter suffix
      ``.weight``.
    - Grouped expert ``linear_fc1``/``linear_fc2`` keys become both TE grouped
      ``weight*`` and sequential ``local_experts.*`` Megatron patterns.
    - ``{expert_id}`` becomes the HF expert wildcard ``*``.
    - One-to-one entries use ``AutoMapping``; three-way attention entries use
      ``QKVMapping``; two-way MLP entries use ``GatedMLPMapping``; expert
      entries without an HF expert placeholder use fused expert primitives.

    Args:
        direct_mapping: Exact Megatron-to-HF parameter names.
        attention_mapping: Layer-local attention declarations.
        mlp_mapping: Layer-local MLP and MoE declarations.

    Returns:
        A current Megatron mapping registry.

    Raises:
        ValueError: If a declaration is ambiguous, has invalid placeholders,
            or normalizes to a duplicate Megatron pattern.
        TypeError: If a dictionary has the wrong value shape.
    """
    compiled: list[MegatronParamMapping] = []
    seen_megatron_patterns: set[str] = set()

    for megatron_param, hf_param in direct_mapping.items():
        _validate_clean_name(megatron_param, label="direct Megatron parameter")
        _validate_clean_name(hf_param, label=f"direct HF target for {megatron_param!r}")
        _validate_no_fields(megatron_param, label="direct Megatron parameter")
        _validate_no_fields(hf_param, label=f"direct HF target for {megatron_param!r}")
        _append_unique(
            compiled,
            seen_megatron_patterns,
            AutoMapping(megatron_param=megatron_param, hf_param=hf_param),
        )

    for legacy_key, targets in attention_mapping.items():
        for mapping in _compile_layer_entry("attention", legacy_key, targets):
            _append_unique(compiled, seen_megatron_patterns, mapping)

    for legacy_key, targets in mlp_mapping.items():
        for mapping in _compile_layer_entry("mlp", legacy_key, targets):
            _append_unique(compiled, seen_megatron_patterns, mapping)

    return MegatronMappingRegistry(*compiled)


def _compile_layer_entry(
    category: str,
    legacy_key: str,
    targets: Sequence[str],
) -> list[MegatronParamMapping]:
    if isinstance(targets, (str, bytes)) or not isinstance(targets, Sequence):
        raise TypeError(f"{category} mapping {legacy_key!r} must contain a sequence of HF targets.")
    if not targets:
        raise ValueError(f"{category} mapping {legacy_key!r} must contain at least one HF target.")
    if any(not isinstance(target, str) for target in targets):
        raise TypeError(f"{category} mapping {legacy_key!r} contains a non-string HF target.")

    megatron_patterns, projection = _normalize_megatron_patterns(legacy_key)
    hf_patterns = [_normalize_hf_pattern(target, legacy_key=legacy_key) for target in targets]
    expert_fields = [_EXPERT_FIELD in _field_names(target) for target in targets]
    is_expert = projection is not None

    if any(expert_fields) and not is_expert:
        raise ValueError(
            f"{category} mapping {legacy_key!r} uses {{{_EXPERT_FIELD}}} but its Megatron key is not an expert "
            "linear_fc1/linear_fc2 declaration."
        )
    if is_expert and any(expert_fields) and not all(expert_fields):
        raise ValueError(
            f"{category} mapping {legacy_key!r} mixes expert-specific and fused HF targets; "
            f"either every target must use {{{_EXPERT_FIELD}}} or none may use it."
        )

    if category == "attention":
        if is_expert:
            raise ValueError(f"attention mapping {legacy_key!r} cannot declare expert projections.")
        if len(hf_patterns) == 1:
            return [AutoMapping(megatron_patterns[0], hf_patterns[0])]
        if len(hf_patterns) == 3:
            roles = _resolve_roles(
                legacy_key=legacy_key,
                category=category,
                patterns=hf_patterns,
                role_tokens={"q": ".q_proj.", "k": ".k_proj.", "v": ".v_proj."},
            )
            return [
                QKVMapping(
                    megatron_param=megatron_patterns[0],
                    q=roles["q"],
                    k=roles["k"],
                    v=roles["v"],
                )
            ]
        raise ValueError(
            f"Ambiguous attention mapping {legacy_key!r}: expected one direct target or three Q/K/V targets, "
            f"got {len(hf_patterns)}."
        )

    if category != "mlp":
        raise ValueError(f"Unsupported legacy mapping category: {category!r}.")

    if len(hf_patterns) == 1:
        if is_expert and not expert_fields[0]:
            mapping_type = FusedGatedExpertMapping if projection == "1" else FusedExpertMapping
            return [mapping_type(megatron_param=pattern, hf_param=hf_patterns[0]) for pattern in megatron_patterns]
        return [AutoMapping(pattern, hf_patterns[0]) for pattern in megatron_patterns]

    if len(hf_patterns) == 2:
        if is_expert and not all(expert_fields):
            raise ValueError(
                f"Ambiguous MLP mapping {legacy_key!r}: two fused HF expert tensors cannot be represented by one "
                "current gated-expert primitive."
            )
        roles = _resolve_roles(
            legacy_key=legacy_key,
            category=category,
            patterns=hf_patterns,
            role_tokens={"gate": ".gate_proj.", "up": ".up_proj."},
        )
        return [
            GatedMLPMapping(megatron_param=pattern, gate=roles["gate"], up=roles["up"])
            for pattern in megatron_patterns
        ]

    raise ValueError(
        f"Ambiguous MLP mapping {legacy_key!r}: expected one direct target or two gate/up targets, "
        f"got {len(hf_patterns)}."
    )


def _normalize_megatron_patterns(legacy_key: str) -> tuple[list[str], str | None]:
    _validate_clean_name(legacy_key, label="layer-local Megatron key")
    _validate_no_fields(legacy_key, label="layer-local Megatron key")
    if legacy_key.startswith("decoder.layers."):
        raise ValueError(
            f"Layer-local Megatron key {legacy_key!r} must be a suffix, not a pre-normalized decoder path."
        )
    if "*" in legacy_key:
        raise ValueError(
            f"Layer-local Megatron key {legacy_key!r} must not contain wildcards; "
            "the compiler adds layer and expert wildcards."
        )

    normalized_key = legacy_key
    if normalized_key.endswith("layernorm"):
        normalized_key = f"{normalized_key}.weight"

    grouped_expert_match = _GROUPED_EXPERT_RE.fullmatch(f"decoder.layers.*.{normalized_key}")
    if grouped_expert_match is None:
        return [f"decoder.layers.*.{normalized_key}"], None

    projection = grouped_expert_match.group("projection")
    grouped_prefix = grouped_expert_match.group("prefix")
    grouped = f"{grouped_prefix}.linear_fc{projection}.weight*"
    sequential = f"{grouped_prefix}.local_experts.*.linear_fc{projection}.weight"
    return [grouped, sequential], projection


def _normalize_hf_pattern(target: str, *, legacy_key: str) -> str:
    _validate_clean_name(target, label=f"HF target for {legacy_key!r}")
    fields = _field_names(target)
    unknown_fields = sorted(fields - _ALLOWED_FIELDS)
    if unknown_fields:
        raise ValueError(
            f"HF target {target!r} for {legacy_key!r} has unsupported placeholder(s): {unknown_fields}. "
            f"Only {{{_LAYER_FIELD}}} and {{{_EXPERT_FIELD}}} are supported."
        )
    if target.count(f"{{{_LAYER_FIELD}}}") != 1:
        raise ValueError(
            f"HF target {target!r} for {legacy_key!r} must contain exactly one {{{_LAYER_FIELD}}} placeholder."
        )
    if target.count(f"{{{_EXPERT_FIELD}}}") > 1:
        raise ValueError(
            f"HF target {target!r} for {legacy_key!r} contains more than one {{{_EXPERT_FIELD}}} placeholder."
        )
    return target.replace(f"{{{_LAYER_FIELD}}}", "*").replace(f"{{{_EXPERT_FIELD}}}", "*")


def _resolve_roles(
    *,
    legacy_key: str,
    category: str,
    patterns: Sequence[str],
    role_tokens: Mapping[str, str],
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for pattern in patterns:
        matching_roles = [role for role, token in role_tokens.items() if token in pattern]
        if len(matching_roles) != 1:
            raise ValueError(
                f"Ambiguous {category} mapping {legacy_key!r}: target {pattern!r} must identify exactly one of "
                f"{sorted(role_tokens)}."
            )
        role = matching_roles[0]
        if role in resolved:
            raise ValueError(f"Ambiguous {category} mapping {legacy_key!r}: role {role!r} is declared more than once.")
        resolved[role] = pattern

    missing_roles = sorted(set(role_tokens) - set(resolved))
    if missing_roles:
        raise ValueError(f"Ambiguous {category} mapping {legacy_key!r}: missing role(s) {missing_roles}.")
    return resolved


def _field_names(value: str) -> set[str]:
    try:
        return {field_name for _, field_name, _, _ in Formatter().parse(value) if field_name is not None}
    except ValueError as error:
        raise ValueError(f"Invalid legacy placeholder syntax in {value!r}: {error}") from error


def _validate_no_fields(value: str, *, label: str) -> None:
    fields = sorted(_field_names(value))
    if fields:
        raise ValueError(f"{label} {value!r} cannot contain placeholder(s): {fields}.")


def _validate_clean_name(value: object, *, label: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string, got {type(value).__name__}.")
    if not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty name without surrounding whitespace, got {value!r}.")


def _append_unique(
    compiled: list[MegatronParamMapping],
    seen_megatron_patterns: set[str],
    mapping: MegatronParamMapping,
) -> None:
    if mapping.megatron_param in seen_megatron_patterns:
        raise ValueError(
            f"Ambiguous legacy declarations normalize to duplicate Megatron pattern {mapping.megatron_param!r}."
        )
    seen_megatron_patterns.add(mapping.megatron_param)
    compiled.append(mapping)
