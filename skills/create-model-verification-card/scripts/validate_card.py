#!/usr/bin/env python3
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

"""Validate model verification card structure, claims, and privacy boundaries."""

from __future__ import annotations

import argparse
import datetime as dt
import ipaddress
import logging
import math
import re
import shlex
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TypeGuard

import yaml
from yaml.tokens import AliasToken, AnchorToken, TagToken


LOG = logging.getLogger(__name__)

STATUSES = frozenset({"unverified", "verified", "unsupported", "not_applicable"})
PRECISIONS = frozenset({"bf16", "fp8_mx", "nvfp4"})
TRAINING_ONLY_PRECISIONS = PRECISIONS - {"bf16"}
REQUIRED_ITEM_NAMES = (
    "hf_to_megatron_cpu",
    "hf_to_megatron_gpu",
    "megatron_to_hf_cpu",
    "megatron_to_hf_gpu",
    "manual_forward_pass",
    "inference",
    "pretrain",
    "sft",
    "sft_export_inference",
    "sft_long_context",
    "peft",
    "checkpoint_resume",
)
OPTIONAL_ITEM_NAMES = ("pretrain_performance",)
ITEM_NAMES = REQUIRED_ITEM_NAMES + OPTIONAL_ITEM_NAMES
MODEL_LEVEL_INDEX_SCOPE = (
    "hf_to_megatron_cpu",
    "hf_to_megatron_gpu",
    "megatron_to_hf_cpu",
    "megatron_to_hf_gpu",
    "manual_forward_pass",
    "inference",
)
TRAINING_INDEX_SCOPE = (
    "pretrain",
    "sft",
    "sft_export_inference",
    "sft_long_context",
    "peft",
    "checkpoint_resume",
)
TRAINING_ITEMS = frozenset(
    {"pretrain", "sft", "sft_long_context", "peft", "checkpoint_resume", "pretrain_performance"}
)
HARDWARE_SCOPED_ITEMS = TRAINING_ITEMS | {"sft_export_inference"}
PUBLIC_HARDWARE_KEYS = frozenset(
    {
        "A100",
        "A800",
        "B100",
        "B200",
        "B300",
        "GB200",
        "GB300",
        "H20",
        "H100",
        "H200",
        "H800",
        "R100",
    }
)
CONVERSION_ITEMS = frozenset({"hf_to_megatron_cpu", "hf_to_megatron_gpu", "megatron_to_hf_cpu", "megatron_to_hf_gpu"})
FEATURE_ITEMS = frozenset({"pretrain", "sft", "sft_long_context", "peft"})
METRIC_NAMES = frozenset(
    {
        "initial_loss",
        "final_loss",
        "last_10_steps_step_time_ms_avg",
        "last_10_steps_model_tflops_per_gpu_avg",
    }
)
FEATURE_KEYS = frozenset({"sequence_packing", "cuda_graph", "context_parallel_size", "moe_dispatcher"})
PACKING_VALUES = frozenset({"offline", "in_batch"})
CUDA_GRAPH_IMPLEMENTATIONS = frozenset({"local", "transformer_engine"})
CUDA_GRAPH_SCOPES = frozenset({"full_iteration", "attn", "mlp", "moe", "moe_router", "moe_preprocess", "mamba"})
MOE_DISPATCHERS = frozenset({"deepep", "hybridep"})
MANUAL_FORWARD_COSINE_THRESHOLD = 0.99
MANUAL_FORWARD_REVISION_PINNING_DATE = dt.date(2026, 7, 20)
UNTUNED_PERFORMANCE_DISCLAIMER = (
    "Performance disclaimer: this model has not been performance-tuned; "
    "reported timing and throughput metrics are sanity checks, not optimized performance results."
)

TOP_LEVEL_KEYS = frozenset({"title", "model", "verification_environment", "summary", "verification_index", "items"})
VERIFICATION_INDEX_KEYS = frozenset({"model_level", "training", "performance"})
MODEL_KEYS = frozenset({"hf_id", "hf_revision", "architecture", "min_transformers_version"})
ENVIRONMENT_KEYS = frozenset({"base_container", "bridge_commit"})
ITEM_KEYS = frozenset(
    {
        "status",
        "precision",
        "bridge_commit",
        "depends_on",
        "command",
        "commands",
        "last_verified",
        "expected_result",
        "enabled_features",
        "metrics",
        "resume_comparison",
    }
)
RESUME_KEYS = frozenset(
    {
        "reference_item",
        "sentinel_steps",
        "loss_relative_tolerance",
        "loss_absolute_tolerance",
        "sentinels_match",
    }
)

FORBIDDEN_KEY_FRAGMENTS = (
    "account",
    "cluster",
    "container",
    "evidence",
    "executor",
    "host",
    "job",
    "launcher",
    "mount",
    "partition",
)

PLACEHOLDER_RE = re.compile(r"\b(?:TODO|TBD|PLACEHOLDER)\b|<[^>]+>", re.IGNORECASE)
HF_ID_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
REVISION_RE = re.compile(r"[0-9a-f]{40}")
VERSION_RE = re.compile(r"\d+\.\d+\.\d+")
PUBLIC_BASE_CONTAINER_RE = re.compile(r"nvcr\.io/nvidia/(?:nemo|pytorch):[A-Za-z0-9._-]+")
URL_RE = re.compile(r"\b[a-z][a-z0-9+.-]*://\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
IPV4_RE = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
ABSOLUTE_PATH_RE = re.compile(r"(?<![A-Za-z0-9_$.])/(?:[A-Za-z0-9._${}-]+(?:/[A-Za-z0-9._${}-]+)*)")
ENVIRONMENT_REFERENCE_RE = re.compile(r"\$(?:\{)?([A-Za-z_][A-Za-z0-9_]*)(?:\})?")
REMOTE_COMMAND_RE = re.compile(r"(?:^|\s)(?:ssh|scp|sftp)(?:\s|$)", re.IGNORECASE)
REMOTE_COPY_RE = re.compile(r"(?:^|\s)[^\s/@:]+@[^\s/:]+:")
JOB_ID_RE = re.compile(r"\b(?:job(?:[_ -]?id)?|scheduler[_ -]?job)\s*[:=#]?\s*\d{4,}\b", re.IGNORECASE)
SECRET_ASSIGNMENT_RE = re.compile(r"\b(?:HF_TOKEN|ACCESS_TOKEN|API_KEY|PASSWORD|SECRET)\s*[:=]", re.IGNORECASE)
ENVIRONMENT_ASSIGNMENT_RE = re.compile(r"(?:^|[\s;&|])[A-Z_][A-Z0-9_]*=(?:\"[^\"]*\"|'[^']*'|[^\s]+)")
RUNTIME_COMMAND_RE = re.compile(
    r"(?:^|[;&|]\s*|\s)(?:srun|sbatch|docker|podman|apptainer|singularity)(?:\s|$)",
    re.IGNORECASE,
)
RUNTIME_ENTRYPOINT_RE = re.compile(r"scripts/training/setup_experiment\.py")
RUNTIME_FLAG_RE = re.compile(
    r"--(?:mount|env|export|account|partition|container(?:-image|-mounts)?|qos|reservation|nodelist|host|"
    r"srun-arg|ssh-tunnel|remote-job-dir|user|time|experiment-name)(?:=|\s|$)",
    re.IGNORECASE,
)
SHELL_SETUP_RE = re.compile(r"(?:^|[;&|]\s*|\s)(?:export\s+[A-Za-z_]|bash\s+-lc\b|cd\s+\S+)", re.IGNORECASE)
HOME_PATH_RE = re.compile(r"(?<![A-Za-z0-9_])~(?:[A-Za-z0-9_.-]+)?/")
REGISTRY_REFERENCE_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?::\d+)?/"
    r"[A-Za-z0-9._/-]+(?::[A-Za-z0-9._-]+)?"
)
SECRET_FLAG_RE = re.compile(r"--(?:api[-_]key|access[-_]token|hf[-_]token|password|secret)(?:=|\s+)", re.IGNORECASE)
IPV6_CANDIDATE_RE = re.compile(r"(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{0,4}:){2,}[0-9A-Fa-f]{0,4}(?![0-9A-Fa-f:])")


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, *, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def _pointer(*parts: str) -> str:
    if not parts:
        return "/"
    escaped = (part.replace("~", "~0").replace("/", "~1") for part in parts)
    return "/" + "/".join(escaped)


def _check_keys(
    value: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    required: frozenset[str],
    path: tuple[str, ...],
    errors: list[str],
) -> None:
    non_string_keys = sorted(repr(key) for key in value if not isinstance(key, str))
    for key in non_string_keys:
        errors.append(f"{_pointer(*path)}: non-string key {key} is not allowed")
    keys = {key for key in value if isinstance(key, str)}
    for key in sorted(keys - allowed):
        errors.append(f"{_pointer(*path, str(key))}: unknown key")
    for key in sorted(required - keys):
        errors.append(f"{_pointer(*path, key)}: required key is missing")


def _as_mapping(value: Any, *, path: tuple[str, ...], errors: list[str]) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{_pointer(*path)}: expected a mapping")
        return None
    if any(not isinstance(key, str) for key in value):
        errors.append(f"{_pointer(*path)}: mapping keys must be strings")
        return None
    return value


def _hardware_variants(
    value: Any,
    *,
    item_name: str,
    errors: list[str],
) -> dict[str, Mapping[str, Any]]:
    """Return hardware-keyed item leaves after validating the group shape."""
    path = ("items", item_name)
    variants = _as_mapping(value, path=path, errors=errors)
    if variants is None:
        return {}
    if any(key in ITEM_KEYS for key in variants):
        errors.append(f"{_pointer(*path)}: hardware-scoped items must be keyed by hardware, for example H100")
        return {}
    if not variants:
        errors.append(f"{_pointer(*path)}: expected at least one hardware entry")
        return {}

    result: dict[str, Mapping[str, Any]] = {}
    normalized_keys: set[str] = set()
    for hardware, leaf_value in variants.items():
        if not isinstance(hardware, str) or hardware not in PUBLIC_HARDWARE_KEYS | {"all"}:
            errors.append(
                f"{_pointer(*path, str(hardware))}: expected a supported public hardware key; "
                f"choose from {sorted(PUBLIC_HARDWARE_KEYS)} or all"
            )
            continue
        normalized = hardware.casefold()
        if normalized in normalized_keys:
            errors.append(f"{_pointer(*path, hardware)}: hardware keys must be case-insensitively unique")
            continue
        normalized_keys.add(normalized)
        leaf = _as_mapping(leaf_value, path=(*path, hardware), errors=errors)
        if leaf is not None:
            result[hardware] = leaf

    if "all" in variants:
        if len(variants) != 1:
            errors.append(f"{_pointer(*path, 'all')}: the global hardware key cannot be mixed with concrete hardware")
        all_leaf = variants.get("all")
        status = all_leaf.get("status") if isinstance(all_leaf, Mapping) else None
        if status not in {"unsupported", "not_applicable"}:
            errors.append(f"{_pointer(*path, 'all', 'status')}: all is reserved for global terminal limitations")
    return result


def _iter_item_leaves(
    items: Mapping[str, Any],
) -> Iterable[tuple[str, str | None, Mapping[str, Any], tuple[str, ...]]]:
    """Yield ordinary items and nested hardware leaves with their JSON-pointer paths."""
    for item_name, value in items.items():
        if not isinstance(value, Mapping):
            continue
        if item_name in HARDWARE_SCOPED_ITEMS and "status" not in value:
            for hardware, leaf in value.items():
                if isinstance(leaf, Mapping):
                    yield item_name, hardware, leaf, ("items", item_name, hardware)
        else:
            yield item_name, None, value, ("items", item_name)


def _item_status(item: Mapping[str, Any] | None) -> str | None:
    """Return a valid item status without duplicating detailed item errors."""
    if item is None:
        return None
    status = item.get("status")
    if isinstance(status, str) and status in STATUSES:
        return status
    return None


def _validate_index_status_buckets(
    value: Any,
    *,
    scope: tuple[str, ...],
    expected_statuses: Mapping[str, str],
    path: tuple[str, ...],
    errors: list[str],
) -> None:
    """Validate one status-bucket directory and compare it with detailed items."""
    buckets = _as_mapping(value, path=path, errors=errors)
    if buckets is None:
        return
    _check_keys(buckets, allowed=STATUSES, required=frozenset(), path=path, errors=errors)

    scope_names = frozenset(scope)
    assignments: dict[str, str] = {}
    assignment_paths: dict[str, tuple[str, ...]] = {}
    for status, members in buckets.items():
        if status not in STATUSES:
            continue
        bucket_path = (*path, status)
        if members == "all":
            errors.append(f"{_pointer(*bucket_path)}: list every item name explicitly; scalar all is not allowed")
            continue
        if not isinstance(members, list) or not members:
            errors.append(f"{_pointer(*bucket_path)}: expected a non-empty item list")
            continue
        for index, item_name in enumerate(members):
            item_path = (*bucket_path, str(index))
            if not isinstance(item_name, str) or item_name not in scope_names:
                errors.append(f"{_pointer(*item_path)}: expected one of {sorted(scope_names)}")
                continue
            if item_name in assignments:
                errors.append(f"{_pointer(*item_path)}: {item_name} must appear exactly once in this scope")
                continue
            assignments[item_name] = status
            assignment_paths[item_name] = bucket_path

    missing_items = [item_name for item_name in scope if item_name not in assignments]
    if missing_items:
        errors.append(f"{_pointer(*path)}: every scope item must appear exactly once; missing {missing_items}")

    for item_name, expected_status in expected_statuses.items():
        indexed_status = assignments.get(item_name)
        if indexed_status is not None and indexed_status != expected_status:
            errors.append(
                f"{_pointer(*assignment_paths[item_name])}: {item_name} is indexed as {indexed_status} "
                f"but detailed items project to {expected_status}"
            )


def _project_training_status(
    hardware_groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    item_name: str,
    hardware: str,
) -> str | None:
    """Project one functional training item onto an indexed hardware target."""
    variants = hardware_groups.get(item_name, {})
    concrete_status = _item_status(variants.get(hardware))
    if concrete_status is not None:
        return concrete_status
    if hardware in variants:
        return None
    terminal_status = _item_status(variants.get("all"))
    if terminal_status in {"unsupported", "not_applicable"}:
        return terminal_status
    return "unverified"


def _validate_verification_index(
    value: Any,
    *,
    items: Mapping[str, Any] | None,
    hardware_groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
    errors: list[str],
) -> None:
    """Validate the compact index as an exact projection of detailed items."""
    path = ("verification_index",)
    verification_index = _as_mapping(value, path=path, errors=errors)
    if verification_index is None:
        return
    _check_keys(
        verification_index,
        allowed=VERIFICATION_INDEX_KEYS,
        required=frozenset({"model_level", "training"}),
        path=path,
        errors=errors,
    )

    model_statuses: dict[str, str] = {}
    if items is not None:
        for item_name in MODEL_LEVEL_INDEX_SCOPE:
            item = items.get(item_name)
            status = _item_status(item if isinstance(item, Mapping) else None)
            if status is not None:
                model_statuses[item_name] = status
    _validate_index_status_buckets(
        verification_index.get("model_level"),
        scope=MODEL_LEVEL_INDEX_SCOPE,
        expected_statuses=model_statuses,
        path=(*path, "model_level"),
        errors=errors,
    )

    training_path = (*path, "training")
    training = _as_mapping(verification_index.get("training"), path=training_path, errors=errors)
    concrete_training_hardware = {
        hardware
        for item_name in TRAINING_INDEX_SCOPE
        for hardware in hardware_groups.get(item_name, {})
        if hardware != "all"
    }
    if training is not None:
        if not training:
            errors.append(f"{_pointer(*training_path)}: expected at least one public hardware target")
        valid_training_hardware: set[str] = set()
        for hardware in training:
            if hardware not in PUBLIC_HARDWARE_KEYS:
                errors.append(
                    f"{_pointer(*training_path, str(hardware))}: expected a supported public hardware key; "
                    f"choose from {sorted(PUBLIC_HARDWARE_KEYS)}"
                )
                continue
            valid_training_hardware.add(hardware)
        for hardware in sorted(concrete_training_hardware - valid_training_hardware):
            errors.append(
                f"{_pointer(*training_path, hardware)}: required because detailed training items use this hardware"
            )
        for hardware in sorted(valid_training_hardware):
            expected_statuses = {
                item_name: status
                for item_name in TRAINING_INDEX_SCOPE
                if (
                    status := _project_training_status(
                        hardware_groups,
                        item_name=item_name,
                        hardware=hardware,
                    )
                )
                is not None
            }
            _validate_index_status_buckets(
                training[hardware],
                scope=TRAINING_INDEX_SCOPE,
                expected_statuses=expected_statuses,
                path=(*training_path, hardware),
                errors=errors,
            )

    performance_path = (*path, "performance")
    performance_variants = {
        hardware: item
        for hardware, item in hardware_groups.get("pretrain_performance", {}).items()
        if hardware != "all"
    }
    if not performance_variants:
        if "performance" in verification_index:
            errors.append(
                f"{_pointer(*performance_path)}: omit performance when pretrain_performance has no concrete leaves"
            )
        return
    if "performance" not in verification_index:
        errors.append(f"{_pointer(*performance_path)}: required to mirror pretrain_performance concrete leaves")
        return

    performance = _as_mapping(verification_index.get("performance"), path=performance_path, errors=errors)
    if performance is None:
        return
    expected_hardware = set(performance_variants)
    actual_hardware = set(performance)
    for hardware in sorted(actual_hardware):
        if hardware not in PUBLIC_HARDWARE_KEYS:
            errors.append(
                f"{_pointer(*performance_path, str(hardware))}: expected a supported public hardware key; "
                f"choose from {sorted(PUBLIC_HARDWARE_KEYS)}"
            )
    for hardware in sorted(expected_hardware - actual_hardware):
        errors.append(f"{_pointer(*performance_path, hardware)}: required to mirror pretrain_performance.{hardware}")
    for hardware in sorted(actual_hardware - expected_hardware):
        errors.append(f"{_pointer(*performance_path, hardware)}: no matching pretrain_performance.{hardware} leaf")
    for hardware in sorted(expected_hardware & actual_hardware):
        indexed_status = performance.get(hardware)
        if not isinstance(indexed_status, str) or indexed_status not in STATUSES:
            errors.append(f"{_pointer(*performance_path, hardware)}: expected one of {sorted(STATUSES)}")
            continue
        expected_status = _item_status(performance_variants[hardware])
        if expected_status is not None and indexed_status != expected_status:
            errors.append(
                f"{_pointer(*performance_path, hardware)}: indexed as {indexed_status} but "
                f"pretrain_performance.{hardware} is {expected_status}"
            )


def _is_iso_date(value: Any) -> bool:
    if isinstance(value, dt.datetime):
        return False
    if isinstance(value, dt.date):
        return True
    if isinstance(value, str):
        try:
            dt.date.fromisoformat(value)
        except ValueError:
            return False
        return True
    return False


def _is_finite_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value))


def _contains_ipv6(value: str) -> bool:
    for match in IPV6_CANDIDATE_RE.finditer(value):
        try:
            address = ipaddress.ip_address(match.group(0))
        except ValueError:
            continue
        if address.version == 6:
            return True
    return False


def _command_value(item: Mapping[str, Any]) -> str | None:
    command = item.get("command")
    if isinstance(command, str) and command.strip():
        return command
    return None


def _command_values(item: Mapping[str, Any]) -> list[str]:
    command = item.get("command")
    if isinstance(command, str) and command.strip():
        return [command]
    commands = item.get("commands")
    if isinstance(commands, list):
        return [value for value in commands if isinstance(value, str) and value.strip()]
    return []


def _validate_enabled_features(value: Any, *, item_path: tuple[str, ...], errors: list[str]) -> None:
    path = (*item_path, "enabled_features")
    features = _as_mapping(value, path=path, errors=errors)
    if features is None:
        return
    _check_keys(features, allowed=FEATURE_KEYS, required=frozenset(), path=path, errors=errors)

    packing = features.get("sequence_packing")
    if packing is not None and (not isinstance(packing, str) or packing not in PACKING_VALUES):
        errors.append(f"{_pointer(*path, 'sequence_packing')}: expected offline or in_batch")

    cuda_graph_value = features.get("cuda_graph")
    if cuda_graph_value is not None:
        cuda_graph = _as_mapping(cuda_graph_value, path=(*path, "cuda_graph"), errors=errors)
        if cuda_graph is not None:
            cuda_path = (*path, "cuda_graph")
            _check_keys(
                cuda_graph,
                allowed=frozenset({"implementation", "scopes"}),
                required=frozenset({"implementation", "scopes"}),
                path=cuda_path,
                errors=errors,
            )
            implementation = cuda_graph.get("implementation")
            if not isinstance(implementation, str) or implementation not in CUDA_GRAPH_IMPLEMENTATIONS:
                errors.append(f"{_pointer(*cuda_path, 'implementation')}: expected local or transformer_engine")
            scopes = cuda_graph.get("scopes")
            if not isinstance(scopes, list) or not scopes:
                errors.append(f"{_pointer(*cuda_path, 'scopes')}: expected a non-empty list")
            elif not all(isinstance(scope, str) for scope in scopes):
                errors.append(f"{_pointer(*cuda_path, 'scopes')}: scopes must be strings")
            else:
                unknown_scopes = sorted(set(scopes) - CUDA_GRAPH_SCOPES)
                if unknown_scopes:
                    errors.append(f"{_pointer(*cuda_path, 'scopes')}: unknown scopes {unknown_scopes}")
                if len(scopes) != len(set(scopes)):
                    errors.append(f"{_pointer(*cuda_path, 'scopes')}: duplicate scopes are not allowed")

    cp_size = features.get("context_parallel_size")
    if cp_size is not None and (not isinstance(cp_size, int) or isinstance(cp_size, bool) or cp_size <= 1):
        errors.append(f"{_pointer(*path, 'context_parallel_size')}: expected an integer greater than one")

    dispatcher = features.get("moe_dispatcher")
    if dispatcher is not None and (not isinstance(dispatcher, str) or dispatcher not in MOE_DISPATCHERS):
        errors.append(f"{_pointer(*path, 'moe_dispatcher')}: expected deepep or hybridep")


def _validate_metrics(
    item: Mapping[str, Any],
    *,
    item_path: tuple[str, ...],
    status: str,
    errors: list[str],
) -> None:
    path = (*item_path, "metrics")
    metrics = _as_mapping(item.get("metrics"), path=path, errors=errors)
    if metrics is None:
        return
    _check_keys(metrics, allowed=METRIC_NAMES, required=METRIC_NAMES, path=path, errors=errors)

    if status == "verified":
        for name in sorted(METRIC_NAMES):
            value = metrics.get(name)
            if not _is_finite_number(value):
                errors.append(f"{_pointer(*path, name)}: verified metrics must be finite numbers")
            elif (
                name
                in {
                    "last_10_steps_step_time_ms_avg",
                    "last_10_steps_model_tflops_per_gpu_avg",
                }
                and float(value) <= 0
            ):
                errors.append(f"{_pointer(*path, name)}: verified performance metrics must be positive")
    elif status in {"unsupported", "not_applicable"}:
        for name in sorted(METRIC_NAMES):
            if metrics.get(name) is not None:
                errors.append(f"{_pointer(*path, name)}: must be null for status {status}")


def _argument_value(command: str, argument: str) -> str | None:
    values = _argument_values(command, argument)
    if not values:
        return None
    return values[0]


def _argument_values(command: str, argument: str) -> list[str]:
    matches = re.findall(rf"(?:^|\s){re.escape(argument)}(?:=|\s+)(\"[^\"]+\"|'[^']+'|[^\s]+)", command)
    return [match.strip("\"'") for match in matches]


def _config_override_values(command: str, field: str) -> list[str]:
    """Return shell-parsed values for a trailing ConfigContainer override."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return []
    prefix = f"{field}="
    normalized_tokens = (token.lstrip("+") for token in tokens)
    return [token[len(prefix) :] for token in normalized_tokens if token.startswith(prefix)]


def _resume_reference_settings(command: str) -> list[tuple[str, str, str | None]] | None:
    """Return order-independent settings that must match a reference run."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None

    ignored_arguments = {
        "--load-dir",
        "--load_dir",
        "--save-dir",
        "--save-interval",
        "--save_dir",
        "--save_interval",
    }
    ignored_checkpoint_overrides = {
        "checkpoint.ckpt_step",
        "checkpoint.finetune",
        "checkpoint.load",
        "checkpoint.load_optim",
        "checkpoint.load_rng",
        "checkpoint.save",
        "checkpoint.save_optim",
        "checkpoint.save_rng",
    }
    ignored_runtime_overrides = {"train.empty_unused_memory_level"}
    settings: list[tuple[str, str, str | None]] = []
    index = 1  # Both commands are validated separately as train.sh invocations.
    while index < len(tokens):
        token = tokens[index]
        normalized = token.lstrip("+")

        if token.startswith("-"):
            if "=" in token:
                argument, value = token.split("=", maxsplit=1)
            else:
                argument = token
                value = None
                if index + 1 < len(tokens):
                    candidate = tokens[index + 1]
                    if not candidate.startswith("-") and "=" not in candidate:
                        value = candidate
                        index += 1
            if argument not in ignored_arguments:
                settings.append(("argument", argument, value))
        elif "=" in normalized:
            field, value = normalized.split("=", maxsplit=1)
            if field not in ignored_checkpoint_overrides and field not in ignored_runtime_overrides:
                settings.append(("override", field, value))
        else:
            settings.append(("positional", normalized, None))
        index += 1

    return sorted(settings, key=lambda setting: (setting[0], setting[1], setting[2] or ""))


def _resume_setting_names(settings: list[tuple[str, str, str | None]]) -> str:
    """Describe mismatched settings without reproducing their possibly private values."""
    names = sorted({"<positional>" if kind == "positional" else name for kind, name, _ in settings})
    return ", ".join(names)


def _has_batch_size_override(command: str) -> bool:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    option_names = {
        "-gb",
        "-mb",
        "--global-batch-size",
        "--global_batch_size",
        "--micro-batch-size",
        "--micro_batch_size",
    }
    config_names = {
        "global_batch_size",
        "micro_batch_size",
        "train.global_batch_size",
        "train.micro_batch_size",
    }
    for token in tokens:
        normalized = token.lstrip("+")
        name = normalized.split("=", 1)[0]
        if name in option_names or name in config_names:
            return True
    return False


def _require_positive_integer_argument(
    command: str,
    argument: str,
    *,
    path: tuple[str, ...],
    errors: list[str],
) -> None:
    values = _argument_values(command, argument)
    if len(values) != 1 or not values[0].isdigit() or int(values[0]) < 1:
        errors.append(f"{_pointer(*path)}: requires exactly one positive integer {argument}")


def _validate_conversion_launcher(
    command: str,
    *,
    operation: str,
    device: str,
    path: tuple[str, ...],
    errors: list[str],
) -> None:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return
    if len(tokens) < 2 or tokens[:2] != ["./scripts/conversion/convert.sh", operation]:
        errors.append(f"{_pointer(*path)}: conversion must use convert.sh {operation}")
    executors = _argument_values(command, "--executor")
    if executors != ["slurm"]:
        errors.append(f"{_pointer(*path)}: conversion must specify --executor slurm exactly once")
    devices = _argument_values(command, "--device")
    if devices != [device]:
        errors.append(f"{_pointer(*path)}: conversion must specify --device {device} exactly once")
    _require_positive_integer_argument(command, "--nodes", path=path, errors=errors)
    gpu_counts = _argument_values(command, "--gpus-per-node")
    if device == "gpu":
        _require_positive_integer_argument(command, "--gpus-per-node", path=path, errors=errors)
    elif gpu_counts != [] and gpu_counts != ["1"]:
        errors.append(f"{_pointer(*path)}: CPU conversion may request at most one shared runtime GPU")
    if any(option in tokens for option in ("--detach", "--dry-run", "--submission-dry-run")):
        errors.append(f"{_pointer(*path)}: verified conversion must wait for completion")


def _validate_training_launcher(command: str, *, item_path: tuple[str, ...], errors: list[str]) -> None:
    path = (*item_path, "command")
    try:
        tokens = shlex.split(command)
    except ValueError:
        return
    if not tokens or tokens[0] != "./scripts/training/train.sh":
        errors.append(f"{_pointer(*path)}: training must use ./scripts/training/train.sh")
    _require_positive_integer_argument(command, "--nodes", path=path, errors=errors)
    _require_positive_integer_argument(command, "--gpus-per-node", path=path, errors=errors)
    if any(option in tokens for option in ("--dry-run", "--submission-dry-run")):
        errors.append(f"{_pointer(*path)}: verified training command must submit the workload")


def _is_inference_launcher(command: str, *, task: str) -> bool:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    return (
        bool(tokens)
        and tokens[0].removeprefix("./") == "scripts/inference/infer.sh"
        and _argument_values(command, "--task") == [task]
    )


def _validate_synchronous_inference_launcher(
    command: str,
    *,
    path: tuple[str, ...],
    errors: list[str],
) -> None:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return
    _require_positive_integer_argument(command, "--nodes", path=path, errors=errors)
    _require_positive_integer_argument(command, "--gpus-per-node", path=path, errors=errors)
    if any(option in tokens for option in ("--detach", "--dry-run", "--submission-dry-run")):
        errors.append(f"{_pointer(*path)}: verified inference must wait for completion")


def _validate_command_text(command: str, *, path: tuple[str, ...], errors: list[str]) -> None:
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        errors.append(f"{_pointer(*path)}: command must be shell-parseable")
        return
    if any(token in {"&", "&&", "|", "||", ";"} for token in tokens):
        errors.append(f"{_pointer(*path)}: each entry must contain exactly one command")
    if _has_batch_size_override(command):
        errors.append(f"{_pointer(*path)}: card commands must use recipe batch sizes")


def _validate_resume(
    item: Mapping[str, Any],
    *,
    item_path: tuple[str, ...],
    status: str,
    errors: list[str],
) -> None:
    path = item_path
    if item.get("depends_on") != "pretrain":
        errors.append(f"{_pointer(*path, 'depends_on')}: must be pretrain")

    if status in {"unsupported", "not_applicable"} and item.get("resume_comparison") is None:
        return

    comparison = _as_mapping(item.get("resume_comparison"), path=(*path, "resume_comparison"), errors=errors)
    if comparison is None:
        return
    comparison_path = (*path, "resume_comparison")
    _check_keys(
        comparison,
        allowed=RESUME_KEYS,
        required=RESUME_KEYS,
        path=comparison_path,
        errors=errors,
    )
    if comparison.get("reference_item") != "pretrain":
        errors.append(f"{_pointer(*comparison_path, 'reference_item')}: must be pretrain")

    for name in ("loss_relative_tolerance", "loss_absolute_tolerance"):
        value = comparison.get(name)
        if not _is_finite_number(value) or float(value) <= 0:
            errors.append(f"{_pointer(*comparison_path, name)}: expected a positive finite number")

    relative_tolerance = comparison.get("loss_relative_tolerance")
    if _is_finite_number(relative_tolerance) and float(relative_tolerance) > 1.0e-2:
        errors.append(f"{_pointer(*comparison_path, 'loss_relative_tolerance')}: must not exceed 1%")

    if status != "verified":
        return

    if not isinstance(item.get("command"), str):
        errors.append(f"{_pointer(*path, 'command')}: verified resume must use one direct command")
        return
    command = item["command"]
    required_fragments = ("checkpoint.ckpt_step=", "--load_dir", "--save_dir")
    for fragment in required_fragments:
        if fragment not in command:
            errors.append(f"{_pointer(*path, 'command')}: missing {fragment}")

    # The inherited checkpoint defaults already provide full-state resume. The
    # command may omit them for readability, but an explicit override must not
    # disable the state needed for a valid continuation.
    required_effective_values = {
        "checkpoint.finetune": "false",
        "checkpoint.load_optim": "true",
        "checkpoint.load_rng": "true",
        "checkpoint.save_optim": "true",
        "checkpoint.save_rng": "true",
    }
    for field, expected_value in required_effective_values.items():
        values = _config_override_values(command, field)
        if len(values) > 1:
            errors.append(f"{_pointer(*path, 'command')}: {field} must not be overridden more than once")
        elif values and values[0].lower() != expected_value:
            errors.append(f"{_pointer(*path, 'command')}: {field} must remain {expected_value}")

    for field in ("checkpoint.load", "checkpoint.save"):
        if _config_override_values(command, field):
            errors.append(
                f"{_pointer(*path, 'command')}: use the canonical --load_dir and --save_dir arguments, not {field}"
            )

    load_dirs = _argument_values(command, "--load_dir")
    save_dirs = _argument_values(command, "--save_dir")
    if len(load_dirs) != 1:
        errors.append(f"{_pointer(*path, 'command')}: verified resume requires exactly one --load_dir")
    if len(save_dirs) != 1:
        errors.append(f"{_pointer(*path, 'command')}: verified resume requires exactly one --save_dir")
    if len(load_dirs) == 1 and len(save_dirs) == 1 and load_dirs[0].rstrip("/") == save_dirs[0].rstrip("/"):
        errors.append(f"{_pointer(*path, 'command')}: load and save directories must be distinct")

    ckpt_match = re.search(r"checkpoint\.ckpt_step=(\d+)", command)
    max_steps_match = re.search(r"--max_steps(?:=|\s+)(\d+)", command)
    sentinel_steps = comparison.get("sentinel_steps")
    if (
        not isinstance(sentinel_steps, list)
        or len(sentinel_steps) != 2
        or not all(isinstance(step, int) and not isinstance(step, bool) for step in sentinel_steps)
    ):
        errors.append(f"{_pointer(*comparison_path, 'sentinel_steps')}: expected two integer steps")
    else:
        if sentinel_steps != sorted(set(sentinel_steps)):
            errors.append(f"{_pointer(*comparison_path, 'sentinel_steps')}: steps must be unique and sorted")
        if ckpt_match is not None and sentinel_steps[0] != int(ckpt_match.group(1)) + 1:
            errors.append(
                f"{_pointer(*comparison_path, 'sentinel_steps')}: first sentinel must be the first resumed step"
            )
        if max_steps_match is not None and sentinel_steps[-1] != int(max_steps_match.group(1)):
            errors.append(f"{_pointer(*comparison_path, 'sentinel_steps')}: final sentinel must equal max_steps")
    if comparison.get("sentinels_match") is not True:
        errors.append(f"{_pointer(*comparison_path, 'sentinels_match')}: must be true when verified")


def _validate_resume_against_pretrain(
    item: Mapping[str, Any],
    pretrain_item: Mapping[str, Any],
    *,
    resume_path: tuple[str, ...],
    pretrain_path: tuple[str, ...],
    default_bridge_commit: str | None,
    errors: list[str],
) -> None:
    """Require a direct resume to use its reference run's checkpoint and settings."""
    if item.get("status") == "verified":
        if pretrain_item.get("status") != "verified":
            errors.append(f"{_pointer(*resume_path, 'depends_on')}: pretrain must be verified first")
        reference_commit = pretrain_item.get("bridge_commit") or default_bridge_commit
        resume_commit = item.get("bridge_commit") or default_bridge_commit
        if reference_commit != resume_commit:
            errors.append(
                f"{_pointer(*resume_path, 'bridge_commit')}: resume and pretrain must use the same Bridge commit"
            )

    resume_command = _command_value(item)
    pretrain_command = _command_value(pretrain_item)
    if resume_command is None or pretrain_command is None:
        return

    resume_command_path = (*resume_path, "command")
    pretrain_command_path = (*pretrain_path, "command")
    reference_save_dirs = _argument_values(pretrain_command, "--save_dir")
    resume_load_dirs = _argument_values(resume_command, "--load_dir")
    if len(reference_save_dirs) != 1:
        errors.append(
            f"{_pointer(*pretrain_command_path)}: checkpoint resume requires exactly one reference --save_dir"
        )
    if len(resume_load_dirs) != 1:
        errors.append(f"{_pointer(*resume_command_path)}: checkpoint resume requires exactly one --load_dir")
    if len(reference_save_dirs) == 1 and len(resume_load_dirs) == 1:
        if reference_save_dirs[0].rstrip("/") != resume_load_dirs[0].rstrip("/"):
            errors.append(f"{_pointer(*resume_command_path)}: --load_dir must equal the pretrain --save_dir")

    if _argument_values(pretrain_command, "--load_dir"):
        errors.append(f"{_pointer(*pretrain_command_path)}: uninterrupted reference pretrain must not use --load_dir")
    if _config_override_values(pretrain_command, "checkpoint.ckpt_step"):
        errors.append(
            f"{_pointer(*pretrain_command_path)}: uninterrupted reference pretrain must not set checkpoint.ckpt_step"
        )
    reference_load_values = _config_override_values(pretrain_command, "checkpoint.load")
    if len(reference_load_values) > 1:
        errors.append(f"{_pointer(*pretrain_command_path)}: checkpoint.load must not be overridden more than once")
    elif reference_load_values and reference_load_values[0].lower() not in {"none", "null"}:
        errors.append(f"{_pointer(*pretrain_command_path)}: uninterrupted reference checkpoint.load must be null")
    if _config_override_values(pretrain_command, "checkpoint.save"):
        errors.append(
            f"{_pointer(*pretrain_command_path)}: use the canonical --save_dir argument, not checkpoint.save"
        )

    required_reference_values = {
        "checkpoint.finetune": "false",
        "checkpoint.save_optim": "true",
        "checkpoint.save_rng": "true",
    }
    for field, expected_value in required_reference_values.items():
        values = _config_override_values(pretrain_command, field)
        if len(values) > 1:
            errors.append(f"{_pointer(*pretrain_command_path)}: {field} must not be overridden more than once")
        elif values and values[0].lower() != expected_value:
            errors.append(
                f"{_pointer(*pretrain_command_path)}: {field} must remain {expected_value} for checkpoint resume"
            )

    reference_settings = _resume_reference_settings(pretrain_command)
    resume_settings = _resume_reference_settings(resume_command)
    if reference_settings is None or resume_settings is None or reference_settings == resume_settings:
        return

    differing_settings = [
        setting
        for setting in {*reference_settings, *resume_settings}
        if reference_settings.count(setting) != resume_settings.count(setting)
    ]
    errors.append(
        f"{_pointer(*resume_command_path)}: reference and resume launch settings must match; "
        f"differences found in {_resume_setting_names(differing_settings)}"
    )


def _validate_inference(
    item: Mapping[str, Any],
    *,
    item_name: str,
    item_path: tuple[str, ...] | None = None,
    status: str,
    errors: list[str],
    command_override: str | None = None,
    command_path: tuple[str, ...] | None = None,
) -> None:
    if status != "verified":
        return
    path = item_path or ("items", item_name)
    resolved_command_path = command_path or (*path, "command")
    command = command_override if command_override is not None else _command_value(item)
    expected = item.get("expected_result")
    if command is None or not isinstance(expected, str):
        return
    try:
        command_tokens = shlex.split(command)
    except ValueError:
        command_tokens = []
    expected_launcher_task = {
        "inference": "vlm-generation",
        "sft_export_inference": "hf-inference",
    }.get(item_name)
    uses_inference_launcher = expected_launcher_task is not None and _is_inference_launcher(
        command, task=expected_launcher_task
    )
    if uses_inference_launcher:
        _validate_synchronous_inference_launcher(command, path=resolved_command_path, errors=errors)
    if not uses_inference_launcher and command_tokens[:2] != ["uv", "run"]:
        errors.append(f"{_pointer(*resolved_command_path)}: inference must use uv run")
    prompts = _argument_values(command, "--prompt")
    if len(prompts) != 1:
        errors.append(f"{_pointer(*resolved_command_path)}: specify --prompt exactly once")
    token_matches = re.findall(r"--max[_-]new[_-]tokens(?:=|\s+)(\d+)", command)
    if not token_matches:
        errors.append(f"{_pointer(*resolved_command_path)}: specify an explicit max_new_tokens value")
        return
    if len(token_matches) != 1:
        errors.append(f"{_pointer(*resolved_command_path)}: specify max_new_tokens exactly once")
        return
    token_count_text = token_matches[0]
    token_count = int(token_count_text)
    if token_count <= 0:
        errors.append(f"{_pointer(*resolved_command_path)}: max_new_tokens must be positive")
    if token_count_text not in expected:
        errors.append(f"{_pointer(*path, 'expected_result')}: state the {token_count_text}-token maximum")
    actual_count_patterns = (
        r"\bexact(?:ly)?\s+(\d+)(?:-token|\s+(?:new\s+|generated\s+)?tokens?|\s+generation\s+steps?)\b",
        r"\b(\d+)-token\s+(?:greedy\s+)?(?:result|output|completion|completions)\b",
        r"\b(?:generated|produced|returned?)\s+(?:exactly\s+)?(\d+)\s+(?:new\s+|generated\s+)?tokens?\b",
        r"\bafter\s+(\d+)\s+(?:new\s+|generated\s+)?tokens?\b",
    )
    actual_counts = {
        int(match) for pattern in actual_count_patterns for match in re.findall(pattern, expected, re.IGNORECASE)
    }
    actual_count = token_count
    if len(actual_counts) != 1:
        errors.append(f"{_pointer(*path, 'expected_result')}: record one actual generated-token count")
    else:
        actual_count = next(iter(actual_counts))
        if actual_count <= 0 or actual_count > token_count:
            errors.append(
                f"{_pointer(*path, 'expected_result')}: generated-token count must be between 1 and {token_count}"
            )
        elif (
            actual_count < token_count
            and re.search(r"\b(?:eos|end[- ]of[- ]sequence)\b", expected, re.IGNORECASE) is None
        ):
            errors.append(f"{_pointer(*path, 'expected_result')}: state that generation stopped at EOS")
    literals = [
        left or right
        for left, right in re.findall(
            r"\bcompletion\b[^\"']{0,200}(?:\"([^\"]+)\"|'([^']+)')",
            expected,
            re.IGNORECASE | re.DOTALL,
        )
    ]
    if not literals:
        errors.append(f"{_pointer(*path, 'expected_result')}: quote the literal completion after the word completion")
    else:
        literal = max(literals, key=len)
        if actual_count > 0 and len(literal.encode()) < actual_count:
            errors.append(
                f"{_pointer(*path, 'expected_result')}: literal completion is too short for {actual_count} tokens"
            )
        if len(prompts) == 1 and literal.strip() == prompts[0].strip():
            errors.append(f"{_pointer(*path, 'expected_result')}: literal completion must not repeat the prompt")
    if re.search(r"\bdo[_-]sample(?:=|\s+)(?:true|1)\b", command, re.IGNORECASE):
        errors.append(f"{_pointer(*resolved_command_path)}: inference verification must be deterministic")


def _validate_manual_forward_pass(
    item: Mapping[str, Any],
    *,
    status: str,
    model_revision: str | None,
    errors: list[str],
) -> None:
    path = ("items", "manual_forward_pass")
    command = _command_value(item)
    expected = item.get("expected_result")
    if command is None:
        return
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = []
    revision_values = _argument_values(command, "--hf-revision")
    revision_flags = [token for token in tokens if token == "--hf-revision" or token.startswith("--hf-revision=")]
    if len(revision_flags) != len(revision_values) or len(revision_values) > 1:
        errors.append(f"{_pointer(*path, 'command')}: specify --hf-revision at most once with a value")
    elif revision_values and revision_values[0] != model_revision:
        errors.append(f"{_pointer(*path, 'command')}: --hf-revision must equal model.hf_revision")
    elif not revision_values:
        last_verified = item.get("last_verified")
        if isinstance(last_verified, dt.datetime):
            last_verified_date = None
        elif isinstance(last_verified, dt.date):
            last_verified_date = last_verified
        elif isinstance(last_verified, str) and _is_iso_date(last_verified):
            last_verified_date = dt.date.fromisoformat(last_verified)
        else:
            last_verified_date = None
        historical_date = (
            status == "verified"
            and last_verified_date is not None
            and last_verified_date < MANUAL_FORWARD_REVISION_PINNING_DATE
        )
        historical = (
            re.search(r"\b(?:grandfathered|historical|predates)\b", expected, re.IGNORECASE)
            if isinstance(expected, str)
            else None
        )
        recorded_revision = (
            re.search(r"\brecorded immutable HF revision\b", expected, re.IGNORECASE)
            if isinstance(expected, str)
            else None
        )
        if not historical_date or historical is None or recorded_revision is None:
            errors.append(
                f"{_pointer(*path, 'command')}: missing --hf-revision is allowed only for verified historical "
                "evidence dated before 2026-07-20 and tied to the recorded immutable HF revision"
            )

    if status != "verified" or not isinstance(expected, str):
        return

    uses_inference_launcher = _is_inference_launcher(command, task="model-comparison")
    if uses_inference_launcher:
        _validate_synchronous_inference_launcher(command, path=(*path, "command"), errors=errors)
    prefix = ["uv", "run", "python", "-m", "torch.distributed.run"]
    if not uses_inference_launcher and tokens[:5] != prefix:
        errors.append(f"{_pointer(*path, 'command')}: manual forward pass must use uv distributed run")
    if not uses_inference_launcher and "examples/conversion/compare_hf_and_megatron/compare.py" not in tokens:
        errors.append(f"{_pointer(*path, 'command')}: use the HF/Megatron comparison helper")
    for argument in ("--hf_model_path", "--megatron_model_path", "--prompt"):
        if len(_argument_values(command, argument)) != 1:
            errors.append(f"{_pointer(*path, 'command')}: specify {argument} exactly once")

    token_match = re.search(
        r"\b(?:token match:\s*true|next[- ]token\b[^.]*\bmatch(?:es|ed)?)", expected, re.IGNORECASE
    )
    token_mismatch = re.search(
        r"\b(?:token match:\s*false|next[- ]token\b[^.]*\b(?:"
        r"(?:not|never)\s+match|(?:doesn|didn)['’]t\s+match|fail(?:s|ed)?\s+to\s+match|mismatch))",
        expected,
        re.IGNORECASE,
    )
    if token_match is None or token_mismatch is not None:
        errors.append(f"{_pointer(*path, 'expected_result')}: record that the next token matches")
    similarity_match = re.search(
        r"cosine similarity(?:\s+is|\s*:)\s*(0(?:\.\d+)?|1(?:\.0+)?)",
        expected,
        re.IGNORECASE,
    )
    if similarity_match is None:
        errors.append(f"{_pointer(*path, 'expected_result')}: record the cosine similarity")
    elif float(similarity_match.group(1)) < MANUAL_FORWARD_COSINE_THRESHOLD:
        errors.append(
            f"{_pointer(*path, 'expected_result')}: cosine similarity must be at least "
            f"{MANUAL_FORWARD_COSINE_THRESHOLD:.2f}"
        )
    number = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    combined_differences = re.search(
        rf"\bmax(?:imum)?\s+and\s+mean\s+(?:absolute\s+)?logit\s+differences?\s*(?:are|:|=)\s*"
        rf"{number}\s+(?:and|,)\s*{number}",
        expected,
        re.IGNORECASE,
    )
    helper_differences = re.search(
        rf"\blogits?\s+diff(?:erences?)?\s*-?\s*max(?:imum)?\s*(?::|=|is|was)\s*{number}\s*,\s*"
        rf"mean\s*(?::|=|is|was)\s*{number}",
        expected,
        re.IGNORECASE,
    )
    for label in ("max", "mean"):
        narrative_difference = re.search(
            rf"\b{label}(?:imum)?\b[^.]*\b(?:absolute\s+)?logit\s+differences?\s*"
            rf"(?:is|are|was|were|:|=)\s*{number}",
            expected,
            re.IGNORECASE,
        )
        if combined_differences is None and helper_differences is None and narrative_difference is None:
            errors.append(
                f"{_pointer(*path, 'expected_result')}: record the numeric {label} absolute logit difference"
            )


def _validate_sft_export_inference(
    item: Mapping[str, Any],
    sft_item: Mapping[str, Any],
    *,
    item_path: tuple[str, ...],
    sft_path: tuple[str, ...],
    errors: list[str],
) -> None:
    path = item_path
    if item.get("depends_on") != "sft":
        errors.append(f"{_pointer(*path, 'depends_on')}: must be sft")
    status = item.get("status")
    if status != "verified":
        return
    if sft_item.get("status") != "verified":
        errors.append(f"{_pointer(*path, 'depends_on')}: sft must be verified first")

    commands = item.get("commands")
    expected = item.get("expected_result")
    if (
        not isinstance(commands, list)
        or len(commands) != 2
        or not all(isinstance(command, str) and command.strip() for command in commands)
        or not isinstance(expected, str)
    ):
        return
    export_command, inference_command = commands
    export_path = (*path, "commands", "0")
    inference_path = (*path, "commands", "1")

    for fragment in (
        "scripts/conversion/convert.sh export",
        "--megatron-path",
        "--hf-path",
    ):
        if fragment not in export_command:
            errors.append(f"{_pointer(*export_path)}: missing {fragment}")
    try:
        inference_tokens = shlex.split(inference_command)
    except ValueError:
        inference_tokens = []
    expected_inference_prefix = [
        "uv",
        "run",
        "python",
        "skills/create-model-verification-card/scripts/verify_hf_inference.py",
    ]
    uses_direct_helper = inference_tokens[:4] == expected_inference_prefix
    uses_inference_launcher = _is_inference_launcher(inference_command, task="hf-inference")
    if not uses_direct_helper and not uses_inference_launcher:
        errors.append(f"{_pointer(*inference_path)}: must run the HF inference verifier through uv or infer.sh")

    export_devices = _argument_values(export_command, "--device")
    if len(export_devices) == 1 and export_devices[0] in {"cpu", "gpu"}:
        _validate_conversion_launcher(
            export_command,
            operation="export",
            device=export_devices[0],
            path=export_path,
            errors=errors,
        )
    else:
        errors.append(f"{_pointer(*export_path)}: SFT export must specify --device cpu or gpu exactly once")
    _validate_inference(
        item,
        item_name="sft_export_inference",
        item_path=item_path,
        status=status,
        errors=errors,
        command_override=inference_command,
        command_path=inference_path,
    )

    sft_command = _command_value(sft_item)
    megatron_path = _argument_value(export_command, "--megatron-path")
    if sft_command is not None:
        save_dir = _argument_value(sft_command, "--save_dir")
        max_steps = _argument_value(sft_command, "--max_steps")
        if save_dir is None or max_steps is None or not max_steps.isdigit():
            errors.append(f"{_pointer(*sft_path, 'command')}: verified SFT export requires a final save directory")
        else:
            expected_megatron_path = f"{save_dir.rstrip('/')}/iter_{int(max_steps):07d}"
            if megatron_path != expected_megatron_path:
                errors.append(f"{_pointer(*export_path)}: export must consume the final SFT checkpoint")

    hf_path = _argument_value(export_command, "--hf-path")
    inference_hf_models = _argument_values(inference_command, "--hf-model")
    if len(inference_hf_models) != 1 or hf_path is None or inference_hf_models[0] != hf_path:
        errors.append(f"{_pointer(*inference_path)}: HF inference must reload the export destination")
    if not re.search(r"\b(?:reload|reloaded|reloads|from_pretrained)\b", expected, re.IGNORECASE):
        errors.append(f"{_pointer(*path, 'expected_result')}: state that the HF export reloads")


def _validate_training_window(
    item: Mapping[str, Any],
    *,
    item_name: str,
    item_path: tuple[str, ...],
    status: str,
    errors: list[str],
) -> None:
    if status != "verified":
        return
    path = (*item_path, "command")
    command = _command_value(item)
    if command is None:
        return
    max_steps_values = _argument_values(command, "--max_steps")
    if len(max_steps_values) != 1 or not max_steps_values[0].isdigit():
        errors.append(f"{_pointer(*path)}: verified training requires exactly one integer --max_steps")
        return
    final_step = int(max_steps_values[0])
    first_step = 1
    if item_name == "checkpoint_resume":
        checkpoint_steps = re.findall(r"checkpoint\.ckpt_step=(\d+)", command)
        if len(checkpoint_steps) != 1:
            errors.append(f"{_pointer(*path)}: verified resume requires exactly one checkpoint.ckpt_step")
            return
        first_step = int(checkpoint_steps[0]) + 1
    if final_step - first_step + 1 < 10:
        errors.append(f"{_pointer(*path)}: last-10 metrics require at least 10 executed optimizer steps")


def _validate_item(
    item_name: str,
    value: Any,
    errors: list[str],
    *,
    path: tuple[str, ...],
    model_revision: str | None,
) -> None:
    item = _as_mapping(value, path=path, errors=errors)
    if item is None:
        return
    required = frozenset({"status", "precision", "last_verified", "expected_result"})
    if item_name == "sft_export_inference":
        required |= frozenset({"commands"})
    else:
        required |= frozenset({"command"})
    if item_name in TRAINING_ITEMS:
        required |= frozenset({"metrics"})
    if item_name in FEATURE_ITEMS:
        required |= frozenset({"enabled_features"})
    if item_name == "checkpoint_resume":
        required |= frozenset({"depends_on", "resume_comparison"})
    if item_name == "sft_export_inference":
        required |= frozenset({"depends_on"})
    _check_keys(item, allowed=ITEM_KEYS, required=required, path=path, errors=errors)

    status = item.get("status")
    if not isinstance(status, str) or status not in STATUSES:
        errors.append(f"{_pointer(*path, 'status')}: expected one of {sorted(STATUSES)}")
        return

    bridge_commit = item.get("bridge_commit")
    if "bridge_commit" in item:
        if status != "verified":
            errors.append(f"{_pointer(*path, 'bridge_commit')}: item overrides are allowed only when verified")
        if not isinstance(bridge_commit, str) or REVISION_RE.fullmatch(bridge_commit) is None:
            errors.append(f"{_pointer(*path, 'bridge_commit')}: expected an immutable 40-hex commit")

    precision = item.get("precision")
    if precision is not None and (not isinstance(precision, str) or precision not in PRECISIONS):
        errors.append(f"{_pointer(*path, 'precision')}: expected one of {sorted(PRECISIONS)} or null")
    elif isinstance(precision, str) and item_name not in TRAINING_ITEMS and precision in TRAINING_ONLY_PRECISIONS:
        errors.append(f"{_pointer(*path, 'precision')}: {precision} is supported only on training items")
    if status == "verified" and precision is None:
        errors.append(f"{_pointer(*path, 'precision')}: verified items require a concrete precision")
    elif status in {"unsupported", "not_applicable"} and precision is not None:
        errors.append(f"{_pointer(*path, 'precision')}: must be null for status {status}")

    command = item.get("command")
    commands = item.get("commands")
    command_entries: list[tuple[tuple[str, ...], str]] = []
    command_field = "commands" if item_name == "sft_export_inference" else "command"
    if item_name == "sft_export_inference":
        if "command" in item:
            errors.append(f"{_pointer(*path, 'command')}: use the ordered commands list for this item")
        if commands is not None:
            if not isinstance(commands, list):
                errors.append(f"{_pointer(*path, 'commands')}: expected a two-string list or null")
            else:
                if len(commands) != 2:
                    errors.append(f"{_pointer(*path, 'commands')}: expected exactly two ordered commands")
                for index, value in enumerate(commands):
                    entry_path = (*path, "commands", str(index))
                    if not isinstance(value, str) or not value.strip():
                        errors.append(f"{_pointer(*entry_path)}: expected a non-empty command string")
                    else:
                        command_entries.append((entry_path, value))
    else:
        if "commands" in item:
            errors.append(f"{_pointer(*path, 'commands')}: only sft_export_inference uses a command list")
        if command is not None and not isinstance(command, str):
            errors.append(f"{_pointer(*path, 'command')}: expected a string or null")
        elif isinstance(command, str):
            command_entries.append(((*path, "command"), command))

    for entry_path, entry in command_entries:
        _validate_command_text(entry, path=entry_path, errors=errors)

    expected = item.get("expected_result")
    if expected is not None and not isinstance(expected, str):
        errors.append(f"{_pointer(*path, 'expected_result')}: expected a string or null")

    if status == "verified":
        complete_commands = _command_values(item)
        expected_count = 2 if item_name == "sft_export_inference" else 1
        if len(complete_commands) != expected_count:
            errors.append(f"{_pointer(*path, command_field)}: verified item requires {expected_count} command(s)")
        for entry_path, entry in command_entries:
            if PLACEHOLDER_RE.search(entry):
                errors.append(f"{_pointer(*entry_path)}: verified command contains a placeholder")
        if not isinstance(expected, str) or not expected.strip():
            errors.append(f"{_pointer(*path, 'expected_result')}: verified items require a concrete result")
        elif PLACEHOLDER_RE.search(expected):
            errors.append(f"{_pointer(*path, 'expected_result')}: verified result contains a placeholder")
        if not _is_iso_date(item.get("last_verified")):
            errors.append(f"{_pointer(*path, 'last_verified')}: verified items require an ISO date")
    elif status in {"unsupported", "not_applicable"}:
        if item.get(command_field) is not None:
            errors.append(f"{_pointer(*path, command_field)}: must be null for status {status}")
        if item.get("last_verified") is not None:
            errors.append(f"{_pointer(*path, 'last_verified')}: must be null for status {status}")
        if not isinstance(expected, str) or not expected.strip() or PLACEHOLDER_RE.search(expected):
            errors.append(f"{_pointer(*path, 'expected_result')}: explain the public limitation")

    if status == "verified" and item_name in CONVERSION_ITEMS:
        complete_command = _command_value(item)
        if complete_command is not None:
            operation = "import" if item_name.startswith("hf_to_megatron") else "export"
            device = "cpu" if item_name.endswith("cpu") else "gpu"
            _validate_conversion_launcher(
                complete_command,
                operation=operation,
                device=device,
                path=(*path, "command"),
                errors=errors,
            )

    if item_name in TRAINING_ITEMS:
        _validate_metrics(item, item_path=path, status=status, errors=errors)
        _validate_training_window(
            item,
            item_name=item_name,
            item_path=path,
            status=status,
            errors=errors,
        )
        if status == "verified":
            complete_command = _command_value(item)
            if complete_command is not None:
                _validate_training_launcher(complete_command, item_path=path, errors=errors)
    elif item.get("metrics") is not None:
        errors.append(f"{_pointer(*path)}: metrics are allowed only on training items")

    if item_name in FEATURE_ITEMS:
        _validate_enabled_features(item.get("enabled_features"), item_path=path, errors=errors)
        if item_name == "sft_long_context" and status == "verified":
            features = item.get("enabled_features")
            if isinstance(features, Mapping):
                if features.get("sequence_packing") not in PACKING_VALUES:
                    errors.append(
                        f"{_pointer(*path, 'enabled_features', 'sequence_packing')}: "
                        "verified long-context SFT requires sequence packing"
                    )
                cp_size = features.get("context_parallel_size")
                if not isinstance(cp_size, int) or isinstance(cp_size, bool) or cp_size <= 1:
                    errors.append(
                        f"{_pointer(*path, 'enabled_features', 'context_parallel_size')}: "
                        "verified long-context SFT requires CP greater than one"
                    )
    elif item.get("enabled_features") is not None:
        errors.append(f"{_pointer(*path, 'enabled_features')}: field is not allowed on this item")

    if item_name == "checkpoint_resume":
        _validate_resume(item, item_path=path, status=status, errors=errors)
    if item_name == "sft_export_inference" and item.get("depends_on") != "sft":
        errors.append(f"{_pointer(*path, 'depends_on')}: must be sft")
    if item_name == "manual_forward_pass":
        _validate_manual_forward_pass(item, status=status, model_revision=model_revision, errors=errors)
    if item_name == "inference":
        _validate_inference(item, item_name=item_name, item_path=path, status=status, errors=errors)


def _walk_keys(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], str]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            yield path, key_text
            yield from _walk_keys(child, (*path, key_text))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_keys(child, (*path, str(index)))


def _validate_privacy(raw: str, card: Mapping[str, Any], deny_terms: tuple[str, ...], errors: list[str]) -> None:
    for path, key in _walk_keys(card):
        if path == ("verification_environment",) and key == "base_container":
            continue
        normalized = key.lower().replace("-", "_")
        if any(fragment in normalized for fragment in FORBIDDEN_KEY_FRAGMENTS):
            errors.append(f"{_pointer(*path, key)}: runtime identity or evidence fields are forbidden")

    privacy_raw = raw
    environment = card.get("verification_environment")
    if isinstance(environment, Mapping):
        base_container = environment.get("base_container")
        if isinstance(base_container, str) and PUBLIC_BASE_CONTAINER_RE.fullmatch(base_container) is not None:
            privacy_raw = privacy_raw.replace(base_container, "")

    if re.search(r"\bcluster\b", privacy_raw, re.IGNORECASE):
        errors.append("/: execution-environment names do not belong in the card")
    if URL_RE.search(privacy_raw):
        errors.append("/: URLs are forbidden; use a public model name or repository-relative path")
    if EMAIL_RE.search(privacy_raw) or IPV4_RE.search(privacy_raw) or _contains_ipv6(privacy_raw):
        errors.append("/: email or IP address detected")
    if REMOTE_COMMAND_RE.search(privacy_raw) or REMOTE_COPY_RE.search(privacy_raw):
        errors.append("/: remote host commands are forbidden")
    if JOB_ID_RE.search(privacy_raw):
        errors.append("/: scheduler job metadata is forbidden")
    if SECRET_ASSIGNMENT_RE.search(privacy_raw):
        errors.append("/: secret assignment detected")
    if HOME_PATH_RE.search(privacy_raw):
        errors.append("/: home-directory paths are forbidden")
    if REGISTRY_REFERENCE_RE.search(privacy_raw):
        errors.append("/: concrete registry references are forbidden")
    commands: list[str] = []
    items = card.get("items")
    if isinstance(items, Mapping):
        for _, _, item, _ in _iter_item_leaves(items):
            command = item.get("command")
            if isinstance(command, str):
                commands.append(command)
            command_list = item.get("commands")
            if isinstance(command_list, list):
                commands.extend(value for value in command_list if isinstance(value, str))
    if any("$(" in command or "`" in command for command in commands):
        errors.append("/: shell command substitution is forbidden in cards")
    if any(SECRET_FLAG_RE.search(command) for command in commands):
        errors.append("/: credential flags are forbidden in cards")
    if any(ENVIRONMENT_ASSIGNMENT_RE.search(command) for command in commands):
        errors.append("/: environment assignments are forbidden in cards")
    if any(RUNTIME_COMMAND_RE.search(command) for command in commands):
        errors.append("/: scheduler and container commands are forbidden in cards")
    if any(RUNTIME_ENTRYPOINT_RE.search(command) for command in commands):
        errors.append("/: use the public launcher rather than its setup implementation")
    if any(RUNTIME_FLAG_RE.search(command) for command in commands):
        errors.append("/: private runtime orchestration flags are forbidden in cards")
    if any(SHELL_SETUP_RE.search(command) for command in commands):
        errors.append("/: shell environment setup is forbidden in cards")
    if "../" in privacy_raw:
        errors.append("/: parent-directory traversal is forbidden in cards")

    raw_without_urls = URL_RE.sub("", privacy_raw)
    for match in ABSOLUTE_PATH_RE.finditer(raw_without_urls):
        errors.append(f"/: absolute path detected at character {match.start()}")

    environment_references = set(ENVIRONMENT_REFERENCE_RE.findall(privacy_raw))
    if environment_references:
        errors.append(
            f"/: {len(environment_references)} environment reference(s) detected; runtime wiring belongs outside the card"
        )

    lowered = privacy_raw.casefold()
    for index, term in enumerate((term for term in deny_terms if term), start=1):
        if term.casefold() in lowered:
            errors.append(f"/: matched caller-supplied deny term #{index}")


def _validate_card(card: Mapping[str, Any], raw: str, deny_terms: tuple[str, ...]) -> list[str]:
    errors: list[str] = []
    _check_keys(card, allowed=TOP_LEVEL_KEYS, required=TOP_LEVEL_KEYS, path=(), errors=errors)

    for name in ("title", "summary"):
        value = card.get(name)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{_pointer(name)}: expected a non-empty string")

    model = _as_mapping(card.get("model"), path=("model",), errors=errors)
    if model is not None:
        _check_keys(model, allowed=MODEL_KEYS, required=MODEL_KEYS, path=("model",), errors=errors)
        if not isinstance(model.get("architecture"), str) or not model.get("architecture", "").strip():
            errors.append(f"{_pointer('model', 'architecture')}: expected a non-empty string")
        min_version = model.get("min_transformers_version")
        if not isinstance(min_version, str) or VERSION_RE.fullmatch(min_version) is None:
            errors.append(f"{_pointer('model', 'min_transformers_version')}: expected MAJOR.MINOR.PATCH")

    environment = _as_mapping(
        card.get("verification_environment"),
        path=("verification_environment",),
        errors=errors,
    )
    if environment is not None:
        _check_keys(
            environment,
            allowed=ENVIRONMENT_KEYS,
            required=ENVIRONMENT_KEYS,
            path=("verification_environment",),
            errors=errors,
        )
        base_container = environment.get("base_container")
        if not isinstance(base_container, str) or PUBLIC_BASE_CONTAINER_RE.fullmatch(base_container) is None:
            errors.append(
                f"{_pointer('verification_environment', 'base_container')}: "
                "expected a public NVIDIA NeMo or PyTorch container"
            )
        bridge_commit = environment.get("bridge_commit")
        if not isinstance(bridge_commit, str) or REVISION_RE.fullmatch(bridge_commit) is None:
            errors.append(
                f"{_pointer('verification_environment', 'bridge_commit')}: expected an immutable 40-hex commit"
            )

    items = _as_mapping(card.get("items"), path=("items",), errors=errors)
    item_leaves: list[tuple[str, str | None, Mapping[str, Any], tuple[str, ...]]] = []
    has_canonical_performance_recipe = False
    concrete_performance_variants: dict[str, Mapping[str, Any]] = {}
    hardware_groups: dict[str, dict[str, Mapping[str, Any]]] = {}
    if items is not None:
        item_names = set(items)
        for name in sorted(item_names - set(ITEM_NAMES)):
            errors.append(f"{_pointer('items', name)}: unknown verification item")
        for name in sorted(set(REQUIRED_ITEM_NAMES) - item_names):
            errors.append(f"{_pointer('items', name)}: required verification item is missing")
        model_revision = model.get("hf_revision") if model is not None else None
        if not isinstance(model_revision, str):
            model_revision = None
        for name in ITEM_NAMES:
            if name not in items:
                continue
            if name in HARDWARE_SCOPED_ITEMS:
                variants = _hardware_variants(items[name], item_name=name, errors=errors)
                hardware_groups[name] = variants
                for hardware, item in variants.items():
                    _validate_item(
                        name,
                        item,
                        errors,
                        path=("items", name, hardware),
                        model_revision=model_revision,
                    )
            else:
                _validate_item(
                    name,
                    items[name],
                    errors,
                    path=("items", name),
                    model_revision=model_revision,
                )

        for hardware, export_item in hardware_groups.get("sft_export_inference", {}).items():
            export_path = ("items", "sft_export_inference", hardware)
            if export_item.get("status") in {"unsupported", "not_applicable"}:
                continue
            sft_item = hardware_groups.get("sft", {}).get(hardware)
            if sft_item is None:
                errors.append(f"{_pointer(*export_path, 'depends_on')}: missing sft.{hardware}")
                continue
            _validate_sft_export_inference(
                export_item,
                sft_item,
                item_path=export_path,
                sft_path=("items", "sft", hardware),
                errors=errors,
            )

        default_bridge_commit = environment.get("bridge_commit") if environment is not None else None
        for hardware, resume_item in hardware_groups.get("checkpoint_resume", {}).items():
            resume_path = ("items", "checkpoint_resume", hardware)
            if resume_item.get("status") in {"unsupported", "not_applicable"}:
                continue
            pretrain_item = hardware_groups.get("pretrain", {}).get(hardware)
            if pretrain_item is None:
                errors.append(f"{_pointer(*resume_path, 'depends_on')}: missing pretrain.{hardware}")
                continue
            _validate_resume_against_pretrain(
                resume_item,
                pretrain_item,
                resume_path=resume_path,
                pretrain_path=("items", "pretrain", hardware),
                default_bridge_commit=default_bridge_commit if isinstance(default_bridge_commit, str) else None,
                errors=errors,
            )

        item_leaves = list(_iter_item_leaves(items))
        performance_variants = hardware_groups.get("pretrain_performance", {})
        concrete_performance_variants = {
            hardware: item for hardware, item in performance_variants.items() if hardware != "all"
        }
        has_canonical_performance_recipe = bool(concrete_performance_variants)
        if "all" in performance_variants:
            errors.append(
                f"{_pointer('items', 'pretrain_performance', 'all')}: "
                "omit pretrain_performance when no canonical hardware recipe exists"
            )
        for hardware, item in concrete_performance_variants.items():
            if item.get("status") not in {"verified", "unverified"}:
                errors.append(
                    f"{_pointer('items', 'pretrain_performance', hardware, 'status')}: "
                    "a canonical performance recipe must be verified or unverified"
                )

        if environment is not None:
            default_bridge_commit = environment.get("bridge_commit")
            for _, _, item, path in item_leaves:
                if (
                    isinstance(default_bridge_commit, str)
                    and "bridge_commit" in item
                    and item.get("bridge_commit") == default_bridge_commit
                ):
                    errors.append(
                        f"{_pointer(*path, 'bridge_commit')}: "
                        "omit a redundant override of verification_environment.bridge_commit"
                    )

    _validate_verification_index(
        card.get("verification_index"),
        items=items,
        hardware_groups=hardware_groups,
        errors=errors,
    )

    summary = card.get("summary")
    if isinstance(summary, str) and items is not None:
        normalized_summary = " ".join(summary.split())
        has_untuned_disclaimer = normalized_summary.startswith(UNTUNED_PERFORMANCE_DISCLAIMER)
        if not has_canonical_performance_recipe and not has_untuned_disclaimer:
            errors.append(
                f"{_pointer('summary')}: cards without a canonical pretrain_performance recipe "
                "must start with the untuned performance disclaimer"
            )
        elif has_canonical_performance_recipe and has_untuned_disclaimer:
            errors.append(
                f"{_pointer('summary')}: remove the untuned performance disclaimer when a canonical "
                "pretrain_performance recipe exists"
            )
        if has_canonical_performance_recipe:
            for hardware in concrete_performance_variants:
                performance_scope = f"pretrain_performance.{hardware}"
                if performance_scope not in normalized_summary:
                    errors.append(f"{_pointer('summary')}: scope the tuned claim to {performance_scope}")

    any_verified = any(item.get("status") == "verified" for _, _, item, _ in item_leaves)
    if model is not None and any_verified:
        hf_id = model.get("hf_id")
        revision = model.get("hf_revision")
        if not isinstance(hf_id, str) or HF_ID_RE.fullmatch(hf_id) is None:
            errors.append(f"{_pointer('model', 'hf_id')}: verified cards require a public ORG/MODEL name")
        elif hf_id == "ORG/MODEL":
            errors.append(f"{_pointer('model', 'hf_id')}: replace the placeholder model name")
        if not isinstance(revision, str) or REVISION_RE.fullmatch(revision) is None:
            errors.append(f"{_pointer('model', 'hf_revision')}: verified cards require an immutable 40-hex revision")
        if model.get("architecture") == "MODEL_ARCHITECTURE":
            errors.append(f"{_pointer('model', 'architecture')}: replace the placeholder architecture")
        if card.get("title") == "model_variant":
            errors.append(f"{_pointer('title')}: replace the placeholder title")

    _validate_privacy(raw, card, deny_terms, errors)
    return sorted(set(errors))


def _load_card(raw: str) -> tuple[Mapping[str, Any] | None, list[str]]:
    errors: list[str] = []
    try:
        for token in yaml.scan(raw):
            if isinstance(token, AnchorToken | AliasToken | TagToken):
                errors.append("/: YAML anchors, aliases, and tags are forbidden")
    except yaml.YAMLError as error:
        return None, [f"/: invalid YAML token stream: {error}"]
    if errors:
        return None, sorted(set(errors))

    try:
        card = yaml.load(raw, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        return None, sorted(set([*errors, f"/: invalid YAML: {error}"]))
    if not isinstance(card, Mapping):
        errors.append("/: card root must be a mapping")
        return None, sorted(set(errors))
    return card, sorted(set(errors))


def _read_denylist(path: Path | None) -> tuple[str, ...]:
    if path is None:
        return ()
    terms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        term = line.strip()
        if term and not term.startswith("#"):
            terms.append(term)
    return tuple(terms)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cards", nargs="+", help="Card paths, or - for standard input")
    parser.add_argument(
        "--deny-term",
        action="append",
        default=[],
        help="Private term to reject without printing it; may be repeated",
    )
    parser.add_argument("--denylist", type=Path, help="Untracked file containing private terms")
    return parser.parse_args()


def main() -> int:
    """Validate all requested cards and return a process exit code."""
    args = _parse_args()
    try:
        deny_terms = (*_read_denylist(args.denylist), *args.deny_term)
    except OSError as error:
        LOG.error("Unable to read denylist: %s", error)
        return 2

    failed = False
    for card_path in args.cards:
        source = "<stdin>" if card_path == "-" else card_path
        try:
            raw = sys.stdin.read() if card_path == "-" else Path(card_path).read_text(encoding="utf-8")
        except OSError as error:
            LOG.error("%s: unable to read card: %s", source, error)
            failed = True
            continue

        card, load_errors = _load_card(raw)
        errors = load_errors if card is None else [*load_errors, *_validate_card(card, raw, deny_terms)]
        if errors:
            failed = True
            for validation_error in sorted(set(errors)):
                LOG.error("%s%s", source, validation_error)
        else:
            LOG.info("%s: valid model verification card", source)
    return 1 if failed else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
