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
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TypeGuard

import yaml
from yaml.tokens import AliasToken, AnchorToken, TagToken


LOG = logging.getLogger(__name__)

STATUSES = frozenset({"unverified", "verified", "unsupported", "not_applicable"})
ITEM_NAMES = (
    "hf_to_megatron_cpu",
    "hf_to_megatron_gpu",
    "megatron_to_hf_cpu",
    "megatron_to_hf_gpu",
    "inference",
    "pretrain",
    "sft",
    "sft_long_context",
    "peft",
    "checkpoint_resume",
    "pretrain_performance",
)
TRAINING_ITEMS = frozenset(
    {"pretrain", "sft", "sft_long_context", "peft", "checkpoint_resume", "pretrain_performance"}
)
FEATURE_ITEMS = frozenset({"pretrain", "sft", "sft_long_context", "peft"})
METRIC_NAMES = frozenset(
    {
        "initial_loss",
        "final_loss",
        "last_half_step_time_ms_avg",
        "last_half_model_tflops_per_gpu_avg",
    }
)
FEATURE_KEYS = frozenset({"sequence_packing", "cuda_graph", "context_parallel_size", "moe_dispatcher"})
PACKING_VALUES = frozenset({"offline", "in_batch"})
CUDA_GRAPH_IMPLEMENTATIONS = frozenset({"local", "transformer_engine"})
CUDA_GRAPH_SCOPES = frozenset({"full_iteration", "attn", "mlp", "moe", "moe_router", "moe_preprocess", "mamba"})
MOE_DISPATCHERS = frozenset({"deepep", "hybridep"})

TOP_LEVEL_KEYS = frozenset({"title", "model", "release", "precision", "summary", "notes", "items"})
MODEL_KEYS = frozenset({"hf_id", "hf_revision", "architecture"})
NOTE_KEYS = frozenset(
    {
        "training_verification",
        "training_metrics",
        "dataset_policy",
        "enabled_features_allowlist",
    }
)
ITEM_KEYS = frozenset(
    {
        "status",
        "depends_on",
        "command",
        "last_verified",
        "expected_result",
        "gpu_type",
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
RUNTIME_ENTRYPOINT_RE = re.compile(r"(?:scripts/training/train\.sh|scripts/training/setup_experiment\.py)")
RUNTIME_FLAG_RE = re.compile(
    r"--(?:mount|env|export|account|partition|container(?:-image|-mounts)?|qos|reservation|nodelist|host|"
    r"srun-arg|ssh-tunnel|remote-job-dir|user|time|experiment-name)(?:=|\s|$)",
    re.IGNORECASE,
)
SLURM_EXECUTOR_RE = re.compile(r"--executor(?:=|\s+)slurm\b", re.IGNORECASE)
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


def _as_string_set(value: Any) -> frozenset[str] | None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    return frozenset(value)


def _validate_feature_allowlist(notes: Mapping[str, Any], errors: list[str]) -> None:
    path = ("notes", "enabled_features_allowlist")
    allowlist = _as_mapping(notes.get("enabled_features_allowlist"), path=path, errors=errors)
    if allowlist is None:
        return
    _check_keys(allowlist, allowed=FEATURE_KEYS, required=FEATURE_KEYS, path=path, errors=errors)
    if _as_string_set(allowlist.get("sequence_packing")) != PACKING_VALUES:
        errors.append(f"{_pointer(*path, 'sequence_packing')}: must list offline and in_batch")
    cuda_graph = _as_mapping(allowlist.get("cuda_graph"), path=(*path, "cuda_graph"), errors=errors)
    if cuda_graph is not None:
        _check_keys(
            cuda_graph,
            allowed=frozenset({"implementation", "scopes"}),
            required=frozenset({"implementation", "scopes"}),
            path=(*path, "cuda_graph"),
            errors=errors,
        )
        if _as_string_set(cuda_graph.get("implementation")) != CUDA_GRAPH_IMPLEMENTATIONS:
            errors.append(f"{_pointer(*path, 'cuda_graph', 'implementation')}: must list local and transformer_engine")
        if _as_string_set(cuda_graph.get("scopes")) != CUDA_GRAPH_SCOPES:
            errors.append(f"{_pointer(*path, 'cuda_graph', 'scopes')}: scopes do not match the fixed allowlist")
    if allowlist.get("context_parallel_size") != "> 1":
        errors.append(f"{_pointer(*path, 'context_parallel_size')}: must be the string '> 1'")
    if _as_string_set(allowlist.get("moe_dispatcher")) != MOE_DISPATCHERS:
        errors.append(f"{_pointer(*path, 'moe_dispatcher')}: must list deepep and hybridep")


def _validate_enabled_features(value: Any, *, item_name: str, errors: list[str]) -> None:
    path = ("items", item_name, "enabled_features")
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


def _validate_metrics(item: Mapping[str, Any], *, item_name: str, status: str, errors: list[str]) -> None:
    path = ("items", item_name, "metrics")
    metrics = _as_mapping(item.get("metrics"), path=path, errors=errors)
    if metrics is None:
        return
    _check_keys(metrics, allowed=METRIC_NAMES, required=METRIC_NAMES, path=path, errors=errors)

    if status == "verified":
        for name in sorted(METRIC_NAMES):
            value = metrics.get(name)
            if not _is_finite_number(value):
                errors.append(f"{_pointer(*path, name)}: verified metrics must be finite numbers")
            elif name in {"last_half_step_time_ms_avg", "last_half_model_tflops_per_gpu_avg"} and float(value) <= 0:
                errors.append(f"{_pointer(*path, name)}: verified performance metrics must be positive")
    elif status in {"unsupported", "not_applicable"}:
        for name in sorted(METRIC_NAMES):
            if metrics.get(name) is not None:
                errors.append(f"{_pointer(*path, name)}: must be null for status {status}")


def _argument_value(command: str, argument: str) -> str | None:
    match = re.search(rf"(?:^|\s){re.escape(argument)}(?:=|\s+)(\"[^\"]+\"|'[^']+'|[^\s]+)", command)
    if match is None:
        return None
    return match.group(1).strip("\"'")


def _validate_resume(item: Mapping[str, Any], *, status: str, errors: list[str]) -> None:
    path = ("items", "checkpoint_resume")
    if item.get("depends_on") != "pretrain":
        errors.append(f"{_pointer(*path, 'depends_on')}: must be pretrain")

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

    if status != "verified":
        return

    if not isinstance(item.get("command"), str):
        errors.append(f"{_pointer(*path, 'command')}: verified resume must use one direct command")
        return
    command = item["command"]
    required_fragments = (
        "checkpoint.load_optim=true",
        "checkpoint.load_rng=true",
        "checkpoint.ckpt_step=",
        "--load_dir",
        "--save_dir",
    )
    for fragment in required_fragments:
        if fragment not in command:
            errors.append(f"{_pointer(*path, 'command')}: missing {fragment}")

    load_dir = _argument_value(command, "--load_dir")
    save_dir = _argument_value(command, "--save_dir")
    if load_dir is not None and save_dir is not None and load_dir == save_dir:
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


def _validate_inference(item: Mapping[str, Any], *, status: str, errors: list[str]) -> None:
    if status != "verified":
        return
    path = ("items", "inference")
    command = _command_value(item)
    expected = item.get("expected_result")
    if command is None or not isinstance(expected, str):
        return
    token_match = re.search(r"--max[_-]new[_-]tokens(?:=|\s+)(\d+)", command)
    if token_match is None:
        errors.append(f"{_pointer(*path, 'command')}: specify an exact max_new_tokens value")
        return
    token_count_text = token_match.group(1)
    token_count = int(token_count_text)
    if token_count <= 0:
        errors.append(f"{_pointer(*path, 'command')}: max_new_tokens must be positive")
    if "exact" not in expected.lower() or token_count_text not in expected:
        errors.append(f"{_pointer(*path, 'expected_result')}: state the exact {token_count_text}-token result")
    literals = [left or right for left, right in re.findall(r'"([^\"]+)"|\'([^\']+)\'', expected, re.DOTALL)]
    if not literals:
        errors.append(f"{_pointer(*path, 'expected_result')}: include the literal completion string")
    elif token_count > 0 and len(max(literals, key=len).encode()) < token_count:
        errors.append(
            f"{_pointer(*path, 'expected_result')}: literal completion is too short for {token_count} tokens"
        )
    repeated = re.search(r"\b(?:twice|two\s+(?:independent\s+)?(?:runs|executions))\b", expected, re.IGNORECASE)
    matched = re.search(
        r"\b(?:byte[- ](?:for[- ])?byte|byte[- ]identical|identical|match(?:es|ed)?)\b", expected, re.IGNORECASE
    )
    if repeated is None or matched is None:
        errors.append(
            f"{_pointer(*path, 'expected_result')}: state that two independent runs produced byte-identical output"
        )
    if re.search(r"\bdo[_-]sample(?:=|\s+)(?:true|1)\b", command, re.IGNORECASE):
        errors.append(f"{_pointer(*path, 'command')}: inference verification must be deterministic")


def _validate_item(item_name: str, value: Any, errors: list[str]) -> None:
    path = ("items", item_name)
    item = _as_mapping(value, path=path, errors=errors)
    if item is None:
        return
    required = frozenset({"status", "command", "last_verified", "expected_result"})
    if item_name in TRAINING_ITEMS:
        required |= frozenset({"gpu_type", "metrics"})
    if item_name in FEATURE_ITEMS:
        required |= frozenset({"enabled_features"})
    if item_name == "checkpoint_resume":
        required |= frozenset({"depends_on", "resume_comparison"})
    _check_keys(item, allowed=ITEM_KEYS, required=required, path=path, errors=errors)

    status = item.get("status")
    if not isinstance(status, str) or status not in STATUSES:
        errors.append(f"{_pointer(*path, 'status')}: expected one of {sorted(STATUSES)}")
        return

    command = item.get("command")
    if command is not None and not isinstance(command, str):
        errors.append(f"{_pointer(*path, 'command')}: expected a string or null")

    expected = item.get("expected_result")
    if expected is not None and not isinstance(expected, str):
        errors.append(f"{_pointer(*path, 'expected_result')}: expected a string or null")

    if status == "verified":
        complete_command = _command_value(item)
        if complete_command is None:
            errors.append(f"{_pointer(*path, 'command')}: verified items require a command")
        elif PLACEHOLDER_RE.search(complete_command):
            errors.append(f"{_pointer(*path, 'command')}: verified command contains a placeholder")
        if not isinstance(expected, str) or not expected.strip():
            errors.append(f"{_pointer(*path, 'expected_result')}: verified items require a concrete result")
        elif PLACEHOLDER_RE.search(expected):
            errors.append(f"{_pointer(*path, 'expected_result')}: verified result contains a placeholder")
        if not _is_iso_date(item.get("last_verified")):
            errors.append(f"{_pointer(*path, 'last_verified')}: verified items require an ISO date")
    elif status in {"unsupported", "not_applicable"}:
        if _command_value(item) is not None:
            errors.append(f"{_pointer(*path, 'command')}: must be null for status {status}")
        if item.get("last_verified") is not None:
            errors.append(f"{_pointer(*path, 'last_verified')}: must be null for status {status}")
        if not isinstance(expected, str) or not expected.strip() or PLACEHOLDER_RE.search(expected):
            errors.append(f"{_pointer(*path, 'expected_result')}: explain the public limitation")

    if item_name in TRAINING_ITEMS:
        gpu_type = item.get("gpu_type")
        if status == "verified" and (not isinstance(gpu_type, str) or not gpu_type.strip()):
            errors.append(f"{_pointer(*path, 'gpu_type')}: verified training requires a public GPU type")
        if status in {"unsupported", "not_applicable"} and gpu_type is not None:
            errors.append(f"{_pointer(*path, 'gpu_type')}: must be null for status {status}")
        _validate_metrics(item, item_name=item_name, status=status, errors=errors)
    elif item.get("metrics") is not None or item.get("gpu_type") is not None:
        errors.append(f"{_pointer(*path)}: metrics and gpu_type are training-only fields")

    if item_name in FEATURE_ITEMS:
        _validate_enabled_features(item.get("enabled_features"), item_name=item_name, errors=errors)
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
        _validate_resume(item, status=status, errors=errors)
    if item_name == "inference":
        _validate_inference(item, status=status, errors=errors)


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
        normalized = key.lower().replace("-", "_")
        if any(fragment in normalized for fragment in FORBIDDEN_KEY_FRAGMENTS):
            errors.append(f"{_pointer(*path, key)}: runtime identity or evidence fields are forbidden")

    if re.search(r"\bcluster\b", raw, re.IGNORECASE):
        errors.append("/: execution-environment names do not belong in the card")
    if URL_RE.search(raw):
        errors.append("/: URLs are forbidden; use a public model name or repository-relative path")
    if EMAIL_RE.search(raw) or IPV4_RE.search(raw) or _contains_ipv6(raw):
        errors.append("/: email or IP address detected")
    if REMOTE_COMMAND_RE.search(raw) or REMOTE_COPY_RE.search(raw):
        errors.append("/: remote host commands are forbidden")
    if JOB_ID_RE.search(raw):
        errors.append("/: scheduler job metadata is forbidden")
    if SECRET_ASSIGNMENT_RE.search(raw):
        errors.append("/: secret assignment detected")
    if HOME_PATH_RE.search(raw):
        errors.append("/: home-directory paths are forbidden")
    if REGISTRY_REFERENCE_RE.search(raw):
        errors.append("/: concrete registry references are forbidden")
    commands: list[str] = []
    items = card.get("items")
    if isinstance(items, Mapping):
        for item in items.values():
            if isinstance(item, Mapping):
                command = item.get("command")
                if isinstance(command, str):
                    commands.append(command)
    if any("$(" in command or "`" in command for command in commands):
        errors.append("/: shell command substitution is forbidden in cards")
    if any(SECRET_FLAG_RE.search(command) for command in commands):
        errors.append("/: credential flags are forbidden in cards")
    if any(ENVIRONMENT_ASSIGNMENT_RE.search(command) for command in commands):
        errors.append("/: environment assignments are forbidden in cards")
    if any(RUNTIME_COMMAND_RE.search(command) for command in commands):
        errors.append("/: scheduler and container commands are forbidden in cards")
    if any(RUNTIME_ENTRYPOINT_RE.search(command) for command in commands):
        errors.append("/: runtime launchers are forbidden; record the rank-local workload")
    if any(RUNTIME_FLAG_RE.search(command) or SLURM_EXECUTOR_RE.search(command) for command in commands):
        errors.append("/: runtime orchestration flags are forbidden in cards")
    if any(SHELL_SETUP_RE.search(command) for command in commands):
        errors.append("/: shell environment setup is forbidden in cards")
    if "../" in raw:
        errors.append("/: parent-directory traversal is forbidden in cards")

    raw_without_urls = URL_RE.sub("", raw)
    for match in ABSOLUTE_PATH_RE.finditer(raw_without_urls):
        errors.append(f"/: absolute path detected at character {match.start()}")

    environment_references = set(ENVIRONMENT_REFERENCE_RE.findall(raw))
    if environment_references:
        errors.append(
            f"/: {len(environment_references)} environment reference(s) detected; runtime wiring belongs outside the card"
        )

    lowered = raw.casefold()
    for index, term in enumerate((term for term in deny_terms if term), start=1):
        if term.casefold() in lowered:
            errors.append(f"/: matched caller-supplied deny term #{index}")


def _validate_card(card: Mapping[str, Any], raw: str, deny_terms: tuple[str, ...]) -> list[str]:
    errors: list[str] = []
    _check_keys(card, allowed=TOP_LEVEL_KEYS, required=TOP_LEVEL_KEYS, path=(), errors=errors)

    for name in ("title", "precision", "summary"):
        value = card.get(name)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{_pointer(name)}: expected a non-empty string")
    if card.get("release") is not None and not isinstance(card.get("release"), str):
        errors.append(f"{_pointer('release')}: expected a string or null")

    model = _as_mapping(card.get("model"), path=("model",), errors=errors)
    if model is not None:
        _check_keys(model, allowed=MODEL_KEYS, required=MODEL_KEYS, path=("model",), errors=errors)
        if not isinstance(model.get("architecture"), str) or not model.get("architecture", "").strip():
            errors.append(f"{_pointer('model', 'architecture')}: expected a non-empty string")

    notes = _as_mapping(card.get("notes"), path=("notes",), errors=errors)
    if notes is not None:
        _check_keys(notes, allowed=NOTE_KEYS, required=NOTE_KEYS, path=("notes",), errors=errors)
        for name in NOTE_KEYS - {"enabled_features_allowlist"}:
            if not isinstance(notes.get(name), str) or not notes.get(name, "").strip():
                errors.append(f"{_pointer('notes', name)}: expected non-empty prose")
        _validate_feature_allowlist(notes, errors)

    items = _as_mapping(card.get("items"), path=("items",), errors=errors)
    if items is not None:
        item_names = set(items)
        for name in sorted(item_names - set(ITEM_NAMES)):
            errors.append(f"{_pointer('items', name)}: unknown verification item")
        for name in sorted(set(ITEM_NAMES) - item_names):
            errors.append(f"{_pointer('items', name)}: required verification item is missing")
        for name in ITEM_NAMES:
            if name in items:
                _validate_item(name, items[name], errors)

    any_verified = bool(
        items and any(isinstance(item, Mapping) and item.get("status") == "verified" for item in items.values())
    )
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
        if card.get("release") == "YY.MM":
            errors.append(f"{_pointer('release')}: replace the placeholder release")

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
