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

"""Regression tests for explicit process-group sequence-parallel scatters."""

import ast
from pathlib import Path

import pytest


_ROOT = Path(__file__).parents[3]
_MODELS_ROOT = _ROOT / "src/megatron/bridge/models"

pytestmark = pytest.mark.unit

# Intentionally-unfixed bare scatter sites go here as
# "src/megatron/bridge/models/...py:<line>": "short rationale".
# Keep this empty unless a model has no explicit TP group available at the call site.
_BARE_SCATTER_EXCLUSIONS: dict[str, str] = {}


def _modeling_sources() -> list[Path]:
    return sorted(
        path
        for path in _MODELS_ROOT.rglob("*.py")
        if any(part.startswith(("modeling", "modelling")) for part in path.relative_to(_MODELS_ROOT).parts)
    )


def _scatter_calls(tree: ast.AST) -> list[ast.Call]:
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "scatter_to_sequence_parallel_region":
            calls.append(node)
        elif isinstance(func, ast.Name) and func.id == "scatter_to_sequence_parallel_region":
            calls.append(node)
    return calls


def test_explicit_process_group_scatter_sites_pass_group():
    missing_group = []
    seen_exclusions = set()

    for path in _modeling_sources():
        relative_path = path.relative_to(_ROOT).as_posix()
        tree = ast.parse(path.read_text(), filename=str(path))
        for call in _scatter_calls(tree):
            if not any(keyword.arg == "group" for keyword in call.keywords):
                location = f"{relative_path}:{call.lineno}"
                if location in _BARE_SCATTER_EXCLUSIONS:
                    seen_exclusions.add(location)
                else:
                    missing_group.append(location)

    assert missing_group == []
    assert set(_BARE_SCATTER_EXCLUSIONS) == seen_exclusions
