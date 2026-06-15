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

pytestmark = pytest.mark.unit

_EXPLICIT_PROCESS_GROUP_SCATTER_FILES = (
    "src/megatron/bridge/models/gemma_vl/modeling_gemma3_vl.py",
    "src/megatron/bridge/models/gemma_vl/modeling_gemma4_vl.py",
    "src/megatron/bridge/models/ministral3/modeling_ministral3.py",
    "src/megatron/bridge/models/qwen_vl/modelling_qwen3_vl/model.py",
    "src/megatron/bridge/models/qwen_vl/modelling_qwen3_vl/text_model.py",
    "src/megatron/bridge/models/qwen_omni/modeling_qwen25_omni/thinker_model.py",
    "src/megatron/bridge/models/qwen_omni/modeling_qwen3_omni/thinker_model.py",
    "src/megatron/bridge/models/qwen3_asr/modeling_qwen3_asr/thinker_model.py",
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

    for relative_path in _EXPLICIT_PROCESS_GROUP_SCATTER_FILES:
        path = _ROOT / relative_path
        tree = ast.parse(path.read_text(), filename=str(path))
        for call in _scatter_calls(tree):
            if not any(keyword.arg == "group" for keyword in call.keywords):
                missing_group.append(f"{relative_path}:{call.lineno}")

    assert missing_group == []
