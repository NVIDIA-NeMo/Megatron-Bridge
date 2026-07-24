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

"""Focused tests for the VLM generation CLI."""

from __future__ import annotations

import ast
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[4] / "examples" / "conversion" / "hf_to_megatron_generate_vlm.py"


def _main_function() -> ast.FunctionDef:
    tree = ast.parse(_SCRIPT.read_text(), filename=str(_SCRIPT))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")


def test_main_initializes_distributed_before_model_parallel():
    executable_statements = [
        statement
        for statement in _main_function().body
        if not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        )
    ]

    assert ast.unparse(executable_statements[0]) == "maybe_initialize_distributed()"


def test_hf_revision_is_forwarded_to_every_loader():
    expected_loaders = {
        ("AutoBridge", "from_hf_pretrained"),
        ("AutoConfig", "from_pretrained"),
        ("AutoProcessor", "from_pretrained"),
        ("AutoTokenizer", "from_pretrained"),
    }
    revision_loaders = set()

    for node in ast.walk(_main_function()):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name):
            continue
        revision = next((keyword.value for keyword in node.keywords if keyword.arg == "revision"), None)
        if revision is not None and ast.unparse(revision) == "args.hf_revision":
            revision_loaders.add((node.func.value.id, node.func.attr))

    assert revision_loaders == expected_loaders
