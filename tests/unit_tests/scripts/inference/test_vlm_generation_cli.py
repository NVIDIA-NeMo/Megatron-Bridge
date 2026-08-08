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

"""Focused tests for the canonical VLM generation CLI."""

from __future__ import annotations

import ast
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "inference" / "vlm_generation.py"


def test_main_initializes_distributed_before_model_parallel() -> None:
    """Standalone workers must initialize distributed before model parallel."""
    tree = ast.parse(_SCRIPT.read_text(), filename=str(_SCRIPT))
    main_function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    executable_statements = [
        statement
        for statement in main_function.body
        if not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        )
    ]

    assert ast.unparse(executable_statements[0]) == "maybe_initialize_distributed()"
