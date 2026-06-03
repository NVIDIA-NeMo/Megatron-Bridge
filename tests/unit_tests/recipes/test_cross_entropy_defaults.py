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

import ast
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SHIPPED_DEFAULT_ROOTS = (
    Path("src/megatron/bridge/recipes"),
    Path("src/megatron/bridge/diffusion/recipes"),
    Path("src/megatron/bridge/models"),
    Path("scripts/performance"),
)


def _iter_shipped_python_files() -> list[Path]:
    files: list[Path] = []
    for root in _SHIPPED_DEFAULT_ROOTS:
        absolute_root = _REPO_ROOT / root
        if absolute_root.exists():
            files.extend(sorted(absolute_root.rglob("*.py")))
    return files


def _is_cross_entropy_impl_target(target: ast.expr) -> bool:
    if isinstance(target, ast.Attribute):
        return target.attr == "cross_entropy_fusion_impl"
    return isinstance(target, ast.Name) and target.id == "cross_entropy_fusion_impl"


def _is_te_literal(value: ast.expr | None) -> bool:
    return isinstance(value, ast.Constant) and value.value == "te"


def _find_te_cross_entropy_defaults(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    failures: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _is_te_literal(node.value):
            for target in node.targets:
                if _is_cross_entropy_impl_target(target):
                    failures.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}")
        elif isinstance(node, ast.AnnAssign) and _is_te_literal(node.value):
            if _is_cross_entropy_impl_target(node.target):
                failures.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}")
        elif isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg == "cross_entropy_fusion_impl" and _is_te_literal(keyword.value):
                    failures.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}")
    return failures


@pytest.mark.unit
def test_shipped_defaults_do_not_select_te_cross_entropy_fusion() -> None:
    """Shipped defaults should prefer native CE fusion because MCore rejects the TE path."""
    failures: list[str] = []
    for path in _iter_shipped_python_files():
        failures.extend(_find_te_cross_entropy_defaults(path))

    assert not failures, "Shipped defaults must use cross_entropy_fusion_impl='native' instead of 'te':\n" + "\n".join(
        failures
    )
