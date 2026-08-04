# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Static contract tests for provider-free recipe construction."""

import ast
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

RECIPE_ROOT = Path(__file__).parents[3] / "src" / "megatron" / "bridge" / "recipes"


def test_recipes_do_not_call_deprecated_autobridge_provider_entry_point() -> None:
    """Require official recipes to construct builder-backed model configs."""
    violations: list[str] = []

    for path in RECIPE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr == "to_megatron_provider":
                violations.append(f"{path.relative_to(RECIPE_ROOT)}:{node.lineno}")

    assert not violations, "Deprecated AutoBridge provider calls remain: " + ", ".join(violations)
