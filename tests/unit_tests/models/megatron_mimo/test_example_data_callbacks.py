# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Static contracts for runtime-grid handoff in MegatronMIMO examples."""

import ast
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).parents[4]
EXAMPLE_PATHS = [
    "examples/megatron_mimo/qwen35_vl/finetune_qwen35_vl.py",
    "examples/megatron_mimo/llava/megatron_mimo_training_llava.py",
    "examples/megatron_mimo/llava/megatron_mimo_training_llava_audio.py",
]


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", EXAMPLE_PATHS)
def test_mimo_example_data_callbacks_use_runtime_infra_grids(relative_path: str) -> None:
    """Require example callbacks to use grids built outside the serializable config."""
    tree = ast.parse((REPOSITORY_ROOT / relative_path).read_text())
    callbacks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_build_data_iterators"
    ]

    assert callbacks, f"No _build_data_iterators callback found in {relative_path}"
    for callback in callbacks:
        rendered = ast.unparse(callback)
        assert "megatron_mimo_infra.module_to_grid_map" in rendered
        assert "cfg.model._grids" not in rendered
