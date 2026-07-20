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

"""Tests for dependency-free flat performance recipe metadata parsing."""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


def _load_module():
    script = Path(__file__).resolve().parents[4] / "scripts" / "training" / "performance_recipe.py"
    spec = importlib.util.spec_from_file_location("test_training_performance_recipe", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
    return module


def test_every_flat_recipe_has_registered_family_and_metadata():
    module = _load_module()
    root = Path(__file__).resolve().parents[4] / "src" / "megatron" / "bridge" / "perf_recipes"
    definition_pattern = re.compile(r"^def ([a-zA-Z0-9_]+_config)\(", re.MULTILINE)
    recipes = [
        (recipe_name, path.relative_to(root).parts[0])
        for path in root.rglob("*.py")
        for recipe_name in definition_pattern.findall(path.read_text())
        if module.PERFORMANCE_RECIPE_PATTERN.search(recipe_name)
    ]

    assert len(recipes) > 300
    for recipe_name, owning_family in recipes:
        metadata = module.available_performance_recipe_metadata(recipe_name)
        assert metadata is not None
        assert metadata.family == owning_family


def test_every_exported_performance_name_has_canonical_metadata():
    module = _load_module()
    recipe_names = module.performance_recipe_names()
    public_modes = {"pretrain": "pretrain", "sft": "sft", "peft": "lora"}

    assert len(recipe_names) > 300
    for recipe_name in recipe_names:
        metadata = module.available_performance_recipe_metadata(recipe_name)
        assert metadata is not None
        module.validate_performance_recipe_scope(
            metadata,
            mode=public_modes[metadata.task],
            step_func=module.performance_recipe_step(metadata),
        )


def test_submission_metadata_is_inferred_from_exact_recipe_name():
    module = _load_module()
    recipe_args = ["--recipe", "qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config"]

    metadata = module.selected_performance_recipe(recipe_args)
    assert metadata is not None
    assert metadata.num_gpus == 16
    assert metadata.family == "qwen"
    assert metadata.task == "pretrain"
    assert module.performance_recipe_step(metadata) == "gpt_step"


def test_metadata_does_not_impose_a_hardware_node_shape():
    module = _load_module()

    metadata = module.performance_recipe_metadata("qwen3_30b_a3b_pretrain_4gpu_h100_bf16_config")

    assert metadata.num_gpus == 4
    assert metadata.hardware == "h100"


@pytest.mark.parametrize(
    ("recipe_name", "task", "step_name"),
    [
        ("llama3_8b_sft_8gpu_gb200_bf16_config", "sft", "gpt_step"),
        ("llama3_70b_peft_8gpu_gb200_bf16_config", "peft", "gpt_step"),
        ("qwen3_vl_30b_a3b_pretrain_16gpu_h100_bf16_config", "pretrain", "qwen3_vl_step"),
        ("wan_14b_pretrain_16gpu_gb200_bf16_config", "pretrain", "wan_step"),
    ],
)
def test_performance_recipe_metadata_selects_task_and_step(recipe_name, task, step_name):
    module = _load_module()

    metadata = module.available_performance_recipe_metadata(recipe_name)

    assert metadata is not None
    assert metadata.task == task
    assert module.performance_recipe_step(metadata) == step_name


@pytest.mark.parametrize(
    "argv",
    [
        ["--model", "qwen3_30b_a3b"],
        ["--recipe", "qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config"],
        ["--recipe", "qwen3_30b_a3b_pretrain_24gpu_h100_bf16_config"],
    ],
)
def test_non_performance_selection_is_not_classified_by_name_shape(argv):
    module = _load_module()

    assert module.selected_performance_recipe(argv) is None


def test_known_cross_package_collisions_have_explicit_performance_precedence():
    module = _load_module()
    root = Path(__file__).resolve().parents[4] / "src" / "megatron" / "bridge"

    def exported_recipe_names(package: str) -> set[str]:
        recipe_names: set[str] = set()
        for init_path in (root / package).rglob("__init__.py"):
            for node in ast.walk(ast.parse(init_path.read_text(), filename=str(init_path))):
                if isinstance(node, ast.alias):
                    candidate = node.asname or node.name
                    if candidate.endswith("_config"):
                        recipe_names.add(candidate)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.endswith("_config"):
                    recipe_names.add(node.value)
        return recipe_names

    collisions = exported_recipe_names("recipes") & exported_recipe_names("perf_recipes")

    assert collisions == (
        module.PERFORMANCE_RECIPE_PRECEDENCE_COLLISIONS | module.LIBRARY_RECIPE_PRECEDENCE_COLLISIONS
    )
    for recipe_name in module.PERFORMANCE_RECIPE_PRECEDENCE_COLLISIONS:
        assert module.available_performance_recipe_metadata(recipe_name) is not None
        assert module.resolved_performance_recipe_metadata(recipe_name) is not None
    for recipe_name in module.LIBRARY_RECIPE_PRECEDENCE_COLLISIONS:
        assert module.available_performance_recipe_metadata(recipe_name) is not None
        assert module.resolved_performance_recipe_metadata(recipe_name) is None


def test_unregistered_recipe_family_is_rejected():
    module = _load_module()

    with pytest.raises(ValueError, match="no registered family prefix"):
        module.performance_recipe_metadata("newmodel_1b_pretrain_8gpu_h100_bf16_config")
