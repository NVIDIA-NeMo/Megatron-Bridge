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


def test_every_flat_recipe_has_registered_family_and_topology():
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
        metadata = module.performance_recipe_metadata(recipe_name)
        assert metadata.num_gpus % metadata.gpus_per_node == 0
        assert metadata.family == owning_family


def test_submission_metadata_requires_explicit_performance_source():
    module = _load_module()
    recipe_args = ["--recipe", "qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config"]

    assert module.selected_performance_recipe(recipe_args) is None
    metadata = module.selected_performance_recipe(["--recipe-source", "performance", *recipe_args])
    assert metadata.num_gpus == 16
    assert metadata.gpus_per_node == 8
    assert metadata.family == "qwen"


def test_unregistered_recipe_family_is_rejected():
    module = _load_module()

    with pytest.raises(ValueError, match="no registered family prefix"):
        module.performance_recipe_metadata("newmodel_1b_pretrain_8gpu_h100_bf16_config")
