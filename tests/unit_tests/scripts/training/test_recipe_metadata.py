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

"""Tests for dependency-free recipe metadata and benchmark selection."""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


def _load_module():
    script = Path(__file__).resolve().parents[4] / "scripts" / "training" / "recipe_metadata.py"
    spec = importlib.util.spec_from_file_location("test_training_recipe_metadata", script)
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
        if module.BENCHMARK_RECIPE_PATTERN.search(recipe_name)
    ]

    assert len(recipes) > 300
    for recipe_name, owning_family in recipes:
        metadata = module.available_benchmark_recipe_metadata(recipe_name)
        assert metadata is not None
        assert metadata.family == owning_family


def test_every_exported_benchmark_name_has_canonical_metadata():
    module = _load_module()
    recipe_names = module.benchmark_recipe_names()
    public_modes = {"pretrain": "pretrain", "sft": "sft", "peft": "lora"}

    assert len(recipe_names) > 300
    for recipe_name in recipe_names:
        metadata = module.available_benchmark_recipe_metadata(recipe_name)
        assert metadata is not None
        module.validate_benchmark_recipe_scope(
            metadata,
            mode=public_modes[metadata.task],
            step_func=module.recipe_step(recipe_name),
        )


def test_submission_metadata_is_inferred_from_exact_recipe_name():
    module = _load_module()
    recipe_args = ["--recipe", "qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config"]

    metadata = module.selected_benchmark_recipe(recipe_args)
    assert metadata is not None
    assert metadata.num_gpus == 16
    assert metadata.family == "qwen"
    assert metadata.task == "pretrain"
    assert module.recipe_step(metadata.recipe_name) == "llm_step"


def test_metadata_does_not_impose_a_hardware_node_shape():
    module = _load_module()

    metadata = module.benchmark_recipe_metadata("qwen3_30b_a3b_pretrain_4gpu_h100_bf16_config")

    assert metadata.num_gpus == 4
    assert metadata.hardware == "h100"


@pytest.mark.parametrize(
    ("recipe_name", "task", "step_name"),
    [
        ("llama3_8b_sft_8gpu_gb200_bf16_config", "sft", "llm_step"),
        ("llama3_70b_peft_8gpu_gb200_bf16_config", "peft", "llm_step"),
        ("qwen3_vl_30b_a3b_pretrain_16gpu_h100_bf16_config", "pretrain", "qwen3_vl_step"),
        ("wan_14b_pretrain_16gpu_gb200_bf16_config", "pretrain", "wan_step"),
    ],
)
def test_benchmark_recipe_metadata_selects_task_and_step(recipe_name, task, step_name):
    module = _load_module()

    metadata = module.available_benchmark_recipe_metadata(recipe_name)

    assert metadata is not None
    assert metadata.task == task
    assert module.recipe_step(recipe_name) == step_name


@pytest.mark.parametrize(
    ("recipe_name", "step_name"),
    [
        ("gpt_oss_20b_pretrain_config", "llm_step"),
        ("nemotron_nano_v2_vl_12b_sft_config", "llava_step"),
        ("nemotron_omni_cord_v2_sft_config", "nemotron_omni_step"),
        ("qwen2_audio_7b_sft_config", "audio_lm_step"),
        ("qwen3_omni_30b_a3b_sft_8gpu_h100_bf16_config", "qwen3_omni_step"),
        ("qwen25_vl_7b_sft_config", "vlm_step"),
        ("qwen3_vl_8b_sft_config", "qwen3_vl_step"),
        ("qwen35_vl_9b_sft_config", "qwen3_vl_step"),
        ("qwen36_vl_35b_a3b_sft_config", "qwen3_vl_step"),
        ("gemma3_vl_4b_sft_config", "vlm_step"),
        ("gemma4_vl_26b_sft_config", "vlm_step"),
        ("glm_45v_sft_config", "vlm_step"),
        ("kimi_k25_vl_sft_config", "vlm_step"),
        ("ministral3_8b_sft_config", "vlm_step"),
        ("step37_sft_flickr8k_config", "step37_flickr8k_step"),
        ("flux_12b_pretrain_config", "flux_step"),
        ("wan_14b_pretrain_config", "wan_step"),
    ],
)
def test_recipe_step_is_source_agnostic(recipe_name, step_name):
    module = _load_module()

    assert module.recipe_step(recipe_name) == step_name


def test_every_registered_non_text_prefix_covers_exported_library_recipes():
    module = _load_module()
    root = Path(__file__).resolve().parents[4] / "src" / "megatron" / "bridge" / "recipes"
    recipe_names: set[str] = set()
    for init_path in root.rglob("__init__.py"):
        for node in ast.walk(ast.parse(init_path.read_text(), filename=str(init_path))):
            if isinstance(node, ast.alias):
                candidate = node.asname or node.name
                if candidate.endswith("_config"):
                    recipe_names.add(candidate)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.endswith("_config"):
                recipe_names.add(node.value)

    for prefix, step_name in module.RECIPE_FORWARD_STEP_PREFIXES:
        matching_names = {name for name in recipe_names if name.startswith(prefix)}
        assert matching_names, f"No exported library recipes matched registered prefix {prefix!r}."
        assert {module.recipe_step(name) for name in matching_names} == {step_name}


def test_text_forward_step_aliases_are_compatible():
    module = _load_module()

    assert module.recipe_steps_match("gpt_step", "llm_step")
    assert module.recipe_steps_match("llm_step", "gpt_step")
    assert not module.recipe_steps_match("vlm_step", "llm_step")


@pytest.mark.parametrize(
    "argv",
    [
        ["--model", "qwen3_30b_a3b"],
        ["--recipe", "qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config"],
        ["--recipe", "qwen3_30b_a3b_pretrain_24gpu_h100_bf16_config"],
    ],
)
def test_non_benchmark_selection_is_not_classified_by_name_shape(argv):
    module = _load_module()

    assert module.selected_benchmark_recipe(argv) is None


def test_known_cross_package_collisions_have_explicit_benchmark_precedence():
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

    assert collisions == (module.BENCHMARK_RECIPE_PRECEDENCE_COLLISIONS | module.LIBRARY_RECIPE_PRECEDENCE_COLLISIONS)
    for recipe_name in module.BENCHMARK_RECIPE_PRECEDENCE_COLLISIONS:
        assert module.available_benchmark_recipe_metadata(recipe_name) is not None
        assert module.resolved_benchmark_recipe_metadata(recipe_name) is not None
    for recipe_name in module.LIBRARY_RECIPE_PRECEDENCE_COLLISIONS:
        assert module.available_benchmark_recipe_metadata(recipe_name) is not None
        assert module.resolved_benchmark_recipe_metadata(recipe_name) is None


def test_unregistered_recipe_family_is_rejected():
    module = _load_module()

    with pytest.raises(ValueError, match="no registered family prefix"):
        module.benchmark_recipe_metadata("newmodel_1b_pretrain_8gpu_h100_bf16_config")
