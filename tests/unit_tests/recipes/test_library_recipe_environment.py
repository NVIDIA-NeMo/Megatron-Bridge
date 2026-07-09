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

"""Tests for recipe-owned library process environment settings."""

import ast
import re
from pathlib import Path
from types import SimpleNamespace

from megatron.bridge.recipes.utils.environment_utils import (
    LIBRARY_PROCESS_ENV_DEFAULTS,
    apply_library_recipe_environment,
    library_recipe_environment,
)


_CANONICAL_RECIPE_NAME = re.compile(
    r".+_(?:pretrain|sft|peft)_\d+gpu_[a-z0-9]+_(?:bf16|fp8cs|fp8mx|fp8sc|nvfp4)(?:_.+)?_config"
)


def _config(*, backend=None, ep=1, nccl_ub=False, env_vars=None):
    return SimpleNamespace(
        env_vars=dict(env_vars or {}),
        model=SimpleNamespace(
            moe_flex_dispatcher_backend=backend,
            expert_model_parallel_size=ep,
        ),
        ddp=SimpleNamespace(nccl_ub=nccl_ub),
    )


def test_library_recipe_environment_owns_executor_defaults():
    config = _config()

    apply_library_recipe_environment(config, gpu="h100")

    assert config.env_vars.items() >= LIBRARY_PROCESS_ENV_DEFAULTS.items()
    assert config.env_vars["TORCH_NCCL_HIGH_PRIORITY"] == 1


def test_library_recipe_environment_recomputes_nccl_ub_and_topology():
    config = _config(backend="hybridep", ep=32, nccl_ub=True)
    apply_library_recipe_environment(config, gpu="gb200")

    assert config.env_vars["NCCL_NVLS_ENABLE"] == 1
    assert config.env_vars["NCCL_CTA_POLICY"] == 1
    assert config.env_vars["NVLINK_DOMAIN_SIZE"] == 72
    assert config.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 32

    config.model.moe_flex_dispatcher_backend = None
    config.ddp.nccl_ub = False
    apply_library_recipe_environment(config, gpu="h100")

    assert config.env_vars["NCCL_NVLS_ENABLE"] == 0
    assert "NCCL_CTA_POLICY" not in config.env_vars
    assert "NVLINK_DOMAIN_SIZE" not in config.env_vars


def test_library_recipe_environment_preserves_explicit_override():
    config = _config(env_vars={"NCCL_NVLS_ENABLE": 7})

    apply_library_recipe_environment(config, gpu="h100", protected_env_names={"NCCL_NVLS_ENABLE"})

    assert config.env_vars["NCCL_NVLS_ENABLE"] == 7


def test_library_recipe_decorator_finalizes_a_direct_builder():
    @library_recipe_environment(model_family_name="qwen")
    def qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config(*, nccl_ub=False):
        return _config(backend="hybridep", ep=8, nccl_ub=nccl_ub)

    resolved = qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config(nccl_ub=True)

    assert resolved.env_vars["NVLINK_DOMAIN_SIZE"] == 8
    assert resolved.env_vars["TORCH_NCCL_AVOID_RECORD_STREAMS"] == 1
    assert resolved.env_vars["NCCL_NVLS_ENABLE"] == 1


def test_every_hardware_library_recipe_builder_is_explicitly_decorated():
    recipe_root = Path(__file__).resolve().parents[3] / "src" / "megatron" / "bridge" / "recipes"
    undecorated = []
    invalid_metadata = []

    for path in recipe_root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or _CANONICAL_RECIPE_NAME.fullmatch(node.name) is None:
                continue
            relative_parts = path.relative_to(recipe_root).parts
            family = relative_parts[0]
            hardware = relative_parts[-2]
            decorators = [
                decorator
                for decorator in node.decorator_list
                if isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "library_recipe_environment"
            ]
            if not decorators:
                undecorated.append(f"{path.relative_to(recipe_root)}:{node.name}")
                continue

            family_keywords = [
                keyword.value.value
                for keyword in decorators[0].keywords
                if keyword.arg == "model_family_name" and isinstance(keyword.value, ast.Constant)
            ]
            recipe_hardware = re.search(r"_\d+gpu_([a-z0-9]+)_", node.name).group(1)
            if family_keywords != [family] or recipe_hardware != hardware:
                invalid_metadata.append(
                    f"{path.relative_to(recipe_root)}:{node.name}: family={family_keywords!r}, hardware={hardware!r}"
                )

    assert not undecorated
    assert not invalid_metadata
