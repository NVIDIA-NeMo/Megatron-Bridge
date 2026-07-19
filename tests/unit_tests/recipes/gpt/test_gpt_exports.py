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

"""Contract tests for GPT recipe exports used by the shared runner."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tests.unit_tests.training.test_run_recipe_qwen3_omni import _load_recipe_runner_module


pytestmark = pytest.mark.unit


def _package(name: str, path: Path | None = None) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = [] if path is None else [str(path)]
    return module


def _load_gpt_package(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, object]:
    recipes_root = Path(__file__).resolve().parents[4] / "src" / "megatron" / "bridge" / "recipes"
    gpt_root = recipes_root / "gpt"
    package_name = "megatron.bridge.recipes.gpt"

    def legacy_recipe() -> None:
        return None

    def vanilla_recipe() -> None:
        return None

    vanilla_module = ModuleType(f"{package_name}.vanilla_gpt")
    vanilla_module.vanilla_gpt_pretrain_config = vanilla_recipe
    gpt3_module = ModuleType(f"{package_name}.gpt3_175b")
    gpt3_module.gpt3_175b_pretrain_config = legacy_recipe

    monkeypatch.setitem(sys.modules, "megatron", _package("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.bridge", _package("megatron.bridge"))
    monkeypatch.setitem(sys.modules, "megatron.bridge.recipes", _package("megatron.bridge.recipes", recipes_root))
    monkeypatch.setitem(sys.modules, vanilla_module.__name__, vanilla_module)
    monkeypatch.setitem(sys.modules, gpt3_module.__name__, gpt3_module)

    spec = importlib.util.spec_from_file_location(
        package_name,
        gpt_root / "__init__.py",
        submodule_search_locations=[str(gpt_root)],
    )
    assert spec is not None and spec.loader is not None
    gpt_package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, package_name, gpt_package)
    spec.loader.exec_module(gpt_package)
    return gpt_package, legacy_recipe


def test_shared_runner_discovers_legacy_gpt3_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    recipe_runner, _ = _load_recipe_runner_module()
    gpt_package, legacy_recipe = _load_gpt_package(monkeypatch)

    root_recipes = _package("megatron.bridge.recipes")
    for name in gpt_package.__all__:
        setattr(root_recipes, name, getattr(gpt_package, name))
    recipe_runner.recipes = root_recipes
    recipe_runner.library_h100_modules = lambda: ()

    assert recipe_runner.find_library_recipe("gpt3_175b_pretrain_config") is legacy_recipe
