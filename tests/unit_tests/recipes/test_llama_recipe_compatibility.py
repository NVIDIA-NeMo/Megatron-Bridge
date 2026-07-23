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
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest


_CANONICAL_MODULE_NAME = "megatron.bridge.recipes.llama.h100.llama3"
_COMPATIBILITY_MODULE_PATH = (
    Path(__file__).parents[3] / "src" / "megatron" / "bridge" / "recipes" / "llama" / "llama3.py"
)
_LOW_PRECISION_RECIPES = (
    ("bf16_with_fp8_current_scaling_mixed", "fp8cs"),
    ("bf16_with_mxfp8_mixed", "fp8mx"),
    ("bf16_with_nvfp4_mixed", "nvfp4"),
)
_RECIPE_ENV_VARS = {"NVTE_FUSED_ATTN": "1"}


def _load_compatibility_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    source = _COMPATIBILITY_MODULE_PATH.read_text()
    syntax_tree = ast.parse(source)
    canonical_module = ModuleType(_CANONICAL_MODULE_NAME)

    def _canonical_recipe() -> str:
        return "canonical"

    for node in ast.walk(syntax_tree):
        if isinstance(node, ast.ImportFrom) and node.module == _CANONICAL_MODULE_NAME:
            for imported_name in node.names:
                setattr(canonical_module, imported_name.name, _canonical_recipe)

    def _low_precision_recipe(name: str) -> Callable[[], dict[str, object]]:
        def _recipe() -> dict[str, object]:
            return {"name": name, "env_vars": _RECIPE_ENV_VARS}

        return _recipe

    canonical_module.llama3_8b_pretrain_2gpu_h100_fp8cs_config = _low_precision_recipe("fp8cs")
    canonical_module.llama3_8b_pretrain_2gpu_h100_fp8mx_config = _low_precision_recipe("fp8mx")
    canonical_module.llama3_8b_pretrain_2gpu_h100_nvfp4_config = _low_precision_recipe("nvfp4")
    monkeypatch.setitem(sys.modules, _CANONICAL_MODULE_NAME, canonical_module)

    spec = importlib.util.spec_from_file_location("llama3_compatibility_under_test", _COMPATIBILITY_MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    compatibility_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compatibility_module)
    return compatibility_module


@pytest.mark.parametrize(("mixed_precision_recipe", "expected_name"), _LOW_PRECISION_RECIPES)
def test_llama3_8b_low_precision_legacy_alias_preserves_selector(
    monkeypatch: pytest.MonkeyPatch, mixed_precision_recipe: str, expected_name: str
) -> None:
    compatibility_module = _load_compatibility_module(monkeypatch)

    config = compatibility_module.llama3_8b_low_precision_pretrain_config(
        mixed_precision_recipe=mixed_precision_recipe
    )

    assert config == {"name": expected_name, "env_vars": _RECIPE_ENV_VARS}


def test_llama3_8b_low_precision_legacy_alias_preserves_parameterless_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compatibility_module = _load_compatibility_module(monkeypatch)

    config = compatibility_module.llama3_8b_low_precision_pretrain_config()

    assert config == {"name": "fp8cs", "env_vars": _RECIPE_ENV_VARS}


def test_unrelated_llama3_legacy_alias_remains_parameterless(monkeypatch: pytest.MonkeyPatch) -> None:
    compatibility_module = _load_compatibility_module(monkeypatch)

    assert compatibility_module.llama3_8b_pretrain_config() == "canonical"
