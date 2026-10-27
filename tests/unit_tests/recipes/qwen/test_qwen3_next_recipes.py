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

"""Qwen3-Next library and perf recipes build on a strict builder-backed model config."""

import inspect
from collections.abc import Callable

import pytest
from transformers import Qwen3NextConfig

from megatron.bridge import perf_recipes
from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.recipes.qwen.h100 import qwen3_next
from tests.unit_tests.recipes.recipe_test_utils import (
    discover_recipe_factories,
    patch_recipe_construction_dependencies,
)


pytestmark = pytest.mark.unit


def _qwen3_next_hf_config() -> Qwen3NextConfig:
    # transformers' default Qwen3-Next config has the 80B-A3B layer layout and MoE shape.
    config = Qwen3NextConfig()
    config.architectures = ["Qwen3NextForCausalLM"]
    return config


class _BuilderOnlyBridge:
    """Return a real strict ModelConfig while rejecting the provider API."""

    @staticmethod
    def from_hf_pretrained(hf_path: str, **kwargs) -> "_BuilderOnlyBridge":
        assert hf_path == "Qwen/Qwen3-Next-80B-A3B-Instruct"
        return _BuilderOnlyBridge()

    def get_model_config(self) -> BridgeGPTModelConfig:
        return AutoBridge.from_hf_config(_qwen3_next_hf_config()).get_model_config()

    def to_megatron_provider(self, load_weights: bool = False):
        raise AssertionError("Qwen3-Next recipes must use get_model_config(), not the legacy provider API")


# The PEFT recipe raises NotImplementedError before building a model config.
_LIBRARY_RECIPES = [
    value
    for name, value in vars(qwen3_next).items()
    if not name.startswith("_") and name.endswith("_config") and inspect.isfunction(value)
    if value.__module__ == qwen3_next.__name__ and "_peft_" not in name
]
# The perf recipes build on the library pretrain recipe.
_PERF_RECIPES = [
    factory for factory in discover_recipe_factories(perf_recipes) if factory.__name__.startswith("qwen3_next_")
]


def test_recipes_are_discovered():
    assert _LIBRARY_RECIPES
    assert _PERF_RECIPES


@pytest.mark.parametrize(
    "recipe_func", [*_LIBRARY_RECIPES, *_PERF_RECIPES], ids=lambda func: f"{func.__module__}.{func.__name__}"
)
def test_each_qwen3_next_recipe_uses_strict_builder_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Recipes only set fields that the builder-backed model config declares."""
    patch_recipe_construction_dependencies(monkeypatch)
    monkeypatch.setattr(qwen3_next, "AutoBridge", _BuilderOnlyBridge)

    cfg = recipe_func()

    assert isinstance(cfg.model, BridgeGPTModelConfig)
    assert cfg.model.experimental_attention_variant == "gated_delta_net"
    assert cfg.model.linear_attention_freq == [1, 1, 1, 0] * 12
    assert cfg.model.num_moe_experts == 512
