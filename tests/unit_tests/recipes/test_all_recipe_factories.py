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

"""Offline construction tests for every public library recipe factory."""

import importlib
import inspect
from collections.abc import Callable

import pytest

from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.training.config import ConfigContainer
from tests.unit_tests.recipes.recipe_test_utils import (
    discover_recipe_factories,
    exported_recipe_factory_keys,
    patch_recipe_construction_dependencies,
    recipe_factory_id,
    recipe_factory_key,
)


pytestmark = pytest.mark.unit

_RECIPES_PACKAGE = importlib.import_module("megatron.bridge.recipes")
_RECIPE_FACTORIES = discover_recipe_factories(
    _RECIPES_PACKAGE,
    exclude_module_prefixes=("megatron.bridge.recipes.utils",),
)
_UNSUPPORTED_FACTORY_KEYS = {
    (
        "megatron.bridge.recipes.qwen.h100.qwen3_next",
        "qwen3_next_80b_a3b_peft_1gpu_h100_bf16_config",
    ),
}
_RUNNABLE_RECIPE_FACTORIES = tuple(
    factory for factory in _RECIPE_FACTORIES if recipe_factory_key(factory) not in _UNSUPPORTED_FACTORY_KEYS
)
_UNSUPPORTED_RECIPE_FACTORIES = tuple(
    factory for factory in _RECIPE_FACTORIES if recipe_factory_key(factory) in _UNSUPPORTED_FACTORY_KEYS
)


@pytest.fixture(autouse=True)
def _keep_recipe_construction_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


def test_all_recipe_factories_are_exported() -> None:
    """Every canonical factory remains available from the public recipe package."""
    canonical = {recipe_factory_key(factory) for factory in _RECIPE_FACTORIES}
    assert exported_recipe_factory_keys(_RECIPES_PACKAGE) == canonical
    assert {recipe_factory_key(factory) for factory in _UNSUPPORTED_RECIPE_FACTORIES} == _UNSUPPORTED_FACTORY_KEYS
    for factory in _RECIPE_FACTORIES:
        assert getattr(_RECIPES_PACKAGE, factory.__name__) is factory


def test_common_pretrain_uses_runtime_tokenizer_vocabulary() -> None:
    """From-scratch pretraining derives the model vocabulary from its tokenizer."""
    assert _pretrain_common().tokenizer.use_tokenizer_vocab_size is True


@pytest.mark.parametrize("recipe_factory", _RUNNABLE_RECIPE_FACTORIES, ids=recipe_factory_id)
def test_recipe_factory_builds_config(recipe_factory: Callable[..., object]) -> None:
    """Every supported recipe can be called with defaults without GPU or network access."""
    inspect.signature(recipe_factory).bind()

    cfg = recipe_factory()

    assert isinstance(cfg, ConfigContainer)
    for section in (
        "model",
        "train",
        "optimizer",
        "scheduler",
        "dataset",
        "logger",
        "tokenizer",
        "checkpoint",
        "rng",
        "ddp",
        "dist",
    ):
        assert getattr(cfg, section) is not None


@pytest.mark.parametrize("recipe_factory", _UNSUPPORTED_RECIPE_FACTORIES, ids=recipe_factory_id)
def test_intentionally_unsupported_recipe_has_clear_error(recipe_factory: Callable[..., object]) -> None:
    """Keep the documented Qwen3-Next PEFT limitation explicit in the exhaustive matrix."""
    inspect.signature(recipe_factory).bind()

    with pytest.raises(NotImplementedError, match="PEFT is not currently supported for Qwen3-Next"):
        recipe_factory()
