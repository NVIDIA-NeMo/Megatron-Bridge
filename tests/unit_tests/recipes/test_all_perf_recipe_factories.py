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

"""Offline construction tests for every public performance recipe factory."""

import importlib
import inspect
import pkgutil
from collections.abc import Callable

import pytest

from megatron.bridge.training.config import ConfigContainer, MockVLMSFTDatasetConfig
from tests.unit_tests.recipes.recipe_test_utils import (
    discover_recipe_factories,
    exported_recipe_factory_keys,
    patch_recipe_construction_dependencies,
    recipe_factory_id,
    recipe_factory_key,
)


pytestmark = pytest.mark.unit

_PERF_RECIPES_PACKAGE = importlib.import_module("megatron.bridge.perf_recipes")
_PERF_RECIPE_FACTORIES = discover_recipe_factories(_PERF_RECIPES_PACKAGE)
_PERF_RECIPE_FAMILY_PACKAGES = tuple(
    importlib.import_module(module_info.name)
    for module_info in pkgutil.iter_modules(
        _PERF_RECIPES_PACKAGE.__path__,
        prefix=f"{_PERF_RECIPES_PACKAGE.__name__}.",
    )
    if module_info.ispkg
)
_PERF_RECIPE_FAMILY_PACKAGES_BY_NAME = {package.__name__: package for package in _PERF_RECIPE_FAMILY_PACKAGES}


@pytest.fixture(autouse=True)
def _keep_recipe_construction_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


def test_all_perf_recipe_factories_are_exported() -> None:
    """Every canonical factory remains available from its public family package."""
    canonical = {recipe_factory_key(factory) for factory in _PERF_RECIPE_FACTORIES}
    exported = set().union(*(exported_recipe_factory_keys(package) for package in _PERF_RECIPE_FAMILY_PACKAGES))
    assert exported == canonical
    for factory in _PERF_RECIPE_FACTORIES:
        family_name = factory.__module__.split(".", maxsplit=4)[3]
        family_package = _PERF_RECIPE_FAMILY_PACKAGES_BY_NAME[f"{_PERF_RECIPES_PACKAGE.__name__}.{family_name}"]
        assert getattr(family_package, factory.__name__) is factory


@pytest.mark.parametrize("recipe_factory", _PERF_RECIPE_FACTORIES, ids=recipe_factory_id)
def test_perf_recipe_factory_builds_config(recipe_factory: Callable[..., object]) -> None:
    """Every performance recipe can be built without GPU or network access."""
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

    assert cfg.tokenizer.use_tokenizer_vocab_size is False

    if "pretrain" in recipe_factory.__name__ and isinstance(cfg.dataset, MockVLMSFTDatasetConfig):
        assert cfg.tokenizer.tokenizer_type == "NullTokenizer"
        assert cfg.tokenizer.vocab_size == cfg.model.vocab_size
