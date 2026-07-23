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

import importlib
import importlib.util
import pkgutil
from collections.abc import Callable

import pytest

from megatron.bridge.training.config import ConfigContainer
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_module_global


def _finetune_perf_recipes() -> list[Callable[[], ConfigContainer]]:
    recipes = []
    llama_package = importlib.import_module("megatron.bridge.perf_recipes.llama")
    for module_info in pkgutil.iter_modules(llama_package.__path__):
        module_name = f"{llama_package.__name__}.{module_info.name}.llama3"
        if not module_info.ispkg or importlib.util.find_spec(module_name) is None:
            continue
        module = importlib.import_module(module_name)
        recipes.extend(
            recipe
            for name in dir(module)
            if ("_sft_" in name or "_lora_" in name)
            and name.endswith("_config")
            and callable(recipe := getattr(module, name))
            and recipe.__module__ == module_name
        )
    return recipes


class _FakeModelCfg:
    cross_entropy_fusion_impl = "te"
    context_parallel_size = 1
    use_te_rng_tracker = False

    def finalize(self) -> None:
        return None


class _FakeBridge:
    @staticmethod
    def from_hf_pretrained(*args, **kwargs) -> "_FakeBridge":
        return _FakeBridge()

    def to_megatron_provider(self, load_weights: bool = False) -> _FakeModelCfg:
        return _FakeModelCfg()


@pytest.mark.unit
@pytest.mark.parametrize("recipe_func", _finetune_perf_recipes(), ids=lambda recipe: recipe.__name__)
def test_llama3_finetune_perf_recipes_use_offline_packing_specs(
    recipe_func: Callable[[], ConfigContainer], monkeypatch: pytest.MonkeyPatch
) -> None:
    base_recipes = importlib.import_module("megatron.bridge.recipes.llama.llama3")
    patch_recipe_module_global(monkeypatch, base_recipes, "AutoBridge", _FakeBridge)

    config = recipe_func()

    assert config.dataset.offline_packing_specs is not None
    assert config.dataset.enable_offline_packing is True
    assert config.dataset.offline_packing_specs.packed_sequence_size == config.dataset.seq_length
    assert not hasattr(config.dataset, "packed_sequence_specs")
    assert config.dataset.dataset_kwargs["pad_to_max_length"] is True
