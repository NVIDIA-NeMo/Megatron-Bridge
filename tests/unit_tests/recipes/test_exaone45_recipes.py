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

import pytest

from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit

_RECIPE_MODULE = importlib.import_module("megatron.bridge.recipes.exaone.h100.exaone45")


@pytest.fixture(autouse=True)
def _patch_recipe_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


@pytest.mark.parametrize(
    "recipe_name",
    [
        "exaone45_vl_33b_sft_16gpu_h100_bf16_config",
        "exaone45_vl_33b_peft_4gpu_h100_bf16_config",
    ],
)
def test_exaone45_micro_batch_one_recipes_disable_in_batch_packing(recipe_name: str) -> None:
    recipe = getattr(_RECIPE_MODULE, recipe_name)()

    assert recipe.train.micro_batch_size == 1
    assert recipe.dataset.enable_in_batch_packing is False
