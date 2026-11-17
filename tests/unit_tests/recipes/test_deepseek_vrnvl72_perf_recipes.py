from tests.unit_tests.recipes.recipe_test_utils import benchmark_factory


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

"""Tests for DeepSeek V3 VRNVL72 performance recipe topology."""


import pytest
from scripts.common.benchmark_parallelism import data_parallel_size, topology_from_config


pytestmark = pytest.mark.unit


def _load_vrnvl72_recipe_module(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies

    patch_recipe_construction_dependencies(monkeypatch)


@pytest.mark.parametrize(
    "recipe_factory_name",
    (
        "deepseek_v3_pretrain_128gpu_vrnvl72_bf16_config",
        "deepseek_v3_pretrain_128gpu_vrnvl72_fp8cs_config",
        "deepseek_v3_pretrain_128gpu_vrnvl72_fp8mx_config",
        "deepseek_v3_pretrain_128gpu_vrnvl72_nvfp4_config",
    ),
)
def test_deepseek_v3_128gpu_vrnvl72_topology_is_valid(
    recipe_factory_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _load_vrnvl72_recipe_module(monkeypatch)
    recipe_factory = benchmark_factory(recipe_factory_name)
    cfg = recipe_factory()

    assert cfg.model.pipeline_model_parallel_size == 2
    assert cfg.model.virtual_pipeline_model_parallel_size == 8
    assert cfg.model.pipeline_model_parallel_size * cfg.model.virtual_pipeline_model_parallel_size == 16
    assert data_parallel_size(num_gpus=128, topology=topology_from_config(cfg.model)) == 64
