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

"""Tests for DeepSeek V3 H100 performance recipe topology."""

import pytest
from scripts.common.benchmark_parallelism import data_parallel_size, topology_from_config

from megatron.bridge.perf_recipes import deepseek
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _offline_recipes(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


@pytest.mark.parametrize("precision", ["bf16", "fp8cs"])
def test_deepseek_v3_64gpu_h100_topology_fits_64_gpus(precision: str) -> None:
    """The 64-GPU H100 recipes build dense and expert grids that divide 64 GPUs."""
    cfg = getattr(deepseek, f"deepseek_v3_pretrain_64gpu_h100_{precision}_config")()
    model = cfg.model

    assert data_parallel_size(num_gpus=64, topology=topology_from_config(model)) == 32
    assert model.num_layers == 13
    assert model.moe_layer_freq == [0] * 3 + [1] * 10
    assert (model.tensor_model_parallel_size, model.expert_model_parallel_size) == (2, 64)
    assert (model.pipeline_model_parallel_size, model.virtual_pipeline_model_parallel_size) == (1, None)
    assert model.pipeline_model_parallel_layout is None
    assert cfg.train.global_batch_size == 1024


@pytest.mark.parametrize("precision", ["bf16", "fp8cs"])
def test_deepseek_v3_1024gpu_h100_topology_unchanged(precision: str) -> None:
    """The 1024-GPU H100 recipes keep the TP2/PP8/VP4/EP64 layout."""
    model = getattr(deepseek, f"deepseek_v3_pretrain_1024gpu_h100_{precision}_config")().model

    assert data_parallel_size(num_gpus=1024, topology=topology_from_config(model)) == 64
    assert (model.pipeline_model_parallel_size, model.virtual_pipeline_model_parallel_size) == (8, 4)
    assert model.pipeline_model_parallel_layout is not None
