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

from pathlib import Path

import pytest
from scripts.common.benchmark_parallelism import data_parallel_size, topology_from_config

from megatron.bridge.perf_recipes import deepseek
from megatron.bridge.perf_recipes.deepseek.h100 import deepseek_v3 as deepseek_h100
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _offline_recipes(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


@pytest.mark.parametrize("precision", ["bf16", "fp8cs"])
def test_deepseek_v3_64gpu_h100_proxy_fits_64_gpus(precision: str) -> None:
    """The 64-GPU H100 proxies build dense and expert grids that divide 64 GPUs."""
    cfg = getattr(deepseek, f"deepseek_v3_pretrain_64gpu_h100_{precision}_proxy_config")()
    model = cfg.model
    topology = topology_from_config(model)

    assert data_parallel_size(num_gpus=64, topology=topology) == 32
    assert 64 % (model.expert_model_parallel_size * model.expert_tensor_parallel_size) == 0
    assert model.num_layers == 13
    assert model.moe_layer_freq == [0] * 3 + [1] * 10
    assert (model.tensor_model_parallel_size, model.expert_model_parallel_size) == (2, 64)
    assert (model.pipeline_model_parallel_size, model.virtual_pipeline_model_parallel_size) == (1, None)
    assert model.pipeline_model_parallel_layout is None
    assert cfg.train.global_batch_size == 1024


@pytest.mark.parametrize("precision", ["bf16", "fp8cs"])
def test_deepseek_v3_64gpu_h100_proxy_keeps_1024gpu_settings(precision: str) -> None:
    """The proxies change only batch size, depth and pipeline layout relative to the 1024-GPU recipes."""
    proxy = getattr(deepseek, f"deepseek_v3_pretrain_64gpu_h100_{precision}_proxy_config")()
    parent = getattr(deepseek, f"deepseek_v3_pretrain_1024gpu_h100_{precision}_config")()

    assert proxy.mixed_precision == parent.mixed_precision
    assert proxy.env_vars == parent.env_vars
    assert proxy.model.recompute_modules == parent.model.recompute_modules
    assert proxy.model.mtp_num_layers == parent.model.mtp_num_layers == 1


@pytest.mark.parametrize("precision", ["bf16", "fp8cs"])
def test_deepseek_v3_64gpu_h100_full_depth_recipes_are_retired(precision: str) -> None:
    """Only the proxies are exported for 64 H100 GPUs; the full-depth 1024-GPU layout cannot run there."""
    name = f"deepseek_v3_pretrain_64gpu_h100_{precision}_config"

    assert not hasattr(deepseek, name)
    assert not hasattr(deepseek_h100, name)


@pytest.mark.parametrize("compute_dtype", ["bf16", "fp8_cs"])
def test_deepseek_v3_64gpu_h100_default_variant_is_not_selectable(
    monkeypatch: pytest.MonkeyPatch, compute_dtype: str
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts" / "performance"))
    from utils.utils import PerfRecipeNotFoundError, get_perf_recipe_by_name

    with pytest.raises(PerfRecipeNotFoundError):
        get_perf_recipe_by_name("deepseek_v3", "pretrain", 64, "h100", compute_dtype)


@pytest.mark.parametrize("compute_dtype", ["bf16", "fp8_cs"])
def test_deepseek_v3_64gpu_h100_proxy_is_available_through_performance_selector(
    monkeypatch: pytest.MonkeyPatch, compute_dtype: str
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts" / "performance"))
    from utils.utils import get_perf_recipe_by_name

    cfg = get_perf_recipe_by_name("deepseek_v3", "pretrain", 64, "h100", compute_dtype, config_variant="proxy")

    assert cfg.model.num_layers == 13
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 64


@pytest.mark.parametrize("precision", ["bf16", "fp8cs"])
def test_deepseek_v3_1024gpu_h100_topology_unchanged(precision: str) -> None:
    """The 1024-GPU H100 recipes keep the TP2/PP8/VP4/EP64 layout."""
    model = getattr(deepseek, f"deepseek_v3_pretrain_1024gpu_h100_{precision}_config")().model

    assert data_parallel_size(num_gpus=1024, topology=topology_from_config(model)) == 64
    assert (model.pipeline_model_parallel_size, model.virtual_pipeline_model_parallel_size) == (8, 4)
    assert model.pipeline_model_parallel_layout is not None
