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

"""Unit tests for Nemotron 3.5 Nano MTP performance recipes."""

from collections.abc import Callable

import pytest

from megatron.bridge.perf_recipes.nemotronh import (
    nemotron_3_5_nano_mtp_pretrain_8gpu_gb200_bf16_config,
    nemotron_3_5_nano_mtp_pretrain_8gpu_gb200_fp8mx_config,
    nemotron_3_5_nano_mtp_pretrain_8gpu_gb200_nvfp4_config,
    nemotron_3_5_nano_mtp_pretrain_16gpu_h100_bf16_config,
    nemotron_3_5_nano_mtp_pretrain_16gpu_h100_fp8cs_config,
)
from megatron.bridge.training.config import ConfigContainer
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit

_H100_RECIPES = (
    nemotron_3_5_nano_mtp_pretrain_16gpu_h100_bf16_config,
    nemotron_3_5_nano_mtp_pretrain_16gpu_h100_fp8cs_config,
)
_GB200_RECIPES = (
    nemotron_3_5_nano_mtp_pretrain_8gpu_gb200_bf16_config,
    nemotron_3_5_nano_mtp_pretrain_8gpu_gb200_fp8mx_config,
    nemotron_3_5_nano_mtp_pretrain_8gpu_gb200_nvfp4_config,
)


@pytest.fixture(autouse=True)
def _keep_recipe_construction_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


@pytest.mark.parametrize("recipe_factory", (*_H100_RECIPES, *_GB200_RECIPES), ids=lambda recipe: recipe.__name__)
def test_perf_recipes_preserve_mtp_and_benchmark_contract(
    recipe_factory: Callable[[], ConfigContainer],
) -> None:
    cfg = recipe_factory()

    assert cfg.model.mtp_num_layers == 2
    assert cfg.model.mtp_use_repeated_layer is True
    assert cfg.model.keep_mtp_spec_in_bf16 is True
    assert cfg.model.mtp_loss_scaling_factor == 0.3
    assert cfg.model.moe_router_force_load_balancing is True
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.seq_length == 8192
    assert cfg.dataset.seq_length == 8192
    assert cfg.comm_overlap.tp_comm_overlap is True
    assert cfg.train.train_iters == 50
    assert cfg.checkpoint.save is None
    assert cfg.checkpoint.async_save is False


@pytest.mark.parametrize("recipe_factory", _H100_RECIPES, ids=lambda recipe: recipe.__name__)
def test_h100_perf_recipe_topology(recipe_factory: Callable[[], ConfigContainer]) -> None:
    cfg = recipe_factory()

    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.train.global_batch_size == 1024
    assert cfg.train.micro_batch_size == 1
    assert cfg.model.cuda_graph_impl == "transformer_engine"
    assert cfg.model.cuda_graph_scope == ["mamba"]
    assert cfg.model.cuda_graph_warmup_steps == 3
    assert cfg.model.recompute_granularity == "selective"
    assert "core_attn" in cfg.model.recompute_modules
    assert "mlp" not in cfg.model.recompute_modules
    assert cfg.model.recompute_method is None
    assert cfg.model.recompute_num_layers is None
    assert cfg.optimizer.optimizer_cpu_offload is False
    assert cfg.optimizer.optimizer_offload_fraction == 0.0
    assert cfg.optimizer.overlap_cpu_optimizer_d2h_h2d is False
    assert cfg.env_vars["NVLINK_DOMAIN_SIZE"] == 8
    assert cfg.env_vars["USE_MNNVL"] == 0


@pytest.mark.parametrize("recipe_factory", _GB200_RECIPES, ids=lambda recipe: recipe.__name__)
def test_gb200_perf_recipe_topology(recipe_factory: Callable[[], ConfigContainer]) -> None:
    cfg = recipe_factory()

    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.train.global_batch_size == 512
    assert cfg.train.micro_batch_size == 2
    assert cfg.model.recompute_granularity is None
    assert cfg.model.recompute_modules is None
    assert cfg.model.recompute_method is None
    assert cfg.model.recompute_num_layers is None
    assert cfg.optimizer.optimizer_cpu_offload is False
    assert cfg.optimizer.optimizer_offload_fraction == 0.0
    assert cfg.optimizer.overlap_cpu_optimizer_d2h_h2d is False
    assert cfg.env_vars["NVLINK_DOMAIN_SIZE"] == 72
    assert cfg.env_vars["USE_MNNVL"] == 1
