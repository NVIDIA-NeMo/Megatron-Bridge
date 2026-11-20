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

"""Optimizer CUDA graph policy for the GB200, GB300 and VRNVL72 full-iteration tracker benchmarks."""

import math
from collections.abc import Callable

import pytest

from megatron.bridge.perf_recipes.deepseek import (
    deepseek_v3_pretrain_256gpu_gb200_fp8mx_config,
    deepseek_v3_pretrain_256gpu_gb200_fp8mx_large_scale_config,
    deepseek_v3_pretrain_256gpu_gb200_nvfp4_config,
    deepseek_v3_pretrain_256gpu_gb300_fp8mx_config,
    deepseek_v3_pretrain_256gpu_gb300_fp8mx_large_scale_config,
    deepseek_v3_pretrain_256gpu_gb300_nvfp4_config,
    deepseek_v3_pretrain_256gpu_vrnvl72_nvfp4_config,
    deepseek_v4_flash_pretrain_128gpu_b300_fp8mx_config,
    deepseek_v4_flash_pretrain_128gpu_gb200_fp8mx_config,
    deepseek_v4_flash_pretrain_128gpu_gb300_fp8mx_config,
    deepseek_v4_flash_pretrain_128gpu_vrnvl72_fp8mx_config,
)
from megatron.bridge.perf_recipes.gpt_oss import gpt_oss_120b_pretrain_64gpu_vrnvl72_fp8mx_config
from megatron.bridge.perf_recipes.llama import (
    llama3_8b_pretrain_8gpu_vrnvl72_fp8mx_config,
    llama3_8b_pretrain_8gpu_vrnvl72_nvfp4_config,
)
from megatron.bridge.perf_recipes.nemotronh import (
    nemotron_3_5_lightning_pretrain_8gpu_gb300_fp8mx_config,
    nemotron_3_5_lightning_pretrain_8gpu_vrnvl72_fp8mx_config,
    nemotron_3_super_pretrain_64gpu_gb200_fp8mx_config,
    nemotron_3_super_pretrain_64gpu_gb200_nvfp4_config,
    nemotron_3_super_pretrain_64gpu_gb300_fp8mx_config,
    nemotron_3_super_pretrain_64gpu_gb300_nvfp4_config,
    nemotron_3_super_pretrain_64gpu_vrnvl72_fp8mx_config,
    nemotron_3_super_pretrain_64gpu_vrnvl72_nvfp4_config,
)
from megatron.bridge.perf_recipes.qwen import (
    qwen3_30b_a3b_pretrain_8gpu_vrnvl72_fp8mx_config,
    qwen3_235b_a22b_pretrain_256gpu_gb200_fp8mx_config,
    qwen3_235b_a22b_pretrain_256gpu_gb200_nvfp4_config,
    qwen3_235b_a22b_pretrain_256gpu_gb300_fp8mx_config,
    qwen3_235b_a22b_pretrain_256gpu_gb300_nvfp4_config,
    qwen3_235b_a22b_pretrain_256gpu_vrnvl72_fp8mx_config,
    qwen3_235b_a22b_pretrain_256gpu_vrnvl72_nvfp4_config,
)
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.utils.cuda_graph import is_full_iteration_cuda_graph
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit

_E2E_OPTIMIZER_CUDA_GRAPH_RECIPES = (
    deepseek_v3_pretrain_256gpu_gb200_fp8mx_config,
    deepseek_v3_pretrain_256gpu_gb200_fp8mx_large_scale_config,
    deepseek_v3_pretrain_256gpu_gb300_fp8mx_config,
    deepseek_v3_pretrain_256gpu_gb300_fp8mx_large_scale_config,
    deepseek_v3_pretrain_256gpu_gb300_nvfp4_config,
    deepseek_v4_flash_pretrain_128gpu_gb200_fp8mx_config,
    deepseek_v4_flash_pretrain_128gpu_gb300_fp8mx_config,
    qwen3_235b_a22b_pretrain_256gpu_gb200_fp8mx_config,
    qwen3_235b_a22b_pretrain_256gpu_gb200_nvfp4_config,
    qwen3_235b_a22b_pretrain_256gpu_gb300_fp8mx_config,
    qwen3_235b_a22b_pretrain_256gpu_gb300_nvfp4_config,
    nemotron_3_5_lightning_pretrain_8gpu_gb300_fp8mx_config,
    nemotron_3_super_pretrain_64gpu_gb200_fp8mx_config,
    nemotron_3_super_pretrain_64gpu_gb200_nvfp4_config,
    nemotron_3_super_pretrain_64gpu_gb300_fp8mx_config,
    nemotron_3_super_pretrain_64gpu_gb300_nvfp4_config,
    # VRNVL72 tracker rows that capture the full iteration.
    llama3_8b_pretrain_8gpu_vrnvl72_fp8mx_config,
    llama3_8b_pretrain_8gpu_vrnvl72_nvfp4_config,
    deepseek_v3_pretrain_256gpu_vrnvl72_nvfp4_config,
    deepseek_v4_flash_pretrain_128gpu_vrnvl72_fp8mx_config,
    gpt_oss_120b_pretrain_64gpu_vrnvl72_fp8mx_config,
    nemotron_3_5_lightning_pretrain_8gpu_vrnvl72_fp8mx_config,
    nemotron_3_super_pretrain_64gpu_vrnvl72_fp8mx_config,
    nemotron_3_super_pretrain_64gpu_vrnvl72_nvfp4_config,
    qwen3_30b_a3b_pretrain_8gpu_vrnvl72_fp8mx_config,
    qwen3_235b_a22b_pretrain_256gpu_vrnvl72_fp8mx_config,
    qwen3_235b_a22b_pretrain_256gpu_vrnvl72_nvfp4_config,
)

_RECIPES_WITHOUT_OPTIMIZER_CUDA_GRAPH = (
    # Built on the GB300 DeepSeek V4 Flash base, but not an E2E-owned tracker row.
    deepseek_v4_flash_pretrain_128gpu_b300_fp8mx_config,
    # Full-iteration tracker row that keeps the optimizer step eager.
    deepseek_v3_pretrain_256gpu_gb200_nvfp4_config,
)


@pytest.fixture(autouse=True)
def _keep_recipe_construction_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


@pytest.mark.parametrize("recipe_factory", _E2E_OPTIMIZER_CUDA_GRAPH_RECIPES, ids=lambda factory: factory.__name__)
def test_e2e_full_iteration_recipes_capture_optimizer_step(
    recipe_factory: Callable[[], ConfigContainer],
) -> None:
    """Full-iteration tracker benchmarks capture the optimizer step with settings the capture supports."""
    cfg = recipe_factory()

    assert cfg.optimizer.optimizer_cuda_graph is True
    assert cfg.model.cuda_graph_use_single_mempool is True
    assert cfg.optimizer.store_param_remainders is False
    assert is_full_iteration_cuda_graph(cfg.model)

    # Only Adam is built as capturable, and these settings read GPU values on the host mid-step.
    assert cfg.optimizer.optimizer == "adam"
    assert cfg.optimizer.log_num_zeros_in_grad is False
    assert not math.isfinite(cfg.optimizer.grad_norm_skip_threshold)


@pytest.mark.parametrize("recipe_factory", _RECIPES_WITHOUT_OPTIMIZER_CUDA_GRAPH, ids=lambda factory: factory.__name__)
def test_recipes_keep_eager_optimizer_step(
    recipe_factory: Callable[[], ConfigContainer],
) -> None:
    """Recipes outside the optimizer CUDA graph policy leave the optimizer step eager."""
    assert not recipe_factory().optimizer.optimizer_cuda_graph
