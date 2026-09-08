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

"""Recompute policy for E2E-owned performance tracker benchmarks."""

from collections.abc import Callable

import pytest

from megatron.bridge.perf_recipes.gpt_oss import (
    gpt_oss_120b_pretrain_64gpu_gb300_bf16_config,
    gpt_oss_120b_pretrain_64gpu_gb300_fp8mx_config,
)
from megatron.bridge.perf_recipes.nemotronh import (
    nemotron_3_5_lightning_pretrain_8gpu_gb200_bf16_config,
    nemotron_3_5_lightning_pretrain_8gpu_gb200_fp8mx_config,
    nemotron_3_5_lightning_pretrain_8gpu_gb300_bf16_config,
    nemotron_3_5_lightning_pretrain_8gpu_gb300_fp8mx_config,
)
from megatron.bridge.perf_recipes.qwen import (
    qwen3_30b_a3b_pretrain_8gpu_gb200_bf16_config,
    qwen3_30b_a3b_pretrain_8gpu_gb200_fp8mx_config,
    qwen3_30b_a3b_pretrain_8gpu_gb300_bf16_config,
    qwen3_30b_a3b_pretrain_8gpu_gb300_fp8mx_config,
    qwen3_235b_a22b_pretrain_256gpu_gb200_bf16_config,
    qwen3_235b_a22b_pretrain_256gpu_gb200_fp8mx_config,
    qwen3_235b_a22b_pretrain_256gpu_gb200_nvfp4_config,
    qwen3_235b_a22b_pretrain_256gpu_gb300_bf16_config,
    qwen3_235b_a22b_pretrain_256gpu_gb300_fp8mx_config,
    qwen3_235b_a22b_pretrain_256gpu_gb300_nvfp4_config,
)
from megatron.bridge.training.config import ConfigContainer
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit

_E2E_NO_RECOMPUTE_RECIPES = (
    gpt_oss_120b_pretrain_64gpu_gb300_bf16_config,
    gpt_oss_120b_pretrain_64gpu_gb300_fp8mx_config,
    qwen3_30b_a3b_pretrain_8gpu_gb200_bf16_config,
    qwen3_30b_a3b_pretrain_8gpu_gb200_fp8mx_config,
    qwen3_30b_a3b_pretrain_8gpu_gb300_bf16_config,
    qwen3_30b_a3b_pretrain_8gpu_gb300_fp8mx_config,
    qwen3_235b_a22b_pretrain_256gpu_gb200_bf16_config,
    qwen3_235b_a22b_pretrain_256gpu_gb200_fp8mx_config,
    qwen3_235b_a22b_pretrain_256gpu_gb200_nvfp4_config,
    qwen3_235b_a22b_pretrain_256gpu_gb300_bf16_config,
    qwen3_235b_a22b_pretrain_256gpu_gb300_fp8mx_config,
    qwen3_235b_a22b_pretrain_256gpu_gb300_nvfp4_config,
    nemotron_3_5_lightning_pretrain_8gpu_gb200_bf16_config,
    nemotron_3_5_lightning_pretrain_8gpu_gb200_fp8mx_config,
    nemotron_3_5_lightning_pretrain_8gpu_gb300_bf16_config,
    nemotron_3_5_lightning_pretrain_8gpu_gb300_fp8mx_config,
)


@pytest.fixture(autouse=True)
def _keep_recipe_construction_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


@pytest.mark.parametrize("recipe_factory", _E2E_NO_RECOMPUTE_RECIPES, ids=lambda factory: factory.__name__)
def test_e2e_perf_recipes_disable_unneeded_recompute(
    recipe_factory: Callable[[], ConfigContainer],
) -> None:
    """E2E benchmark recipes do not inherit the implicit core-attention default."""
    assert recipe_factory().model.recompute_modules == []
