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

"""Unit tests for DeepSeek flat performance recipes."""

import pytest

from megatron.bridge.perf_recipes.deepseek import (
    deepseek_v3_pretrain_256gpu_vr200_bf16_config,
    deepseek_v3_pretrain_256gpu_vr200_nvfp4_config,
)
from megatron.bridge.training.config import ConfigContainer
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _keep_recipe_construction_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


def _assert_memory_safe_vr200_layout(cfg: ConfigContainer) -> None:
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.virtual_pipeline_model_parallel_size == 4
    assert cfg.model.expert_model_parallel_size == 64
    assert cfg.train.micro_batch_size == 1
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 64


def test_deepseek_v3_256gpu_vr200_bf16_uses_memory_safe_layout() -> None:
    cfg = deepseek_v3_pretrain_256gpu_vr200_bf16_config()

    _assert_memory_safe_vr200_layout(cfg)
    assert cfg.model.recompute_modules == ["mla_up_proj"]


def test_deepseek_v3_256gpu_vr200_nvfp4_uses_memory_safe_layout() -> None:
    cfg = deepseek_v3_pretrain_256gpu_vr200_nvfp4_config()

    _assert_memory_safe_vr200_layout(cfg)
    assert cfg.model.recompute_modules == ["mlp"]
    assert cfg.optimizer.overlap_param_gather_with_optimizer_step is False
    assert cfg.comm_overlap.overlap_param_gather_with_optimizer_step is None
