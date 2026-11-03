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
"""The GPT-OSS 120B BF16 B200 and B300 perf recipes run NCCL EP with each EP8 group inside one NVL8 domain."""

import importlib

import pytest

from megatron.bridge.perf_recipes.environment import HYBRID_EP_ENV_NAMES
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit

# (module under megatron.bridge.perf_recipes, recipe factory, expected moe_use_grouped_tensor).
# B300 runs TE CUDA graphs and keeps grouped tensors; the eager B200 recipe was slower with them.
_GPT_OSS_NVL8_NCCLEP_RECIPES = (
    ("gpt_oss.b200.gpt_oss", "gpt_oss_120b_pretrain_64gpu_b200_bf16_config", False),
    ("gpt_oss.b300.gpt_oss", "gpt_oss_120b_pretrain_64gpu_b300_bf16_config", True),
)

# Import at collection time: the offline-construction fixture patches ``AutoBridge`` in every recipe module
# that is already loaded, so a lazy import inside a test would escape the patch.
_RECIPE_MODULES = {
    module_name: importlib.import_module(f"megatron.bridge.perf_recipes.{module_name}")
    for module_name, _, _ in _GPT_OSS_NVL8_NCCLEP_RECIPES
}


@pytest.fixture(autouse=True)
def _keep_recipe_construction_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


@pytest.mark.parametrize(
    ("module_name", "factory_name", "grouped_tensor"), _GPT_OSS_NVL8_NCCLEP_RECIPES, ids=lambda value: str(value)
)
def test_gpt_oss_nvl8_bf16_recipes_use_nccl_ep(module_name: str, factory_name: str, grouped_tensor: bool) -> None:
    cfg = getattr(_RECIPE_MODULES[module_name], factory_name)()

    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "ncclep"
    assert cfg.model.moe_use_grouped_tensor is grouped_tensor
    # Pull/push mode is only valid while each EP group stays inside one NVL8 domain.
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.env_vars["NCCL_EP_HT_EM_PULL_PUSH"] == 1
    assert cfg.env_vars.keys().isdisjoint(HYBRID_EP_ENV_NAMES)
