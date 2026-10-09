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
"""Every GB200 MoE perf recipe switched to NCCL EP carries the full NCCL EP stack, and the GB200 recipes
left on HybridEP, several of which share a base with a switched recipe, do not pick it up."""

import importlib

import pytest

from megatron.bridge.perf_recipes.environment import HYBRID_EP_ENV_NAMES
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit

_NCCL_EP_ENV = {"NCCL_EP_HT_EM_PULL_PUSH": 1}

# (module under megatron.bridge.perf_recipes, recipe factory) for the GB200 recipes defaulted to NCCL EP: the
# nemo-ci GB200 benchmarks whose GB300 counterparts already run NCCL EP.
_GB200_NCCLEP_RECIPES = (
    ("deepseek.gb200.deepseek_v3", "deepseek_v3_pretrain_256gpu_gb200_bf16_config"),
    ("deepseek.gb200.deepseek_v3", "deepseek_v3_pretrain_256gpu_gb200_fp8mx_config"),
    ("deepseek.gb200.deepseek_v3", "deepseek_v3_pretrain_256gpu_gb200_nvfp4_config"),
    ("deepseek.gb200.deepseek_v3", "deepseek_v3_pretrain_256gpu_gb200_fp8mx_large_scale_config"),
    ("gpt_oss.gb200.gpt_oss", "gpt_oss_120b_pretrain_64gpu_gb200_fp8mx_config"),
    ("kimi.gb200.kimi_k2", "kimi_k2_pretrain_256gpu_gb200_fp8mx_config"),
    ("nemotronh.gb200.nemotronh", "nemotron_3_super_pretrain_64gpu_gb200_bf16_config"),
    ("nemotronh.gb200.nemotronh", "nemotron_3_super_pretrain_64gpu_gb200_fp8mx_config"),
    ("nemotronh.gb200.nemotronh", "nemotron_3_super_pretrain_64gpu_gb200_nvfp4_config"),
    ("nemotronh.gb200.nemotronh", "nemotron_3_ultra_pretrain_256gpu_gb200_fp8mx_config"),
    ("nemotronh.gb200.nemotronh", "nemotron_3_nano_pretrain_8gpu_gb200_bf16_config"),
    ("nemotronh.gb200.nemotronh", "nemotron_3_nano_pretrain_8gpu_gb200_fp8mx_config"),
    ("nemotronh.gb200.nemotronh", "nemotron_3_5_lightning_pretrain_8gpu_gb200_bf16_config"),
    ("nemotronh.gb200.nemotronh", "nemotron_3_5_lightning_pretrain_8gpu_gb200_fp8mx_config"),
    ("qwen.gb200.qwen3_moe", "qwen3_30b_a3b_pretrain_8gpu_gb200_bf16_config"),
    ("qwen.gb200.qwen3_moe", "qwen3_30b_a3b_pretrain_8gpu_gb200_fp8mx_config"),
    ("qwen.gb200.qwen3_moe", "qwen3_30b_a3b_pretrain_32gpu_gb200_bf16_config"),
    ("qwen.gb200.qwen3_moe", "qwen3_235b_a22b_pretrain_256gpu_gb200_bf16_config"),
    ("qwen.gb200.qwen3_moe", "qwen3_235b_a22b_pretrain_256gpu_gb200_fp8mx_config"),
    ("qwen.gb200.qwen3_moe", "qwen3_235b_a22b_pretrain_256gpu_gb200_fp8mx_large_scale_config"),
    ("qwen.gb200.qwen3_moe", "qwen3_235b_a22b_pretrain_256gpu_gb200_nvfp4_config"),
)

# GB200 recipes that stay on HybridEP. Most build on the same base or parent recipe as a recipe above (the
# shared DeepSeek BF16 builder, the Nano/Lightning MXFP8 base, the FP8-CS parents of the NVFP4 recipes), so
# the NCCL EP overlay must not leak into them.
_GB200_HYBRIDEP_RECIPES = (
    ("deepseek.gb200.deepseek_v3", "deepseek_v3_pretrain_256gpu_gb200_fp8cs_config"),
    ("deepseek.gb200.deepseek_v3", "deepseek_v3_pretrain_256gpu_gb200_fp8mx_partial_cg_dev_config"),
    ("nemotronh.gb200.nemotronh", "nemotron_3_ultra_pretrain_256gpu_gb200_nvfp4_config"),
    ("nemotronh.gb200.nemotronh", "nemotron_3_nano_pretrain_8gpu_gb200_nvfp4_config"),
    ("nemotronh.gb200.nemotronh", "nemotron_3_5_lightning_pretrain_8gpu_gb200_fp8mx_fsdp_config"),
    ("nemotronh.gb200.nemotronh", "nemotron_3_5_lightning_pretrain_8gpu_gb200_nvfp4_config"),
    ("qwen.gb200.qwen3_moe", "qwen3_30b_a3b_pretrain_8gpu_gb200_fp8cs_config"),
    ("qwen.gb200.qwen3_moe", "qwen3_30b_a3b_pretrain_8gpu_gb200_nvfp4_config"),
    ("qwen.gb200.qwen3_moe", "qwen3_30b_a3b_pretrain_32gpu_gb200_fp8cs_config"),
    ("qwen.gb200.qwen3_moe", "qwen3_235b_a22b_pretrain_256gpu_gb200_fp8cs_config"),
    ("qwen.gb200.qwen3_moe", "qwen3_235b_a22b_pretrain_64gpu_gb200_nvfp4_full_iteration_config"),
)


# Import the recipe modules at collection time: the offline-construction fixture patches ``AutoBridge`` in
# every recipe module that is already loaded, so a lazy import inside a test would escape the patch.
_RECIPE_MODULES = {
    module_name: importlib.import_module(f"megatron.bridge.perf_recipes.{module_name}")
    for module_name in sorted({name for name, _ in (*_GB200_NCCLEP_RECIPES, *_GB200_HYBRIDEP_RECIPES)})
}


def _build(module_name: str, factory_name: str):
    return getattr(_RECIPE_MODULES[module_name], factory_name)()


@pytest.fixture(autouse=True)
def _keep_recipe_construction_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


@pytest.mark.parametrize(("module_name", "factory_name"), _GB200_NCCLEP_RECIPES, ids=lambda value: value)
def test_gb200_moe_perf_recipes_default_to_the_nccl_ep_stack(module_name: str, factory_name: str) -> None:
    cfg = _build(module_name, factory_name)

    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "ncclep"
    # Device-side expert token counts; the legacy grouped-MLP path host-syncs tokens_per_expert per layer.
    assert cfg.model.moe_use_grouped_tensor is True
    # The plugin runs in hierarchical-topology pull mode, and nothing reads the HybridEP tuning anymore.
    assert {name: cfg.env_vars[name] for name in _NCCL_EP_ENV} == _NCCL_EP_ENV
    assert cfg.env_vars.keys().isdisjoint(HYBRID_EP_ENV_NAMES)


@pytest.mark.parametrize(("module_name", "factory_name"), _GB200_HYBRIDEP_RECIPES, ids=lambda value: value)
def test_gb200_hybridep_recipes_do_not_inherit_nccl_ep(module_name: str, factory_name: str) -> None:
    cfg = _build(module_name, factory_name)

    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.env_vars.keys().isdisjoint(_NCCL_EP_ENV)
    assert cfg.env_vars.keys() >= HYBRID_EP_ENV_NAMES
