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
"""NCCL EP perf recipes carry the full dispatcher stack; VR200 aliases stay on HybridEP."""

import importlib

import pytest

from megatron.bridge.perf_recipes.environment import HYBRID_EP_ENV_NAMES
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit

_NCCL_EP_ENV = {"NCCL_EP_HT_EM_PULL_PUSH": 1}

# (module under megatron.bridge.perf_recipes, recipe factory) for the GB300 recipes defaulted to NCCL EP.
_GB300_NCCLEP_RECIPES = (
    ("deepseek.gb300.deepseek_v3", "deepseek_v3_pretrain_256gpu_gb300_bf16_config"),
    ("deepseek.gb300.deepseek_v3", "deepseek_v3_pretrain_256gpu_gb300_fp8mx_config"),
    ("deepseek.gb300.deepseek_v3", "deepseek_v3_pretrain_256gpu_gb300_nvfp4_config"),
    ("deepseek.gb300.deepseek_v3", "deepseek_v3_pretrain_64gpu_gb300_fp8mx_fsdp_config"),
    ("deepseek.gb300.deepseek_v3", "deepseek_v3_pretrain_128gpu_gb300_fp8mx_hsdp_config"),
    ("deepseek.gb300.deepseek_v3", "deepseek_v3_pretrain_256gpu_gb300_fp8mx_large_scale_config"),
    ("gpt_oss.gb300.gpt_oss", "gpt_oss_120b_pretrain_64gpu_gb300_fp8mx_config"),
    ("kimi.gb300.kimi_k2", "kimi_k2_pretrain_256gpu_gb300_fp8mx_config"),
    ("nemotronh.gb300.nemotronh", "nemotron_3_super_pretrain_64gpu_gb300_bf16_config"),
    ("nemotronh.gb300.nemotronh", "nemotron_3_super_pretrain_64gpu_gb300_fp8mx_config"),
    ("nemotronh.gb300.nemotronh", "nemotron_3_super_pretrain_64gpu_gb300_nvfp4_config"),
    ("nemotronh.gb300.nemotronh", "nemotron_3_ultra_pretrain_256gpu_gb300_fp8mx_config"),
    ("nemotronh.gb300.nemotronh", "nemotron_3_nano_pretrain_8gpu_gb300_bf16_config"),
    ("nemotronh.gb300.nemotronh", "nemotron_3_nano_pretrain_8gpu_gb300_fp8mx_config"),
    ("nemotronh.gb300.nemotronh", "nemotron_3_5_lightning_pretrain_8gpu_gb300_bf16_config"),
    ("nemotronh.gb300.nemotronh", "nemotron_3_5_lightning_pretrain_8gpu_gb300_fp8mx_config"),
    ("qwen.gb300.qwen3_moe", "qwen3_30b_a3b_pretrain_8gpu_gb300_bf16_config"),
    ("qwen.gb300.qwen3_moe", "qwen3_30b_a3b_pretrain_8gpu_gb300_fp8mx_config"),
    ("qwen.gb300.qwen3_moe", "qwen3_30b_a3b_pretrain_32gpu_gb300_bf16_config"),
    ("qwen.gb300.qwen3_moe", "qwen3_235b_a22b_pretrain_256gpu_gb300_bf16_config"),
    ("qwen.gb300.qwen3_moe", "qwen3_235b_a22b_pretrain_256gpu_gb300_fp8mx_config"),
    ("qwen.gb300.qwen3_moe", "qwen3_235b_a22b_pretrain_256gpu_gb300_fp8mx_large_scale_config"),
    ("qwen.gb300.qwen3_moe", "qwen3_235b_a22b_pretrain_256gpu_gb300_nvfp4_config"),
    ("qwen_vl.gb300.qwen35_vl", "qwen35_vl_35b_a3b_pretrain_8gpu_gb300_bf16_config"),
)

# Ultra NVL8 recipes use the same NCCL EP stack as GB300.
_ULTRA_NVL8_RECIPES = (
    ("nemotronh.b200.nemotronh", "nemotron_3_ultra_pretrain_256gpu_b200_fp8mx_config"),
    ("nemotronh.b300.nemotronh", "nemotron_3_ultra_pretrain_256gpu_b300_fp8mx_config"),
)

# VR200 aliases of the same recipes: they reuse the shared HybridEP builders and must not pick up NCCL EP.
_VR200_ALIASES = (
    ("deepseek.vr200.deepseek_v3", "deepseek_v3_pretrain_256gpu_vr200_bf16_config"),
    ("deepseek.vr200.deepseek_v3", "deepseek_v3_pretrain_256gpu_vr200_fp8mx_config"),
    ("deepseek.vr200.deepseek_v3", "deepseek_v3_pretrain_256gpu_vr200_nvfp4_config"),
    ("gpt_oss.vr200.gpt_oss", "gpt_oss_120b_pretrain_64gpu_vr200_fp8mx_config"),
    ("kimi.vr200.kimi_k2", "kimi_k2_pretrain_256gpu_vr200_fp8mx_config"),
    ("nemotronh.vr200.nemotronh", "nemotron_3_nano_pretrain_8gpu_vr200_bf16_config"),
    ("nemotronh.vr200.nemotronh", "nemotron_3_super_pretrain_64gpu_vr200_bf16_config"),
    ("nemotronh.vr200.nemotronh", "nemotron_3_super_pretrain_64gpu_vr200_nvfp4_config"),
    ("nemotronh.vr200.nemotronh", "nemotron_3_ultra_pretrain_256gpu_vr200_fp8mx_config"),
    ("nemotronh.vr200.nemotronh", "nemotron_3_5_lightning_pretrain_8gpu_vr200_bf16_config"),
    ("nemotronh.vr200.nemotronh", "nemotron_3_5_lightning_pretrain_8gpu_vr200_fp8mx_config"),
    ("deepseek.vr200.deepseek_v3", "deepseek_v3_pretrain_64gpu_vr200_fp8mx_proxy_config"),
    ("deepseek.vr200.deepseek_v3", "deepseek_v3_pretrain_64gpu_vr200_nvfp4_proxy_config"),
    ("qwen.vr200.qwen3_moe", "qwen3_235b_a22b_pretrain_256gpu_vr200_bf16_config"),
    ("qwen.vr200.qwen3_moe", "qwen3_235b_a22b_pretrain_256gpu_vr200_fp8mx_config"),
    ("qwen.vr200.qwen3_moe", "qwen3_235b_a22b_pretrain_256gpu_vr200_nvfp4_config"),
    ("qwen.vr200.qwen3_moe", "qwen3_30b_a3b_pretrain_8gpu_vr200_bf16_config"),
    ("qwen.vr200.qwen3_moe", "qwen3_30b_a3b_pretrain_8gpu_vr200_fp8mx_config"),
    ("qwen_vl.vr200.qwen35_vl", "qwen35_vl_35b_a3b_pretrain_8gpu_vr200_bf16_config"),
)


# Import the recipe modules at collection time: the offline-construction fixture patches ``AutoBridge`` in
# every recipe module that is already loaded, so a lazy import inside a test would escape the patch.
_RECIPE_MODULES = {
    module_name: importlib.import_module(f"megatron.bridge.perf_recipes.{module_name}")
    for module_name in sorted({name for name, _ in (*_GB300_NCCLEP_RECIPES, *_ULTRA_NVL8_RECIPES, *_VR200_ALIASES)})
}


def _build(module_name: str, factory_name: str):
    return getattr(_RECIPE_MODULES[module_name], factory_name)()


@pytest.fixture(autouse=True)
def _keep_recipe_construction_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


@pytest.mark.parametrize(
    ("module_name", "factory_name"), (*_GB300_NCCLEP_RECIPES, *_ULTRA_NVL8_RECIPES), ids=lambda value: value
)
def test_moe_perf_recipes_default_to_the_nccl_ep_stack(module_name: str, factory_name: str) -> None:
    cfg = _build(module_name, factory_name)

    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "ncclep"
    # Device-side expert token counts; the legacy grouped-MLP path host-syncs tokens_per_expert per layer.
    assert cfg.model.moe_use_grouped_tensor is True
    # The plugin runs in hierarchical-topology pull mode, and nothing reads the HybridEP tuning anymore.
    assert {name: cfg.env_vars[name] for name in _NCCL_EP_ENV} == _NCCL_EP_ENV
    assert cfg.env_vars.keys().isdisjoint(HYBRID_EP_ENV_NAMES)


@pytest.mark.parametrize(("module_name", "factory_name"), _VR200_ALIASES, ids=lambda value: value)
def test_vr200_aliases_do_not_inherit_nccl_ep(module_name: str, factory_name: str) -> None:
    cfg = _build(module_name, factory_name)

    # The aliases keep the dispatcher their HybridEP base had upstream (flex/hybridep, or alltoall for the
    # DeepSeek BF16 and NVFP4 recipes); the GB300 NCCL EP overlay must never leak into them.
    assert cfg.model.moe_flex_dispatcher_backend != "ncclep"
    assert cfg.env_vars.keys().isdisjoint(_NCCL_EP_ENV)
    if cfg.model.moe_token_dispatcher_type == "flex":
        assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
        assert cfg.env_vars.keys() >= HYBRID_EP_ENV_NAMES
    else:
        assert cfg.model.moe_token_dispatcher_type == "alltoall"


@pytest.mark.parametrize(("module_name", "factory_name"), _ULTRA_NVL8_RECIPES, ids=lambda value: value)
def test_ultra_nvl8_perf_recipes_use_full_fsdp(module_name: str, factory_name: str) -> None:
    cfg = _build(module_name, factory_name)

    tp = 2 if ".b200." in module_name else 1
    assert cfg.model.tensor_model_parallel_size == tp
    assert cfg.model.sequence_parallel is (tp > 1)
    assert cfg.model.expert_model_parallel_size == 64
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.seq_length == 8192
    assert cfg.train.global_batch_size == 256
    assert cfg.train.micro_batch_size == 1
    assert cfg.ddp.use_megatron_fsdp is True
    assert cfg.ddp.num_distributed_optimizer_instances == 1
    assert cfg.ddp.data_parallel_sharding_strategy == "optim_grads_params"
    assert cfg.ddp.outer_dp_sharding_strategy == "no_shard"
    assert cfg.model.fine_grained_activation_offloading is True
    assert cfg.model.offload_modules == ["fused_group_mlp"]
    assert cfg.model.min_offloaded_tensor_size == (350_000_000 if tp == 2 else 500_000_000)
    assert cfg.model.recompute_modules == ["moe_act"]
    assert cfg.env_vars["NVTE_CPU_OFFLOAD_V1"] == 1
    assert cfg.env_vars["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == 1

    exports = importlib.import_module("megatron.bridge.perf_recipes.nemotronh")
    assert getattr(exports, factory_name) is getattr(_RECIPE_MODULES[module_name], factory_name)
