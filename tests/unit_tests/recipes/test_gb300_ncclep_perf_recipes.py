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
"""Every GB300 MoE perf recipe switched to NCCL EP carries the full NCCL EP stack, and the VRNVL72
aliases built on the shared HybridEP bases stay on HybridEP. VRNVL72 aliases built on an NCCL EP base
carry the same NCCL EP stack."""

import importlib

import pytest
from megatron.training.determinism import DETERMINISM_ENV_VAR_DEFAULTS

from megatron.bridge.perf_recipes.environment import HYBRID_EP_ENV_NAMES
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit

_NCCL_EP_ENV = {"NCCL_EP_HT_EM_PULL_PUSH": 1}
_SSM_DETERMINISM_ENV = {"MAMBA_DETERMINISTIC": "1", "CAUSAL_CONV1D_DETERMINISTIC": "1"}

# (module under megatron.bridge.perf_recipes, recipe factory) for the GB300 recipes defaulted to NCCL EP.
_GB300_NCCLEP_RECIPES = (
    ("deepseek.gb300.deepseek_v3", "deepseek_v3_pretrain_256gpu_gb300_bf16_config"),
    ("deepseek.gb300.deepseek_v3", "deepseek_v3_pretrain_256gpu_gb300_fp8mx_config"),
    ("deepseek.gb300.deepseek_v3", "deepseek_v3_pretrain_256gpu_gb300_nvfp4_config"),
    ("deepseek.gb300.deepseek_v3", "deepseek_v3_pretrain_64gpu_gb300_fp8mx_fsdp_config"),
    ("deepseek.gb300.deepseek_v3", "deepseek_v3_pretrain_256gpu_gb300_fp8mx_hsdp_mfsdpv2_config"),
    ("deepseek.gb300.deepseek_v3", "deepseek_v3_pretrain_256gpu_gb300_fp8mx_large_scale_config"),
    ("gpt_oss.gb300.gpt_oss", "gpt_oss_120b_pretrain_64gpu_gb300_fp8mx_config"),
    ("kimi.gb300.kimi_k2", "kimi_k2_pretrain_256gpu_gb300_fp8mx_config"),
    ("nemotronh.gb300.nemotronh", "nemotron_3_super_pretrain_64gpu_gb300_bf16_config"),
    ("nemotronh.gb300.nemotronh", "nemotron_3_super_pretrain_64gpu_gb300_fp8mx_config"),
    ("nemotronh.gb300.nemotronh", "nemotron_3_super_pretrain_64gpu_gb300_nvfp4_config"),
    ("nemotronh.gb300.nemotronh", "nemotron_3_ultra_pretrain_256gpu_gb300_fp8mx_config"),
    ("nemotronh.gb300.nemotronh", "nemotron_3_ultra_pretrain_256gpu_gb300_nvfp4_config"),
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

# VRNVL72 aliases that share the GB300 NCCL EP builder instead of a HybridEP base.
_VRNVL72_NCCLEP_RECIPES = (
    ("nemotronh.vrnvl72.nemotronh", "nemotron_3_ultra_pretrain_256gpu_vrnvl72_fp8mx_config"),
    ("nemotronh.vrnvl72.nemotronh", "nemotron_3_ultra_pretrain_256gpu_vrnvl72_nvfp4_config"),
)

# VRNVL72 aliases of the same recipes: they reuse the shared HybridEP builders and must not pick up NCCL EP.
_VRNVL72_ALIASES = (
    ("deepseek.vrnvl72.deepseek_v3", "deepseek_v3_pretrain_256gpu_vrnvl72_bf16_config"),
    ("deepseek.vrnvl72.deepseek_v3", "deepseek_v3_pretrain_256gpu_vrnvl72_fp8mx_config"),
    ("deepseek.vrnvl72.deepseek_v3", "deepseek_v3_pretrain_256gpu_vrnvl72_nvfp4_config"),
    ("gpt_oss.vrnvl72.gpt_oss", "gpt_oss_120b_pretrain_64gpu_vrnvl72_fp8mx_config"),
    ("kimi.vrnvl72.kimi_k2", "kimi_k2_pretrain_256gpu_vrnvl72_fp8mx_config"),
    ("nemotronh.vrnvl72.nemotronh", "nemotron_3_nano_pretrain_8gpu_vrnvl72_bf16_config"),
    ("nemotronh.vrnvl72.nemotronh", "nemotron_3_super_pretrain_64gpu_vrnvl72_bf16_config"),
    ("nemotronh.vrnvl72.nemotronh", "nemotron_3_super_pretrain_64gpu_vrnvl72_nvfp4_config"),
    ("nemotronh.vrnvl72.nemotronh", "nemotron_3_5_lightning_pretrain_8gpu_vrnvl72_bf16_config"),
    ("nemotronh.vrnvl72.nemotronh", "nemotron_3_5_lightning_pretrain_8gpu_vrnvl72_fp8mx_config"),
    ("deepseek.vrnvl72.deepseek_v3", "deepseek_v3_pretrain_64gpu_vrnvl72_fp8mx_proxy_config"),
    ("deepseek.vrnvl72.deepseek_v3", "deepseek_v3_pretrain_64gpu_vrnvl72_nvfp4_proxy_config"),
    ("deepseek.vrnvl72.deepseek_v3", "deepseek_v3_pretrain_4gpu_vrnvl72_fp8mx_proxy_config"),
    ("deepseek.vrnvl72.deepseek_v3", "deepseek_v3_pretrain_4gpu_vrnvl72_nvfp4_proxy_config"),
    ("qwen.vrnvl72.qwen3_moe", "qwen3_235b_a22b_pretrain_256gpu_vrnvl72_bf16_config"),
    ("qwen.vrnvl72.qwen3_moe", "qwen3_235b_a22b_pretrain_256gpu_vrnvl72_fp8mx_config"),
    ("qwen.vrnvl72.qwen3_moe", "qwen3_235b_a22b_pretrain_256gpu_vrnvl72_nvfp4_config"),
    ("qwen.vrnvl72.qwen3_moe", "qwen3_30b_a3b_pretrain_8gpu_vrnvl72_bf16_config"),
    ("qwen.vrnvl72.qwen3_moe", "qwen3_30b_a3b_pretrain_8gpu_vrnvl72_fp8mx_config"),
    ("qwen_vl.vrnvl72.qwen35_vl", "qwen35_vl_35b_a3b_pretrain_8gpu_vrnvl72_bf16_config"),
)


# Import the recipe modules at collection time: the offline-construction fixture patches ``AutoBridge`` in
# every recipe module that is already loaded, so a lazy import inside a test would escape the patch.
_RECIPE_MODULES = {
    module_name: importlib.import_module(f"megatron.bridge.perf_recipes.{module_name}")
    for module_name in sorted({name for name, _ in (*_GB300_NCCLEP_RECIPES, *_VRNVL72_ALIASES)})
}


def _build(module_name: str, factory_name: str):
    return getattr(_RECIPE_MODULES[module_name], factory_name)()


@pytest.fixture(autouse=True)
def _keep_recipe_construction_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


@pytest.mark.parametrize(
    ("module_name", "factory_name"), (*_GB300_NCCLEP_RECIPES, *_VRNVL72_NCCLEP_RECIPES), ids=lambda value: value
)
def test_gb300_moe_perf_recipes_default_to_the_nccl_ep_stack(module_name: str, factory_name: str) -> None:
    cfg = _build(module_name, factory_name)

    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "ncclep"
    # Device-side expert token counts; the legacy grouped-MLP path host-syncs tokens_per_expert per layer.
    assert cfg.model.moe_use_grouped_tensor is True
    # The plugin runs in hierarchical-topology pull mode, and nothing reads the HybridEP tuning anymore.
    assert {name: cfg.env_vars[name] for name in _NCCL_EP_ENV} == _NCCL_EP_ENV
    assert cfg.env_vars.keys().isdisjoint(HYBRID_EP_ENV_NAMES)


def test_gb300_deepseek_hsdp_uses_hybridep() -> None:
    cfg = _build("deepseek.gb300.deepseek_v3", "deepseek_v3_pretrain_256gpu_gb300_fp8mx_hsdp_config")

    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.moe_use_grouped_tensor is True
    assert cfg.env_vars.keys() >= HYBRID_EP_ENV_NAMES
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == cfg.model.expert_model_parallel_size == 64
    assert cfg.env_vars.keys().isdisjoint(_NCCL_EP_ENV)


@pytest.mark.parametrize(("module_name", "factory_name"), _VRNVL72_ALIASES, ids=lambda value: value)
def test_vrnvl72_aliases_do_not_inherit_nccl_ep(module_name: str, factory_name: str) -> None:
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


def test_deepseek_v3_gb300_fp8mx_deterministic_is_base_plus_determinism_overrides() -> None:
    module = "deepseek.gb300.deepseek_v3"
    base = _build(module, "deepseek_v3_pretrain_256gpu_gb300_fp8mx_config")
    cfg = _build(module, "deepseek_v3_pretrain_256gpu_gb300_fp8mx_deterministic_config")

    assert cfg.model.deterministic_mode is True
    assert cfg.model.cross_entropy_loss_fusion is False
    assert cfg.model.moe_router_aux_loss_fusion is False
    assert cfg.model.moe_router_fusion == base.model.moe_router_fusion
    # Same NCCL EP stack as the base recipe; only the determinism overrides differ.
    assert cfg.model.moe_token_dispatcher_type == base.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == base.model.moe_flex_dispatcher_backend == "ncclep"
    assert cfg.model.moe_use_grouped_tensor is base.model.moe_use_grouped_tensor is True
    assert cfg.env_vars.keys().isdisjoint(HYBRID_EP_ENV_NAMES)
    assert cfg.env_vars == {**base.env_vars, **DETERMINISM_ENV_VAR_DEFAULTS, **_SSM_DETERMINISM_ENV}
