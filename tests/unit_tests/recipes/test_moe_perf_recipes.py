# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import importlib

import pytest

from megatron.bridge.perf_recipes.deepseek.gb200.deepseek_v3 import (
    deepseek_v3_pretrain_256gpu_gb200_fp8mx_config,
)
from megatron.bridge.perf_recipes.gpt_oss.gb200.gpt_oss import (
    gpt_oss_120b_pretrain_64gpu_gb200_fp8mx_config,
)
from megatron.bridge.perf_recipes.qwen.gb200.qwen3_moe import (
    qwen3_30b_a3b_pretrain_8gpu_gb200_fp8mx_config,
    qwen3_235b_a22b_pretrain_64gpu_gb200_fp8mx_config,
)
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_module_global


class _FakeModelCfg:
    """Minimal model provider for Qwen and DeepSeek recipe construction."""

    def __init__(self):
        self.apply_rope_fusion = False
        self.context_parallel_size = 1
        self.cross_entropy_fusion_impl = "native"
        self.make_vocab_size_divisible_by = 128
        self.moe_expert_rank_capacity_factor = None
        self.moe_pad_experts_for_cuda_graph_inference = False
        self.moe_paged_stash = False
        self.num_moe_experts = 0
        self.rotary_base = 10000.0
        self.vocab_size = 1024

    def finalize(self):
        return None


class _FakeBridge:
    @staticmethod
    def from_hf_pretrained(hf_path: str, **kwargs):
        return _FakeBridge()

    def to_megatron_provider(self, load_weights: bool = False):
        return _FakeModelCfg()


@pytest.fixture(autouse=True)
def _patch_hf_bridges(monkeypatch: pytest.MonkeyPatch):
    """Keep perf-recipe tests independent of the HF cache and network."""
    for module_name in (
        "megatron.bridge.recipes.deepseek.deepseek_v3",
        "megatron.bridge.recipes.gpt_oss.h100.gpt_oss",
        "megatron.bridge.recipes.qwen.qwen3_moe",
    ):
        module = importlib.import_module(module_name)
        patch_recipe_module_global(monkeypatch, module, "AutoBridge", _FakeBridge)


@pytest.mark.parametrize(
    "recipe",
    [
        deepseek_v3_pretrain_256gpu_gb200_fp8mx_config,
        gpt_oss_120b_pretrain_64gpu_gb200_fp8mx_config,
        qwen3_30b_a3b_pretrain_8gpu_gb200_fp8mx_config,
    ],
)
def test_moe_perf_recipes_use_natural_router_loads(recipe):
    cfg = recipe()

    assert cfg.model.moe_router_force_load_balancing is False
    assert cfg.model.use_transformer_engine_op_fuser is True


def test_qwen3_30b_gb200_natural_routing_avoids_full_iteration_graphs():
    cfg = qwen3_30b_a3b_pretrain_8gpu_gb200_fp8mx_config()

    assert cfg.model.cuda_graph_impl == "transformer_engine"
    assert cfg.model.cuda_graph_scope == ["attn", "moe_router", "moe_preprocess"]
    assert cfg.model.moe_paged_stash is False
    assert cfg.model.moe_pad_experts_for_cuda_graph_inference is False
    assert cfg.model.moe_expert_rank_capacity_factor is None


def test_qwen3_existing_full_iteration_graph_settings_are_preserved():
    cfg = qwen3_235b_a22b_pretrain_64gpu_gb200_fp8mx_config()

    assert cfg.model.cuda_graph_impl == "full_iteration"
    assert cfg.model.cuda_graph_scope == []
    assert cfg.model.moe_paged_stash is True
    assert cfg.model.moe_pad_experts_for_cuda_graph_inference is True
    assert cfg.model.moe_expert_rank_capacity_factor == 1.5
    assert cfg.rng.te_rng_tracker is True
    assert cfg.model.use_te_rng_tracker is True
    assert cfg.model.use_transformer_engine_op_fuser is True
    assert cfg.comm_overlap.tp_comm_overlap is True
    assert cfg.comm_overlap.overlap_moe_expert_parallel_comm is True
    assert cfg.comm_overlap.delay_wgrad_compute is True
