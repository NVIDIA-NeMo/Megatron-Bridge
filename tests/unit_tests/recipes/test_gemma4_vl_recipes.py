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

"""Unit tests for Gemma 4 VL recipe configuration builders."""

import importlib

import pytest

from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_module_global


_gemma4_vl_module = importlib.import_module("megatron.bridge.recipes.gemma4_vl.gemma4_vl")
_gemma4_vl_h100_module = importlib.import_module("megatron.bridge.recipes.gemma4_vl.h100.gemma4_vl")


class _FakeModelCfg:
    """Fake model configuration for testing."""

    def finalize(self):
        return None


class _FakeAutoBridge:
    """Fake AutoBridge for testing."""

    @staticmethod
    def from_hf_pretrained(hf_path: str):
        return _FakeAutoBridge()

    def to_megatron_provider(self, load_weights: bool = False):
        return _FakeModelCfg()


def test_gemma4_vl_sft_uses_long_distributed_timeout(monkeypatch: pytest.MonkeyPatch):
    """Full Gemma 4 VL SFT should allow long checkpoint-save finalization."""
    patch_recipe_module_global(monkeypatch, _gemma4_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _gemma4_vl_module.gemma4_vl_26b_sft_config()

    assert cfg.dist.distributed_timeout_minutes == 90


def test_gemma4_vl_sft_canonical_recipe_requires_8_gpu_topology(monkeypatch: pytest.MonkeyPatch):
    """The canonical full-SFT recipe should resolve to an eight-rank MoE mesh."""
    patch_recipe_module_global(monkeypatch, _gemma4_vl_h100_module, "AutoBridge", _FakeAutoBridge)

    cfg = _gemma4_vl_h100_module.gemma4_vl_26b_sft_8gpu_h100_bf16_config()
    cfg.model.finalize()

    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.model.expert_tensor_parallel_size == 1
    required_world_size = cfg.model.pipeline_model_parallel_size * max(
        cfg.model.tensor_model_parallel_size * cfg.model.context_parallel_size,
        cfg.model.expert_model_parallel_size * cfg.model.expert_tensor_parallel_size,
    )
    assert required_world_size == 8


def test_gemma4_vl_long_context_sft_uses_packing_and_cp(monkeypatch: pytest.MonkeyPatch):
    """The long-context recipe should own a valid 8K packed CP=2 workload."""
    patch_recipe_module_global(monkeypatch, _gemma4_vl_h100_module, "AutoBridge", _FakeAutoBridge)

    cfg = _gemma4_vl_h100_module.gemma4_vl_26b_sft_long_context_8gpu_h100_bf16_config()
    cfg.model.finalize()

    assert cfg.model.seq_length == 8192
    assert cfg.dataset.seq_length == 8192
    assert cfg.model.context_parallel_size == 2
    assert cfg.model.calculate_per_token_loss is True
    assert cfg.dataset.enable_in_batch_packing is True
    assert cfg.train.micro_batch_size == 2
    assert cfg.ddp.average_in_collective is False
    required_world_size = cfg.model.pipeline_model_parallel_size * max(
        cfg.model.tensor_model_parallel_size * cfg.model.context_parallel_size,
        cfg.model.expert_model_parallel_size * cfg.model.expert_tensor_parallel_size,
    )
    assert required_world_size == 8
