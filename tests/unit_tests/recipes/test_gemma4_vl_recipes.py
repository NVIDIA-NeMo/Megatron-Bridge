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


def test_gemma4_vl_sft_uses_memory_stable_8gpu_contract(monkeypatch: pytest.MonkeyPatch):
    """Full Gemma 4 VL SFT should use the measured single-node memory contract."""
    patch_recipe_module_global(monkeypatch, _gemma4_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _gemma4_vl_module.gemma4_vl_26b_sft_config()

    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.sequence_parallel is True
    assert cfg.model.recompute_granularity == "full"
    assert cfg.model.recompute_modules is None
    assert cfg.model.recompute_method == "uniform"
    assert cfg.model.recompute_num_layers == 1
    assert cfg.model.cuda_graph_impl == "none"
    assert cfg.model.freeze_vision_model is True
    assert cfg.model.freeze_vision_projection is False
    assert cfg.model.freeze_language_model is False
    assert cfg.train.micro_batch_size == 1
    assert cfg.train.global_batch_size == 32
    assert cfg.env_vars["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_gemma4_vl_peft_uses_memory_stable_4gpu_contract(monkeypatch: pytest.MonkeyPatch):
    """Gemma 4 VL PEFT should use the measured single-node LoRA contract."""
    patch_recipe_module_global(monkeypatch, _gemma4_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _gemma4_vl_module.gemma4_vl_26b_peft_config()

    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 4
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.sequence_parallel is True
    assert cfg.model.recompute_granularity is None
    assert cfg.model.recompute_modules is None
    assert cfg.model.cuda_graph_impl == "none"
    assert cfg.train.micro_batch_size == 1
    assert cfg.train.global_batch_size == 32
    assert cfg.peft.target_modules == ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
    assert cfg.peft.dim == 32
    assert cfg.peft.alpha == 32
    assert cfg.peft.dropout == 0.0
    assert cfg.peft.sequence_parallel_input_regather is False
    assert cfg.peft.share_expert_adapters is True
    assert cfg.env_vars["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
