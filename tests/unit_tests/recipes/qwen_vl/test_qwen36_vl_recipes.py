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

"""Tests for exact Qwen3.6 vision-language recipes."""

from __future__ import annotations

import importlib
import inspect

import pytest


pytestmark = pytest.mark.unit

_module = importlib.import_module("megatron.bridge.recipes.qwen_vl.h100.qwen36_vl")
_MODEL = "Qwen/Qwen3.6-35B-A3B"
_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"  # pragma: allowlist secret


class _FakeModelCfg:
    def finalize(self):
        return None


class _FakeBridge:
    def to_megatron_provider(self, load_weights=False):
        assert load_weights is False
        return _FakeModelCfg()


class _FakeAutoBridge:
    calls = []

    @classmethod
    def from_hf_pretrained(cls, hf_path, **kwargs):
        cls.calls.append((hf_path, kwargs))
        return _FakeBridge()


@pytest.mark.parametrize(
    ("recipe_name", "tp", "ep", "peft"),
    [
        ("qwen36_vl_35b_a3b_sft_16gpu_h100_bf16_config", 2, 16, False),
        ("qwen36_vl_35b_a3b_peft_4gpu_h100_bf16_config", 2, 4, True),
    ],
)
def test_qwen36_vl_recipe_defaults(monkeypatch, recipe_name, tp, ep, peft):
    monkeypatch.setattr(_module, "AutoBridge", _FakeAutoBridge)
    _FakeAutoBridge.calls.clear()

    recipe = getattr(_module, recipe_name)
    cfg = recipe()

    assert not inspect.signature(recipe).parameters
    assert _FakeAutoBridge.calls == [(_MODEL, {"revision": _REVISION})]
    assert cfg.model.tensor_model_parallel_size == tp
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == ep
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.mtp_num_layers == 1
    assert cfg.model.seq_length == 4096
    assert cfg.train.global_batch_size == 32
    assert cfg.train.micro_batch_size == 4
    assert cfg.dataset.hf_processor_path == _MODEL
    assert cfg.dataset.hf_processor_kwargs == {"revision": _REVISION}
    assert cfg.dataset.enable_in_batch_packing is False
    assert cfg.dataset.defer_in_batch_packing_to_step is True
    assert (cfg.peft is not None) is peft
