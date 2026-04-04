# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Smoke tests for Qwen2.5-Omni recipe builders."""

import importlib

import pytest

_qwen25_vl = importlib.import_module("megatron.bridge.recipes.qwen_vl.qwen25_vl")


class _FakeModelCfg:
    def __init__(self):
        self.cross_entropy_fusion_impl = "te"

    def finalize(self):
        return None


class _FakeBridge:
    def to_megatron_provider(self, load_weights: bool = False):
        return _FakeModelCfg()

    @staticmethod
    def from_hf_pretrained(hf_path: str, **kwargs):
        return _FakeBridge()


def _assert_basic_config(cfg):
    from megatron.bridge.training.config import ConfigContainer

    assert isinstance(cfg, ConfigContainer)
    assert cfg.model is not None
    assert cfg.train is not None
    assert cfg.optimizer is not None
    assert cfg.scheduler is not None
    assert cfg.dataset is not None
    assert cfg.logger is not None
    assert cfg.tokenizer is not None
    assert cfg.checkpoint is not None
    assert cfg.rng is not None
    assert cfg.train.global_batch_size >= 1
    assert cfg.train.micro_batch_size >= 1


def test_qwen25_omni_7b_finetune_config_builds(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_qwen25_vl, "AutoBridge", _FakeBridge)
    cfg = _qwen25_vl.qwen25_omni_7b_finetune_config(peft_scheme=None)
    _assert_basic_config(cfg)
    assert cfg.model.tensor_model_parallel_size >= 1


def test_qwen25_omni_7b_pretrain_config_builds(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_qwen25_vl, "AutoBridge", _FakeBridge)
    cfg = _qwen25_vl.qwen25_omni_7b_pretrain_config()
    _assert_basic_config(cfg)
    assert cfg.model.freeze_language_model is True
    assert cfg.model.freeze_vision_model is True
