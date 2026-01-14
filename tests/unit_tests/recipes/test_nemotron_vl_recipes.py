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

import importlib

import pytest
import torch


_nemotron_module = importlib.import_module("megatron.bridge.recipes.nemotron_vl.nemotron_nano_v2_vl")


def _safe_overrides() -> dict:
    """Create safe test overrides for Nemotron VL recipe functions."""
    return {
        "name": "unit_nemotron_vl",
        "dir": ".",
        "hf_model_path": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
        "train_iters": 10,
        "global_batch_size": 2,
        "micro_batch_size": 1,
        "seq_length": 64,
        "lr": 1e-4,
        "min_lr": 1e-5,
        "lr_warmup_iters": 2,
        "tensor_parallelism": 1,
        "pipeline_parallelism": 1,
        "context_parallelism": 1,
        "sequence_parallelism": False,
    }


class _FakeModelCfg:
    """Fake model configuration for testing."""

    def __init__(self):
        self.tensor_model_parallel_size = 1
        self.pipeline_model_parallel_size = 1
        self.pipeline_dtype = None
        self.virtual_pipeline_model_parallel_size = None
        self.context_parallel_size = 1
        self.sequence_parallel = False
        self.seq_length = 64
        self.freeze_language_model = False
        self.freeze_vision_model = False
        self.freeze_vision_projection = False

    def finalize(self):
        return None


class _FakeAutoBridge:
    """Fake AutoBridge for testing to avoid HF downloads and I/O."""

    @staticmethod
    def from_hf_pretrained(hf_path: str, *args, **kwargs):
        return _FakeAutoBridge()

    def to_megatron_provider(self, load_weights: bool = False):
        return _FakeModelCfg()


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
    assert cfg.dataset.seq_length >= 1


def test_nemotron_vl_pretrain_builds_config(monkeypatch: pytest.MonkeyPatch):
    """Test that pretrain_config builds a valid configuration and sets basic fields."""
    monkeypatch.setattr(_nemotron_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides()
    cfg = _nemotron_module.nemotron_nano_v2_vl_12b_pretrain_config(**overrides)

    _assert_basic_config(cfg)

    # Dataset provider should be HF-based
    from megatron.bridge.data.vlm_datasets import HFDatasetConversationProvider

    assert isinstance(cfg.dataset, HFDatasetConversationProvider)

    # Null tokenizer is used
    assert getattr(cfg.tokenizer, "tokenizer_type", None) == "NullTokenizer"

    # Parallelism settings should be wired into model cfg
    assert getattr(cfg.model, "tensor_model_parallel_size", 0) == overrides["tensor_parallelism"]
    assert getattr(cfg.model, "pipeline_model_parallel_size", 0) == overrides["pipeline_parallelism"]
    assert getattr(cfg.model, "context_parallel_size", 0) == overrides["context_parallelism"]
    assert getattr(cfg.model, "sequence_parallel", None) is overrides["sequence_parallelism"]
    assert getattr(cfg.model, "seq_length", 0) == overrides["seq_length"]


def test_nemotron_vl_pretrain_pipeline_dtype(monkeypatch: pytest.MonkeyPatch):
    """Test that pipeline_parallelism_dtype is respected."""
    monkeypatch.setattr(_nemotron_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides()
    overrides["pipeline_parallelism_dtype"] = torch.bfloat16

    cfg = _nemotron_module.nemotron_nano_v2_vl_12b_pretrain_config(**overrides)

    assert getattr(cfg.model, "pipeline_dtype", None) is torch.bfloat16


def test_nemotron_vl_finetune_with_lora(monkeypatch: pytest.MonkeyPatch):
    """Test finetune_config wiring including LoRA when enabled."""
    monkeypatch.setattr(_nemotron_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides()
    cfg = _nemotron_module.nemotron_nano_v2_vl_12b_finetune_config(
        pretrained_checkpoint="/fake/ckpt",
        lora_on_language_model=True,
        lora_on_vision_model=False,
        **overrides,
    )

    _assert_basic_config(cfg)

    # Check that checkpoint wiring includes the pretrained checkpoint
    assert getattr(cfg.checkpoint, "pretrained_checkpoint", None) == "/fake/ckpt"

    # LoRA should be configured
    from megatron.bridge.peft.lora import VLMLoRA

    assert isinstance(getattr(cfg, "peft", None), VLMLoRA)

    # Finetune defaults applied (since overrides didn't provide finetune-specific lr)
    assert hasattr(cfg.optimizer, "lr") and cfg.optimizer.lr == 5e-5
    assert hasattr(cfg.optimizer, "min_lr") and cfg.optimizer.min_lr == 5e-6
    assert getattr(cfg.model, "tensor_model_parallel_size", None) == 2


def test_nemotron_vl_finetune_without_lora(monkeypatch: pytest.MonkeyPatch):
    """Test finetune_config when LoRA is disabled."""
    monkeypatch.setattr(_nemotron_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides()
    del overrides["lr"]
    del overrides["min_lr"]
    cfg = _nemotron_module.nemotron_nano_v2_vl_12b_finetune_config(
        pretrained_checkpoint="/fake/ckpt",
        lora_on_language_model=False,
        **overrides,
    )

    _assert_basic_config(cfg)

    # No PEFT configured
    assert getattr(cfg, "peft", None) is None

    # Finetune defaults applied when not explicitly provided in overrides
    assert hasattr(cfg.optimizer, "lr") and cfg.optimizer.lr == 1e-5
    assert hasattr(cfg.optimizer, "min_lr") and cfg.optimizer.min_lr == 1e-6


def test_nemotron_vl_finetune_custom_save_dir(monkeypatch: pytest.MonkeyPatch):
    """Test that save_checkpoint_dir overrides are respected in finetune_config."""
    monkeypatch.setattr(_nemotron_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides()
    cfg = _nemotron_module.nemotron_nano_v2_vl_12b_finetune_config(
        pretrained_checkpoint="/fake/ckpt",
        save_checkpoint_dir="/fake/save",
        **overrides,
    )

    assert getattr(cfg.checkpoint, "save", None) == "/fake/save"
    assert getattr(cfg.checkpoint, "load", None) == "/fake/save"
