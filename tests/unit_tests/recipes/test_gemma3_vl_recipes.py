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

#
# Test purpose:
# - Parametrize over all exported Gemma3-VL recipe functions in `megatron.bridge.recipes.gemma3_vl.gemma3_vl`.
# - For each recipe, monkeypatch AutoBridge and the provider to avoid I/O.
# - Build a config and assert it forms a valid `ConfigContainer`.
# - Verify dataset provider selection and sanity-check parallelism fields.
#

import importlib
from typing import Callable

import pytest
import torch


_gemma3_vl_module = importlib.import_module("megatron.bridge.recipes.gemma3_vl.gemma3_vl")

# SFT configs (parameterless)
_GEMMA3_VL_SFT_FUNCS = [
    _gemma3_vl_module.gemma3_vl_4b_sft_config,
    _gemma3_vl_module.gemma3_vl_12b_sft_config,
    _gemma3_vl_module.gemma3_vl_27b_sft_config,
]

# PEFT configs (take peft_scheme parameter)
_GEMMA3_VL_PEFT_FUNCS = [
    _gemma3_vl_module.gemma3_vl_4b_peft_config,
    _gemma3_vl_module.gemma3_vl_12b_peft_config,
    _gemma3_vl_module.gemma3_vl_27b_peft_config,
]

# All recipe functions
_GEMMA3_VL_ALL_FUNCS = _GEMMA3_VL_SFT_FUNCS + _GEMMA3_VL_PEFT_FUNCS


class _FakeModelCfg:
    """Fake model configuration for testing."""

    def __init__(self):
        # Set default attributes that recipes might set
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
    """Fake AutoBridge for testing."""

    @staticmethod
    def from_hf_pretrained(hf_path: str):
        """Mock from_hf_pretrained method."""
        return _FakeAutoBridge()

    def to_megatron_provider(self, load_weights: bool = False):
        """Return a fake model config."""
        return _FakeModelCfg()


def _assert_basic_config(cfg):
    """Assert that a config has all required components."""
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


@pytest.mark.parametrize("recipe_func", _GEMMA3_VL_SFT_FUNCS)
def test_each_gemma3_vl_sft_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that each Gemma3-VL SFT recipe function builds a valid configuration."""
    # Monkeypatch AutoBridge to return a fake model config
    monkeypatch.setattr(_gemma3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = recipe_func()

    _assert_basic_config(cfg)

    # Check that NullTokenizer is used
    if hasattr(cfg, "tokenizer") and hasattr(cfg.tokenizer, "tokenizer_type"):
        assert cfg.tokenizer.tokenizer_type == "NullTokenizer"

    # Verify parallelism settings
    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1

    # Verify freeze settings are set
    assert hasattr(cfg.model, "freeze_language_model")
    assert hasattr(cfg.model, "freeze_vision_model")
    assert hasattr(cfg.model, "freeze_vision_projection")

    # SFT configs should not have PEFT
    assert cfg.peft is None


@pytest.mark.parametrize("recipe_func", _GEMMA3_VL_PEFT_FUNCS)
def test_each_gemma3_vl_peft_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that each Gemma3-VL PEFT recipe function builds a valid configuration."""
    # Monkeypatch AutoBridge to return a fake model config
    monkeypatch.setattr(_gemma3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = recipe_func()  # Default peft_scheme="lora"

    _assert_basic_config(cfg)

    # Check that NullTokenizer is used
    if hasattr(cfg, "tokenizer") and hasattr(cfg.tokenizer, "tokenizer_type"):
        assert cfg.tokenizer.tokenizer_type == "NullTokenizer"

    # Verify parallelism settings
    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1

    # Verify freeze settings are set
    assert hasattr(cfg.model, "freeze_language_model")
    assert hasattr(cfg.model, "freeze_vision_model")
    assert hasattr(cfg.model, "freeze_vision_projection")

    # PEFT configs should have PEFT configured
    assert cfg.peft is not None
    assert hasattr(cfg.peft, "dim")
    assert hasattr(cfg.peft, "alpha")


@pytest.mark.parametrize("recipe_func", _GEMMA3_VL_PEFT_FUNCS)
@pytest.mark.parametrize("peft_scheme", ["lora", "dora"])
def test_gemma3_vl_peft_schemes(recipe_func: Callable, peft_scheme: str, monkeypatch: pytest.MonkeyPatch):
    """Test that different PEFT schemes are correctly applied for Gemma3-VL models."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_gemma3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = recipe_func(peft_scheme=peft_scheme)

    _assert_basic_config(cfg)

    # Check PEFT config presence
    assert cfg.peft is not None
    # Verify PEFT config has expected attributes
    assert hasattr(cfg.peft, "dim")
    assert hasattr(cfg.peft, "alpha")


def test_gemma3_vl_4b_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 4B SFT has correct default parallelism and learning rate."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_gemma3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _gemma3_vl_module.gemma3_vl_4b_sft_config()

    _assert_basic_config(cfg)

    # For full SFT, 4B should use TP=1, PP=1
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.peft is None


def test_gemma3_vl_4b_peft_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 4B LoRA has correct default parallelism."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_gemma3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _gemma3_vl_module.gemma3_vl_4b_peft_config(peft_scheme="lora")

    _assert_basic_config(cfg)

    # For LoRA, 4B should use TP=1, PP=1
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1

    # Check PEFT config
    assert cfg.peft is not None
    assert cfg.peft.dim == 32
    assert cfg.peft.alpha == 32


def test_gemma3_vl_4b_peft_dora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 4B DoRA has correct default parallelism."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_gemma3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _gemma3_vl_module.gemma3_vl_4b_peft_config(peft_scheme="dora")

    _assert_basic_config(cfg)

    # For DoRA, 4B should use same parallelism as LoRA
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1

    # Check PEFT config (DoRA has alpha=64 by default, unlike LoRA's alpha=32)
    assert cfg.peft is not None
    assert cfg.peft.dim == 32
    assert cfg.peft.alpha == 64


def test_gemma3_vl_12b_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 12B SFT has correct default parallelism."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_gemma3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _gemma3_vl_module.gemma3_vl_12b_sft_config()

    _assert_basic_config(cfg)

    # For full SFT, 12B should use TP=4, PP=1
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.peft is None


def test_gemma3_vl_12b_peft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 12B PEFT has correct default parallelism."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_gemma3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _gemma3_vl_module.gemma3_vl_12b_peft_config()

    _assert_basic_config(cfg)

    # For LoRA, 12B should use TP=1, PP=1
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1

    # Check PEFT config
    assert cfg.peft is not None


def test_gemma3_vl_27b_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 27B SFT has correct default parallelism and pipeline_dtype."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_gemma3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _gemma3_vl_module.gemma3_vl_27b_sft_config()

    _assert_basic_config(cfg)

    # For full SFT, 27B should use TP=8, PP=2
    assert cfg.model.tensor_model_parallel_size == 8
    assert cfg.model.pipeline_model_parallel_size == 2
    assert cfg.peft is None

    # For full SFT, pipeline_dtype should be set to bfloat16
    assert cfg.model.pipeline_dtype == torch.bfloat16


def test_gemma3_vl_27b_peft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 27B PEFT has correct default parallelism."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_gemma3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _gemma3_vl_module.gemma3_vl_27b_peft_config()

    _assert_basic_config(cfg)

    # For LoRA, 27B should use TP=4, PP=1
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 1

    # Check PEFT config
    assert cfg.peft is not None

    # For LoRA, pipeline_dtype should NOT be set
    assert cfg.model.pipeline_dtype is None


def test_gemma3_vl_27b_peft_dora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 27B DoRA has correct default parallelism."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_gemma3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _gemma3_vl_module.gemma3_vl_27b_peft_config(peft_scheme="dora")

    _assert_basic_config(cfg)

    # For DoRA, 27B should use same parallelism as LoRA (TP=4, PP=1)
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 1

    # Check PEFT config
    assert cfg.peft is not None

    # For DoRA, pipeline_dtype should NOT be set
    assert cfg.model.pipeline_dtype is None


def test_gemma3_vl_sft_has_hf_dataset_provider(monkeypatch: pytest.MonkeyPatch):
    """Test that SFT configs use HFDatasetConversationProvider by default."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_gemma3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _gemma3_vl_module.gemma3_vl_4b_sft_config()

    from megatron.bridge.data.vlm_datasets.hf_provider import HFDatasetConversationProvider

    assert isinstance(cfg.dataset, HFDatasetConversationProvider)


def test_gemma3_vl_peft_has_hf_dataset_provider(monkeypatch: pytest.MonkeyPatch):
    """Test that PEFT configs use HFDatasetConversationProvider by default."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_gemma3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _gemma3_vl_module.gemma3_vl_4b_peft_config()

    from megatron.bridge.data.vlm_datasets.hf_provider import HFDatasetConversationProvider

    assert isinstance(cfg.dataset, HFDatasetConversationProvider)


def test_gemma3_vl_sft_freeze_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that SFT configs have freeze options set to False by default."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_gemma3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _gemma3_vl_module.gemma3_vl_4b_sft_config()

    # Default freeze options should be False for full SFT
    assert cfg.model.freeze_language_model is False
    assert cfg.model.freeze_vision_model is False
    assert cfg.model.freeze_vision_projection is False


def test_gemma3_vl_peft_freeze_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that PEFT configs have freeze options set to False by default."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_gemma3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _gemma3_vl_module.gemma3_vl_4b_peft_config()

    # Default freeze options should be False for PEFT
    assert cfg.model.freeze_language_model is False
    assert cfg.model.freeze_vision_model is False
    assert cfg.model.freeze_vision_projection is False


def test_gemma3_vl_precision_config(monkeypatch: pytest.MonkeyPatch):
    """Test that precision config is correctly set."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_gemma3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _gemma3_vl_module.gemma3_vl_4b_sft_config()

    _assert_basic_config(cfg)

    # Default should be bf16_mixed
    assert cfg.mixed_precision == "bf16_mixed"


def test_gemma3_vl_ddp_config(monkeypatch: pytest.MonkeyPatch):
    """Test that DDP config is correctly set for VLMs."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_gemma3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _gemma3_vl_module.gemma3_vl_4b_sft_config()

    _assert_basic_config(cfg)

    # VLMs should have overlap disabled
    assert cfg.ddp.overlap_grad_reduce is False
    assert cfg.ddp.overlap_param_gather is False
    assert cfg.ddp.check_for_nan_in_grad is True
    assert cfg.ddp.use_distributed_optimizer is True
