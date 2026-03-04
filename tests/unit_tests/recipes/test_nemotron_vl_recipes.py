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
# - Parametrize over all exported Nemotron VL recipe functions in `megatron.bridge.recipes.nemotron_vl`.
# - For each recipe, monkeypatch AutoBridge and the provider to avoid I/O.
# - Build a config and assert it forms a valid `ConfigContainer`.
# - Verify dataset provider selection and sanity-check parallelism fields.
#

import importlib
from typing import Callable

import pytest


_nemotron_vl_module = importlib.import_module("megatron.bridge.recipes.nemotron_vl.nemotron_nano_v2_vl")

# SFT configs (parameterless)
_NEMOTRON_VL_SFT_FUNCS = [
    _nemotron_vl_module.nemotron_nano_v2_vl_12b_sft_config,
]

# PEFT configs (take peft_scheme parameter)
_NEMOTRON_VL_PEFT_FUNCS = [
    _nemotron_vl_module.nemotron_nano_v2_vl_12b_peft_config,
]


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
    def from_hf_pretrained(hf_path: str, **kwargs):
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


@pytest.mark.parametrize("recipe_func", _NEMOTRON_VL_SFT_FUNCS)
def test_each_nemotron_vl_sft_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that each Nemotron VL SFT recipe function builds a valid configuration."""
    # Monkeypatch AutoBridge to return a fake model config
    monkeypatch.setattr(_nemotron_vl_module, "AutoBridge", _FakeAutoBridge)

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


@pytest.mark.parametrize("recipe_func", _NEMOTRON_VL_PEFT_FUNCS)
def test_each_nemotron_vl_peft_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that each Nemotron VL PEFT recipe function builds a valid configuration."""
    # Monkeypatch AutoBridge to return a fake model config
    monkeypatch.setattr(_nemotron_vl_module, "AutoBridge", _FakeAutoBridge)

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


def test_nemotron_vl_12b_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 12B SFT has correct default parallelism."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_nemotron_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _nemotron_vl_module.nemotron_nano_v2_vl_12b_sft_config()

    _assert_basic_config(cfg)

    # For full SFT, 12B should use TP=4, PP=1
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.peft is None


def test_nemotron_vl_12b_peft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 12B PEFT has correct default parallelism."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_nemotron_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _nemotron_vl_module.nemotron_nano_v2_vl_12b_peft_config()

    _assert_basic_config(cfg)

    # For PEFT, 12B should use TP=2, PP=1
    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1

    # Check PEFT config (uses VLMLoRA)
    assert cfg.peft is not None


def test_nemotron_vl_sft_has_hf_dataset_provider(monkeypatch: pytest.MonkeyPatch):
    """Test that SFT configs use HFDatasetConversationProvider by default."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_nemotron_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _nemotron_vl_module.nemotron_nano_v2_vl_12b_sft_config()

    from megatron.bridge.data.vlm_datasets.hf_provider import HFDatasetConversationProvider

    assert isinstance(cfg.dataset, HFDatasetConversationProvider)


def test_nemotron_vl_peft_has_hf_dataset_provider(monkeypatch: pytest.MonkeyPatch):
    """Test that PEFT configs use HFDatasetConversationProvider by default."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_nemotron_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _nemotron_vl_module.nemotron_nano_v2_vl_12b_peft_config()

    from megatron.bridge.data.vlm_datasets.hf_provider import HFDatasetConversationProvider

    assert isinstance(cfg.dataset, HFDatasetConversationProvider)


def test_nemotron_vl_sft_freeze_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that SFT configs have freeze options set to False by default."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_nemotron_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _nemotron_vl_module.nemotron_nano_v2_vl_12b_sft_config()

    # Default freeze options should be False for full SFT
    assert cfg.model.freeze_language_model is False
    assert cfg.model.freeze_vision_model is False
    assert cfg.model.freeze_vision_projection is False


def test_nemotron_vl_peft_freeze_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that PEFT configs have freeze options set to False by default."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_nemotron_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _nemotron_vl_module.nemotron_nano_v2_vl_12b_peft_config()

    # Default freeze options should be False for PEFT
    assert cfg.model.freeze_language_model is False
    assert cfg.model.freeze_vision_model is False
    assert cfg.model.freeze_vision_projection is False


def test_nemotron_vl_precision_config(monkeypatch: pytest.MonkeyPatch):
    """Test that precision config is correctly set."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_nemotron_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _nemotron_vl_module.nemotron_nano_v2_vl_12b_sft_config()

    _assert_basic_config(cfg)

    # Default should be bf16_mixed
    assert cfg.mixed_precision == "bf16_mixed"


def test_nemotron_vl_ddp_config(monkeypatch: pytest.MonkeyPatch):
    """Test that DDP config is correctly set for VLMs."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_nemotron_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _nemotron_vl_module.nemotron_nano_v2_vl_12b_sft_config()

    _assert_basic_config(cfg)

    # VLMs should have overlap disabled
    assert cfg.ddp.overlap_grad_reduce is False
    assert cfg.ddp.overlap_param_gather is False
    assert cfg.ddp.check_for_nan_in_grad is True
    assert cfg.ddp.use_distributed_optimizer is True


def test_nemotron_vl_peft_uses_vlm_lora(monkeypatch: pytest.MonkeyPatch):
    """Test that Nemotron Nano V2 VL uses VLMLoRA for PEFT."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_nemotron_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _nemotron_vl_module.nemotron_nano_v2_vl_12b_peft_config()

    _assert_basic_config(cfg)

    # Check PEFT config is present (should be VLMLoRA)
    assert cfg.peft is not None

    # Check PEFT type is VLMLoRA
    from megatron.bridge.peft.lora import VLMLoRA

    assert isinstance(cfg.peft, VLMLoRA)


def test_nemotron_vl_sft_training_params(monkeypatch: pytest.MonkeyPatch):
    """Test that training parameters are correctly set for SFT."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_nemotron_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _nemotron_vl_module.nemotron_nano_v2_vl_12b_sft_config()

    _assert_basic_config(cfg)

    # Check training parameters
    assert cfg.train.train_iters == 2000
    assert cfg.train.micro_batch_size == 1


def test_nemotron_vl_peft_training_params(monkeypatch: pytest.MonkeyPatch):
    """Test that training parameters are correctly set for PEFT."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_nemotron_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _nemotron_vl_module.nemotron_nano_v2_vl_12b_peft_config()

    _assert_basic_config(cfg)

    # Check training parameters (should match SFT after update)
    assert cfg.train.train_iters == 2000
    assert cfg.train.micro_batch_size == 1
