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
# - Parametrize over all exported GLM-4.5V recipe functions in `megatron.bridge.recipes.glm_vl.glm_45v`.
# - For each recipe, monkeypatch AutoBridge and the provider to avoid I/O.
# - Build a config and assert it forms a valid `ConfigContainer`.
# - Verify dataset provider selection and sanity-check parallelism fields.
# - Test MoE-specific settings for this MoE VLM model.
#

import importlib
from typing import Callable

import pytest


_glm_45v_module = importlib.import_module("megatron.bridge.recipes.glm_vl.glm_45v")

# SFT configs (parameterless)
_GLM_45V_SFT_FUNCS = [
    _glm_45v_module.glm_45v_sft_config,
]

# PEFT configs (take peft_scheme parameter)
_GLM_45V_PEFT_FUNCS = [
    _glm_45v_module.glm_45v_peft_config,
]

# All recipe functions
_GLM_45V_ALL_FUNCS = _GLM_45V_SFT_FUNCS + _GLM_45V_PEFT_FUNCS


class _FakeModelCfg:
    """Fake model configuration for testing."""

    def __init__(self):
        # Set default attributes that recipes might set
        self.tensor_model_parallel_size = 1
        self.pipeline_model_parallel_size = 1
        self.pipeline_dtype = None
        self.virtual_pipeline_model_parallel_size = None
        self.context_parallel_size = 1
        self.expert_model_parallel_size = 1
        self.sequence_parallel = False
        self.seq_length = 64
        self.freeze_language_model = False
        self.freeze_vision_model = False
        self.freeze_vision_projection = False
        # MoE-specific
        self.moe_token_dispatcher_type = None
        self.moe_flex_dispatcher_backend = None
        self.moe_hybridep_num_sms = None
        self.moe_router_fusion = False
        self.moe_permute_fusion = False
        self.moe_grouped_gemm = False
        self.moe_router_padding_for_fp8 = False
        self.moe_shared_expert_overlap = False
        self.moe_router_force_load_balancing = False

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


@pytest.mark.parametrize("recipe_func", _GLM_45V_SFT_FUNCS)
def test_each_glm_45v_sft_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that each GLM-4.5V SFT recipe function builds a valid configuration."""
    # Monkeypatch AutoBridge to return a fake model config
    monkeypatch.setattr(_glm_45v_module, "AutoBridge", _FakeAutoBridge)

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


@pytest.mark.parametrize("recipe_func", _GLM_45V_PEFT_FUNCS)
def test_each_glm_45v_peft_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that each GLM-4.5V PEFT recipe function builds a valid configuration."""
    # Monkeypatch AutoBridge to return a fake model config
    monkeypatch.setattr(_glm_45v_module, "AutoBridge", _FakeAutoBridge)

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


@pytest.mark.parametrize("peft_scheme", ["lora", "dora"])
def test_glm_45v_peft_schemes(peft_scheme: str, monkeypatch: pytest.MonkeyPatch):
    """Test that different PEFT schemes are correctly applied for GLM-4.5V model."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_glm_45v_module, "AutoBridge", _FakeAutoBridge)

    cfg = _glm_45v_module.glm_45v_peft_config(peft_scheme=peft_scheme)

    _assert_basic_config(cfg)

    # Check PEFT config presence
    assert cfg.peft is not None
    # Verify PEFT config has expected attributes
    assert hasattr(cfg.peft, "dim")
    assert hasattr(cfg.peft, "alpha")


def test_glm_45v_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that GLM-4.5V SFT has correct default parallelism and MoE settings."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_glm_45v_module, "AutoBridge", _FakeAutoBridge)

    cfg = _glm_45v_module.glm_45v_sft_config()

    _assert_basic_config(cfg)

    # For full SFT, GLM-4.5V should use TP=1, PP=8, EP=16
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 8
    assert cfg.peft is None

    # Check expert_model_parallel_size for MoE model
    assert cfg.model.expert_model_parallel_size == 16


def test_glm_45v_peft_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that GLM-4.5V LoRA has correct default parallelism."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_glm_45v_module, "AutoBridge", _FakeAutoBridge)

    cfg = _glm_45v_module.glm_45v_peft_config(peft_scheme="lora")

    _assert_basic_config(cfg)

    # For LoRA, GLM-4.5V should use TP=1, PP=8
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 8

    # Check PEFT config
    assert cfg.peft is not None
    assert cfg.peft.dim == 32
    assert cfg.peft.alpha == 32


def test_glm_45v_peft_dora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that GLM-4.5V DoRA has correct default parallelism."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_glm_45v_module, "AutoBridge", _FakeAutoBridge)

    cfg = _glm_45v_module.glm_45v_peft_config(peft_scheme="dora")

    _assert_basic_config(cfg)

    # For DoRA, should use same parallelism as LoRA (TP=1, PP=8)
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 8

    # Check PEFT config (DoRA has alpha=64 by default, unlike LoRA's alpha=32)
    assert cfg.peft is not None
    assert cfg.peft.dim == 32
    assert cfg.peft.alpha == 64


def test_glm_45v_sft_has_hf_dataset_provider(monkeypatch: pytest.MonkeyPatch):
    """Test that SFT configs use HFDatasetConversationProvider by default."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_glm_45v_module, "AutoBridge", _FakeAutoBridge)

    cfg = _glm_45v_module.glm_45v_sft_config()

    from megatron.bridge.data.vlm_datasets.hf_provider import HFDatasetConversationProvider

    assert isinstance(cfg.dataset, HFDatasetConversationProvider)


def test_glm_45v_peft_has_hf_dataset_provider(monkeypatch: pytest.MonkeyPatch):
    """Test that PEFT configs use HFDatasetConversationProvider by default."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_glm_45v_module, "AutoBridge", _FakeAutoBridge)

    cfg = _glm_45v_module.glm_45v_peft_config()

    from megatron.bridge.data.vlm_datasets.hf_provider import HFDatasetConversationProvider

    assert isinstance(cfg.dataset, HFDatasetConversationProvider)


def test_glm_45v_sft_freeze_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that SFT configs have freeze options set to False by default."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_glm_45v_module, "AutoBridge", _FakeAutoBridge)

    cfg = _glm_45v_module.glm_45v_sft_config()

    # Default freeze options should be False for full SFT
    assert cfg.model.freeze_language_model is False
    assert cfg.model.freeze_vision_model is False
    assert cfg.model.freeze_vision_projection is False


def test_glm_45v_peft_freeze_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that PEFT configs have freeze options set to False by default."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_glm_45v_module, "AutoBridge", _FakeAutoBridge)

    cfg = _glm_45v_module.glm_45v_peft_config()

    # Default freeze options should be False for PEFT
    assert cfg.model.freeze_language_model is False
    assert cfg.model.freeze_vision_model is False
    assert cfg.model.freeze_vision_projection is False


def test_glm_45v_precision_config(monkeypatch: pytest.MonkeyPatch):
    """Test that precision config is correctly set."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_glm_45v_module, "AutoBridge", _FakeAutoBridge)

    cfg = _glm_45v_module.glm_45v_sft_config()

    _assert_basic_config(cfg)

    # Default should be bf16_mixed
    assert cfg.mixed_precision == "bf16_mixed"


def test_glm_45v_ddp_config(monkeypatch: pytest.MonkeyPatch):
    """Test that DDP config is correctly set for VLMs."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_glm_45v_module, "AutoBridge", _FakeAutoBridge)

    cfg = _glm_45v_module.glm_45v_sft_config()

    _assert_basic_config(cfg)

    # VLMs should have overlap disabled
    assert cfg.ddp.overlap_grad_reduce is False
    assert cfg.ddp.overlap_param_gather is False
    assert cfg.ddp.check_for_nan_in_grad is True
    assert cfg.ddp.use_distributed_optimizer is True


def test_glm_45v_moe_settings(monkeypatch: pytest.MonkeyPatch):
    """Test that MoE-specific settings are correctly configured."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_glm_45v_module, "AutoBridge", _FakeAutoBridge)

    cfg = _glm_45v_module.glm_45v_sft_config()

    _assert_basic_config(cfg)

    # Check MoE-specific settings
    assert hasattr(cfg.model, "moe_token_dispatcher_type")
    assert hasattr(cfg.model, "moe_flex_dispatcher_backend")
    assert hasattr(cfg.model, "moe_hybridep_num_sms")
    assert hasattr(cfg.model, "moe_router_fusion")
    assert hasattr(cfg.model, "moe_permute_fusion")
    assert hasattr(cfg.model, "moe_grouped_gemm")
    assert hasattr(cfg.model, "moe_router_padding_for_fp8")
    assert hasattr(cfg.model, "moe_shared_expert_overlap")
    assert hasattr(cfg.model, "moe_router_force_load_balancing")

    # Verify default MoE kernel settings
    assert cfg.model.moe_router_fusion is False
    assert cfg.model.moe_permute_fusion is True
    assert cfg.model.moe_grouped_gemm is True


def test_glm_45v_pipeline_layout_function_exists():
    """Test that pipeline layout function is exported."""
    assert hasattr(_glm_45v_module, "set_glm_45v_pipeline_model_parallel_layout")
    assert callable(_glm_45v_module.set_glm_45v_pipeline_model_parallel_layout)


def test_glm_45v_sft_uses_pipeline_layout(monkeypatch: pytest.MonkeyPatch):
    """Test that SFT config has pipeline model parallel layout set."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_glm_45v_module, "AutoBridge", _FakeAutoBridge)

    cfg = _glm_45v_module.glm_45v_sft_config()

    _assert_basic_config(cfg)

    # PP should be set when pipeline layout is used
    assert cfg.model.pipeline_model_parallel_size >= 1
    # Check if pipeline_model_parallel_layout is set
    assert hasattr(cfg.model, "pipeline_model_parallel_layout")
