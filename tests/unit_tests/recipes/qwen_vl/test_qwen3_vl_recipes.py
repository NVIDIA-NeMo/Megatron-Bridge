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
# - Parametrize over all exported Qwen3-VL recipe functions in `megatron.bridge.recipes.qwen_vl.qwen3_vl`.
# - For each recipe, monkeypatch AutoBridge and the provider to avoid I/O.
# - Build a config and assert it forms a valid `ConfigContainer`.
# - Verify dataset provider selection and sanity-check parallelism fields.
# - Test MoE-specific settings for Qwen3-VL MoE models.
#

import importlib
from typing import Callable

import pytest


_qwen3_vl_module = importlib.import_module("megatron.bridge.recipes.qwen_vl.qwen3_vl")

# Pretrain mock configs (accept **user_kwargs)
_QWEN3_VL_PRETRAIN_MOCK_FUNCS = [
    _qwen3_vl_module.qwen3_vl_8b_pretrain_mock_config,
    _qwen3_vl_module.qwen3_vl_30b_a3b_pretrain_mock_config,
    _qwen3_vl_module.qwen3_vl_235b_a22b_pretrain_mock_config,
]

# SFT configs (parameterless)
_QWEN3_VL_SFT_FUNCS = [
    _qwen3_vl_module.qwen3_vl_8b_sft_config,
    _qwen3_vl_module.qwen3_vl_30b_a3b_sft_config,
    _qwen3_vl_module.qwen3_vl_235b_a22b_sft_config,
]

# PEFT configs (take peft_scheme parameter)
_QWEN3_VL_PEFT_FUNCS = [
    _qwen3_vl_module.qwen3_vl_8b_peft_config,
    _qwen3_vl_module.qwen3_vl_30b_a3b_peft_config,
    _qwen3_vl_module.qwen3_vl_235b_a22b_peft_config,
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


@pytest.mark.parametrize("recipe_func", _QWEN3_VL_SFT_FUNCS)
def test_each_qwen3_vl_sft_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that each Qwen3-VL SFT recipe function builds a valid configuration."""
    # Monkeypatch AutoBridge to return a fake model config
    monkeypatch.setattr(_qwen3_vl_module, "AutoBridge", _FakeAutoBridge)

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


@pytest.mark.parametrize("recipe_func", _QWEN3_VL_PEFT_FUNCS)
def test_each_qwen3_vl_peft_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that each Qwen3-VL PEFT recipe function builds a valid configuration."""
    # Monkeypatch AutoBridge to return a fake model config
    monkeypatch.setattr(_qwen3_vl_module, "AutoBridge", _FakeAutoBridge)

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


@pytest.mark.parametrize("recipe_func", _QWEN3_VL_PEFT_FUNCS)
@pytest.mark.parametrize("peft_scheme", ["lora", "dora"])
def test_qwen3_vl_peft_schemes(recipe_func: Callable, peft_scheme: str, monkeypatch: pytest.MonkeyPatch):
    """Test that different PEFT schemes are correctly applied for Qwen3-VL models."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_qwen3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = recipe_func(peft_scheme=peft_scheme)

    _assert_basic_config(cfg)

    # Check PEFT config presence
    assert cfg.peft is not None
    # Verify PEFT config has expected attributes
    assert hasattr(cfg.peft, "dim")
    assert hasattr(cfg.peft, "alpha")


def test_qwen3_vl_8b_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 8B SFT has correct default parallelism."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_qwen3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen3_vl_module.qwen3_vl_8b_sft_config()

    _assert_basic_config(cfg)

    # For full SFT, 8B should use TP=2, PP=1
    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.peft is None


def test_qwen3_vl_8b_peft_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 8B LoRA has correct default parallelism."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_qwen3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen3_vl_module.qwen3_vl_8b_peft_config(peft_scheme="lora")

    _assert_basic_config(cfg)

    # For LoRA, 8B should use TP=1, PP=1
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1

    # Check PEFT config
    assert cfg.peft is not None
    assert cfg.peft.dim == 32
    assert cfg.peft.alpha == 32


def test_qwen3_vl_8b_peft_dora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 8B DoRA has correct default parallelism."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_qwen3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen3_vl_module.qwen3_vl_8b_peft_config(peft_scheme="dora")

    _assert_basic_config(cfg)

    # For DoRA, 8B should use same parallelism as LoRA
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1

    # Check PEFT config (DoRA has alpha=64 by default, unlike LoRA's alpha=32)
    assert cfg.peft is not None
    assert cfg.peft.dim == 32
    assert cfg.peft.alpha == 64


def test_qwen3_vl_30b_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 30B-A3B SFT has correct default parallelism and MoE settings."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_qwen3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen3_vl_module.qwen3_vl_30b_a3b_sft_config()

    _assert_basic_config(cfg)

    # For full SFT, 30B-A3B should use TP=1, PP=1, EP=8
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.peft is None

    # Check expert_model_parallel_size for MoE model
    assert cfg.model.expert_model_parallel_size == 8


def test_qwen3_vl_30b_peft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 30B-A3B PEFT has correct default parallelism."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_qwen3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen3_vl_module.qwen3_vl_30b_a3b_peft_config()

    _assert_basic_config(cfg)

    # For LoRA, 30B-A3B should use TP=1, PP=1, EP=4
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 4

    # Check PEFT config
    assert cfg.peft is not None


def test_qwen3_vl_235b_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 235B-A22B SFT has correct default parallelism and MoE settings."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_qwen3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen3_vl_module.qwen3_vl_235b_a22b_sft_config()

    _assert_basic_config(cfg)

    # For full SFT, 235B-A22B should use TP=4, PP=1, EP=32
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.peft is None

    # Check expert_model_parallel_size for MoE model
    assert cfg.model.expert_model_parallel_size == 32


def test_qwen3_vl_235b_peft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 235B-A22B PEFT has correct default parallelism."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_qwen3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen3_vl_module.qwen3_vl_235b_a22b_peft_config()

    _assert_basic_config(cfg)

    # For LoRA, 235B-A22B should use TP=1, PP=1, EP=16
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 16

    # Check PEFT config
    assert cfg.peft is not None


def test_qwen3_vl_sft_has_hf_dataset_provider(monkeypatch: pytest.MonkeyPatch):
    """Test that SFT configs use HFDatasetConversationProvider by default."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_qwen3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen3_vl_module.qwen3_vl_8b_sft_config()

    from megatron.bridge.data.vlm_datasets.hf_provider import HFDatasetConversationProvider

    assert isinstance(cfg.dataset, HFDatasetConversationProvider)


def test_qwen3_vl_peft_has_hf_dataset_provider(monkeypatch: pytest.MonkeyPatch):
    """Test that PEFT configs use HFDatasetConversationProvider by default."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_qwen3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen3_vl_module.qwen3_vl_8b_peft_config()

    from megatron.bridge.data.vlm_datasets.hf_provider import HFDatasetConversationProvider

    assert isinstance(cfg.dataset, HFDatasetConversationProvider)


def test_qwen3_vl_sft_freeze_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that SFT configs have freeze options set to False by default."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_qwen3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen3_vl_module.qwen3_vl_8b_sft_config()

    # Default freeze options should be False for full SFT
    assert cfg.model.freeze_language_model is False
    assert cfg.model.freeze_vision_model is False
    assert cfg.model.freeze_vision_projection is False


def test_qwen3_vl_peft_freeze_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that PEFT configs have freeze options set to False by default."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_qwen3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen3_vl_module.qwen3_vl_8b_peft_config()

    # Default freeze options should be False for PEFT
    assert cfg.model.freeze_language_model is False
    assert cfg.model.freeze_vision_model is False
    assert cfg.model.freeze_vision_projection is False


def test_qwen3_vl_precision_config(monkeypatch: pytest.MonkeyPatch):
    """Test that precision config is correctly set."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_qwen3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen3_vl_module.qwen3_vl_8b_sft_config()

    _assert_basic_config(cfg)

    # Default should be bf16_mixed
    assert cfg.mixed_precision == "bf16_mixed"


def test_qwen3_vl_ddp_config(monkeypatch: pytest.MonkeyPatch):
    """Test that DDP config is correctly set for VLMs."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_qwen3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen3_vl_module.qwen3_vl_8b_sft_config()

    _assert_basic_config(cfg)

    # VLMs should have overlap disabled
    assert cfg.ddp.overlap_grad_reduce is False
    assert cfg.ddp.overlap_param_gather is False
    assert cfg.ddp.check_for_nan_in_grad is True
    assert cfg.ddp.use_distributed_optimizer is True


def test_qwen3_vl_moe_settings_30b(monkeypatch: pytest.MonkeyPatch):
    """Test that MoE-specific settings are correctly configured for 30B-A3B model."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_qwen3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen3_vl_module.qwen3_vl_30b_a3b_sft_config()

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


def test_qwen3_vl_moe_settings_235b(monkeypatch: pytest.MonkeyPatch):
    """Test that MoE-specific settings are correctly configured for 235B-A22B model."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_qwen3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen3_vl_module.qwen3_vl_235b_a22b_sft_config()

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


def test_qwen3_vl_8b_is_dense_model(monkeypatch: pytest.MonkeyPatch):
    """Test that 8B is a dense model without MoE-specific parallelism."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_qwen3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen3_vl_module.qwen3_vl_8b_sft_config()

    _assert_basic_config(cfg)

    # 8B should be dense model with EP=1
    assert cfg.model.expert_model_parallel_size == 1

    # Verify dense model kernel settings
    assert cfg.model.moe_router_fusion is False
    assert cfg.model.moe_permute_fusion is False
    assert cfg.model.moe_grouped_gemm is False


# =============================================================================
# Qwen3-VL 8B PEFT Energon Config Tests
# =============================================================================


def _patch_energon_deps(monkeypatch):
    """Monkeypatch AutoBridge and HF tokenizer/processor for energon config tests."""
    monkeypatch.setattr(_qwen3_vl_module, "AutoBridge", _FakeAutoBridge)
    monkeypatch.setattr(
        _qwen3_vl_module,
        "AutoTokenizer",
        type(
            "FakeAutoTokenizer",
            (),
            {
                "from_pretrained": staticmethod(lambda *a, **kw: None),
            },
        ),
    )
    monkeypatch.setattr(
        _qwen3_vl_module,
        "Qwen3VLProcessor",
        type(
            "FakeProcessor",
            (),
            {
                "from_pretrained": staticmethod(lambda *a, **kw: None),
            },
        ),
    )


def test_qwen3_vl_8b_peft_energon_builds_config(monkeypatch: pytest.MonkeyPatch):
    """Test that the energon PEFT config builds a valid ConfigContainer."""
    _patch_energon_deps(monkeypatch)

    cfg = _qwen3_vl_module.qwen3_vl_8b_peft_energon_config()

    _assert_basic_config(cfg)
    assert cfg.peft is not None


def test_qwen3_vl_8b_peft_energon_uses_energon_provider(monkeypatch: pytest.MonkeyPatch):
    """Test that the energon config uses EnergonProvider as dataset."""
    _patch_energon_deps(monkeypatch)

    cfg = _qwen3_vl_module.qwen3_vl_8b_peft_energon_config()

    from megatron.bridge.data.energon.energon_provider import EnergonProvider

    assert isinstance(cfg.dataset, EnergonProvider)


def test_qwen3_vl_8b_peft_energon_dataset_params(monkeypatch: pytest.MonkeyPatch):
    """Test that the energon dataset has correct seq_length, batch sizes."""
    _patch_energon_deps(monkeypatch)

    cfg = _qwen3_vl_module.qwen3_vl_8b_peft_energon_config()

    assert cfg.dataset.seq_length == 4096
    assert cfg.dataset.micro_batch_size == cfg.train.micro_batch_size
    assert cfg.dataset.global_batch_size == cfg.train.global_batch_size


@pytest.mark.parametrize("peft_scheme", ["lora", "dora"])
def test_qwen3_vl_8b_peft_energon_schemes(peft_scheme: str, monkeypatch: pytest.MonkeyPatch):
    """Test that lora and dora schemes work with energon config."""
    _patch_energon_deps(monkeypatch)

    cfg = _qwen3_vl_module.qwen3_vl_8b_peft_energon_config(peft_scheme=peft_scheme)

    _assert_basic_config(cfg)
    assert cfg.peft is not None
    assert hasattr(cfg.peft, "dim")
    assert hasattr(cfg.peft, "alpha")


def test_qwen3_vl_8b_peft_energon_parallelism(monkeypatch: pytest.MonkeyPatch):
    """Test that energon config inherits 8B PEFT parallelism (TP=1, PP=1)."""
    _patch_energon_deps(monkeypatch)

    cfg = _qwen3_vl_module.qwen3_vl_8b_peft_energon_config()

    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 1


def test_qwen3_vl_8b_peft_energon_precision(monkeypatch: pytest.MonkeyPatch):
    """Test that energon config uses bf16_mixed precision."""
    _patch_energon_deps(monkeypatch)

    cfg = _qwen3_vl_module.qwen3_vl_8b_peft_energon_config()

    assert cfg.mixed_precision == "bf16_mixed"


def test_qwen3_vl_8b_peft_energon_freeze_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that energon PEFT config has freeze options set to False."""
    _patch_energon_deps(monkeypatch)

    cfg = _qwen3_vl_module.qwen3_vl_8b_peft_energon_config()

    assert cfg.model.freeze_language_model is False
    assert cfg.model.freeze_vision_model is False
    assert cfg.model.freeze_vision_projection is False


def test_qwen3_vl_8b_peft_energon_task_encoder(monkeypatch: pytest.MonkeyPatch):
    """Test that energon config creates a QwenVLTaskEncoder in the dataset."""
    _patch_energon_deps(monkeypatch)

    cfg = _qwen3_vl_module.qwen3_vl_8b_peft_energon_config()

    from megatron.bridge.recipes.qwen_vl.data.energon.task_encoder import QwenVLTaskEncoder

    assert isinstance(cfg.dataset.task_encoder, QwenVLTaskEncoder)


# =============================================================================
# Qwen3-VL Pretrain Mock Config Tests
# =============================================================================


@pytest.mark.parametrize("recipe_func", _QWEN3_VL_PRETRAIN_MOCK_FUNCS)
def test_each_qwen3_vl_pretrain_mock_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that each Qwen3-VL pretrain mock recipe function builds a valid ConfigContainer."""
    monkeypatch.setattr(_qwen3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = recipe_func()

    _assert_basic_config(cfg)

    assert cfg.tokenizer.tokenizer_type == "NullTokenizer"
    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1

    assert cfg.model.freeze_language_model is True
    assert cfg.model.freeze_vision_model is True
    assert cfg.model.freeze_vision_projection is False

    assert cfg.peft is None


@pytest.mark.parametrize("recipe_func", _QWEN3_VL_PRETRAIN_MOCK_FUNCS)
def test_qwen3_vl_pretrain_mock_uses_mock_dataset(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that pretrain mock configs use MockVLMConversationProvider."""
    monkeypatch.setattr(_qwen3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = recipe_func()

    from megatron.bridge.data.vlm_datasets.mock_provider import MockVLMConversationProvider

    assert isinstance(cfg.dataset, MockVLMConversationProvider)


def test_qwen3_vl_8b_pretrain_mock_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 8B pretrain mock has correct default parallelism and training params."""
    monkeypatch.setattr(_qwen3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen3_vl_module.qwen3_vl_8b_pretrain_mock_config()

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.pipeline_dtype is None
    assert cfg.model.expert_model_parallel_size == 1

    assert cfg.model.freeze_language_model is True
    assert cfg.model.freeze_vision_model is True
    assert cfg.model.freeze_vision_projection is False

    assert cfg.train.train_iters == 300000
    assert cfg.train.global_batch_size == 32
    assert cfg.train.micro_batch_size == 2
    assert cfg.optimizer.lr == 3e-4
    assert cfg.mixed_precision == "bf16_mixed"
    assert cfg.peft is None


def test_qwen3_vl_30b_a3b_pretrain_mock_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 30B-A3B pretrain mock has correct default parallelism for MoE."""
    monkeypatch.setattr(_qwen3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen3_vl_module.qwen3_vl_30b_a3b_pretrain_mock_config()

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 2
    assert cfg.model.pipeline_dtype is not None  # PP > 1 => bf16
    assert cfg.model.expert_model_parallel_size == 4
    assert cfg.model.sequence_parallel is True

    assert cfg.model.freeze_language_model is True
    assert cfg.model.freeze_vision_model is True
    assert cfg.model.freeze_vision_projection is False


def test_qwen3_vl_235b_a22b_pretrain_mock_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 235B-A22B pretrain mock has correct default parallelism for large MoE."""
    monkeypatch.setattr(_qwen3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen3_vl_module.qwen3_vl_235b_a22b_pretrain_mock_config()

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 16
    assert cfg.model.pipeline_dtype is not None  # PP > 1 => bf16
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.model.context_parallel_size == 2
    assert cfg.model.sequence_parallel is True


def test_qwen3_vl_pretrain_mock_ddp_config(monkeypatch: pytest.MonkeyPatch):
    """Test that pretrain mock DDP config is correctly set."""
    monkeypatch.setattr(_qwen3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen3_vl_module.qwen3_vl_8b_pretrain_mock_config()

    assert cfg.ddp.overlap_grad_reduce is False
    assert cfg.ddp.overlap_param_gather is False
    assert cfg.ddp.check_for_nan_in_grad is True
    assert cfg.ddp.grad_reduce_in_fp32 is True
    assert cfg.ddp.use_distributed_optimizer is True


def test_qwen3_vl_pretrain_mock_user_kwargs_override(monkeypatch: pytest.MonkeyPatch):
    """Test that user kwargs properly override recommended defaults."""
    monkeypatch.setattr(_qwen3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen3_vl_module.qwen3_vl_8b_pretrain_mock_config(
        train_iters=1000,
        global_batch_size=16,
        micro_batch_size=1,
        lr=1e-4,
        seq_length=2048,
    )

    _assert_basic_config(cfg)

    assert cfg.train.train_iters == 1000
    assert cfg.train.global_batch_size == 16
    assert cfg.train.micro_batch_size == 1
    assert cfg.optimizer.lr == 1e-4
    assert cfg.model.seq_length == 2048


def test_qwen3_vl_pretrain_mock_non_mock_raises(monkeypatch: pytest.MonkeyPatch):
    """Test that mock=False raises ValueError since real datasets are not yet supported."""
    monkeypatch.setattr(_qwen3_vl_module, "AutoBridge", _FakeAutoBridge)

    with pytest.raises(ValueError, match="Non-mock dataset not yet supported"):
        _qwen3_vl_module.qwen3_vl_8b_pretrain_mock_config(mock=False)


def test_qwen3_vl_pretrain_mock_checkpoint_config(monkeypatch: pytest.MonkeyPatch):
    """Test that pretrain mock checkpoint config is correctly set."""
    monkeypatch.setattr(_qwen3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen3_vl_module.qwen3_vl_8b_pretrain_mock_config()

    assert cfg.checkpoint.ckpt_format == "torch_dist"
    assert cfg.checkpoint.save_interval == 500
    assert cfg.checkpoint.fully_parallel_save is True


def test_qwen3_vl_pretrain_mock_rng_seed(monkeypatch: pytest.MonkeyPatch):
    """Test that pretrain mock RNG seed is set."""
    monkeypatch.setattr(_qwen3_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen3_vl_module.qwen3_vl_8b_pretrain_mock_config()

    assert cfg.rng.seed == 1234
