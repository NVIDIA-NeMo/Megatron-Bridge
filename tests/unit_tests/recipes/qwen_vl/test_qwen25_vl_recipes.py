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
# - Parametrize over all exported Qwen2.5-VL recipe functions in `megatron.bridge.recipes.qwen_vl.qwen25_vl`.
# - For each recipe, monkeypatch AutoBridge and the provider to avoid I/O.
# - Build a config and assert it forms a valid `ConfigContainer`.
# - Verify dataset provider selection and sanity-check parallelism fields.
#

import importlib
from typing import Callable

import pytest
import torch


_qwen25_vl_module = importlib.import_module("megatron.bridge.recipes.qwen_vl.qwen25_vl")

# SFT configs (parameterless)
_QWEN25_VL_SFT_FUNCS = [
    _qwen25_vl_module.qwen25_vl_3b_sft_config,
    _qwen25_vl_module.qwen25_vl_7b_sft_config,
    _qwen25_vl_module.qwen25_vl_32b_sft_config,
    _qwen25_vl_module.qwen25_vl_72b_sft_config,
]

# PEFT configs (take peft_scheme parameter)
_QWEN25_VL_PEFT_FUNCS = [
    _qwen25_vl_module.qwen25_vl_3b_peft_config,
    _qwen25_vl_module.qwen25_vl_7b_peft_config,
    _qwen25_vl_module.qwen25_vl_32b_peft_config,
    _qwen25_vl_module.qwen25_vl_72b_peft_config,
]


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


@pytest.mark.parametrize("recipe_func", _QWEN25_VL_SFT_FUNCS)
def test_each_qwen25_vl_sft_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that each Qwen2.5-VL SFT recipe function builds a valid configuration."""
    monkeypatch.setattr(_qwen25_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = recipe_func()

    _assert_basic_config(cfg)

    if hasattr(cfg, "tokenizer") and hasattr(cfg.tokenizer, "tokenizer_type"):
        assert cfg.tokenizer.tokenizer_type == "NullTokenizer"

    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1

    assert hasattr(cfg.model, "freeze_language_model")
    assert hasattr(cfg.model, "freeze_vision_model")
    assert hasattr(cfg.model, "freeze_vision_projection")

    assert cfg.peft is None


@pytest.mark.parametrize("recipe_func", _QWEN25_VL_PEFT_FUNCS)
def test_each_qwen25_vl_peft_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that each Qwen2.5-VL PEFT recipe function builds a valid configuration."""
    monkeypatch.setattr(_qwen25_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = recipe_func()

    _assert_basic_config(cfg)

    if hasattr(cfg, "tokenizer") and hasattr(cfg.tokenizer, "tokenizer_type"):
        assert cfg.tokenizer.tokenizer_type == "NullTokenizer"

    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1

    assert hasattr(cfg.model, "freeze_language_model")
    assert hasattr(cfg.model, "freeze_vision_model")
    assert hasattr(cfg.model, "freeze_vision_projection")

    assert cfg.peft is not None
    assert hasattr(cfg.peft, "dim")
    assert hasattr(cfg.peft, "alpha")


@pytest.mark.parametrize("recipe_func", _QWEN25_VL_PEFT_FUNCS)
@pytest.mark.parametrize("peft_scheme", ["lora", "dora"])
def test_qwen25_vl_peft_schemes(recipe_func: Callable, peft_scheme: str, monkeypatch: pytest.MonkeyPatch):
    """Test that different PEFT schemes are correctly applied for Qwen2.5-VL models."""
    monkeypatch.setattr(_qwen25_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = recipe_func(peft_scheme=peft_scheme)

    _assert_basic_config(cfg)

    assert cfg.peft is not None
    assert hasattr(cfg.peft, "dim")
    assert hasattr(cfg.peft, "alpha")


def test_qwen25_vl_3b_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 3B SFT has correct default parallelism."""
    monkeypatch.setattr(_qwen25_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen25_vl_module.qwen25_vl_3b_sft_config()

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.peft is None


def test_qwen25_vl_3b_peft_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 3B LoRA has correct default parallelism."""
    monkeypatch.setattr(_qwen25_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen25_vl_module.qwen25_vl_3b_peft_config(peft_scheme="lora")

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1

    assert cfg.peft is not None
    assert cfg.peft.dim == 32
    assert cfg.peft.alpha == 32


def test_qwen25_vl_3b_peft_dora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 3B DoRA has correct default parallelism."""
    monkeypatch.setattr(_qwen25_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen25_vl_module.qwen25_vl_3b_peft_config(peft_scheme="dora")

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1

    assert cfg.peft is not None
    assert cfg.peft.dim == 32
    assert cfg.peft.alpha == 64


def test_qwen25_vl_7b_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 7B SFT has correct default parallelism."""
    monkeypatch.setattr(_qwen25_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen25_vl_module.qwen25_vl_7b_sft_config()

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.peft is None


def test_qwen25_vl_7b_peft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 7B PEFT has correct default parallelism."""
    monkeypatch.setattr(_qwen25_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen25_vl_module.qwen25_vl_7b_peft_config()

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.peft is not None


def test_qwen25_vl_32b_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 32B SFT has correct default parallelism."""
    monkeypatch.setattr(_qwen25_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen25_vl_module.qwen25_vl_32b_sft_config()

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 8
    assert cfg.model.pipeline_model_parallel_size == 2
    assert cfg.model.pipeline_dtype == torch.bfloat16
    assert cfg.peft is None


def test_qwen25_vl_32b_peft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 32B PEFT has correct default parallelism."""
    monkeypatch.setattr(_qwen25_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen25_vl_module.qwen25_vl_32b_peft_config()

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.peft is not None


def test_qwen25_vl_72b_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 72B SFT has correct default parallelism."""
    monkeypatch.setattr(_qwen25_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen25_vl_module.qwen25_vl_72b_sft_config()

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 8
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.pipeline_dtype == torch.bfloat16
    assert cfg.peft is None


def test_qwen25_vl_72b_peft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 72B PEFT has correct default parallelism."""
    monkeypatch.setattr(_qwen25_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen25_vl_module.qwen25_vl_72b_peft_config()

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.peft is not None


def test_qwen25_vl_sft_has_hf_dataset_provider(monkeypatch: pytest.MonkeyPatch):
    """Test that SFT configs use HFDatasetConversationProvider by default."""
    monkeypatch.setattr(_qwen25_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen25_vl_module.qwen25_vl_3b_sft_config()

    from megatron.bridge.data.vlm_datasets.hf_provider import HFDatasetConversationProvider

    assert isinstance(cfg.dataset, HFDatasetConversationProvider)


def test_qwen25_vl_peft_has_hf_dataset_provider(monkeypatch: pytest.MonkeyPatch):
    """Test that PEFT configs use HFDatasetConversationProvider by default."""
    monkeypatch.setattr(_qwen25_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen25_vl_module.qwen25_vl_3b_peft_config()

    from megatron.bridge.data.vlm_datasets.hf_provider import HFDatasetConversationProvider

    assert isinstance(cfg.dataset, HFDatasetConversationProvider)


def test_qwen25_vl_sft_freeze_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that SFT configs have freeze options set to False by default."""
    monkeypatch.setattr(_qwen25_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen25_vl_module.qwen25_vl_3b_sft_config()

    assert cfg.model.freeze_language_model is False
    assert cfg.model.freeze_vision_model is False
    assert cfg.model.freeze_vision_projection is False


def test_qwen25_vl_peft_freeze_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that PEFT configs have freeze options set to False by default."""
    monkeypatch.setattr(_qwen25_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen25_vl_module.qwen25_vl_3b_peft_config()

    assert cfg.model.freeze_language_model is False
    assert cfg.model.freeze_vision_model is False
    assert cfg.model.freeze_vision_projection is False


def test_qwen25_vl_precision_config(monkeypatch: pytest.MonkeyPatch):
    """Test that precision config is correctly set."""
    monkeypatch.setattr(_qwen25_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen25_vl_module.qwen25_vl_3b_sft_config()

    _assert_basic_config(cfg)

    assert cfg.mixed_precision == "bf16_mixed"


def test_qwen25_vl_ddp_config(monkeypatch: pytest.MonkeyPatch):
    """Test that DDP config is correctly set for VLMs."""
    monkeypatch.setattr(_qwen25_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen25_vl_module.qwen25_vl_3b_sft_config()

    _assert_basic_config(cfg)

    assert cfg.ddp.overlap_grad_reduce is False
    assert cfg.ddp.overlap_param_gather is False
    assert cfg.ddp.check_for_nan_in_grad is True
    assert cfg.ddp.use_distributed_optimizer is True


def test_qwen25_vl_optimizer_precision_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that optimizer precision settings are correctly configured."""
    monkeypatch.setattr(_qwen25_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen25_vl_module.qwen25_vl_3b_sft_config()

    _assert_basic_config(cfg)

    assert cfg.optimizer.use_precision_aware_optimizer is False
    assert cfg.optimizer.main_grads_dtype == torch.float32
    assert cfg.optimizer.main_params_dtype == torch.float32
    assert cfg.optimizer.exp_avg_dtype == torch.float32
    assert cfg.optimizer.exp_avg_sq_dtype == torch.float32


def test_qwen25_vl_training_config(monkeypatch: pytest.MonkeyPatch):
    """Test that training configuration is correctly set."""
    monkeypatch.setattr(_qwen25_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen25_vl_module.qwen25_vl_3b_sft_config()

    _assert_basic_config(cfg)

    assert cfg.train.train_iters == 300000
    assert cfg.train.global_batch_size == 32
    assert cfg.train.micro_batch_size == 2
    assert cfg.train.manual_gc is True
    assert cfg.train.manual_gc_interval == 100


def test_qwen25_vl_validation_config(monkeypatch: pytest.MonkeyPatch):
    """Test that validation configuration is correctly set."""
    monkeypatch.setattr(_qwen25_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen25_vl_module.qwen25_vl_3b_sft_config()

    _assert_basic_config(cfg)

    assert cfg.validation.eval_interval == 500
    assert cfg.validation.eval_iters == 32


def test_qwen25_vl_sft_learning_rate(monkeypatch: pytest.MonkeyPatch):
    """Test that SFT has lower learning rate than PEFT."""
    monkeypatch.setattr(_qwen25_vl_module, "AutoBridge", _FakeAutoBridge)

    sft_cfg = _qwen25_vl_module.qwen25_vl_3b_sft_config()
    peft_cfg = _qwen25_vl_module.qwen25_vl_3b_peft_config()

    # SFT should have lower LR (5e-6) compared to PEFT (1e-4)
    assert sft_cfg.optimizer.lr < peft_cfg.optimizer.lr


def test_qwen25_vl_kernel_settings(monkeypatch: pytest.MonkeyPatch):
    """Test that kernel settings are correctly configured."""
    monkeypatch.setattr(_qwen25_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen25_vl_module.qwen25_vl_3b_sft_config()

    _assert_basic_config(cfg)

    assert cfg.model.attention_backend == "auto"
    assert cfg.model.cross_entropy_loss_fusion is True
    assert cfg.model.cross_entropy_fusion_impl == "native"


def test_qwen25_vl_cuda_graph_settings(monkeypatch: pytest.MonkeyPatch):
    """Test that CUDA graph settings are correctly configured."""
    monkeypatch.setattr(_qwen25_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen25_vl_module.qwen25_vl_3b_sft_config()

    _assert_basic_config(cfg)

    assert cfg.model.cuda_graph_impl == "none"
    assert cfg.model.cuda_graph_scope == "full"
    assert cfg.model.cuda_graph_warmup_steps == 3


def test_qwen25_vl_transformer_impl(monkeypatch: pytest.MonkeyPatch):
    """Test that transformer implementation is set correctly."""
    monkeypatch.setattr(_qwen25_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen25_vl_module.qwen25_vl_3b_sft_config()

    _assert_basic_config(cfg)

    assert cfg.model.transformer_impl == "transformer_engine"


def test_qwen25_vl_memory_saving_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that memory saving settings are disabled by default."""
    monkeypatch.setattr(_qwen25_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen25_vl_module.qwen25_vl_3b_sft_config()

    _assert_basic_config(cfg)

    assert cfg.model.recompute_granularity is None
    assert cfg.model.recompute_modules is None
    assert cfg.model.fine_grained_activation_offloading is False
    assert cfg.model.offload_modules is None
