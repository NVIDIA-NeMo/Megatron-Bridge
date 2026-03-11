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
from typing import Callable

import pytest


_llama_module = importlib.import_module("megatron.bridge.recipes.llama")
_LLAMA_RECIPE_FUNCS = [
    getattr(_llama_module, name)
    for name in getattr(_llama_module, "__all__", [])
    if callable(getattr(_llama_module, name, None))
]


# Llama3 SFT-specific tests
_LLAMA3_SFT_FUNCS = [
    getattr(_llama_module, name)
    for name in [
        "llama32_1b_sft_config",
        "llama32_3b_sft_config",
        "llama3_8b_sft_config",
        "llama31_8b_sft_config",
        "llama3_70b_sft_config",
        "llama31_70b_sft_config",
        "llama31_405b_sft_config",
        "llama33_70b_sft_config",
        "llama34_scout_17b_16e_sft_config",
        "llama34_maverick_17b_128e_sft_config",
    ]
    if callable(getattr(_llama_module, name, None))
]


# Llama3 PEFT-specific tests
_LLAMA3_PEFT_FUNCS = [
    getattr(_llama_module, name)
    for name in [
        "llama32_1b_peft_config",
        "llama32_3b_peft_config",
        "llama3_8b_peft_config",
        "llama31_8b_peft_config",
        "llama3_70b_peft_config",
        "llama31_70b_peft_config",
        "llama31_405b_peft_config",
        "llama33_70b_peft_config",
        "llama34_scout_17b_16e_peft_config",
        "llama34_maverick_17b_128e_peft_config",
    ]
    if callable(getattr(_llama_module, name, None))
]


def _safe_overrides_for(name: str) -> dict:
    """Return overrides for recipe functions.

    Pretrain configs use the new parameterless API (return empty dict).
    SFT/PEFT configs also use parameterless API now.
    Special case: low_precision pretrain configs still require mixed_precision_recipe.
    """
    lname = name.lower()

    # Exception: low_precision recipes still require mixed_precision_recipe argument
    if "low_precision" in lname:
        return {"mixed_precision_recipe": "bf16_with_fp8_current_scaling_mixed"}

    return {}


class _FakeModelCfg:
    def __init__(self):
        self.cross_entropy_fusion_impl = "te"
        self.context_parallel_size = 1

    def finalize(self):
        return None


class _FakeBridge:
    def __init__(self):
        pass

    def to_megatron_provider(self, load_weights: bool = False):
        return _FakeModelCfg()

    @staticmethod
    def from_hf_pretrained(hf_path: str, **kwargs):
        return _FakeBridge()


def _apply_test_overrides(cfg, name: str):
    """Apply test-friendly overrides to a config after creation."""
    lname = name.lower()

    # Apply common test overrides
    cfg.train.train_iters = 10
    cfg.train.micro_batch_size = 1
    cfg.dataset.seq_length = 64
    cfg.scheduler.min_lr = 1e-5
    cfg.scheduler.lr_warmup_iters = 2
    cfg.optimizer.lr = 1e-4
    cfg.logger.name = f"unit_{name}"
    cfg.logger.dir = "."

    # 405B has special global_batch_size defaults, don't override
    if "405b" not in lname:
        cfg.train.global_batch_size = 2

    return cfg


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

    if hasattr(cfg.dataset, "seq_length"):
        assert cfg.dataset.seq_length >= 1
    else:
        # Some other dataset type
        assert cfg.dataset is not None


@pytest.mark.parametrize("recipe_func", _LLAMA_RECIPE_FUNCS)
def test_each_llama_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    func_name = recipe_func.__name__
    is_peft = "peft" in func_name.lower()
    is_sft = "sft" in func_name.lower()
    is_low_precision = "low_precision" in func_name.lower()

    # New API: SFT/PEFT configs are parameterless (PEFT has optional peft_scheme)
    if is_peft:
        cfg = recipe_func(peft_scheme="lora")
    elif is_low_precision:
        overrides = _safe_overrides_for(func_name)
        cfg = recipe_func(**overrides)
    else:
        cfg = recipe_func()

    _assert_basic_config(cfg)

    # Ensure tokenizer is properly configured
    is_sft_or_peft = is_sft or is_peft
    if is_sft_or_peft:
        # SFT/PEFT recipes always use HF tokenizer
        assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
        assert cfg.tokenizer.tokenizer_model is not None
    else:
        # Pretrain recipes use either NullTokenizer or HuggingFaceTokenizer
        if cfg.tokenizer.tokenizer_type == "NullTokenizer":
            assert cfg.tokenizer.vocab_size is not None
        else:
            assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
            assert cfg.tokenizer.tokenizer_model is not None

    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1

    if "llama3" in recipe_func.__name__.lower():
        # Pretrain configs use "te", SFT/PEFT configs use "native"
        expected_impl = (
            "native" if ("sft" in recipe_func.__name__.lower() or "peft" in recipe_func.__name__.lower()) else "te"
        )
        assert cfg.model.cross_entropy_fusion_impl == expected_impl


@pytest.mark.parametrize("recipe_func", _LLAMA3_SFT_FUNCS)
def test_llama3_sft_config_builds(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that each Llama3 SFT recipe builds a valid config."""
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    cfg = recipe_func()
    _apply_test_overrides(cfg, recipe_func.__name__)

    _assert_basic_config(cfg)

    # SFT always uses HF tokenizer
    assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
    assert cfg.tokenizer.tokenizer_model is not None

    # Check parallelism
    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1

    # SFT should not have PEFT config
    assert cfg.peft is None


@pytest.mark.parametrize("recipe_func", _LLAMA3_PEFT_FUNCS)
def test_llama3_peft_config_builds(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that each Llama3 PEFT recipe builds a valid config."""
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    cfg = recipe_func(peft_scheme="lora")
    _apply_test_overrides(cfg, recipe_func.__name__)

    _assert_basic_config(cfg)

    # PEFT always uses HF tokenizer
    assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
    assert cfg.tokenizer.tokenizer_model is not None

    # Check parallelism
    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1

    # PEFT should have PEFT config
    assert cfg.peft is not None


@pytest.mark.parametrize("recipe_func", _LLAMA3_PEFT_FUNCS)
@pytest.mark.parametrize("peft_scheme", ["lora", "dora"])
def test_llama3_peft_schemes(recipe_func: Callable, peft_scheme: str, monkeypatch: pytest.MonkeyPatch):
    """Test that PEFT configurations are correctly applied with different schemes."""
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    cfg = recipe_func(peft_scheme=peft_scheme)
    _apply_test_overrides(cfg, recipe_func.__name__)

    _assert_basic_config(cfg)

    # Check PEFT config presence
    assert cfg.peft is not None


@pytest.mark.parametrize("packed", [True, False])
def test_llama3_8b_sft_packed_sequence(packed: bool, monkeypatch: pytest.MonkeyPatch):
    """Test that packed sequence configuration works correctly."""
    from megatron.bridge.recipes.llama import llama3_8b_sft_config

    mod = importlib.import_module("megatron.bridge.recipes.llama.llama3")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    cfg = llama3_8b_sft_config()
    _apply_test_overrides(cfg, "llama3_8b_sft_config")

    # Modify packed_sequence after creation
    cfg.dataset.packed_sequence = packed

    _assert_basic_config(cfg)


def test_llama31_405b_has_account_for_settings(monkeypatch: pytest.MonkeyPatch):
    """Test that 405B model has account_for settings enabled."""
    from megatron.bridge.recipes.llama import llama31_405b_sft_config

    mod = importlib.import_module("megatron.bridge.recipes.llama.llama3")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    cfg = llama31_405b_sft_config()
    _apply_test_overrides(cfg, "llama31_405b_sft_config")

    _assert_basic_config(cfg)

    # Check account_for settings
    assert cfg.model.account_for_embedding_in_pipeline_split is True
    assert cfg.model.account_for_loss_in_pipeline_split is True


def test_llama31_405b_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 405B LoRA has correct default parallelism (performance mode)."""
    from megatron.bridge.recipes.llama import llama31_405b_peft_config

    mod = importlib.import_module("megatron.bridge.recipes.llama.llama3")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    cfg = llama31_405b_peft_config(peft_scheme="lora")
    _apply_test_overrides(cfg, "llama31_405b_peft_config")

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 8
    assert cfg.model.virtual_pipeline_model_parallel_size == 8
    assert cfg.train.global_batch_size == 32


def test_llama31_405b_full_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 405B full SFT has correct default parallelism."""
    from megatron.bridge.recipes.llama import llama31_405b_sft_config

    mod = importlib.import_module("megatron.bridge.recipes.llama.llama3")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    cfg = llama31_405b_sft_config()
    _apply_test_overrides(cfg, "llama31_405b_sft_config")

    _assert_basic_config(cfg)

    # For full SFT, 405B should use TP=8, PP=14
    assert cfg.model.tensor_model_parallel_size == 8
    assert cfg.model.pipeline_model_parallel_size == 16
    assert cfg.train.global_batch_size == 16


def test_llama3_8b_full_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 8B full SFT has correct default parallelism."""
    from megatron.bridge.recipes.llama import llama3_8b_sft_config

    mod = importlib.import_module("megatron.bridge.recipes.llama.llama3")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    cfg = llama3_8b_sft_config()
    _apply_test_overrides(cfg, "llama3_8b_sft_config")

    _assert_basic_config(cfg)

    # For full SFT, 8B should use TP=2
    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1

    # Check manual GC is enabled
    assert cfg.train.manual_gc is True


def test_llama3_8b_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 8B LoRA has correct default parallelism and performance optimizations."""
    from megatron.bridge.recipes.llama import llama3_8b_peft_config

    mod = importlib.import_module("megatron.bridge.recipes.llama.llama3")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    cfg = llama3_8b_peft_config(peft_scheme="lora")
    _apply_test_overrides(cfg, "llama3_8b_peft_config")

    _assert_basic_config(cfg)

    # For LoRA, 8B should use TP=1
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1

    # Check PEFT-specific performance settings
    assert cfg.model.cross_entropy_loss_fusion is False  # Disabled for PEFT
    assert cfg.optimizer.use_distributed_optimizer is False  # Disabled for PEFT

    # Check manual GC is enabled
    assert cfg.train.manual_gc is True
    assert cfg.train.manual_gc_interval == 100


def test_llama3_70b_full_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 70B full SFT has correct default parallelism."""
    from megatron.bridge.recipes.llama import llama3_70b_sft_config

    mod = importlib.import_module("megatron.bridge.recipes.llama.llama3")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    cfg = llama3_70b_sft_config()
    _apply_test_overrides(cfg, "llama3_70b_sft_config")

    _assert_basic_config(cfg)

    # For full SFT, 70B should use TP=8, PP=4
    assert cfg.model.tensor_model_parallel_size == 8
    assert cfg.model.pipeline_model_parallel_size == 4


def test_llama3_70b_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 70B LoRA has correct default parallelism."""
    from megatron.bridge.recipes.llama import llama3_70b_peft_config

    mod = importlib.import_module("megatron.bridge.recipes.llama.llama3")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    cfg = llama3_70b_peft_config(peft_scheme="lora")
    _apply_test_overrides(cfg, "llama3_70b_peft_config")

    _assert_basic_config(cfg)

    # For LoRA, 70B should use TP=8
    assert cfg.model.tensor_model_parallel_size == 8


def test_llama3_8b_dora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 8B DoRA has correct default parallelism and performance optimizations."""
    from megatron.bridge.recipes.llama import llama3_8b_peft_config

    mod = importlib.import_module("megatron.bridge.recipes.llama.llama3")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    cfg = llama3_8b_peft_config(peft_scheme="dora")
    _apply_test_overrides(cfg, "llama3_8b_peft_config")

    _assert_basic_config(cfg)

    # For DoRA, 8B should use TP=1 (same as LoRA)
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1

    # Check PEFT-specific performance settings
    assert cfg.model.cross_entropy_loss_fusion is False  # Disabled for PEFT
    assert cfg.optimizer.use_distributed_optimizer is False  # Disabled for PEFT

    # Check manual GC is enabled
    assert cfg.train.manual_gc is True
    assert cfg.train.manual_gc_interval == 100


def test_llama3_70b_dora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 70B DoRA has correct default parallelism."""
    from megatron.bridge.recipes.llama import llama3_70b_peft_config

    mod = importlib.import_module("megatron.bridge.recipes.llama.llama3")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    cfg = llama3_70b_peft_config(peft_scheme="dora")
    _apply_test_overrides(cfg, "llama3_70b_peft_config")

    _assert_basic_config(cfg)

    # For DoRA, 70B should use TP=8 (same as LoRA)
    assert cfg.model.tensor_model_parallel_size == 8


def test_llama31_405b_dora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 405B DoRA has correct default parallelism (performance mode)."""
    from megatron.bridge.recipes.llama import llama31_405b_peft_config

    mod = importlib.import_module("megatron.bridge.recipes.llama.llama3")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    cfg = llama31_405b_peft_config(peft_scheme="dora")
    _apply_test_overrides(cfg, "llama31_405b_peft_config")

    _assert_basic_config(cfg)

    # For DoRA, 405B should use same parallelism as LoRA
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 8
    assert cfg.model.virtual_pipeline_model_parallel_size == 8
    assert cfg.train.global_batch_size == 32


def test_llama3_8b_low_precision_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 8B low precision configs have correct defaults."""
    from megatron.bridge.recipes.llama import llama3_8b_low_precision_pretrain_config

    mod = importlib.import_module("megatron.bridge.recipes.llama.llama3")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for("llama3_8b_low_precision_pretrain_config")

    cfg = llama3_8b_low_precision_pretrain_config(**overrides)

    _assert_basic_config(cfg)

    # For low precision, 8B should use correct defaults
    assert cfg.optimizer.lr == 6e-4
    assert cfg.optimizer.min_lr == 6e-6
    assert cfg.optimizer.adam_eps == 1e-8
    assert cfg.train.micro_batch_size == 1
    assert cfg.train.global_batch_size == 768


def test_llama3_8b_low_precision_nvfp4_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 8B low precision NVFP4 has correct default BF16 layer configuration."""
    from megatron.bridge.recipes.llama import llama3_8b_low_precision_pretrain_config

    mod = importlib.import_module("megatron.bridge.recipes.llama.llama3")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for("llama3_8b_low_precision_pretrain_config")
    # change the mixed precision recipe to NVFP4
    overrides["mixed_precision_recipe"] = "bf16_with_nvfp4_mixed"

    cfg = llama3_8b_low_precision_pretrain_config(**overrides)

    _assert_basic_config(cfg)

    # For NVFP4, 8B should use BF16 for the last 4 layers
    assert cfg.mixed_precision.first_last_layers_bf16 is True
    assert cfg.mixed_precision.num_layers_at_start_in_bf16 == 0
    assert cfg.mixed_precision.num_layers_at_end_in_bf16 == 4
