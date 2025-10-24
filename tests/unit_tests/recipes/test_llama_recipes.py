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


def _safe_overrides_for(name: str) -> dict:
    overrides = {
        "name": f"unit_{name}",
        "dir": ".",
        "mock": True,
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
        "use_null_tokenizer": True,
    }

    # Large models/variants may set additional flags in recipes; keep harmless defaults
    lname = name.lower()
    if "70b" in lname or "405b" in lname:
        overrides.update(
            {
                "virtual_pipeline_parallelism": None,
                "sequence_parallelism": True,
            }
        )

    return overrides


class _FakeModelCfg:
    def finalize(self):
        return None


class _FakeBridge:
    def __init__(self):
        pass

    def to_megatron_provider(self, load_weights: bool = False):
        return _FakeModelCfg()

    @staticmethod
    def from_hf_pretrained(hf_path: str):
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
    assert cfg.dataset.sequence_length >= 1


@pytest.mark.parametrize("recipe_func", _LLAMA_RECIPE_FUNCS)
def test_each_llama_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for(recipe_func.__name__)

    cfg = recipe_func(**overrides)

    _assert_basic_config(cfg)

    if overrides.get("use_null_tokenizer") and hasattr(cfg, "tokenizer") and hasattr(cfg.tokenizer, "tokenizer_type"):
        assert cfg.tokenizer.tokenizer_type == "NullTokenizer"

    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1


# Llama3 finetune-specific tests
_LLAMA3_FINETUNE_FUNCS = [
    getattr(_llama_module, name)
    for name in [
        "llama32_1b_finetune_config",
        "llama32_3b_finetune_config",
        "llama3_8b_finetune_config",
        "llama31_8b_finetune_config",
        "llama3_70b_finetune_config",
        "llama31_70b_finetune_config",
        "llama31_405b_finetune_config",
    ]
    if callable(getattr(_llama_module, name, None))
]


@pytest.mark.parametrize("recipe_func", _LLAMA3_FINETUNE_FUNCS)
def test_llama3_finetune_config_builds(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that each Llama3 finetune recipe builds a valid config."""
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for(recipe_func.__name__)
    cfg = recipe_func(**overrides)

    _assert_basic_config(cfg)

    # Finetuning always uses HF tokenizer
    assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
    assert cfg.tokenizer.tokenizer_model is not None

    # Check parallelism
    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1


@pytest.mark.parametrize("recipe_func", _LLAMA3_FINETUNE_FUNCS)
@pytest.mark.parametrize("peft", ["lora", "none"])
def test_llama3_finetune_peft_vs_full_sft(recipe_func: Callable, peft: str, monkeypatch: pytest.MonkeyPatch):
    """Test that PEFT and full SFT configurations are correctly applied."""
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for(recipe_func.__name__)
    overrides["peft"] = peft

    cfg = recipe_func(**overrides)

    _assert_basic_config(cfg)

    # Check PEFT config presence
    if peft == "lora":
        assert cfg.peft is not None
    elif peft == "none":
        assert cfg.peft is None


@pytest.mark.parametrize("packed", [True, False])
def test_llama3_8b_finetune_packed_sequence(packed: bool, monkeypatch: pytest.MonkeyPatch):
    """Test that packed sequence configuration works correctly."""
    from megatron.bridge.recipes.llama import llama3_8b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.llama.llama3")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for("llama3_8b_finetune_config")
    overrides["packed_sequence"] = packed

    cfg = llama3_8b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # Packed sequence affects global batch size default
    if packed and "global_batch_size" not in overrides:
        # Would default to 8 for packed
        pass
    else:
        # Uses explicit override
        assert cfg.train.global_batch_size == overrides["global_batch_size"]


def test_llama31_405b_has_recompute(monkeypatch: pytest.MonkeyPatch):
    """Test that 405B model has recompute enabled."""
    from megatron.bridge.recipes.llama import llama31_405b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.llama.llama3")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for("llama31_405b_finetune_config")
    cfg = llama31_405b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # Check recompute settings
    assert hasattr(cfg.model, "recompute_granularity")
    assert cfg.model.recompute_granularity == "full"
    assert cfg.model.recompute_method == "uniform"
    assert cfg.model.recompute_num_layers == 1

    # Check account_for settings
    assert cfg.model.account_for_embedding_in_pipeline_split is True
    assert cfg.model.account_for_loss_in_pipeline_split is True


def test_llama31_405b_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 405B LoRA has correct default parallelism (performance mode)."""
    from megatron.bridge.recipes.llama import llama31_405b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.llama.llama3")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for("llama31_405b_finetune_config")
    overrides["peft"] = "lora"

    cfg = llama31_405b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For LoRA, 405B performance mode uses TP=4, PP=4, VPP=4
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.virtual_pipeline_model_parallel_size == 4
    assert cfg.model.sequence_parallel is True
    assert cfg.train.global_batch_size == 6


def test_llama31_405b_full_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 405B full SFT has correct default parallelism."""
    from megatron.bridge.recipes.llama import llama31_405b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.llama.llama3")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for("llama31_405b_finetune_config")
    overrides["peft"] = "none"

    cfg = llama31_405b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For full SFT, 405B should use TP=8, PP=14
    assert cfg.model.tensor_model_parallel_size == 8
    assert cfg.model.pipeline_model_parallel_size == 14
    assert cfg.model.virtual_pipeline_model_parallel_size is None
    assert cfg.model.sequence_parallel is True
    assert cfg.train.global_batch_size == 6  # 405B uses smaller batch


def test_llama3_8b_full_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 8B full SFT has correct default parallelism and performance optimizations."""
    from megatron.bridge.recipes.llama import llama3_8b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.llama.llama3")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for("llama3_8b_finetune_config")
    overrides["peft"] = "none"

    cfg = llama3_8b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For full SFT, 8B should use TP=2
    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1

    # Check full SFT-specific performance settings
    assert cfg.ddp.grad_reduce_in_fp32 is False  # Performance optimization
    assert cfg.ddp.overlap_grad_reduce is True
    assert cfg.ddp.overlap_param_gather is True

    # Check manual GC is enabled
    assert cfg.train.manual_gc is True


def test_llama3_8b_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 8B LoRA has correct default parallelism and performance optimizations."""
    from megatron.bridge.recipes.llama import llama3_8b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.llama.llama3")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for("llama3_8b_finetune_config")
    overrides["peft"] = "lora"

    cfg = llama3_8b_finetune_config(**overrides)

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
    from megatron.bridge.recipes.llama import llama3_70b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.llama.llama3")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for("llama3_70b_finetune_config")
    overrides["peft"] = "none"

    cfg = llama3_70b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For full SFT, 70B should use TP=8, PP=4
    assert cfg.model.tensor_model_parallel_size == 8
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.virtual_pipeline_model_parallel_size == 5
    assert cfg.model.sequence_parallel is True


def test_llama3_70b_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 70B LoRA has correct default parallelism."""
    from megatron.bridge.recipes.llama import llama3_70b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.llama.llama3")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for("llama3_70b_finetune_config")
    overrides["peft"] = "lora"

    cfg = llama3_70b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For LoRA, 70B should use TP=8, PP=1
    assert cfg.model.tensor_model_parallel_size == 8
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.sequence_parallel is True
