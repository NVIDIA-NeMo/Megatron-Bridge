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


def _safe_overrides_for(name: str) -> dict:
    # Detect if this is a finetune recipe
    is_finetune = "finetune" in name.lower()
    lname = name.lower()

    overrides = {
        "name": f"unit_{name}",
        "dir": ".",
        "train_iters": 10,
        "micro_batch_size": 1,
        "seq_length": 64,
        "min_lr": 1e-5,
        "lr_warmup_iters": 2,
    }

    # 405B has special default for global_batch_size (6), don't override it
    if "405b" not in lname:
        overrides["global_batch_size"] = 2

    if is_finetune:
        # Finetuning-specific overrides
        overrides.update(
            {
                "finetune_lr": 1e-4,
                # Note: Finetuning always uses HF tokenizer, never null tokenizer
                # Note: Finetuning recipes set parallelism internally based on PEFT vs full SFT
            }
        )
    else:
        # Pretrain-specific overrides
        overrides.update(
            {
                "mock": True,
                "lr": 1e-4,
                "use_null_tokenizer": True,
                "tensor_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "context_parallel_size": 1,
            }
        )
        # Low precision recipes require an additional mixed_precision_recipe argument
        if "low_precision" in lname:
            overrides.update(
                {
                    "mixed_precision_recipe": "bf16_with_fp8_current_scaling_mixed",
                }
            )

        # Large models/variants may set additional flags in pretrain recipes
        if "70b" in lname or "405b" in lname:
            overrides.update(
                {
                    "virtual_pipeline_model_parallel_size": None,
                    "sequence_parallel": True,
                }
            )

    return overrides


class _FakeModelCfg:
    def __init__(self):
        self.cross_entropy_fusion_impl = "te"

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

    # Check sequence length (different attribute names for different dataset types)
    if hasattr(cfg.dataset, "sequence_length"):
        assert cfg.dataset.sequence_length >= 1  # GPTDatasetConfig
    elif hasattr(cfg.dataset, "seq_length"):
        assert cfg.dataset.seq_length >= 1  # FinetuningDatasetConfig / HFDatasetConfig
    else:
        # Some other dataset type
        assert cfg.dataset is not None


@pytest.mark.parametrize("recipe_func", _LLAMA_RECIPE_FUNCS)
def test_each_llama_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for(recipe_func.__name__)

    cfg = recipe_func(**overrides)

    _assert_basic_config(cfg)

    # Ensure tokenizer choice matches recipe type
    is_finetune = "finetune" in recipe_func.__name__.lower()
    if is_finetune:
        # Finetuning recipes always use HF tokenizer
        assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
        assert cfg.tokenizer.tokenizer_model is not None
    else:
        # Pretrain recipes honor use_null_tokenizer override
        if overrides.get("use_null_tokenizer"):
            assert cfg.tokenizer.tokenizer_type == "NullTokenizer"
        else:
            assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
            assert cfg.tokenizer.tokenizer_model is not None

    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1

    if "llama3" in recipe_func.__name__.lower():
        assert cfg.model.cross_entropy_fusion_impl == "te"


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
@pytest.mark.parametrize("peft", ["lora", "dora", "none"])
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
    if peft in ["lora", "dora"]:
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


def test_llama31_405b_has_account_for_settings(monkeypatch: pytest.MonkeyPatch):
    """Test that 405B model has account_for settings enabled."""
    from megatron.bridge.recipes.llama import llama31_405b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.llama.llama3")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for("llama31_405b_finetune_config")
    cfg = llama31_405b_finetune_config(**overrides)

    _assert_basic_config(cfg)

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

    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 8
    assert cfg.model.virtual_pipeline_model_parallel_size == 8
    assert cfg.train.global_batch_size == 32


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
    assert cfg.model.pipeline_model_parallel_size == 16
    assert cfg.train.global_batch_size == 16


def test_llama3_8b_full_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 8B full SFT has correct default parallelism."""
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


def test_llama3_70b_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 70B LoRA has correct default parallelism."""
    from megatron.bridge.recipes.llama import llama3_70b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.llama.llama3")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for("llama3_70b_finetune_config")
    overrides["peft"] = "lora"

    cfg = llama3_70b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For LoRA, 70B should use TP=8
    assert cfg.model.tensor_model_parallel_size == 8


def test_llama3_8b_dora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 8B DoRA has correct default parallelism and performance optimizations."""
    from megatron.bridge.recipes.llama import llama3_8b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.llama.llama3")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for("llama3_8b_finetune_config")
    overrides["peft"] = "dora"

    cfg = llama3_8b_finetune_config(**overrides)

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
    from megatron.bridge.recipes.llama import llama3_70b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.llama.llama3")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for("llama3_70b_finetune_config")
    overrides["peft"] = "dora"

    cfg = llama3_70b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For DoRA, 70B should use TP=8 (same as LoRA)
    assert cfg.model.tensor_model_parallel_size == 8


def test_llama31_405b_dora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 405B DoRA has correct default parallelism (performance mode)."""
    from megatron.bridge.recipes.llama import llama31_405b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.llama.llama3")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for("llama31_405b_finetune_config")
    overrides["peft"] = "dora"

    cfg = llama31_405b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For DoRA, 405B should use same parallelism as LoRA
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 8
    assert cfg.model.virtual_pipeline_model_parallel_size == 8
    assert cfg.train.global_batch_size == 32
