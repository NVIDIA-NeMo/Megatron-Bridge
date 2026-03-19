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
# - Parametrize over all exported Moonlight recipe functions in `megatron.bridge.recipes.moonlight`.
# - For each recipe, monkeypatch `MoonlightModelProvider16B` with a lightweight fake to avoid I/O.
# - Build a config with small, safe overrides and assert it forms a valid `ConfigContainer`.
# - Verify tokenizer selection and sanity-check parallelism fields.
#

import importlib
from typing import Callable

import pytest


_moonlight_module = importlib.import_module("megatron.bridge.recipes.moonlight")
_MOONLIGHT_RECIPE_FUNCS = [
    getattr(_moonlight_module, name)
    for name in getattr(_moonlight_module, "__all__", [])
    if callable(getattr(_moonlight_module, name, None))
]

# Moonlight SFT-specific tests
_MOONLIGHT_SFT_FUNCS = [
    getattr(_moonlight_module, name)
    for name in ["moonlight_16b_sft_config"]
    if callable(getattr(_moonlight_module, name, None))
]

# Moonlight PEFT-specific tests
_MOONLIGHT_PEFT_FUNCS = [
    getattr(_moonlight_module, name)
    for name in ["moonlight_16b_peft_config"]
    if callable(getattr(_moonlight_module, name, None))
]


def _safe_overrides_for(name: str) -> dict:
    """Return overrides for recipe functions.

    All configs now use the new parameterless API (return empty dict).
    """
    return {}


def _apply_test_overrides(cfg, name: str):
    """Apply test-friendly overrides to a config after creation."""
    # Apply common test overrides
    cfg.train.train_iters = 10
    cfg.train.micro_batch_size = 1
    cfg.dataset.seq_length = 64
    cfg.scheduler.min_lr = 1e-5
    cfg.scheduler.lr_warmup_iters = 2
    cfg.optimizer.lr = 1e-4
    cfg.logger.name = f"unit_{name}"
    cfg.logger.dir = "."
    cfg.train.global_batch_size = 2
    cfg.tokenizer.tokenizer_model = "moonshotai/Moonlight-16B-A3B"

    return cfg


class _FakeMoonlightModelProvider16B:
    """Fake MoonlightModelProvider16B for testing without model I/O."""

    def __init__(self, *args, **kwargs):
        # Store all the kwargs that would be passed to the real provider
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Set required attributes
        self.vocab_size = 151936  # Default vocab size for Moonlight
        self.account_for_embedding_in_pipeline_split = False
        self.account_for_loss_in_pipeline_split = False
        self.num_layers_in_first_pipeline_stage = None
        self.num_layers_in_last_pipeline_stage = None
        self.moe_permute_fusion = True
        self.apply_rope_fusion = False
        self.pipeline_model_parallel_layout = None
        self.moe_token_dispatcher_type = "alltoall"
        self.moe_enable_deepep = False
        self.moe_shared_expert_overlap = True

        # Set parallelism defaults if not provided
        if not hasattr(self, "tensor_model_parallel_size"):
            self.tensor_model_parallel_size = 1
        if not hasattr(self, "pipeline_model_parallel_size"):
            self.pipeline_model_parallel_size = 1
        if not hasattr(self, "context_parallel_size"):
            self.context_parallel_size = 1
        if not hasattr(self, "expert_model_parallel_size"):
            self.expert_model_parallel_size = 1

    def finalize(self):
        return None


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


@pytest.mark.parametrize("recipe_func", _MOONLIGHT_RECIPE_FUNCS)
def test_each_moonlight_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)

    # Monkeypatch the MoonlightModelProvider16B class
    monkeypatch.setattr(mod, "MLAModelProvider", _FakeMoonlightModelProvider16B)

    func_name = recipe_func.__name__
    is_peft = "peft" in func_name.lower()
    is_sft = "sft" in func_name.lower()

    # New API: SFT configs are parameterless, PEFT has optional peft_scheme
    if is_peft:
        cfg = recipe_func(peft_scheme="lora")
    else:
        cfg = recipe_func()

    _assert_basic_config(cfg)

    # Ensure tokenizer choice matches recipe type
    is_sft_or_peft = is_sft or is_peft
    if is_sft_or_peft:
        # SFT/PEFT recipes always use HF tokenizer
        assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
        assert cfg.tokenizer.tokenizer_model is not None
    else:
        # Pretrain recipes use NullTokenizer
        assert cfg.tokenizer.tokenizer_type == "NullTokenizer"

    # Check parallelism settings
    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "expert_model_parallel_size", 1) >= 1


@pytest.mark.parametrize("recipe_func", _MOONLIGHT_SFT_FUNCS)
def test_moonlight_sft_config_builds(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that each Moonlight SFT recipe builds a valid config."""
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    monkeypatch.setattr(mod, "MLAModelProvider", _FakeMoonlightModelProvider16B)

    cfg = recipe_func()
    _apply_test_overrides(cfg, recipe_func.__name__)

    _assert_basic_config(cfg)

    # SFT always uses HF tokenizer
    assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
    assert cfg.tokenizer.tokenizer_model is not None

    # Check parallelism
    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "expert_model_parallel_size", 1) >= 1

    # SFT should not have PEFT config
    assert cfg.peft is None


@pytest.mark.parametrize("recipe_func", _MOONLIGHT_PEFT_FUNCS)
def test_moonlight_peft_config_builds(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that each Moonlight PEFT recipe builds a valid config."""
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    monkeypatch.setattr(mod, "MLAModelProvider", _FakeMoonlightModelProvider16B)

    cfg = recipe_func(peft_scheme="lora")
    _apply_test_overrides(cfg, recipe_func.__name__)

    _assert_basic_config(cfg)

    # PEFT always uses HF tokenizer
    assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
    assert cfg.tokenizer.tokenizer_model is not None

    # Check parallelism
    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "expert_model_parallel_size", 1) >= 1

    # PEFT should have PEFT config
    assert cfg.peft is not None


@pytest.mark.parametrize("recipe_func", _MOONLIGHT_PEFT_FUNCS)
@pytest.mark.parametrize("peft_scheme", ["lora", "dora"])
def test_moonlight_peft_schemes(recipe_func: Callable, peft_scheme: str, monkeypatch: pytest.MonkeyPatch):
    """Test that PEFT configurations are correctly applied with different schemes."""
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    monkeypatch.setattr(mod, "MLAModelProvider", _FakeMoonlightModelProvider16B)

    cfg = recipe_func(peft_scheme=peft_scheme)
    _apply_test_overrides(cfg, recipe_func.__name__)

    _assert_basic_config(cfg)

    # Check PEFT config presence
    assert cfg.peft is not None


def test_moonlight_16b_peft_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that Moonlight-16B LoRA has correct default parallelism."""
    from megatron.bridge.recipes.moonlight import moonlight_16b_peft_config

    mod = importlib.import_module("megatron.bridge.recipes.moonlight.moonlight_16b")
    monkeypatch.setattr(mod, "MLAModelProvider", _FakeMoonlightModelProvider16B)

    cfg = moonlight_16b_peft_config(peft_scheme="lora")
    _apply_test_overrides(cfg, "moonlight_16b_peft_config")

    _assert_basic_config(cfg)

    # For LoRA, Moonlight-16B should use TP=1, PP=1, EP=2
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 2
    assert cfg.model.sequence_parallel is False

    # Check manual GC is enabled
    assert cfg.train.manual_gc is True
    assert cfg.train.manual_gc_interval == 5


def test_moonlight_16b_peft_dora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that Moonlight-16B DoRA has correct default parallelism."""
    from megatron.bridge.recipes.moonlight import moonlight_16b_peft_config

    mod = importlib.import_module("megatron.bridge.recipes.moonlight.moonlight_16b")
    monkeypatch.setattr(mod, "MLAModelProvider", _FakeMoonlightModelProvider16B)

    cfg = moonlight_16b_peft_config(peft_scheme="dora")
    _apply_test_overrides(cfg, "moonlight_16b_peft_config")

    _assert_basic_config(cfg)

    # For DoRA, Moonlight-16B should use TP=1, PP=1, EP=2 (same as LoRA)
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 2
    assert cfg.model.sequence_parallel is False

    # Check manual GC is enabled
    assert cfg.train.manual_gc is True
    assert cfg.train.manual_gc_interval == 5


def test_moonlight_16b_sft_full_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that Moonlight-16B full SFT has correct default parallelism."""
    from megatron.bridge.recipes.moonlight import moonlight_16b_sft_config

    mod = importlib.import_module("megatron.bridge.recipes.moonlight.moonlight_16b")
    monkeypatch.setattr(mod, "MLAModelProvider", _FakeMoonlightModelProvider16B)

    cfg = moonlight_16b_sft_config()
    _apply_test_overrides(cfg, "moonlight_16b_sft_config")

    _assert_basic_config(cfg)

    # For full SFT, Moonlight-16B should use TP=2, PP=1, EP=8
    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.model.sequence_parallel is True

    # Check manual GC is enabled
    assert cfg.train.manual_gc is True
    assert cfg.train.manual_gc_interval == 5


def test_moonlight_16b_sft_precision_aware_optimizer(monkeypatch: pytest.MonkeyPatch):
    """Test that Moonlight-16B SFT uses precision-aware optimizer settings."""
    from megatron.bridge.recipes.moonlight import moonlight_16b_sft_config

    mod = importlib.import_module("megatron.bridge.recipes.moonlight.moonlight_16b")
    monkeypatch.setattr(mod, "MLAModelProvider", _FakeMoonlightModelProvider16B)

    cfg = moonlight_16b_sft_config()
    _apply_test_overrides(cfg, "moonlight_16b_sft_config")

    _assert_basic_config(cfg)

    # Check precision-aware optimizer settings
    assert cfg.optimizer.use_precision_aware_optimizer is True
    import torch

    assert cfg.optimizer.main_params_dtype == torch.float32
    assert cfg.optimizer.main_grads_dtype == torch.bfloat16
    assert cfg.optimizer.exp_avg_dtype == torch.bfloat16
    assert cfg.optimizer.exp_avg_sq_dtype == torch.bfloat16


def test_moonlight_16b_sft_tokenizer_with_trust_remote_code(monkeypatch: pytest.MonkeyPatch):
    """Test that Moonlight-16B SFT uses HF tokenizer with trust_remote_code."""
    from megatron.bridge.recipes.moonlight import moonlight_16b_sft_config

    mod = importlib.import_module("megatron.bridge.recipes.moonlight.moonlight_16b")
    monkeypatch.setattr(mod, "MLAModelProvider", _FakeMoonlightModelProvider16B)

    cfg = moonlight_16b_sft_config()

    _assert_basic_config(cfg)

    # Check tokenizer settings
    assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
    assert cfg.tokenizer.tokenizer_model == "moonshotai/Moonlight-16B-A3B"
    assert cfg.tokenizer.hf_tokenizer_kwargs == {"trust_remote_code": True}
