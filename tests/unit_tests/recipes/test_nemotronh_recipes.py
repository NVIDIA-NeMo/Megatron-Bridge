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
# - Parametrize over all exported NemotronH recipe functions in `megatron.bridge.recipes.nemotronh`.
# - For each recipe, monkeypatch `AutoBridge` with a lightweight fake to avoid I/O.
# - Build a config with small, safe overrides and assert it forms a valid `ConfigContainer`.
# - Verify tokenizer selection: pretrain recipes honor `use_null_tokenizer`, finetune recipes always use HF tokenizer.
# - Sanity-check parallelism fields and finetuning-specific requirements.
#

import importlib
from typing import Callable

import pytest


_nemotronh_module = importlib.import_module("megatron.bridge.recipes.nemotronh")
_NEMOTRONH_RECIPE_FUNCS = [
    getattr(_nemotronh_module, name)
    for name in getattr(_nemotronh_module, "__all__", [])
    if callable(getattr(_nemotronh_module, name, None))
]


def _safe_overrides_for(name: str) -> dict:
    """Return overrides for recipe functions.

    Pretrain configs use the new parameterless API (return empty dict).
    Finetune configs still accept parameters.
    """
    is_finetune = "finetune" in name.lower()

    if is_finetune:
        # Finetuning-specific overrides - finetune configs still accept parameters
        overrides = {
            "name": f"unit_{name}",
            "dir": ".",
            "train_iters": 10,
            "global_batch_size": 2,
            "micro_batch_size": 1,
            "seq_length": 64,
            "finetune_lr": 1e-4,
            "min_lr": 1e-5,
            "lr_warmup_iters": 2,
            "peft": None,
            "pretrained_checkpoint": "/fake/checkpoint/path",
        }
    else:
        # Pretrain configs use the new parameterless API
        overrides = {}

    return overrides


def _assert_basic_config(cfg):
    from megatron.bridge.training.config import ConfigContainer

    assert isinstance(cfg, ConfigContainer)
    # Required top-level sections
    assert cfg.model is not None
    assert cfg.train is not None
    assert cfg.optimizer is not None
    assert cfg.scheduler is not None
    assert cfg.dataset is not None
    assert cfg.logger is not None
    assert cfg.tokenizer is not None
    assert cfg.checkpoint is not None
    assert cfg.rng is not None

    # A few critical fields
    assert cfg.train.global_batch_size >= 1
    assert cfg.train.micro_batch_size >= 1

    # Check sequence length (different attribute names for different dataset types)
    if hasattr(cfg.dataset, "sequence_length"):
        assert cfg.dataset.sequence_length >= 1  # GPTDatasetConfig legacy
    elif hasattr(cfg.dataset, "seq_length"):
        assert cfg.dataset.seq_length >= 1  # FinetuningDatasetConfig / HFDatasetConfig
    else:
        # Some other dataset type
        assert cfg.dataset is not None


@pytest.mark.parametrize("recipe_func", _NEMOTRONH_RECIPE_FUNCS)
def test_each_nemotronh_recipe_builds_config(recipe_func: Callable):
    """Test that each NemotronH recipe builds a valid config."""
    # Note: NemotronH recipes don't use AutoBridge, so no patching needed
    # They directly instantiate model providers

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
        # Pretrain recipes use either NullTokenizer or HuggingFaceTokenizer
        if cfg.tokenizer.tokenizer_type == "NullTokenizer":
            assert cfg.tokenizer.vocab_size is not None
        else:
            assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
            assert cfg.tokenizer.tokenizer_model is not None

    # Parallelism and shaping
    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1

    # Finetuning-specific assertions
    if is_finetune:
        # Should have pretrained_checkpoint set (even if fake)
        assert cfg.checkpoint.pretrained_checkpoint is not None
        # Should have PEFT config (or None if disabled in test)
        assert hasattr(cfg, "peft")  # peft field should exist
        # Dataset should be configured (SQuAD by default)
        assert cfg.dataset is not None


# NemotronH finetune-specific tests (all model sizes)
_NEMOTRONH_FINETUNE_FUNCS = [
    getattr(_nemotronh_module, name)
    for name in [
        "nemotronh_4b_finetune_config",
        "nemotronh_8b_finetune_config",
        "nemotronh_47b_finetune_config",
        "nemotronh_56b_finetune_config",
    ]
    if callable(getattr(_nemotronh_module, name, None))
]


@pytest.mark.parametrize("recipe_func", _NEMOTRONH_FINETUNE_FUNCS)
@pytest.mark.parametrize("peft", ["lora", "none"])
def test_nemotronh_finetune_peft_vs_full_sft(recipe_func: Callable, peft: str):
    """Test that PEFT and full SFT configurations are correctly applied for NemotronH models."""
    # Note: NemotronH recipes don't use AutoBridge, so no patching needed

    overrides = _safe_overrides_for(recipe_func.__name__)
    overrides["peft"] = peft

    cfg = recipe_func(**overrides)

    _assert_basic_config(cfg)

    # Check PEFT config presence
    if peft == "lora":
        assert cfg.peft is not None
    elif peft == "none":
        assert cfg.peft is None


# Nemotron Nano v2 finetune-specific tests (9B and 12B models)
_NEMOTRON_NANO_V2_FINETUNE_FUNCS = [
    getattr(_nemotronh_module, name)
    for name in [
        "nemotron_nano_9b_v2_finetune_config",
        "nemotron_nano_12b_v2_finetune_config",
    ]
    if callable(getattr(_nemotronh_module, name, None))
]


@pytest.mark.parametrize("recipe_func", _NEMOTRON_NANO_V2_FINETUNE_FUNCS)
@pytest.mark.parametrize("peft", ["lora", "none"])
def test_nemotron_nano_v2_finetune_peft_vs_full_sft(recipe_func: Callable, peft: str):
    """Test that PEFT and full SFT configurations are correctly applied for Nemotron Nano v2 models."""
    # Note: Nemotron Nano v2 recipes don't use AutoBridge, so no patching needed

    overrides = _safe_overrides_for(recipe_func.__name__)
    overrides["peft"] = peft

    cfg = recipe_func(**overrides)

    _assert_basic_config(cfg)

    # Check PEFT config presence
    if peft == "lora":
        assert cfg.peft is not None
    elif peft == "none":
        assert cfg.peft is None


def test_nemotron_nano_9b_v2_lora_defaults():
    """Test that Nemotron Nano 9B v2 LoRA has correct default parallelism."""
    from megatron.bridge.recipes.nemotronh import nemotron_nano_9b_v2_finetune_config

    overrides = _safe_overrides_for("nemotron_nano_9b_v2_finetune_config")
    overrides["peft"] = "lora"

    cfg = nemotron_nano_9b_v2_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For LoRA, Nemotron Nano 9B v2 should use TP=1, PP=1
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.sequence_parallel is False

    # Check PEFT config
    assert cfg.peft is not None
    assert cfg.peft.dim == 32
    assert cfg.peft.alpha == 32
    assert cfg.peft.target_modules == ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj"]


def test_nemotron_nano_9b_v2_full_sft_defaults():
    """Test that Nemotron Nano 9B v2 full SFT has correct default parallelism."""
    from megatron.bridge.recipes.nemotronh import nemotron_nano_9b_v2_finetune_config

    overrides = _safe_overrides_for("nemotron_nano_9b_v2_finetune_config")
    overrides["peft"] = "none"

    cfg = nemotron_nano_9b_v2_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For full SFT, Nemotron Nano 9B v2 should use TP=2, PP=1
    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.sequence_parallel is True
    assert cfg.peft is None


def test_nemotron_nano_12b_v2_lora_defaults():
    """Test that Nemotron Nano 12B v2 LoRA has correct default parallelism."""
    from megatron.bridge.recipes.nemotronh import nemotron_nano_12b_v2_finetune_config

    overrides = _safe_overrides_for("nemotron_nano_12b_v2_finetune_config")
    overrides["peft"] = "lora"

    cfg = nemotron_nano_12b_v2_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For LoRA, Nemotron Nano 12B v2 should use TP=1, PP=1
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.sequence_parallel is False

    # Check PEFT config
    assert cfg.peft is not None
    assert cfg.peft.dim == 32
    assert cfg.peft.alpha == 32
    assert cfg.peft.target_modules == ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj"]


def test_nemotron_nano_12b_v2_full_sft_defaults():
    """Test that Nemotron Nano 12B v2 full SFT has correct default parallelism."""
    from megatron.bridge.recipes.nemotronh import nemotron_nano_12b_v2_finetune_config

    overrides = _safe_overrides_for("nemotron_nano_12b_v2_finetune_config")
    overrides["peft"] = "none"

    cfg = nemotron_nano_12b_v2_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For full SFT, Nemotron Nano 12B v2 should use TP=4, PP=1
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.sequence_parallel is True
    assert cfg.peft is None


def test_nemotronh_4b_lora_defaults():
    """Test that NemotronH 4B LoRA has correct default parallelism."""
    from megatron.bridge.recipes.nemotronh import nemotronh_4b_finetune_config

    overrides = _safe_overrides_for("nemotronh_4b_finetune_config")
    overrides["peft"] = "lora"

    cfg = nemotronh_4b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For LoRA, NemotronH 4B should use TP=1, PP=1
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.sequence_parallel is False

    # Check PEFT config
    assert cfg.peft is not None
    assert cfg.peft.dim == 32
    assert cfg.peft.alpha == 32
    assert cfg.peft.target_modules == ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj"]


def test_nemotronh_4b_full_sft_defaults():
    """Test that NemotronH 4B full SFT has correct default parallelism."""
    from megatron.bridge.recipes.nemotronh import nemotronh_4b_finetune_config

    overrides = _safe_overrides_for("nemotronh_4b_finetune_config")
    overrides["peft"] = "none"

    cfg = nemotronh_4b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For full SFT, NemotronH 4B should use TP=1, PP=1 (no change from LoRA)
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.sequence_parallel is False
    assert cfg.peft is None


def test_nemotronh_8b_lora_defaults():
    """Test that NemotronH 8B LoRA has correct default parallelism."""
    from megatron.bridge.recipes.nemotronh import nemotronh_8b_finetune_config

    overrides = _safe_overrides_for("nemotronh_8b_finetune_config")
    overrides["peft"] = "lora"

    cfg = nemotronh_8b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For LoRA, NemotronH 8B should use TP=1, PP=1
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.sequence_parallel is False

    # Check PEFT config
    assert cfg.peft is not None
    assert cfg.peft.dim == 32
    assert cfg.peft.alpha == 32
    assert cfg.peft.target_modules == ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj"]


def test_nemotronh_8b_full_sft_defaults():
    """Test that NemotronH 8B full SFT has correct default parallelism."""
    from megatron.bridge.recipes.nemotronh import nemotronh_8b_finetune_config

    overrides = _safe_overrides_for("nemotronh_8b_finetune_config")
    overrides["peft"] = "none"

    cfg = nemotronh_8b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For full SFT, NemotronH 8B should use TP=2, PP=1
    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.sequence_parallel is True
    assert cfg.peft is None


def test_nemotronh_47b_lora_defaults():
    """Test that NemotronH 47B LoRA has correct default parallelism."""
    from megatron.bridge.recipes.nemotronh import nemotronh_47b_finetune_config

    overrides = _safe_overrides_for("nemotronh_47b_finetune_config")
    overrides["peft"] = "lora"

    cfg = nemotronh_47b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For LoRA, NemotronH 47B should use TP=4, PP=1
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.sequence_parallel is False

    # Check PEFT config
    assert cfg.peft is not None
    assert cfg.peft.dim == 32
    assert cfg.peft.alpha == 32
    assert cfg.peft.target_modules == ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj"]


def test_nemotronh_47b_full_sft_defaults():
    """Test that NemotronH 47B full SFT has correct default parallelism."""
    from megatron.bridge.recipes.nemotronh import nemotronh_47b_finetune_config

    overrides = _safe_overrides_for("nemotronh_47b_finetune_config")
    overrides["peft"] = "none"

    cfg = nemotronh_47b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For full SFT, NemotronH 47B should use TP=8, PP=2
    assert cfg.model.tensor_model_parallel_size == 8
    assert cfg.model.pipeline_model_parallel_size == 2
    assert cfg.model.sequence_parallel is True
    assert cfg.peft is None


def test_nemotronh_56b_lora_defaults():
    """Test that NemotronH 56B LoRA has correct default parallelism."""
    from megatron.bridge.recipes.nemotronh import nemotronh_56b_finetune_config

    overrides = _safe_overrides_for("nemotronh_56b_finetune_config")
    overrides["peft"] = "lora"

    cfg = nemotronh_56b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For LoRA, NemotronH 56B should use TP=4, PP=1
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.sequence_parallel is False

    # Check PEFT config
    assert cfg.peft is not None
    assert cfg.peft.dim == 32
    assert cfg.peft.alpha == 32
    assert cfg.peft.target_modules == ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj"]


def test_nemotronh_56b_full_sft_defaults():
    """Test that NemotronH 56B full SFT has correct default parallelism."""
    from megatron.bridge.recipes.nemotronh import nemotronh_56b_finetune_config

    overrides = _safe_overrides_for("nemotronh_56b_finetune_config")
    overrides["peft"] = "none"

    cfg = nemotronh_56b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For full SFT, NemotronH 56B should use TP=8, PP=1
    assert cfg.model.tensor_model_parallel_size == 8
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.sequence_parallel is True
    assert cfg.peft is None
