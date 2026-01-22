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
# - Parametrize over all exported Gemma3 recipe functions in `megatron.bridge.recipes.gemma`.
# - For each recipe, monkeypatch the provider class with a lightweight fake to avoid I/O.
# - Build a config with small, safe overrides and assert it forms a valid `ConfigContainer`.
# - Verify tokenizer selection honors `use_null_tokenizer`, and sanity-check parallelism fields.
#

import importlib
from collections.abc import Callable

import pytest


_gemma_module = importlib.import_module("megatron.bridge.recipes.gemma")
_GEMMA3_RECIPE_FUNCS = [
    getattr(_gemma_module, name)
    for name in getattr(_gemma_module, "__all__", [])
    if callable(getattr(_gemma_module, name, None)) and "gemma3" in name.lower()
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
            "min_lr": 1e-5,
            "lr_warmup_iters": 2,
            "finetune_lr": 1e-4,
            "pretrained_checkpoint": "/fake/checkpoint/path",
        }
    else:
        # Pretrain configs use the new parameterless API
        overrides = {}

    return overrides


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
        self.account_for_embedding_in_pipeline_split = False
        self.account_for_loss_in_pipeline_split = False
        # Finetuning-specific attributes
        self.cross_entropy_loss_fusion = True
        self.vocab_size = 256000  # Gemma3 vocab size

    def finalize(self):
        return None


class _FakeBridge:
    """Fake AutoBridge for testing finetune configs."""

    def __init__(self):
        pass

    def to_megatron_provider(self, load_weights: bool = False):
        return _FakeModelCfg()

    @staticmethod
    def from_hf_pretrained(hf_path: str, **kwargs):
        return _FakeBridge()


class _FakeTokenizer:
    """Fake HuggingFace tokenizer for testing."""

    def __len__(self):
        return 256000  # Gemma3 tokenizer vocab size


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


@pytest.mark.parametrize("recipe_func", _GEMMA3_RECIPE_FUNCS)
def test_each_gemma3_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that each Gemma3 recipe function builds a valid configuration."""
    # Monkeypatch the provider classes to return fake model configs
    from megatron.bridge.models.gemma import gemma3_provider

    # Create a fake provider class that returns a fake model config
    class FakeProvider(_FakeModelCfg):
        def __init__(self, *args, **kwargs):
            super().__init__()

    # Monkeypatch all provider classes
    monkeypatch.setattr(gemma3_provider, "Gemma3ModelProvider1B", FakeProvider)
    monkeypatch.setattr(gemma3_provider, "Gemma3ModelProvider4B", FakeProvider)
    monkeypatch.setattr(gemma3_provider, "Gemma3ModelProvider12B", FakeProvider)
    monkeypatch.setattr(gemma3_provider, "Gemma3ModelProvider27B", FakeProvider)

    # For finetune recipes, also monkeypatch AutoBridge and AutoTokenizer
    is_finetune = "finetune" in recipe_func.__name__.lower()
    if is_finetune:
        module_name = recipe_func.__module__
        mod = importlib.import_module(module_name)
        monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

        # Mock AutoTokenizer to avoid HF I/O
        import transformers

        monkeypatch.setattr(
            transformers,
            "AutoTokenizer",
            type("FakeAutoTokenizer", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: _FakeTokenizer())}),
        )

    overrides = _safe_overrides_for(recipe_func.__name__)

    cfg = recipe_func(**overrides)

    _assert_basic_config(cfg)

    # Ensure tokenizer is properly configured
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

    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1


# Gemma3 finetune-specific tests
_GEMMA3_FINETUNE_FUNCS = [
    getattr(_gemma_module, name)
    for name in [
        "gemma3_1b_finetune_config",
    ]
    if callable(getattr(_gemma_module, name, None))
]


@pytest.mark.parametrize("recipe_func", _GEMMA3_FINETUNE_FUNCS)
@pytest.mark.parametrize("peft", ["lora", "dora", "none"])
def test_gemma3_finetune_peft_vs_full_sft(recipe_func: Callable, peft: str, monkeypatch: pytest.MonkeyPatch):
    """Test that PEFT and full SFT configurations are correctly applied for Gemma3 models."""
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    # Mock AutoTokenizer to avoid HF I/O
    import transformers

    monkeypatch.setattr(
        transformers,
        "AutoTokenizer",
        type("FakeAutoTokenizer", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: _FakeTokenizer())}),
    )

    overrides = _safe_overrides_for(recipe_func.__name__)
    overrides["peft"] = peft

    cfg = recipe_func(**overrides)

    _assert_basic_config(cfg)

    # Check PEFT config presence
    if peft in ["lora", "dora"]:
        assert cfg.peft is not None
    elif peft == "none":
        assert cfg.peft is None


def test_gemma3_1b_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 1B LoRA has correct default parallelism and performance optimizations."""
    from megatron.bridge.recipes.gemma import gemma3_1b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.gemma.gemma3")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    # Mock AutoTokenizer to avoid HF I/O
    import transformers

    monkeypatch.setattr(
        transformers,
        "AutoTokenizer",
        type("FakeAutoTokenizer", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: _FakeTokenizer())}),
    )

    overrides = _safe_overrides_for("gemma3_1b_finetune_config")
    overrides["peft"] = "lora"

    cfg = gemma3_1b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For LoRA, 1B should use TP=1, PP=1
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1

    # Check PEFT config
    assert cfg.peft is not None
    assert cfg.peft.dim == 8
    assert cfg.peft.alpha == 16

    # Check PEFT-specific performance settings
    assert cfg.model.cross_entropy_loss_fusion is False
    assert cfg.optimizer.use_distributed_optimizer is False


def test_gemma3_1b_dora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 1B DoRA has correct default parallelism and performance optimizations."""
    from megatron.bridge.recipes.gemma import gemma3_1b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.gemma.gemma3")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    # Mock AutoTokenizer to avoid HF I/O
    import transformers

    monkeypatch.setattr(
        transformers,
        "AutoTokenizer",
        type("FakeAutoTokenizer", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: _FakeTokenizer())}),
    )

    overrides = _safe_overrides_for("gemma3_1b_finetune_config")
    overrides["peft"] = "dora"

    cfg = gemma3_1b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For DoRA, 1B should use same parallelism as LoRA
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1

    # Check PEFT config
    assert cfg.peft is not None
    assert cfg.peft.dim == 8
    assert cfg.peft.alpha == 16

    # Check PEFT-specific performance settings
    assert cfg.model.cross_entropy_loss_fusion is False
    assert cfg.optimizer.use_distributed_optimizer is False


def test_gemma3_1b_full_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 1B full SFT has correct default parallelism."""
    from megatron.bridge.recipes.gemma import gemma3_1b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.gemma.gemma3")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    # Mock AutoTokenizer to avoid HF I/O
    import transformers

    monkeypatch.setattr(
        transformers,
        "AutoTokenizer",
        type("FakeAutoTokenizer", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: _FakeTokenizer())}),
    )

    overrides = _safe_overrides_for("gemma3_1b_finetune_config")
    overrides["peft"] = "none"

    cfg = gemma3_1b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For full SFT, 1B should use TP=1, PP=1
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.peft is None


@pytest.mark.parametrize("packed", [True, False])
def test_gemma3_1b_finetune_packed_sequence(packed: bool, monkeypatch: pytest.MonkeyPatch):
    """Test that packed sequence configuration works correctly."""
    from megatron.bridge.recipes.gemma import gemma3_1b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.gemma.gemma3")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    # Mock AutoTokenizer to avoid HF I/O
    import transformers

    monkeypatch.setattr(
        transformers,
        "AutoTokenizer",
        type("FakeAutoTokenizer", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: _FakeTokenizer())}),
    )

    overrides = _safe_overrides_for("gemma3_1b_finetune_config")
    overrides["packed_sequence"] = packed

    cfg = gemma3_1b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # Packed sequence affects default seq_length (4096 vs 2048)
    # But we override seq_length in tests, so just verify config is valid
    assert cfg.dataset is not None
