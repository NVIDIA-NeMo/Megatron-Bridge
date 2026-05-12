# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
# - Parametrize over all exported Gemma2 recipe functions in
#   `megatron.bridge.recipes.gemma`.
# - Monkeypatch AutoBridge with a lightweight fake so we never reach
#   HuggingFace; mock AutoTokenizer so SFT/PEFT recipes don't try to
#   load a real tokenizer.
# - Build a config with the parameterless recipe API and assert it
#   forms a valid ConfigContainer with the expected tokenizer type
#   and parallelism shape.
# - Companion file to test_gemma3_recipes.py; together they cover the
#   full gemma module.
#

import importlib
from typing import Callable
from unittest.mock import MagicMock

import pytest


_gemma_module = importlib.import_module("megatron.bridge.recipes.gemma")
_GEMMA2_RECIPE_FUNCS = [
    getattr(_gemma_module, name)
    for name in getattr(_gemma_module, "__all__", [])
    if callable(getattr(_gemma_module, name, None)) and "gemma2" in name.lower()
]


class _FakeModelCfg:
    """Fake model config returned by the patched AutoBridge.

    Carries the attributes that the gemma2 recipes assign onto
    `cfg.model` (cross_entropy_fusion_impl, parallelism fields, etc.)
    so the recipes can mutate it without exploding.
    """

    def __init__(self):
        self.tensor_model_parallel_size = 1
        self.pipeline_model_parallel_size = 1
        self.pipeline_dtype = None
        self.virtual_pipeline_model_parallel_size = None
        self.context_parallel_size = 1
        self.sequence_parallel = False
        self.seq_length = 64
        self.account_for_embedding_in_pipeline_split = False
        self.account_for_loss_in_pipeline_split = False
        self.cross_entropy_loss_fusion = True
        self.cross_entropy_fusion_impl = "te"  # recipes flip this to "native"
        self.vocab_size = 256000  # Gemma2 vocab size

    def finalize(self):
        return None


class _FakeBridge:
    """Stand-in for AutoBridge that returns a _FakeModelCfg without I/O."""

    def __init__(self):
        pass

    def to_megatron_provider(self, load_weights: bool = False):
        return _FakeModelCfg()

    @staticmethod
    def from_hf_pretrained(hf_path: str, **kwargs):
        return _FakeBridge()


def _mock_auto_tokenizer(monkeypatch: pytest.MonkeyPatch, vocab_size: int = 256000) -> None:
    """Install a transformers.AutoTokenizer mock so SFT/PEFT recipes work offline."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.__len__ = MagicMock(return_value=vocab_size)

    mock_auto_tokenizer = MagicMock()
    mock_auto_tokenizer.from_pretrained = MagicMock(return_value=mock_tokenizer)

    monkeypatch.setattr("transformers.AutoTokenizer", mock_auto_tokenizer)


def _assert_basic_config(cfg) -> None:
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


@pytest.mark.parametrize("recipe_func", _GEMMA2_RECIPE_FUNCS)
def test_each_gemma2_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Every exported Gemma2 recipe should build a valid ConfigContainer."""
    # Patch AutoBridge inside the recipe module so the pretrain/sft/peft
    # paths all see the fake before they call from_hf_pretrained.
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)
    _mock_auto_tokenizer(monkeypatch)

    cfg = recipe_func()
    _assert_basic_config(cfg)

    is_sft_or_peft = "sft" in recipe_func.__name__.lower() or "peft" in recipe_func.__name__.lower()
    if is_sft_or_peft:
        # SFT and PEFT recipes always use the HF tokenizer
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


# Gemma2 SFT-specific tests
_GEMMA2_SFT_FUNCS = [
    getattr(_gemma_module, name)
    for name in [
        "gemma2_2b_sft_config",
        "gemma2_9b_sft_config",
        "gemma2_27b_sft_config",
    ]
    if callable(getattr(_gemma_module, name, None))
]


# Gemma2 PEFT-specific tests
_GEMMA2_PEFT_FUNCS = [
    getattr(_gemma_module, name)
    for name in [
        "gemma2_2b_peft_config",
        "gemma2_9b_peft_config",
        "gemma2_27b_peft_config",
    ]
    if callable(getattr(_gemma_module, name, None))
]


@pytest.mark.parametrize("recipe_func", _GEMMA2_SFT_FUNCS)
def test_gemma2_sft_config_builds(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Every Gemma2 SFT recipe builds a valid config and excludes PEFT."""
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)
    _mock_auto_tokenizer(monkeypatch)

    cfg = recipe_func()
    _assert_basic_config(cfg)

    assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
    assert cfg.tokenizer.tokenizer_model is not None
    # SFT must not carry a PEFT config
    assert cfg.peft is None


@pytest.mark.parametrize("recipe_func", _GEMMA2_PEFT_FUNCS)
def test_gemma2_peft_config_builds(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Every Gemma2 PEFT recipe builds a valid config and includes a PEFT block."""
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)
    _mock_auto_tokenizer(monkeypatch)

    cfg = recipe_func()
    _assert_basic_config(cfg)

    assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
    assert cfg.tokenizer.tokenizer_model is not None
    # PEFT recipes must produce a non-None PEFT block
    assert cfg.peft is not None


@pytest.mark.parametrize("recipe_func", _GEMMA2_PEFT_FUNCS)
@pytest.mark.parametrize("peft_scheme", ["lora", "dora"])
def test_gemma2_peft_schemes(recipe_func: Callable, peft_scheme: str, monkeypatch: pytest.MonkeyPatch):
    """LoRA and DoRA schemes both produce a populated PEFT block."""
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)
    _mock_auto_tokenizer(monkeypatch)

    cfg = recipe_func(peft_scheme=peft_scheme)
    _assert_basic_config(cfg)
    assert cfg.peft is not None


def test_gemma2_uses_native_cross_entropy_fusion(monkeypatch: pytest.MonkeyPatch):
    """All gemma2 pretrain recipes set cross_entropy_fusion_impl='native'.

    This matches the comment in gemma2.py noting 'Gemma2 uses native'. Locks
    in the pretrain-path override so a future refactor doesn't silently flip
    it back to 'te' on this model family.
    """
    from megatron.bridge.recipes.gemma import gemma2_2b_pretrain_config

    mod = importlib.import_module("megatron.bridge.recipes.gemma.gemma2")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)
    _mock_auto_tokenizer(monkeypatch)

    cfg = gemma2_2b_pretrain_config()
    _assert_basic_config(cfg)
    assert cfg.model.cross_entropy_fusion_impl == "native"
