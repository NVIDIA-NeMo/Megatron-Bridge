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
from collections.abc import Callable

import pytest


_qwen_module = importlib.import_module("megatron.bridge.recipes.qwen")
_QWEN2_RECIPE_FUNCS = [
    getattr(_qwen_module, name)
    for name in getattr(_qwen_module, "__all__", [])
    if callable(getattr(_qwen_module, name, None)) and "qwen2" in name.lower()
]


# Qwen2/2.5 finetune-specific tests
_QWEN2_FINETUNE_FUNCS = [
    getattr(_qwen_module, name)
    for name in [
        "qwen2_500m_finetune_config",
        "qwen2_1p5b_finetune_config",
        "qwen2_7b_finetune_config",
        "qwen2_72b_finetune_config",
        "qwen25_500m_finetune_config",
        "qwen25_1p5b_finetune_config",
        "qwen25_7b_finetune_config",
        "qwen25_14b_finetune_config",
        "qwen25_32b_finetune_config",
        "qwen25_72b_finetune_config",
    ]
    if callable(getattr(_qwen_module, name, None))
]


def _safe_overrides_for(name: str) -> dict:
    """Return overrides for recipe functions.

    Pretrain configs use the new parameterless API (return empty dict).
    Finetune configs still accept parameters.
    """
    is_finetune = "finetune" in name.lower()
    lname = name.lower()

    if is_finetune:
        # Finetuning-specific overrides - finetune configs still accept parameters
        overrides = {
            "name": f"unit_{name}",
            "dir": ".",
            "train_iters": 10,
            "micro_batch_size": 1,
            "seq_length": 64,
            "min_lr": 1e-5,
            "lr_warmup_iters": 2,
            "finetune_lr": 1e-4,
        }
        # 72B has special global_batch_size defaults
        if "72b" not in lname:
            overrides["global_batch_size"] = 2
    else:
        # Pretrain configs use the new parameterless API
        overrides = {}

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
    def from_hf_pretrained(hf_path: str, **kwargs):
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


@pytest.mark.parametrize("recipe_func", _QWEN2_RECIPE_FUNCS)
def test_each_qwen2_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
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
        # Pretrain recipes use either NullTokenizer or HuggingFaceTokenizer
        if cfg.tokenizer.tokenizer_type == "NullTokenizer":
            assert cfg.tokenizer.vocab_size is not None
        else:
            assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
            assert cfg.tokenizer.tokenizer_model is not None

    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1


@pytest.mark.parametrize("recipe_func", _QWEN2_FINETUNE_FUNCS)
def test_qwen2_finetune_config_builds(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that each Qwen2/2.5 finetune recipe builds a valid config."""
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


@pytest.mark.parametrize("recipe_func", _QWEN2_FINETUNE_FUNCS)
@pytest.mark.parametrize("peft", ["lora", "dora", "none"])
def test_qwen2_finetune_peft_vs_full_sft(recipe_func: Callable, peft: str, monkeypatch: pytest.MonkeyPatch):
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
def test_qwen2_7b_finetune_packed_sequence(packed: bool, monkeypatch: pytest.MonkeyPatch):
    """Test that packed sequence configuration works correctly."""
    from megatron.bridge.recipes.qwen import qwen2_7b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.qwen.qwen2")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for("qwen2_7b_finetune_config")
    overrides["packed_sequence"] = packed

    cfg = qwen2_7b_finetune_config(**overrides)

    _assert_basic_config(cfg)


def test_qwen2_7b_full_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 7B full SFT has correct default parallelism."""
    from megatron.bridge.recipes.qwen import qwen2_7b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.qwen.qwen2")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for("qwen2_7b_finetune_config")
    overrides["peft"] = "none"

    cfg = qwen2_7b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For full SFT, 7B should use TP=2
    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1


def test_qwen2_7b_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 7B LoRA has correct default parallelism."""
    from megatron.bridge.recipes.qwen import qwen2_7b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.qwen.qwen2")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for("qwen2_7b_finetune_config")
    overrides["peft"] = "lora"

    cfg = qwen2_7b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For LoRA, 7B should use TP=1
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1


def test_qwen2_72b_full_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 72B full SFT has correct default parallelism."""
    from megatron.bridge.recipes.qwen import qwen2_72b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.qwen.qwen2")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for("qwen2_72b_finetune_config")
    overrides["peft"] = "none"

    cfg = qwen2_72b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For full SFT, 72B should use TP=8, PP=4
    assert cfg.model.tensor_model_parallel_size == 8
    assert cfg.model.pipeline_model_parallel_size == 4


def test_qwen2_72b_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 72B LoRA has correct default parallelism."""
    from megatron.bridge.recipes.qwen import qwen2_72b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.qwen.qwen2")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for("qwen2_72b_finetune_config")
    overrides["peft"] = "lora"

    cfg = qwen2_72b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For LoRA, 72B should use TP=8, PP=1
    assert cfg.model.tensor_model_parallel_size == 8
    assert cfg.model.pipeline_model_parallel_size == 1


def test_qwen25_7b_full_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that Qwen2.5 7B full SFT has correct default parallelism."""
    from megatron.bridge.recipes.qwen import qwen25_7b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.qwen.qwen2")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for("qwen25_7b_finetune_config")
    overrides["peft"] = "none"

    cfg = qwen25_7b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For full SFT, 7B should use TP=2
    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1


def test_qwen25_7b_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that Qwen2.5 7B LoRA has correct default parallelism."""
    from megatron.bridge.recipes.qwen import qwen25_7b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.qwen.qwen2")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for("qwen25_7b_finetune_config")
    overrides["peft"] = "lora"

    cfg = qwen25_7b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For LoRA, 7B should use TP=1
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1


def test_qwen25_14b_full_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that Qwen2.5 14B full SFT has correct default parallelism."""
    from megatron.bridge.recipes.qwen import qwen25_14b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.qwen.qwen2")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for("qwen25_14b_finetune_config")
    overrides["peft"] = "none"

    cfg = qwen25_14b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For full SFT, 14B should use TP=4
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 1


def test_qwen25_14b_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that Qwen2.5 14B LoRA has correct default parallelism."""
    from megatron.bridge.recipes.qwen import qwen25_14b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.qwen.qwen2")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for("qwen25_14b_finetune_config")
    overrides["peft"] = "lora"

    cfg = qwen25_14b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For LoRA, 14B should use TP=1
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1


def test_qwen25_32b_full_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that Qwen2.5 32B full SFT has correct default parallelism."""
    from megatron.bridge.recipes.qwen import qwen25_32b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.qwen.qwen2")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for("qwen25_32b_finetune_config")
    overrides["peft"] = "none"

    cfg = qwen25_32b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For full SFT, 32B should use TP=8, PP=2
    assert cfg.model.tensor_model_parallel_size == 8
    assert cfg.model.pipeline_model_parallel_size == 2


def test_qwen25_32b_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that Qwen2.5 32B LoRA has correct default parallelism."""
    from megatron.bridge.recipes.qwen import qwen25_32b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.qwen.qwen2")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for("qwen25_32b_finetune_config")
    overrides["peft"] = "lora"

    cfg = qwen25_32b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For LoRA, 32B should use TP=8, PP=1
    assert cfg.model.tensor_model_parallel_size == 8
    assert cfg.model.pipeline_model_parallel_size == 1


def test_qwen25_72b_full_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that Qwen2.5 72B full SFT has correct default parallelism."""
    from megatron.bridge.recipes.qwen import qwen25_72b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.qwen.qwen2")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for("qwen25_72b_finetune_config")
    overrides["peft"] = "none"

    cfg = qwen25_72b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For full SFT, 72B should use TP=8, PP=4
    assert cfg.model.tensor_model_parallel_size == 8
    assert cfg.model.pipeline_model_parallel_size == 4


def test_qwen25_72b_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that Qwen2.5 72B LoRA has correct default parallelism."""
    from megatron.bridge.recipes.qwen import qwen25_72b_finetune_config

    mod = importlib.import_module("megatron.bridge.recipes.qwen.qwen2")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for("qwen25_72b_finetune_config")
    overrides["peft"] = "lora"

    cfg = qwen25_72b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For LoRA, 72B should use TP=8, PP=1
    assert cfg.model.tensor_model_parallel_size == 8
    assert cfg.model.pipeline_model_parallel_size == 1
