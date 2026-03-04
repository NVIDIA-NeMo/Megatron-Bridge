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
# - Parametrize over all exported Qwen3.5-VL recipe functions.
# - Monkeypatch AutoBridge and the provider to avoid I/O and heavy model init.
# - Build a config with small, safe overrides and assert it forms a valid ConfigContainer.
# - Verify dataset provider selection, parallelism fields, freeze options, and PEFT defaults.
#

import importlib
from typing import Callable

import pytest
import torch


_qwen35_vl_module = importlib.import_module("megatron.bridge.recipes.qwen_vl.qwen35_vl")
_QWEN35_VL_RECIPE_FUNCS = [
    _qwen35_vl_module.qwen35_vl_800m_finetune_config,
    _qwen35_vl_module.qwen35_vl_2b_finetune_config,
    _qwen35_vl_module.qwen35_vl_4b_finetune_config,
    _qwen35_vl_module.qwen35_vl_9b_finetune_config,
    _qwen35_vl_module.qwen35_vl_27b_finetune_config,
    _qwen35_vl_module.qwen35_vl_35b_a3b_finetune_config,
    _qwen35_vl_module.qwen35_vl_122b_a10b_finetune_config,
    _qwen35_vl_module.qwen35_vl_397b_a17b_finetune_config,
]


def _safe_overrides_for(name: str) -> dict:
    """Create safe test overrides for a given recipe function name."""
    overrides = {
        "name": f"unit_{name}",
        "dir": ".",
        "dataset_type": "mock",
        "train_iters": 10,
        "global_batch_size": 2,
        "micro_batch_size": 1,
        "seq_length": 64,
        "lr": 1e-4,
        "min_lr": 1e-5,
        "lr_warmup_iters": 2,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "use_null_tokenizer": True,
    }
    return overrides


class _FakeModelCfg:
    """Fake model configuration for testing."""

    def __init__(self):
        self.tensor_model_parallel_size = 1
        self.pipeline_model_parallel_size = 1
        self.pipeline_dtype = None
        self.virtual_pipeline_model_parallel_size = None
        self.context_parallel_size = 1
        self.expert_model_parallel_size = 1
        self.expert_tensor_parallel_size = 1
        self.sequence_parallel = False
        self.seq_length = 64
        self.freeze_language_model = False
        self.freeze_vision_model = False
        self.freeze_vision_projection = False
        self.account_for_embedding_in_pipeline_split = False
        self.account_for_loss_in_pipeline_split = False
        self.recompute_granularity = None
        self.recompute_method = None
        self.recompute_num_layers = None

    def validate_parallelism(self):
        return None

    def finalize(self):
        return None


class _FakeAutoBridge:
    """Fake AutoBridge for testing."""

    @staticmethod
    def from_hf_pretrained(hf_path: str, **kwargs):
        return _FakeAutoBridge()

    def to_megatron_provider(self, load_weights: bool = False):
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

    if hasattr(cfg.dataset, "seq_length"):
        assert cfg.dataset.seq_length >= 1


# ---------------------------------------------------------------------------
# Basic recipe building tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("recipe_func", _QWEN35_VL_RECIPE_FUNCS)
def test_each_qwen35_vl_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Each Qwen3.5-VL recipe function should build a valid ConfigContainer."""
    monkeypatch.setattr(_qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for(recipe_func.__name__)
    cfg = recipe_func(**overrides)

    _assert_basic_config(cfg)

    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1
    assert hasattr(cfg.model, "freeze_language_model")
    assert hasattr(cfg.model, "freeze_vision_model")
    assert hasattr(cfg.model, "freeze_vision_projection")


# ---------------------------------------------------------------------------
# Dataset type selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dataset_type", ["mock", "hf", "preloaded"])
def test_qwen35_vl_dataset_type_selection(dataset_type: str, monkeypatch: pytest.MonkeyPatch):
    """Different dataset_type values should produce the correct dataset provider."""
    monkeypatch.setattr(_qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("qwen35_vl_27b_finetune_config")
    overrides["dataset_type"] = dataset_type

    if dataset_type == "preloaded":
        overrides["train_data_path"] = ["/fake/train.json"]
        overrides["valid_data_path"] = ["/fake/valid.json"]
        overrides["test_data_path"] = ["/fake/test.json"]
        overrides["image_folder"] = "/fake/images"

    cfg = _qwen35_vl_module.qwen35_vl_27b_finetune_config(**overrides)

    from megatron.bridge.data.vlm_datasets import (
        HFDatasetConversationProvider,
        MockVLMConversationProvider,
        PreloadedVLMConversationProvider,
    )

    if dataset_type == "mock":
        assert isinstance(cfg.dataset, MockVLMConversationProvider)
    elif dataset_type == "hf":
        assert isinstance(cfg.dataset, HFDatasetConversationProvider)
    elif dataset_type == "preloaded":
        assert isinstance(cfg.dataset, PreloadedVLMConversationProvider)


# ---------------------------------------------------------------------------
# Training scenarios: SFT freeze combinations
# ---------------------------------------------------------------------------


def test_sft_nothing_frozen(monkeypatch: pytest.MonkeyPatch):
    """Scenario 1: Full SFT with nothing frozen — all modules trainable."""
    monkeypatch.setattr(_qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("qwen35_vl_27b_finetune_config")
    overrides["peft"] = None
    overrides["freeze_language_model"] = False
    overrides["freeze_vision_model"] = False
    overrides["freeze_vision_projection"] = False

    cfg = _qwen35_vl_module.qwen35_vl_27b_finetune_config(**overrides)

    _assert_basic_config(cfg)
    assert cfg.peft is None
    assert cfg.model.freeze_language_model is False
    assert cfg.model.freeze_vision_model is False
    assert cfg.model.freeze_vision_projection is False


def test_sft_language_model_frozen(monkeypatch: pytest.MonkeyPatch):
    """Scenario 2: SFT with language model frozen — train vision + projection."""
    monkeypatch.setattr(_qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("qwen35_vl_27b_finetune_config")
    overrides["peft"] = None
    overrides["freeze_language_model"] = True
    overrides["freeze_vision_model"] = False
    overrides["freeze_vision_projection"] = False

    cfg = _qwen35_vl_module.qwen35_vl_27b_finetune_config(**overrides)

    _assert_basic_config(cfg)
    assert cfg.peft is None
    assert cfg.model.freeze_language_model is True
    assert cfg.model.freeze_vision_model is False
    assert cfg.model.freeze_vision_projection is False


def test_sft_vision_and_language_frozen(monkeypatch: pytest.MonkeyPatch):
    """Scenario 3: SFT with vision + language frozen — train projection only."""
    monkeypatch.setattr(_qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("qwen35_vl_27b_finetune_config")
    overrides["peft"] = None
    overrides["freeze_language_model"] = True
    overrides["freeze_vision_model"] = True
    overrides["freeze_vision_projection"] = False

    cfg = _qwen35_vl_module.qwen35_vl_27b_finetune_config(**overrides)

    _assert_basic_config(cfg)
    assert cfg.peft is None
    assert cfg.model.freeze_language_model is True
    assert cfg.model.freeze_vision_model is True
    assert cfg.model.freeze_vision_projection is False


# ---------------------------------------------------------------------------
# Training scenarios: PEFT + freeze combinations
# ---------------------------------------------------------------------------


def test_peft_lora_language_only(monkeypatch: pytest.MonkeyPatch):
    """Scenario 4: LoRA adapters on all modules, vision base weights frozen.

    Default LoRA targets linear_qkv/proj/fc1/fc2 in both vision and language.
    Freezing vision base weights means only LoRA adapter deltas are trainable
    on the vision side, while the language model base weights remain trainable
    as well (unless also frozen via freeze_language_model).  The typical
    "language-only PEFT" pattern freezes vision + projection and adds LoRA
    adapters; the language base weights are also frozen by the recipe default,
    but LoRA adapter weights on the language side are always trainable.
    """
    monkeypatch.setattr(_qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("qwen35_vl_27b_finetune_config")
    overrides["peft"] = "lora"
    overrides["freeze_language_model"] = True
    overrides["freeze_vision_model"] = True
    overrides["freeze_vision_projection"] = True

    cfg = _qwen35_vl_module.qwen35_vl_27b_finetune_config(**overrides)

    _assert_basic_config(cfg)
    assert cfg.peft is not None
    assert cfg.peft.dim == 32
    assert cfg.model.freeze_language_model is True
    assert cfg.model.freeze_vision_model is True
    assert cfg.model.freeze_vision_projection is True


def test_peft_lora_vision_and_language(monkeypatch: pytest.MonkeyPatch):
    """Scenario 5: LoRA adapters with nothing frozen — adapters on all modules.

    LoRA targets linear_qkv/proj/fc1/fc2 in both vision and language.
    With nothing frozen, all base weights and all adapter weights are trainable.
    """
    monkeypatch.setattr(_qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("qwen35_vl_27b_finetune_config")
    overrides["peft"] = "lora"
    overrides["freeze_language_model"] = False
    overrides["freeze_vision_model"] = False
    overrides["freeze_vision_projection"] = False

    cfg = _qwen35_vl_module.qwen35_vl_27b_finetune_config(**overrides)

    _assert_basic_config(cfg)
    assert cfg.peft is not None
    assert cfg.peft.dim == 32
    assert cfg.model.freeze_language_model is False
    assert cfg.model.freeze_vision_model is False
    assert cfg.model.freeze_vision_projection is False


# ---------------------------------------------------------------------------
# PEFT vs full SFT (parametrized across all recipes)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("recipe_func", _QWEN35_VL_RECIPE_FUNCS)
@pytest.mark.parametrize("peft", ["lora", "dora", None])
def test_qwen35_vl_finetune_peft_vs_full_sft(recipe_func, peft, monkeypatch: pytest.MonkeyPatch):
    """PEFT and full SFT configurations should be correctly applied."""
    monkeypatch.setattr(_qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for(recipe_func.__name__)
    overrides["peft"] = peft

    cfg = recipe_func(**overrides)

    _assert_basic_config(cfg)

    if peft in ["lora", "dora"]:
        assert cfg.peft is not None
        assert hasattr(cfg.peft, "dim")
        assert hasattr(cfg.peft, "alpha")
    elif peft is None:
        assert cfg.peft is None


# ---------------------------------------------------------------------------
# 800M dense defaults
# ---------------------------------------------------------------------------


def test_qwen35_vl_800m_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """800M LoRA should have correct default parallelism and learning rate."""
    monkeypatch.setattr(_qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("qwen35_vl_800m_finetune_config")
    overrides["peft"] = "lora"
    overrides.pop("tensor_model_parallel_size", None)
    overrides.pop("pipeline_model_parallel_size", None)

    cfg = _qwen35_vl_module.qwen35_vl_800m_finetune_config(**overrides)

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.peft is not None
    assert cfg.optimizer.lr == 1e-4


def test_qwen35_vl_800m_full_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """800M full SFT should have correct default parallelism and learning rate."""
    monkeypatch.setattr(_qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("qwen35_vl_800m_finetune_config")
    overrides["peft"] = None
    overrides.pop("tensor_model_parallel_size", None)
    overrides.pop("pipeline_model_parallel_size", None)

    cfg = _qwen35_vl_module.qwen35_vl_800m_finetune_config(**overrides)

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.peft is None
    assert cfg.optimizer.lr == 5e-6


# ---------------------------------------------------------------------------
# 2B dense defaults
# ---------------------------------------------------------------------------


def test_qwen35_vl_2b_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """2B LoRA should have correct default parallelism and learning rate."""
    monkeypatch.setattr(_qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("qwen35_vl_2b_finetune_config")
    overrides["peft"] = "lora"
    overrides.pop("tensor_model_parallel_size", None)
    overrides.pop("pipeline_model_parallel_size", None)

    cfg = _qwen35_vl_module.qwen35_vl_2b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.peft is not None
    assert cfg.optimizer.lr == 1e-4


def test_qwen35_vl_2b_full_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """2B full SFT should have correct default parallelism and learning rate."""
    monkeypatch.setattr(_qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("qwen35_vl_2b_finetune_config")
    overrides["peft"] = None
    overrides.pop("tensor_model_parallel_size", None)
    overrides.pop("pipeline_model_parallel_size", None)

    cfg = _qwen35_vl_module.qwen35_vl_2b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.peft is None
    assert cfg.optimizer.lr == 5e-6


# ---------------------------------------------------------------------------
# 4B dense defaults
# ---------------------------------------------------------------------------


def test_qwen35_vl_4b_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """4B LoRA should have correct default parallelism and learning rate."""
    monkeypatch.setattr(_qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("qwen35_vl_4b_finetune_config")
    overrides["peft"] = "lora"
    overrides.pop("tensor_model_parallel_size", None)
    overrides.pop("pipeline_model_parallel_size", None)

    cfg = _qwen35_vl_module.qwen35_vl_4b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.peft is not None
    assert cfg.optimizer.lr == 1e-4


def test_qwen35_vl_4b_full_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """4B full SFT should have correct default parallelism and learning rate."""
    monkeypatch.setattr(_qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("qwen35_vl_4b_finetune_config")
    overrides["peft"] = None
    overrides.pop("tensor_model_parallel_size", None)
    overrides.pop("pipeline_model_parallel_size", None)

    cfg = _qwen35_vl_module.qwen35_vl_4b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.peft is None
    assert cfg.optimizer.lr == 5e-6


# ---------------------------------------------------------------------------
# 9B dense defaults
# ---------------------------------------------------------------------------


def test_qwen35_vl_9b_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """9B LoRA should have correct default parallelism and learning rate."""
    monkeypatch.setattr(_qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("qwen35_vl_9b_finetune_config")
    overrides["peft"] = "lora"
    overrides.pop("tensor_model_parallel_size", None)
    overrides.pop("pipeline_model_parallel_size", None)

    cfg = _qwen35_vl_module.qwen35_vl_9b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.peft is not None
    assert cfg.optimizer.lr == 1e-4


def test_qwen35_vl_9b_full_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """9B full SFT should have correct default parallelism and learning rate."""
    monkeypatch.setattr(_qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("qwen35_vl_9b_finetune_config")
    overrides["peft"] = None
    overrides.pop("tensor_model_parallel_size", None)
    overrides.pop("pipeline_model_parallel_size", None)

    cfg = _qwen35_vl_module.qwen35_vl_9b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.peft is None
    assert cfg.optimizer.lr == 5e-6


# ---------------------------------------------------------------------------
# 27B dense defaults
# ---------------------------------------------------------------------------


def test_qwen35_vl_27b_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """27B LoRA should have correct default parallelism and learning rate."""
    monkeypatch.setattr(_qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("qwen35_vl_27b_finetune_config")
    overrides["peft"] = "lora"
    overrides.pop("tensor_model_parallel_size", None)
    overrides.pop("pipeline_model_parallel_size", None)

    cfg = _qwen35_vl_module.qwen35_vl_27b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.peft is not None
    assert cfg.peft.dim == 32
    assert cfg.peft.alpha == 32
    assert cfg.optimizer.lr == 1e-4
    assert cfg.model.pipeline_dtype is None


def test_qwen35_vl_27b_dora_defaults(monkeypatch: pytest.MonkeyPatch):
    """27B DoRA should have correct default parallelism and learning rate."""
    monkeypatch.setattr(_qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("qwen35_vl_27b_finetune_config")
    overrides["peft"] = "dora"
    overrides.pop("tensor_model_parallel_size", None)
    overrides.pop("pipeline_model_parallel_size", None)

    cfg = _qwen35_vl_module.qwen35_vl_27b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.peft is not None
    assert cfg.peft.dim == 32
    assert cfg.peft.alpha == 64


def test_qwen35_vl_27b_full_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """27B full SFT should have correct default parallelism and learning rate."""
    monkeypatch.setattr(_qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("qwen35_vl_27b_finetune_config")
    overrides["peft"] = None
    overrides.pop("tensor_model_parallel_size", None)
    overrides.pop("pipeline_model_parallel_size", None)

    cfg = _qwen35_vl_module.qwen35_vl_27b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.peft is None
    assert cfg.optimizer.lr == 5e-6
    assert cfg.model.pipeline_dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# 35B-A3B MoE defaults
# ---------------------------------------------------------------------------


def test_qwen35_vl_35b_a3b_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """35B-A3B LoRA should have correct default parallelism and learning rate."""
    monkeypatch.setattr(_qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("qwen35_vl_35b_a3b_finetune_config")
    overrides["peft"] = "lora"
    overrides.pop("tensor_model_parallel_size", None)
    overrides.pop("pipeline_model_parallel_size", None)
    overrides.pop("expert_model_parallel_size", None)

    cfg = _qwen35_vl_module.qwen35_vl_35b_a3b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 4
    assert cfg.peft is not None
    assert cfg.optimizer.lr == 2e-4


def test_qwen35_vl_35b_a3b_full_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """35B-A3B full SFT should have correct default parallelism and learning rate."""
    monkeypatch.setattr(_qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("qwen35_vl_35b_a3b_finetune_config")
    overrides["peft"] = None
    overrides.pop("tensor_model_parallel_size", None)
    overrides.pop("pipeline_model_parallel_size", None)
    overrides.pop("expert_model_parallel_size", None)

    cfg = _qwen35_vl_module.qwen35_vl_35b_a3b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 16
    assert cfg.peft is None
    assert cfg.optimizer.lr == 2e-5
    assert cfg.model.pipeline_dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# 122B-A10B MoE defaults
# ---------------------------------------------------------------------------


def test_qwen35_vl_122b_a10b_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """122B-A10B LoRA should have correct default parallelism and learning rate."""
    monkeypatch.setattr(_qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("qwen35_vl_122b_a10b_finetune_config")
    overrides["peft"] = "lora"
    overrides.pop("tensor_model_parallel_size", None)
    overrides.pop("pipeline_model_parallel_size", None)
    overrides.pop("expert_model_parallel_size", None)

    cfg = _qwen35_vl_module.qwen35_vl_122b_a10b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.peft is not None
    assert cfg.optimizer.lr == 2e-4
    assert cfg.model.pipeline_dtype == torch.bfloat16


def test_qwen35_vl_122b_a10b_full_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """122B-A10B full SFT should have correct default parallelism and learning rate."""
    monkeypatch.setattr(_qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("qwen35_vl_122b_a10b_finetune_config")
    overrides["peft"] = None
    overrides.pop("tensor_model_parallel_size", None)
    overrides.pop("pipeline_model_parallel_size", None)
    overrides.pop("expert_model_parallel_size", None)

    cfg = _qwen35_vl_module.qwen35_vl_122b_a10b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 6
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.peft is None
    assert cfg.optimizer.lr == 2e-5
    assert cfg.model.pipeline_dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# 397B MoE defaults
# ---------------------------------------------------------------------------


def test_qwen35_vl_397b_a17b_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """397B-A17B LoRA should have correct default parallelism and learning rate."""
    monkeypatch.setattr(_qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("qwen35_vl_397b_a17b_finetune_config")
    overrides["peft"] = "lora"
    overrides.pop("tensor_model_parallel_size", None)
    overrides.pop("pipeline_model_parallel_size", None)
    overrides.pop("expert_model_parallel_size", None)

    cfg = _qwen35_vl_module.qwen35_vl_397b_a17b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 32
    assert cfg.peft is not None
    assert cfg.optimizer.lr == 2e-4


def test_qwen35_vl_397b_a17b_full_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """397B-A17B full SFT should have correct default parallelism and learning rate."""
    monkeypatch.setattr(_qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("qwen35_vl_397b_a17b_finetune_config")
    overrides["peft"] = None
    overrides.pop("tensor_model_parallel_size", None)
    overrides.pop("pipeline_model_parallel_size", None)
    overrides.pop("expert_model_parallel_size", None)

    cfg = _qwen35_vl_module.qwen35_vl_397b_a17b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.expert_model_parallel_size == 32
    assert cfg.peft is None
    assert cfg.optimizer.lr == 2e-5
    assert cfg.model.pipeline_dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Custom overrides
# ---------------------------------------------------------------------------


def test_qwen35_vl_custom_finetune_lr(monkeypatch: pytest.MonkeyPatch):
    """Custom finetune_lr should override default learning rate."""
    monkeypatch.setattr(_qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("qwen35_vl_27b_finetune_config")
    overrides["peft"] = "lora"
    overrides["finetune_lr"] = 2e-4

    cfg = _qwen35_vl_module.qwen35_vl_27b_finetune_config(**overrides)

    _assert_basic_config(cfg)
    assert cfg.optimizer.lr == 2e-4


def test_qwen35_vl_recompute_option(monkeypatch: pytest.MonkeyPatch):
    """enable_recompute should set recompute fields on the model config."""
    monkeypatch.setattr(_qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("qwen35_vl_27b_finetune_config")
    overrides["enable_recompute"] = True

    cfg = _qwen35_vl_module.qwen35_vl_27b_finetune_config(**overrides)

    _assert_basic_config(cfg)
    assert cfg.model.recompute_granularity == "full"
    assert cfg.model.recompute_method == "uniform"
    assert cfg.model.recompute_num_layers == 1


def test_qwen35_vl_invalid_dataset_type(monkeypatch: pytest.MonkeyPatch):
    """An unsupported dataset_type should raise ValueError."""
    monkeypatch.setattr(_qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("qwen35_vl_27b_finetune_config")
    overrides["dataset_type"] = "unsupported"

    with pytest.raises(ValueError, match="Unsupported dataset_type"):
        _qwen35_vl_module.qwen35_vl_27b_finetune_config(**overrides)
