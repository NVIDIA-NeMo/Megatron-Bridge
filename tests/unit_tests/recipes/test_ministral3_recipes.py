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
# - Parametrize over all exported Ministral3 recipe functions in `megatron.bridge.recipes.ministral3.ministral3`.
# - For each recipe, monkeypatch AutoBridge and the provider to avoid I/O.
# - Build a config with small, safe overrides and assert it forms a valid `ConfigContainer`.
# - Verify dataset provider selection and sanity-check parallelism fields.
#

import importlib
from collections.abc import Callable

import pytest


_ministral3_module = importlib.import_module("megatron.bridge.recipes.ministral3.ministral3")
_MINISTRAL3_RECIPE_FUNCS = [
    _ministral3_module.ministral3_3b_finetune_config,
    _ministral3_module.ministral3_8b_finetune_config,
    _ministral3_module.ministral3_14b_finetune_config,
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
        "finetune_lr": 1e-4,
        "min_lr": 1e-5,
        "lr_warmup_iters": 2,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
    }

    # Large models may set additional flags in recipes; keep harmless defaults
    lname = name.lower()
    if "8b" in lname or "14b" in lname:
        overrides.update(
            {
                "virtual_pipeline_model_parallel_size": None,
                "sequence_parallel": False,
            }
        )

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


@pytest.mark.parametrize("recipe_func", _MINISTRAL3_RECIPE_FUNCS)
def test_each_ministral3_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that each Ministral3 recipe function builds a valid configuration."""
    # Monkeypatch AutoBridge to return a fake model config
    monkeypatch.setattr(_ministral3_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for(recipe_func.__name__)

    cfg = recipe_func(**overrides)

    _assert_basic_config(cfg)

    # Check that NullTokenizer is used
    if hasattr(cfg, "tokenizer") and hasattr(cfg.tokenizer, "tokenizer_type"):
        assert cfg.tokenizer.tokenizer_type == "NullTokenizer"

    # Verify parallelism settings
    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1

    # Verify freeze settings are set
    assert hasattr(cfg.model, "freeze_language_model")
    assert hasattr(cfg.model, "freeze_vision_model")
    assert hasattr(cfg.model, "freeze_vision_projection")


@pytest.mark.parametrize("dataset_type", ["mock", "hf", "preloaded"])
def test_ministral3_dataset_type_selection(dataset_type: str, monkeypatch: pytest.MonkeyPatch):
    """Test that different dataset_type values produce correct dataset providers."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_ministral3_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("ministral3_3b_finetune_config")
    overrides["dataset_type"] = dataset_type

    # For preloaded, we need to provide data paths
    if dataset_type == "preloaded":
        overrides["train_data_path"] = ["/fake/train.json"]
        overrides["valid_data_path"] = ["/fake/valid.json"]
        overrides["test_data_path"] = ["/fake/test.json"]
        overrides["image_folder"] = "/fake/images"

    cfg = _ministral3_module.ministral3_3b_finetune_config(**overrides)

    # Check that appropriate dataset provider is used
    from megatron.bridge.data.vlm_datasets.hf_provider import HFDatasetConversationProvider
    from megatron.bridge.data.vlm_datasets.mock_provider import MockVLMConversationProvider
    from megatron.bridge.data.vlm_datasets.preloaded_provider import PreloadedVLMConversationProvider

    if dataset_type == "mock":
        assert isinstance(cfg.dataset, MockVLMConversationProvider)
    elif dataset_type == "hf":
        assert isinstance(cfg.dataset, HFDatasetConversationProvider)
    elif dataset_type == "preloaded":
        assert isinstance(cfg.dataset, PreloadedVLMConversationProvider)


def test_ministral3_freeze_options(monkeypatch: pytest.MonkeyPatch):
    """Test that freeze options are correctly passed to the model config."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_ministral3_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("ministral3_3b_finetune_config")
    overrides["freeze_language_model"] = True
    overrides["freeze_vision_model"] = True
    overrides["freeze_vision_projection"] = False

    cfg = _ministral3_module.ministral3_3b_finetune_config(**overrides)

    assert cfg.model.freeze_language_model is True
    assert cfg.model.freeze_vision_model is True
    assert cfg.model.freeze_vision_projection is False


# PEFT-specific tests
_MINISTRAL3_FINETUNE_FUNCS = [
    _ministral3_module.ministral3_3b_finetune_config,
    _ministral3_module.ministral3_8b_finetune_config,
    _ministral3_module.ministral3_14b_finetune_config,
]


@pytest.mark.parametrize("recipe_func", _MINISTRAL3_FINETUNE_FUNCS)
@pytest.mark.parametrize("peft", ["lora", "dora", None])
def test_ministral3_finetune_peft_vs_full_sft(recipe_func, peft, monkeypatch: pytest.MonkeyPatch):
    """Test that PEFT and full SFT configurations are correctly applied for Ministral3 models."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_ministral3_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for(recipe_func.__name__)
    overrides["peft"] = peft

    cfg = recipe_func(**overrides)

    _assert_basic_config(cfg)

    # Check PEFT config presence
    if peft in ["lora", "dora"]:
        assert cfg.peft is not None
        # Verify PEFT config has expected attributes
        assert hasattr(cfg.peft, "dim")
        assert hasattr(cfg.peft, "alpha")
    elif peft is None:
        assert cfg.peft is None


def test_ministral3_3b_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 3B LoRA has correct default parallelism and learning rate."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_ministral3_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("ministral3_3b_finetune_config")
    overrides["peft"] = "lora"
    # Remove TP/PP overrides to test recipe defaults
    overrides.pop("tensor_model_parallel_size", None)
    overrides.pop("pipeline_model_parallel_size", None)
    # Remove finetune_lr to test default
    overrides.pop("finetune_lr", None)

    cfg = _ministral3_module.ministral3_3b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For LoRA, 3B should use TP=1, PP=1
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1

    # Check PEFT config
    assert cfg.peft is not None
    assert cfg.peft.dim == 32
    assert cfg.peft.alpha == 32

    # Check that learning rate defaults to 1e-4 for LoRA
    assert cfg.optimizer.lr == 1e-4


def test_ministral3_3b_dora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 3B DoRA has correct default parallelism and learning rate."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_ministral3_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("ministral3_3b_finetune_config")
    overrides["peft"] = "dora"
    # Remove TP/PP overrides to test recipe defaults
    overrides.pop("tensor_model_parallel_size", None)
    overrides.pop("pipeline_model_parallel_size", None)

    cfg = _ministral3_module.ministral3_3b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For DoRA, 3B should use same parallelism as LoRA
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1

    # Check PEFT config (DoRA has alpha=64 by default, unlike LoRA's alpha=32)
    assert cfg.peft is not None
    assert cfg.peft.dim == 32
    assert cfg.peft.alpha == 64


def test_ministral3_3b_full_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 3B full SFT has correct default parallelism and learning rate."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_ministral3_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("ministral3_3b_finetune_config")
    overrides["peft"] = None
    # Remove TP/PP overrides to test recipe defaults
    overrides.pop("tensor_model_parallel_size", None)
    overrides.pop("pipeline_model_parallel_size", None)
    # Remove finetune_lr to test default
    overrides.pop("finetune_lr", None)

    cfg = _ministral3_module.ministral3_3b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For full SFT, 3B should use TP=1, PP=1
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.peft is None

    # Check that learning rate defaults to 5e-6 for full SFT
    assert cfg.optimizer.lr == 5e-6


def test_ministral3_8b_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 8B LoRA has correct default parallelism."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_ministral3_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("ministral3_8b_finetune_config")
    overrides["peft"] = "lora"
    # Remove TP/PP overrides to test recipe defaults
    overrides.pop("tensor_model_parallel_size", None)
    overrides.pop("pipeline_model_parallel_size", None)

    cfg = _ministral3_module.ministral3_8b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For LoRA, 8B should use TP=1, PP=1
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1

    # Check PEFT config
    assert cfg.peft is not None


def test_ministral3_8b_full_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 8B full SFT has correct default parallelism."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_ministral3_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("ministral3_8b_finetune_config")
    overrides["peft"] = None
    # Remove TP/PP overrides to test recipe defaults
    overrides.pop("tensor_model_parallel_size", None)
    overrides.pop("pipeline_model_parallel_size", None)

    cfg = _ministral3_module.ministral3_8b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For full SFT, 8B should use TP=2, PP=1
    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.peft is None


def test_ministral3_14b_lora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 14B LoRA has correct default parallelism."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_ministral3_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("ministral3_14b_finetune_config")
    overrides["peft"] = "lora"
    # Remove TP/PP overrides to test recipe defaults
    overrides.pop("tensor_model_parallel_size", None)
    overrides.pop("pipeline_model_parallel_size", None)

    cfg = _ministral3_module.ministral3_14b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For LoRA, 14B should use TP=2, PP=1
    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1

    # Check PEFT config
    assert cfg.peft is not None


def test_ministral3_14b_full_sft_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 14B full SFT has correct default parallelism."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_ministral3_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("ministral3_14b_finetune_config")
    overrides["peft"] = None
    # Remove TP/PP overrides to test recipe defaults
    overrides.pop("tensor_model_parallel_size", None)
    overrides.pop("pipeline_model_parallel_size", None)

    cfg = _ministral3_module.ministral3_14b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For full SFT, 14B should use TP=4, PP=1
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.peft is None


def test_ministral3_14b_dora_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that 14B DoRA has correct default parallelism."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_ministral3_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("ministral3_14b_finetune_config")
    overrides["peft"] = "dora"
    # Remove TP/PP overrides to test recipe defaults
    overrides.pop("tensor_model_parallel_size", None)
    overrides.pop("pipeline_model_parallel_size", None)

    cfg = _ministral3_module.ministral3_14b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # For DoRA, 14B should use same parallelism as LoRA (TP=2, PP=1)
    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 1

    # Check PEFT config
    assert cfg.peft is not None


def test_ministral3_custom_finetune_lr(monkeypatch: pytest.MonkeyPatch):
    """Test that custom finetune_lr overrides default learning rate."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_ministral3_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("ministral3_3b_finetune_config")
    overrides["peft"] = "lora"
    overrides["finetune_lr"] = 2e-4  # Custom learning rate

    cfg = _ministral3_module.ministral3_3b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # Check that custom learning rate is used
    assert cfg.optimizer.lr == 2e-4


def test_ministral3_peft_with_freeze_options(monkeypatch: pytest.MonkeyPatch):
    """Test that PEFT can be combined with freeze options."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_ministral3_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("ministral3_3b_finetune_config")
    overrides["peft"] = "lora"
    overrides["freeze_language_model"] = True
    overrides["freeze_vision_model"] = False
    overrides["freeze_vision_projection"] = True

    cfg = _ministral3_module.ministral3_3b_finetune_config(**overrides)

    _assert_basic_config(cfg)

    # Check PEFT config
    assert cfg.peft is not None

    # Check freeze options
    assert cfg.model.freeze_language_model is True
    assert cfg.model.freeze_vision_model is False
    assert cfg.model.freeze_vision_projection is True
