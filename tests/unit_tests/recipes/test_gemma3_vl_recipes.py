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
# - Parametrize over all exported Gemma3-VL recipe functions in `megatron.bridge.recipes.gemma3_vl.gemma3_vl`.
# - For each recipe, monkeypatch AutoBridge and the provider to avoid I/O.
# - Build a config with small, safe overrides and assert it forms a valid `ConfigContainer`.
# - Verify dataset provider selection and sanity-check parallelism fields.
#

import importlib
from typing import Callable

import pytest


_gemma3_vl_module = importlib.import_module("megatron.bridge.recipes.gemma3_vl.gemma3_vl")
_GEMMA3_VL_RECIPE_FUNCS = [
    _gemma3_vl_module.gemma3_vl_4b_finetune_config,
    _gemma3_vl_module.gemma3_vl_12b_finetune_config,
    _gemma3_vl_module.gemma3_vl_27b_finetune_config,
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
        "tensor_parallelism": 1,
        "pipeline_parallelism": 1,
        "context_parallelism": 1,
    }

    # Large models/variants may set additional flags in recipes; keep harmless defaults
    lname = name.lower()
    if "12b" in lname or "27b" in lname:
        overrides.update(
            {
                "virtual_pipeline_parallelism": None,
                "sequence_parallelism": True,
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
    assert cfg.dataset.sequence_length >= 1


@pytest.mark.parametrize("recipe_func", _GEMMA3_VL_RECIPE_FUNCS)
def test_each_gemma3_vl_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that each Gemma3-VL recipe function builds a valid configuration."""
    # Monkeypatch AutoBridge to return a fake model config
    monkeypatch.setattr(_gemma3_vl_module, "AutoBridge", _FakeAutoBridge)

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
def test_gemma3_vl_dataset_type_selection(dataset_type: str, monkeypatch: pytest.MonkeyPatch):
    """Test that different dataset_type values produce correct dataset providers."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_gemma3_vl_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("gemma3_vl_4b_finetune_config")
    overrides["dataset_type"] = dataset_type

    # For preloaded, we need to provide data paths
    if dataset_type == "preloaded":
        overrides["train_data_path"] = ["/fake/train.json"]
        overrides["valid_data_path"] = ["/fake/valid.json"]
        overrides["test_data_path"] = ["/fake/test.json"]
        overrides["image_folder"] = "/fake/images"

    cfg = _gemma3_vl_module.gemma3_vl_4b_finetune_config(**overrides)

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


def test_gemma3_vl_freeze_options(monkeypatch: pytest.MonkeyPatch):
    """Test that freeze options are correctly passed to the model config."""
    # Monkeypatch AutoBridge
    monkeypatch.setattr(_gemma3_vl_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("gemma3_vl_4b_finetune_config")
    overrides["freeze_language_model"] = True
    overrides["freeze_vision_model"] = True
    overrides["freeze_vision_projection"] = False

    cfg = _gemma3_vl_module.gemma3_vl_4b_finetune_config(**overrides)

    assert cfg.model.freeze_language_model is True
    assert cfg.model.freeze_vision_model is True
    assert cfg.model.freeze_vision_projection is False


def test_gemma3_vl_27b_pipeline_dtype(monkeypatch: pytest.MonkeyPatch):
    """Test that 27B model sets pipeline_parallelism_dtype correctly."""
    import torch

    # Monkeypatch AutoBridge
    monkeypatch.setattr(_gemma3_vl_module, "AutoBridge", _FakeAutoBridge)

    overrides = _safe_overrides_for("gemma3_vl_27b_finetune_config")

    cfg = _gemma3_vl_module.gemma3_vl_27b_finetune_config(**overrides)

    # The 27B model should set pipeline_dtype to bfloat16 by default
    assert cfg.model.pipeline_dtype == torch.bfloat16
