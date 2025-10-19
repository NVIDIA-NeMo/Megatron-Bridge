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
from typing import Callable

import pytest


_gemma_module = importlib.import_module("megatron.bridge.recipes.gemma")
_GEMMA3_RECIPE_FUNCS = [
    getattr(_gemma_module, name)
    for name in getattr(_gemma_module, "__all__", [])
    if callable(getattr(_gemma_module, name, None))
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
        self.account_for_embedding_in_pipeline_split = False
        self.account_for_loss_in_pipeline_split = False

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
    assert cfg.dataset.sequence_length >= 1


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

    overrides = _safe_overrides_for(recipe_func.__name__)

    cfg = recipe_func(**overrides)

    _assert_basic_config(cfg)

    if overrides.get("use_null_tokenizer") and hasattr(cfg, "tokenizer") and hasattr(cfg.tokenizer, "tokenizer_type"):
        assert cfg.tokenizer.tokenizer_type == "NullTokenizer"

    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1
