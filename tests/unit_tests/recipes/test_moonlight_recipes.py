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
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "sequence_parallel": False,
        "recompute_granularity": "selective",
        "enable_deepep": False,
        "apply_rope_fusion": False,
        "optimizer_type": "adam",
    }

    return overrides


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
    assert cfg.dataset.sequence_length >= 1


@pytest.mark.parametrize("recipe_func", _MOONLIGHT_RECIPE_FUNCS)
def test_each_moonlight_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)

    # Monkeypatch the MoonlightModelProvider16B class
    monkeypatch.setattr(mod, "MoonlightModelProvider16B", _FakeMoonlightModelProvider16B)

    overrides = _safe_overrides_for(recipe_func.__name__)

    cfg = recipe_func(**overrides)

    _assert_basic_config(cfg)

    # Moonlight uses NullTokenizer
    if hasattr(cfg, "tokenizer") and hasattr(cfg.tokenizer, "tokenizer_type"):
        assert cfg.tokenizer.tokenizer_type == "NullTokenizer"

    # Check parallelism settings
    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "expert_model_parallel_size", 1) >= 1
