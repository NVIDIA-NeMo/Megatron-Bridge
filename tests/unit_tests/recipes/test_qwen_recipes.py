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
# - Parametrize over all exported Qwen recipe functions in `megatron.bridge.recipes.qwen`.
# - For each recipe, monkeypatch `AutoBridge` with a lightweight fake to avoid I/O.
# - Build a config with small, safe overrides and assert it forms a valid `ConfigContainer`.
# - Verify tokenizer selection honors `use_null_tokenizer`, and sanity-check parallelism fields.
#

import importlib
from typing import Callable

import pytest


_qwen_module = importlib.import_module("megatron.bridge.recipes.qwen")
_QWEN_RECIPE_FUNCS = [
    getattr(_qwen_module, name)
    for name in getattr(_qwen_module, "__all__", [])
    if callable(getattr(_qwen_module, name, None))
]


def _safe_overrides_for(name: str) -> dict:
    # Minimal, dependency-light overrides for fast unit testing
    overrides = {
        "name": f"unit_{name}",
        "dir": ".",  # keep paths local
        "mock": True,  # use mock data paths
        "train_iters": 10,
        "global_batch_size": 2,
        "micro_batch_size": 1,
        "seq_length": 64,
        "lr": 1e-4,
        "min_lr": 1e-5,
        "lr_warmup_iters": 2,
        # Keep parallelism tiny so provider shaping is trivial
        "tensor_parallelism": 1,
        "pipeline_parallelism": 1,
        "context_parallelism": 1,
        # Prefer NullTokenizer in tests to avoid HF tokenizer I/O
        "use_null_tokenizer": True,
    }

    # For MoE recipes, ensure expert settings are small/valid
    lname = name.lower()
    if "a3b" in lname or "a22b" in lname or "moe" in lname:
        overrides.update(
            {
                "expert_parallelism": 2,
                "expert_tensor_parallelism": 1,
                "sequence_parallelism": True,
            }
        )

    return overrides


class _FakeModelCfg:
    # Minimal provider to accept attribute assignments used in recipes
    def finalize(self):
        # qwen3 recipe may call finalize(); make it a no-op
        return None


class _FakeBridge:
    def __init__(self):
        pass

    def to_megatron_provider(self, load_weights: bool = False):
        return _FakeModelCfg()

    @staticmethod
    def from_hf_pretrained(hf_path: str):
        # Ignore hf_path; return a bridge that yields a fake provider
        return _FakeBridge()


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
    assert cfg.dataset.sequence_length >= 1


@pytest.mark.parametrize("recipe_func", _QWEN_RECIPE_FUNCS)
def test_each_qwen_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    # Monkeypatch AutoBridge in the specific module where the recipe function is defined
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for(recipe_func.__name__)

    cfg = recipe_func(**overrides)

    _assert_basic_config(cfg)

    # Ensure tokenizer choice matches override
    if overrides.get("use_null_tokenizer"):
        assert cfg.tokenizer.tokenizer_type == "NullTokenizer"
        assert cfg.tokenizer.vocab_size is not None
    else:
        assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
        assert cfg.tokenizer.tokenizer_model is not None

    # Parallelism and shaping
    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1
