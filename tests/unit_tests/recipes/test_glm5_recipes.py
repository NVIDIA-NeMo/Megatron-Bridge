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

"""Unit tests for the GLM-5.2 753B pretrain recipes."""

import importlib
from typing import Callable

import pytest

from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_module_global


pytestmark = pytest.mark.unit


_glm_module = importlib.import_module("megatron.bridge.recipes.glm")
_GLM5_RECIPE_FUNCS = [
    getattr(_glm_module, name) for name in ("glm52_753b_pretrain_config", "glm5_2_753b_pretrain_config")
]


class _FakeModelCfg:
    """Minimal settable model config so the recipe builds without HF I/O."""

    def __init__(self):
        self.vocab_size = 151552  # GLM vocab size
        self.make_vocab_size_divisible_by = 1280


class _FakeBridge:
    """Stand-in for AutoBridge that avoids any Hugging Face download."""

    @staticmethod
    def from_hf_pretrained(hf_path: str, **kwargs):
        return _FakeBridge()

    def to_megatron_provider(self, load_weights: bool = False):
        return _FakeModelCfg()


@pytest.mark.parametrize("recipe_func", _GLM5_RECIPE_FUNCS)
def test_each_glm5_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Each GLM-5.2 pretrain recipe (canonical + alias) builds a valid ConfigContainer."""
    mod = importlib.import_module(recipe_func.__module__)
    patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)

    cfg = recipe_func()

    # Basic container validity.
    for sub in ("model", "train", "scheduler", "dataset", "tokenizer", "checkpoint", "rng", "optimizer"):
        assert getattr(cfg, sub) is not None
    assert cfg.train.global_batch_size >= 1
    assert cfg.train.micro_batch_size >= 1
    assert cfg.dataset.seq_length >= 1

    # Mock-data pretraining uses NullTokenizer with a preserved vocab size.
    assert cfg.tokenizer.tokenizer_type == "NullTokenizer"
    assert cfg.tokenizer.vocab_size is not None

    # DSA invariants: Megatron-Core requires CP=1 and rejects RoPE fusion for the
    # "dsa" attention variant, so these must not regress.
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.apply_rope_fusion is False
    # MTP is disabled for pretraining.
    assert cfg.model.mtp_num_layers is None

    # Standard recipe env baseline (expandable allocator).
    assert cfg.env_vars.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True"
