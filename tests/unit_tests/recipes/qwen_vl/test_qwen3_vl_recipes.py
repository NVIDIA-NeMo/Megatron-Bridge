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


_qwen3_vl_module = importlib.import_module("megatron.bridge.recipes.qwen_vl.qwen3_vl")


def _collect_recipe_functions(mod) -> list[Callable]:
    # Prefer explicit exports
    exported_names = getattr(mod, "__all__", None)
    candidates: list[Callable] = []

    if exported_names:
        for name in exported_names:
            fn = getattr(mod, name, None)
            if callable(fn) and (name.endswith("_config") or "qwen3" in name.lower() or "qwen" in name.lower()):
                candidates.append(fn)
    else:
        # Fallback: discover by convention
        for name in dir(mod):
            if name.startswith("_"):
                continue
            fn = getattr(mod, name, None)
            if callable(fn) and name.endswith("_config"):
                candidates.append(fn)

    # De-dupe while preserving order
    seen = set()
    unique = []
    for fn in candidates:
        if fn.__name__ not in seen:
            unique.append(fn)
            seen.add(fn.__name__)
    return unique


_QWEN3_VL_RECIPE_FUNCS: list[Callable] = _collect_recipe_functions(_qwen3_vl_module)


def _safe_overrides_for(name: str) -> dict:
    overrides = {
        "name": f"unit_{name}",
        "dir": ".",
        "train_iters": 5,
        "micro_batch_size": 1,
        "seq_length": 64,
        "min_lr": 1e-5,
        "lr_warmup_iters": 2,
        "mock": True,
        "lr": 1e-4,
        "use_null_tokenizer": True,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
    }

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

    # Different dataset configs may expose length as sequence_length or seq_length;
    # for multimodal datasets there may be no such attribute. Only assert presence when available.
    if hasattr(cfg.dataset, "sequence_length"):
        assert cfg.dataset.sequence_length >= 1
    elif hasattr(cfg.dataset, "seq_length"):
        assert cfg.dataset.seq_length >= 1
    else:
        assert cfg.dataset is not None


@pytest.mark.parametrize("recipe_func", _QWEN3_VL_RECIPE_FUNCS)
def test_each_qwen3_vl_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    # Monkeypatch AutoBridge used inside the recipe module to avoid heavyweight init
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for(recipe_func.__name__)

    cfg = recipe_func(**overrides)

    _assert_basic_config(cfg)

    # Minimal sanity checks on parallelism fields being set to sane values
    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1
