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

"""
Unit tests for Step-3.5-Flash recipe configuration builders.

Patterned after DeepSeek / OLMoE recipe tests: monkeypatch ``AutoBridge`` to a
lightweight fake provider so we exercise the recipe wiring without HF Hub I/O
or real Megatron-Core layer construction. Also monkeypatches
``apply_flex_dispatcher_backend`` so the recipe does not poke CUDA device
properties during tests.
"""

import importlib
from typing import Callable

import pytest
import torch

from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_module_global


_step35_module = importlib.import_module("megatron.bridge.recipes.stepfun.step35")
_STEP35_RECIPE_FUNCS = [
    getattr(_step35_module, name)
    for name in getattr(importlib.import_module("megatron.bridge.recipes.stepfun"), "__all__", [])
    if callable(getattr(_step35_module, name, None))
]


class _FakeModelCfg:
    """Stand-in for ``Step35ModelProvider`` returned by ``AutoBridge``."""

    def __init__(self):
        # Attributes the recipe reads back after assignment (e.g. for
        # apply_flex_dispatcher_backend) need plausible defaults.
        self.num_moe_experts = 288
        self.moe_flex_dispatcher_backend = None
        self.seq_length = 4096
        # Fields the recipe never sets explicitly but downstream code may read.
        self.apply_rope_fusion = False
        self.pipeline_model_parallel_layout = None

    def finalize(self):
        return None


class _FakeBridge:
    def to_megatron_provider(self, load_weights: bool = False):
        # Recipe always passes load_weights=False, but we don't gate on it.
        return _FakeModelCfg()

    @staticmethod
    def from_hf_pretrained(hf_path: str):
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


def _patch_recipe_env(monkeypatch, mod):
    """Patch ``AutoBridge`` and ``apply_flex_dispatcher_backend`` in the recipe
    module so the recipe builds without hitting HF Hub or CUDA."""
    patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)
    patch_recipe_module_global(monkeypatch, mod, "apply_flex_dispatcher_backend", lambda *a, **kw: None)


# ---------------------------------------------------------------------------
# Parametrized "each recipe builds" sanity test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("recipe_func", _STEP35_RECIPE_FUNCS)
def test_each_step35_recipe_builds_config(recipe_func: Callable, monkeypatch):
    mod = importlib.import_module(recipe_func.__module__)
    _patch_recipe_env(monkeypatch, mod)

    cfg = recipe_func()
    _assert_basic_config(cfg)


# ---------------------------------------------------------------------------
# Defaults specific to the Step-3.5-Flash pretrain recipe
# ---------------------------------------------------------------------------


def test_step35_196b_pretrain_defaults(monkeypatch):
    """The pretrain recipe ships with the alignment-run parallelism layout:
    TP=1, PP=8, CP=8, EP=8, MoE alltoall + permute_fusion + grouped_gemm, etc."""
    from megatron.bridge.recipes.stepfun.step35 import step35_196b_a11b_pretrain_config

    mod = importlib.import_module("megatron.bridge.recipes.stepfun.step35")
    _patch_recipe_env(monkeypatch, mod)

    cfg = step35_196b_a11b_pretrain_config()
    _assert_basic_config(cfg)

    m = cfg.model
    # Parallelism (alignment / resume layout)
    assert m.tensor_model_parallel_size == 1
    assert m.pipeline_model_parallel_size == 8
    assert m.context_parallel_size == 8
    assert m.expert_model_parallel_size == 8
    assert m.expert_tensor_parallel_size == 1
    assert m.sequence_parallel is True
    assert m.seq_length == 4096
    assert m.pipeline_dtype is torch.bfloat16

    # MoE wiring
    assert m.moe_grouped_gemm is True
    assert m.moe_permute_fusion is True
    assert m.moe_token_dispatcher_type == "alltoall"
    assert m.moe_flex_dispatcher_backend == "deepep"

    # Memory / recompute
    assert m.recompute_granularity == "full"
    assert m.recompute_method == "uniform"
    assert m.recompute_num_layers == 1

    # DDP
    assert cfg.ddp.use_distributed_optimizer is True
    assert cfg.ddp.use_megatron_fsdp is False


def test_step35_196b_recipe_uses_stepfun_tokenizer(monkeypatch):
    from megatron.bridge.recipes.stepfun.step35 import step35_196b_a11b_pretrain_config

    mod = importlib.import_module("megatron.bridge.recipes.stepfun.step35")
    _patch_recipe_env(monkeypatch, mod)

    cfg = step35_196b_a11b_pretrain_config()
    assert cfg.tokenizer.tokenizer_model == "stepfun-ai/Step-3.5-Flash"
