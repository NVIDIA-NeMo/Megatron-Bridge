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

"""
Unit tests for GPT-OSS recipe configuration builders.

For each exported recipe function in ``megatron.bridge.recipes.gpt_oss``:
- Call the config factory (parameterless API).
- Assert a valid ``ConfigContainer`` is returned.
- For FP8 current-scaling variants, verify that the FP8 mixed-precision
  preset and the ``moe_router_padding_for_fp8`` flag are applied.
"""

import importlib
from typing import Callable

import pytest


_gpt_oss_module = importlib.import_module("megatron.bridge.recipes.gpt_oss")
_ALL_RECIPE_FUNCS = [
    getattr(_gpt_oss_module, name)
    for name in getattr(_gpt_oss_module, "__all__", [])
    if callable(getattr(_gpt_oss_module, name, None))
]

_FP8_RECIPE_FUNCS = [f for f in _ALL_RECIPE_FUNCS if "fp8" in f.__name__]
_BASE_RECIPE_FUNCS = [f for f in _ALL_RECIPE_FUNCS if "fp8" not in f.__name__]


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
    assert cfg.dataset.seq_length >= 1


@pytest.mark.parametrize("recipe_func", _BASE_RECIPE_FUNCS, ids=lambda f: f.__name__)
def test_base_recipe_builds_valid_config(recipe_func: Callable):
    """Each base GPT-OSS recipe should return a valid ConfigContainer."""
    if "peft" in recipe_func.__name__:
        cfg = recipe_func(peft_scheme="lora")
    else:
        cfg = recipe_func()
    _assert_basic_config(cfg)


@pytest.mark.parametrize("recipe_func", _FP8_RECIPE_FUNCS, ids=lambda f: f.__name__)
def test_fp8_recipe_builds_valid_config(recipe_func: Callable):
    """Each FP8 current-scaling GPT-OSS recipe should return a valid ConfigContainer."""
    if "peft" in recipe_func.__name__:
        cfg = recipe_func(peft_scheme="lora")
    else:
        cfg = recipe_func()
    _assert_basic_config(cfg)


@pytest.mark.parametrize("recipe_func", _FP8_RECIPE_FUNCS, ids=lambda f: f.__name__)
def test_fp8_recipe_enables_fp8_mixed_precision(recipe_func: Callable):
    """FP8 variants must set the bf16_with_fp8_current_scaling_mixed preset."""
    if "peft" in recipe_func.__name__:
        cfg = recipe_func(peft_scheme="lora")
    else:
        cfg = recipe_func()
    assert cfg.mixed_precision == "bf16_with_fp8_current_scaling_mixed", (
        f"{recipe_func.__name__}: expected mixed_precision=bf16_with_fp8_current_scaling_mixed, "
        f"got {cfg.mixed_precision!r}"
    )


@pytest.mark.parametrize("recipe_func", _FP8_RECIPE_FUNCS, ids=lambda f: f.__name__)
def test_fp8_recipe_enables_moe_router_padding(recipe_func: Callable):
    """FP8 variants must enable moe_router_padding_for_fp8 on the model config."""
    if "peft" in recipe_func.__name__:
        cfg = recipe_func(peft_scheme="lora")
    else:
        cfg = recipe_func()
    assert getattr(cfg.model, "moe_router_padding_for_fp8", False) is True, (
        f"{recipe_func.__name__}: expected moe_router_padding_for_fp8=True"
    )


def test_fp8_pretrain_inherits_base_pretrain_config():
    """gpt_oss_20b_pretrain_fp8_current_scaling_config should inherit base pretrain settings."""
    from megatron.bridge.recipes.gpt_oss import (
        gpt_oss_20b_pretrain_config,
        gpt_oss_20b_pretrain_fp8_current_scaling_config,
    )

    base = gpt_oss_20b_pretrain_config()
    fp8 = gpt_oss_20b_pretrain_fp8_current_scaling_config()

    assert fp8.model.num_layers == base.model.num_layers
    assert fp8.train.global_batch_size == base.train.global_batch_size
    assert fp8.dataset.seq_length == base.dataset.seq_length
    assert fp8.mixed_precision != base.mixed_precision
    assert fp8.model.moe_router_padding_for_fp8 is True
    assert base.model.moe_router_padding_for_fp8 is False


def test_fp8_sft_inherits_base_sft_config():
    """gpt_oss_20b_sft_fp8_current_scaling_config should inherit base SFT settings."""
    from megatron.bridge.recipes.gpt_oss import (
        gpt_oss_20b_sft_config,
        gpt_oss_20b_sft_fp8_current_scaling_config,
    )

    base = gpt_oss_20b_sft_config()
    fp8 = gpt_oss_20b_sft_fp8_current_scaling_config()

    assert fp8.model.num_layers == base.model.num_layers
    assert fp8.train.global_batch_size == base.train.global_batch_size
    assert fp8.mixed_precision != base.mixed_precision
    assert fp8.model.moe_router_padding_for_fp8 is True
    assert base.model.moe_router_padding_for_fp8 is False


def test_fp8_peft_inherits_base_peft_config():
    """gpt_oss_20b_peft_fp8_current_scaling_config should inherit base PEFT settings."""
    from megatron.bridge.recipes.gpt_oss import (
        gpt_oss_20b_peft_config,
        gpt_oss_20b_peft_fp8_current_scaling_config,
    )

    base = gpt_oss_20b_peft_config(peft_scheme="lora")
    fp8 = gpt_oss_20b_peft_fp8_current_scaling_config(peft_scheme="lora")

    assert fp8.model.num_layers == base.model.num_layers
    assert fp8.peft is not None
    assert base.peft is not None
    assert fp8.mixed_precision != base.mixed_precision
    assert fp8.model.moe_router_padding_for_fp8 is True
    assert base.model.moe_router_padding_for_fp8 is False


@pytest.mark.parametrize("peft_scheme", ["lora", "dora"])
def test_fp8_peft_supports_lora_and_dora(peft_scheme: str):
    """FP8 PEFT config should work with both lora and dora schemes."""
    from megatron.bridge.recipes.gpt_oss import gpt_oss_20b_peft_fp8_current_scaling_config

    cfg = gpt_oss_20b_peft_fp8_current_scaling_config(peft_scheme=peft_scheme)
    _assert_basic_config(cfg)
    assert cfg.peft is not None
    assert cfg.mixed_precision == "bf16_with_fp8_current_scaling_mixed"
