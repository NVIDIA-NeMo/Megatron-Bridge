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

"""Unit tests for Nemotron 3 Nano with MTP performance recipes."""

from collections.abc import Callable
from inspect import signature

import pytest

from megatron.bridge.perf_recipes.nemotronh import (
    nemotron_3_nano_mtp_pretrain_8gpu_gb200_bf16_config,
    nemotron_3_nano_mtp_pretrain_8gpu_gb200_fp8mx_config,
    nemotron_3_nano_mtp_pretrain_8gpu_gb200_nvfp4_config,
    nemotron_3_nano_mtp_pretrain_16gpu_h100_bf16_config,
    nemotron_3_nano_mtp_pretrain_16gpu_h100_fp8cs_config,
    nemotron_3_nano_pretrain_8gpu_gb200_bf16_config,
    nemotron_3_nano_pretrain_8gpu_gb200_fp8mx_config,
    nemotron_3_nano_pretrain_8gpu_gb200_nvfp4_config,
    nemotron_3_nano_pretrain_16gpu_h100_bf16_config,
    nemotron_3_nano_pretrain_16gpu_h100_fp8cs_config,
)
from megatron.bridge.training.config import ConfigContainer


pytestmark = pytest.mark.unit

_H100_RECIPES = (
    nemotron_3_nano_mtp_pretrain_16gpu_h100_bf16_config,
    nemotron_3_nano_mtp_pretrain_16gpu_h100_fp8cs_config,
)
_GB200_RECIPES = (
    nemotron_3_nano_mtp_pretrain_8gpu_gb200_bf16_config,
    nemotron_3_nano_mtp_pretrain_8gpu_gb200_fp8mx_config,
    nemotron_3_nano_mtp_pretrain_8gpu_gb200_nvfp4_config,
)
_NON_MTP_RECIPES = (
    nemotron_3_nano_pretrain_16gpu_h100_bf16_config,
    nemotron_3_nano_pretrain_16gpu_h100_fp8cs_config,
    nemotron_3_nano_pretrain_8gpu_gb200_bf16_config,
    nemotron_3_nano_pretrain_8gpu_gb200_fp8mx_config,
    nemotron_3_nano_pretrain_8gpu_gb200_nvfp4_config,
)
_MTP_BASE_RECIPE_PAIRS = (
    (
        nemotron_3_nano_mtp_pretrain_16gpu_h100_bf16_config,
        nemotron_3_nano_pretrain_16gpu_h100_bf16_config,
    ),
    (
        nemotron_3_nano_mtp_pretrain_16gpu_h100_fp8cs_config,
        nemotron_3_nano_pretrain_16gpu_h100_fp8cs_config,
    ),
    (
        nemotron_3_nano_mtp_pretrain_8gpu_gb200_bf16_config,
        nemotron_3_nano_pretrain_8gpu_gb200_bf16_config,
    ),
    (
        nemotron_3_nano_mtp_pretrain_8gpu_gb200_fp8mx_config,
        nemotron_3_nano_pretrain_8gpu_gb200_fp8mx_config,
    ),
    (
        nemotron_3_nano_mtp_pretrain_8gpu_gb200_nvfp4_config,
        nemotron_3_nano_pretrain_8gpu_gb200_nvfp4_config,
    ),
)


@pytest.mark.parametrize("recipe_factory", _NON_MTP_RECIPES, ids=lambda recipe: recipe.__name__)
def test_standard_perf_recipes_do_not_expose_mtp_flag(recipe_factory: Callable[[], ConfigContainer]) -> None:
    """Standard performance recipes remain parameterless and non-MTP."""
    assert "enable_mtp" not in signature(recipe_factory).parameters
    assert recipe_factory().model.mtp_num_layers == 0


@pytest.mark.parametrize("recipe_factory", (*_H100_RECIPES, *_GB200_RECIPES), ids=lambda recipe: recipe.__name__)
def test_perf_recipes_enable_mtp(recipe_factory: Callable[[], ConfigContainer]) -> None:
    """Each MTP performance variant preserves the shared Nano MTP block."""
    cfg = recipe_factory()

    assert cfg.model.mtp_num_layers == 2
    assert cfg.model.mtp_hybrid_override_pattern == "*E"
    assert cfg.model.mtp_use_repeated_layer is True
    assert cfg.model.keep_mtp_spec_in_bf16 is True
    assert cfg.model.mtp_loss_scaling_factor == 0.3
    assert cfg.model.moe_router_force_load_balancing is True
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"


@pytest.mark.parametrize(
    ("mtp_recipe_factory", "base_recipe_factory"),
    _MTP_BASE_RECIPE_PAIRS,
    ids=[mtp_recipe.__name__ for mtp_recipe, _ in _MTP_BASE_RECIPE_PAIRS],
)
def test_perf_recipes_inherit_non_mtp_policy(
    mtp_recipe_factory: Callable[[], ConfigContainer],
    base_recipe_factory: Callable[[], ConfigContainer],
) -> None:
    """MTP variants inherit environment, loss normalization, and RNG policy."""
    mtp_cfg = mtp_recipe_factory()
    base_cfg = base_recipe_factory()

    assert mtp_cfg.env_vars == base_cfg.env_vars
    assert mtp_cfg.model.calculate_per_token_loss == base_cfg.model.calculate_per_token_loss
    assert mtp_cfg.model.use_te_rng_tracker == base_cfg.model.use_te_rng_tracker
    assert mtp_cfg.tokenizer.tokenizer_model != base_cfg.tokenizer.tokenizer_model


@pytest.mark.parametrize("recipe_factory", _H100_RECIPES, ids=lambda recipe: recipe.__name__)
def test_h100_perf_recipe_topology(recipe_factory: Callable[[], ConfigContainer]) -> None:
    """H100 MTP variants retain the existing Nano performance topology."""
    cfg = recipe_factory()

    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.train.global_batch_size == 1024
    assert cfg.train.micro_batch_size == 1
    assert cfg.model.recompute_granularity == "selective"
    assert cfg.env_vars["NVLINK_DOMAIN_SIZE"] == 8
    assert cfg.env_vars["USE_MNNVL"] == 0


@pytest.mark.parametrize("recipe_factory", _GB200_RECIPES, ids=lambda recipe: recipe.__name__)
def test_gb200_perf_recipe_topology(recipe_factory: Callable[[], ConfigContainer]) -> None:
    """GB200 MTP variants retain the existing Nano performance topology."""
    cfg = recipe_factory()

    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.train.global_batch_size == 512
    assert cfg.train.micro_batch_size == 2
    assert cfg.model.recompute_granularity is None
    assert cfg.env_vars["NVLINK_DOMAIN_SIZE"] == 72
    assert cfg.env_vars["USE_MNNVL"] == 1
