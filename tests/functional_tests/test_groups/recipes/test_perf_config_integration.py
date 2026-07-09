# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Construction tests for canonical flat performance recipes.

These tests deliberately import ``megatron.bridge.perf_recipes`` directly.
Legacy adapters under ``scripts/performance/configs`` are compatibility code
and must not be the source of truth for recipe CI.
"""

import pytest

from megatron.bridge.perf_recipes.deepseek.h100.deepseek_v3 import (
    deepseek_v3_pretrain_64gpu_h100_bf16_config,
)
from megatron.bridge.perf_recipes.gpt_oss.h100.gpt_oss import (
    gpt_oss_120b_pretrain_64gpu_h100_bf16_config,
)
from megatron.bridge.perf_recipes.llama.h100.llama3 import (
    llama3_8b_pretrain_8gpu_h100_bf16_config,
    llama3_8b_pretrain_8gpu_h100_fp8cs_config,
    llama3_70b_pretrain_64gpu_h100_bf16_config,
)
from megatron.bridge.perf_recipes.llama.h100.llama31 import (
    llama31_405b_pretrain_1024gpu_h100_bf16_config,
)
from megatron.bridge.perf_recipes.nemotronh.gb300.nemotronh import (
    nemotron_3_super_pretrain_64gpu_gb300_bf16_config,
    nemotron_3_super_pretrain_64gpu_gb300_nvfp4_config,
)
from megatron.bridge.perf_recipes.nemotronh.h100.nemotronh import (
    nemotronh_56b_pretrain_64gpu_h100_fp8cs_config,
)
from megatron.bridge.perf_recipes.qwen.h100.qwen3_moe import (
    qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config,
    qwen3_next_80b_a3b_pretrain_128gpu_h100_bf16_config,
)


PERF_RECIPE_CASES = [
    llama3_8b_pretrain_8gpu_h100_bf16_config,
    llama3_70b_pretrain_64gpu_h100_bf16_config,
    llama31_405b_pretrain_1024gpu_h100_bf16_config,
    deepseek_v3_pretrain_64gpu_h100_bf16_config,
    qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config,
    qwen3_next_80b_a3b_pretrain_128gpu_h100_bf16_config,
    nemotronh_56b_pretrain_64gpu_h100_fp8cs_config,
    nemotron_3_super_pretrain_64gpu_gb300_bf16_config,
    gpt_oss_120b_pretrain_64gpu_h100_bf16_config,
]


@pytest.mark.parametrize("recipe_func", PERF_RECIPE_CASES, ids=lambda recipe: recipe.__name__)
def test_canonical_perf_recipe_instantiation(recipe_func):
    """Every selected flat performance recipe should build a complete config."""
    config = recipe_func()

    assert config.model is not None
    assert config.mixed_precision is not None
    assert config.train is not None
    assert config.dataset is not None
    assert config.train.train_iters == 50
    assert config.checkpoint.save is None


def test_perf_recipe_precision_variants():
    """Precision is part of the selected recipe rather than a legacy wrapper argument."""
    bf16_config = llama3_8b_pretrain_8gpu_h100_bf16_config()
    fp8_config = llama3_8b_pretrain_8gpu_h100_fp8cs_config()

    assert bf16_config.mixed_precision.fp8 is None
    assert fp8_config.mixed_precision.fp8 == "e4m3"
    assert fp8_config.mixed_precision.fp8_recipe == "tensorwise"


def test_nemotron_super_nvfp4_recipe():
    """The canonical NVFP4 recipe retains Nemotron-specific BF16 boundary layers."""
    config = nemotron_3_super_pretrain_64gpu_gb300_nvfp4_config()

    assert config.mixed_precision.first_last_layers_bf16 is True
    assert config.mixed_precision.num_layers_at_end_in_bf16 == 14


def test_canonical_perf_recipe_allows_runtime_overrides():
    """A flat recipe remains a normal ConfigContainer after construction."""
    config = llama3_8b_pretrain_8gpu_h100_bf16_config()

    config.train.train_iters = 100
    config.train.global_batch_size = 16

    assert config.train.train_iters == 100
    assert config.train.global_batch_size == 16
