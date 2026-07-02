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

"""Tests for Qwen3.5-VL performance workload presets."""

import sys
from pathlib import Path

import pytest

from megatron.bridge.perf_recipes.qwen_vl.h100.qwen35_vl import (
    qwen35_vl_122b_a10b_pretrain_128gpu_h100_bf16_config,
    qwen35_vl_122b_a10b_pretrain_128gpu_h100_fp8cs_config,
)


pytestmark = pytest.mark.unit

_PERF_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts" / "performance"
if str(_PERF_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_SCRIPTS_DIR))

from configs.qwen_vl.qwen35_vl_workload_base_configs import (  # noqa: E402
    QWEN35_VL_122B_A10B_PRETRAIN_CONFIG_H100_BF16,
    QWEN35_VL_122B_A10B_PRETRAIN_CONFIG_H100_FP8_CS,
)


@pytest.mark.parametrize(
    ("legacy_config", "recipe_fn"),
    [
        (
            QWEN35_VL_122B_A10B_PRETRAIN_CONFIG_H100_BF16,
            qwen35_vl_122b_a10b_pretrain_128gpu_h100_bf16_config,
        ),
        (
            QWEN35_VL_122B_A10B_PRETRAIN_CONFIG_H100_FP8_CS,
            qwen35_vl_122b_a10b_pretrain_128gpu_h100_fp8cs_config,
        ),
    ],
)
def test_qwen35_vl_122b_h100_pipeline_layout(legacy_config, recipe_fn):
    num_layers = 48
    config = recipe_fn()
    pp_size = config.model.pipeline_model_parallel_size
    vp_size = config.model.virtual_pipeline_model_parallel_size

    assert num_layers % pp_size == 0
    assert vp_size is not None
    assert (num_layers // pp_size) % vp_size == 0
    assert pp_size == legacy_config.pipeline_model_parallel_size
    assert vp_size == legacy_config.virtual_pipeline_model_parallel_size
