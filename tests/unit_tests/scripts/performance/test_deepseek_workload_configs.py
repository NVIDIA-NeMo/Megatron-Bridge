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

"""Tests for DeepSeek performance workload presets."""

import sys
from pathlib import Path

import pytest


_PERF_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts" / "performance"
if str(_PERF_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_SCRIPTS_DIR))

from configs.deepseek.deepseek_workload_base_configs import (  # noqa: E402
    DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_FP8_MX_V1,
    DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_NVFP4_V1,
    DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_NVFP4_V2,
    DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_FP8_MX_V1,
    DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_NVFP4_V1,
    DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_NVFP4_V2,
)


_PARALLEL_MAPPING_FIELDS = (
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "context_parallel_size",
    "virtual_pipeline_model_parallel_size",
    "expert_model_parallel_size",
    "expert_tensor_parallel_size",
    "pp_layout",
)


@pytest.mark.parametrize(
    "nvfp4_config,mxfp8_config",
    [
        (DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_NVFP4_V1, DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_FP8_MX_V1),
        (DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_NVFP4_V1, DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_FP8_MX_V1),
    ],
)
def test_nvfp4_v1_mirrors_mxfp8_mapping_and_optimized_features(nvfp4_config, mxfp8_config):
    for field in _PARALLEL_MAPPING_FIELDS:
        assert getattr(nvfp4_config, field) == getattr(mxfp8_config, field)

    assert nvfp4_config.cuda_graph_impl == "full_iteration"
    assert nvfp4_config.cutedsl_fused_grouped_mlp is True
    assert nvfp4_config.fp8_dot_product_attention is True
    assert nvfp4_config.moe_a2a_overlap is False


@pytest.mark.parametrize(
    "v1_config,v2_config",
    [
        (DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_NVFP4_V1, DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_NVFP4_V2),
        (DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_NVFP4_V1, DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_NVFP4_V2),
    ],
)
def test_nvfp4_v2_only_changes_global_batch_size(v1_config, v2_config):
    assert v1_config.global_batch_size == 2048
    assert v2_config.global_batch_size == 4096
    assert vars(v2_config) == {**vars(v1_config), "global_batch_size": 4096}
