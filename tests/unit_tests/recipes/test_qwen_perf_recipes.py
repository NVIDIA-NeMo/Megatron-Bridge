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

"""Tests for Qwen3 NVFP4 perf recipes: tp_comm_overlap must be disabled for NVFP4.

NVFP4's fp4_param_gather path is incompatible with TP comm overlap, so every
NVFP4 pretrain config must set ``comm_overlap.tp_comm_overlap = False`` while
non-NVFP4 (FP8 current-scaling) siblings keep it enabled.
"""

import pytest
from scripts.common.benchmark_parallelism import data_parallel_size, topology_from_config

from megatron.bridge.perf_recipes.qwen import (
    qwen3_30b_a3b_pretrain_8gpu_b200_fp8cs_config,
    qwen3_30b_a3b_pretrain_8gpu_b200_nvfp4_config,
    qwen3_30b_a3b_pretrain_8gpu_b300_fp8cs_config,
    qwen3_30b_a3b_pretrain_8gpu_b300_nvfp4_config,
    qwen3_30b_a3b_pretrain_8gpu_gb200_fp8cs_config,
    qwen3_30b_a3b_pretrain_8gpu_gb200_nvfp4_config,
    qwen3_30b_a3b_pretrain_8gpu_gb300_fp8cs_config,
    qwen3_30b_a3b_pretrain_8gpu_gb300_nvfp4_config,
    qwen3_30b_a3b_pretrain_8gpu_vr200_nvfp4_config,
    qwen3_235b_a22b_pretrain_64gpu_b200_fp8cs_config,
    qwen3_235b_a22b_pretrain_64gpu_b200_nvfp4_config,
    qwen3_235b_a22b_pretrain_64gpu_b300_fp8cs_config,
    qwen3_235b_a22b_pretrain_64gpu_b300_nvfp4_config,
    qwen3_235b_a22b_pretrain_64gpu_gb200_bf16_config,
    qwen3_235b_a22b_pretrain_64gpu_gb200_fp8cs_config,
    qwen3_235b_a22b_pretrain_64gpu_gb200_fp8mx_config,
    qwen3_235b_a22b_pretrain_64gpu_gb200_nvfp4_config,
    qwen3_235b_a22b_pretrain_64gpu_gb300_bf16_config,
    qwen3_235b_a22b_pretrain_64gpu_gb300_fp8cs_config,
    qwen3_235b_a22b_pretrain_64gpu_gb300_fp8mx_config,
    qwen3_235b_a22b_pretrain_64gpu_gb300_nvfp4_config,
    qwen3_235b_a22b_pretrain_256gpu_b200_fp8cs_config,
    qwen3_235b_a22b_pretrain_256gpu_b200_nvfp4_config,
    qwen3_235b_a22b_pretrain_256gpu_b300_fp8cs_config,
    qwen3_235b_a22b_pretrain_256gpu_b300_nvfp4_config,
    qwen3_235b_a22b_pretrain_256gpu_gb200_fp8cs_config,
    qwen3_235b_a22b_pretrain_256gpu_gb200_nvfp4_config,
    qwen3_235b_a22b_pretrain_256gpu_gb300_fp8cs_config,
    qwen3_235b_a22b_pretrain_256gpu_gb300_nvfp4_config,
    qwen3_235b_a22b_pretrain_256gpu_vr200_nvfp4_config,
)


@pytest.mark.parametrize(
    ("recipe_func", "expected_data_parallel_size", "expected_hybridep_domain_size"),
    [
        (qwen3_235b_a22b_pretrain_64gpu_gb200_bf16_config, 8, 8),
        (qwen3_235b_a22b_pretrain_64gpu_gb200_fp8cs_config, 8, 8),
        (qwen3_235b_a22b_pretrain_64gpu_gb200_fp8mx_config, 8, 8),
        (qwen3_235b_a22b_pretrain_64gpu_gb200_nvfp4_config, 8, 8),
        (qwen3_235b_a22b_pretrain_64gpu_gb300_bf16_config, 32, 32),
        (qwen3_235b_a22b_pretrain_64gpu_gb300_fp8cs_config, 32, 32),
        (qwen3_235b_a22b_pretrain_64gpu_gb300_fp8mx_config, 32, 32),
        (qwen3_235b_a22b_pretrain_64gpu_gb300_nvfp4_config, 32, 32),
    ],
)
def test_qwen3_235b_64gpu_blackwell_topology_is_valid(
    recipe_func, expected_data_parallel_size, expected_hybridep_domain_size
):
    cfg = recipe_func()

    assert data_parallel_size(num_gpus=64, topology=topology_from_config(cfg.model)) == expected_data_parallel_size
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == expected_hybridep_domain_size


@pytest.mark.parametrize(
    "nvfp4_config_fn",
    [
        qwen3_30b_a3b_pretrain_8gpu_gb200_nvfp4_config,
        qwen3_30b_a3b_pretrain_8gpu_gb300_nvfp4_config,
        qwen3_30b_a3b_pretrain_8gpu_b200_nvfp4_config,
        qwen3_30b_a3b_pretrain_8gpu_b300_nvfp4_config,
        qwen3_30b_a3b_pretrain_8gpu_vr200_nvfp4_config,
        qwen3_235b_a22b_pretrain_64gpu_gb200_nvfp4_config,
        qwen3_235b_a22b_pretrain_256gpu_gb200_nvfp4_config,
        qwen3_235b_a22b_pretrain_64gpu_gb300_nvfp4_config,
        qwen3_235b_a22b_pretrain_256gpu_gb300_nvfp4_config,
        qwen3_235b_a22b_pretrain_64gpu_b200_nvfp4_config,
        qwen3_235b_a22b_pretrain_256gpu_b200_nvfp4_config,
        qwen3_235b_a22b_pretrain_64gpu_b300_nvfp4_config,
        qwen3_235b_a22b_pretrain_256gpu_b300_nvfp4_config,
        qwen3_235b_a22b_pretrain_256gpu_vr200_nvfp4_config,
    ],
)
def test_nvfp4_disables_tp_comm_overlap(nvfp4_config_fn):
    cfg = nvfp4_config_fn()
    assert cfg.comm_overlap.tp_comm_overlap is False, (
        f"{nvfp4_config_fn.__name__}: expected tp_comm_overlap=False for NVFP4, got {cfg.comm_overlap.tp_comm_overlap}"
    )


@pytest.mark.parametrize(
    "fp8cs_config_fn",
    [
        qwen3_30b_a3b_pretrain_8gpu_gb200_fp8cs_config,
        qwen3_30b_a3b_pretrain_8gpu_gb300_fp8cs_config,
        qwen3_30b_a3b_pretrain_8gpu_b200_fp8cs_config,
        qwen3_30b_a3b_pretrain_8gpu_b300_fp8cs_config,
        qwen3_235b_a22b_pretrain_64gpu_gb200_fp8cs_config,
        qwen3_235b_a22b_pretrain_256gpu_gb200_fp8cs_config,
        qwen3_235b_a22b_pretrain_64gpu_gb300_fp8cs_config,
        qwen3_235b_a22b_pretrain_256gpu_gb300_fp8cs_config,
        qwen3_235b_a22b_pretrain_64gpu_b200_fp8cs_config,
        qwen3_235b_a22b_pretrain_256gpu_b200_fp8cs_config,
        qwen3_235b_a22b_pretrain_64gpu_b300_fp8cs_config,
        qwen3_235b_a22b_pretrain_256gpu_b300_fp8cs_config,
    ],
)
def test_non_nvfp4_preserves_tp_comm_overlap(fp8cs_config_fn):
    """Regression: the NVFP4 fix must not affect FP8 current-scaling siblings."""
    cfg = fp8cs_config_fn()
    assert cfg.comm_overlap.tp_comm_overlap is True, (
        f"{fp8cs_config_fn.__name__}: expected tp_comm_overlap=True for FP8-CS, got {cfg.comm_overlap.tp_comm_overlap}"
    )
