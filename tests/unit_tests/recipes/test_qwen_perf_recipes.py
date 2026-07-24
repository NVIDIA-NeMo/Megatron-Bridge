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
    qwen3_235b_a22b_pretrain_64gpu_gb200_fp8cs_config,
    qwen3_235b_a22b_pretrain_64gpu_gb200_nvfp4_config,
    qwen3_235b_a22b_pretrain_64gpu_gb300_fp8cs_config,
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
