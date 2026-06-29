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
"""VR200 performance recipes for Qwen3 MoE."""

from megatron.bridge.perf_recipes.qwen.common import (
    ConfigContainer,
)
from megatron.bridge.perf_recipes.qwen.gb300.qwen3_moe import (
    qwen3_30b_a3b_pretrain_8gpu_gb300_bf16_config,
    qwen3_30b_a3b_pretrain_8gpu_gb300_fp8mx_config,
    qwen3_235b_a22b_pretrain_256gpu_gb300_bf16_config,
    qwen3_235b_a22b_pretrain_256gpu_gb300_fp8mx_config,
    qwen3_235b_a22b_pretrain_256gpu_gb300_nvfp4_config,
)


def qwen3_235b_a22b_pretrain_256gpu_vr200_bf16_config() -> ConfigContainer:
    """Qwen3 235B A22B pretrain: 256× VR200, BF16 (alias of GB300)."""
    return qwen3_235b_a22b_pretrain_256gpu_gb300_bf16_config()


def qwen3_235b_a22b_pretrain_256gpu_vr200_fp8mx_config() -> ConfigContainer:
    """Qwen3 235B A22B pretrain: 256× VR200, FP8-MX (alias of GB300)."""
    return qwen3_235b_a22b_pretrain_256gpu_gb300_fp8mx_config()


def qwen3_235b_a22b_pretrain_256gpu_vr200_nvfp4_config() -> ConfigContainer:
    """Qwen3 235B A22B pretrain: 256× VR200, NVFP4 (alias of GB300)."""
    return qwen3_235b_a22b_pretrain_256gpu_gb300_nvfp4_config()


def qwen3_30b_a3b_pretrain_8gpu_vr200_bf16_config() -> ConfigContainer:
    """Qwen3 30B-A3B pretrain: 8× VR200, BF16 (alias of GB300)."""
    return qwen3_30b_a3b_pretrain_8gpu_gb300_bf16_config()


def qwen3_30b_a3b_pretrain_8gpu_vr200_fp8mx_config() -> ConfigContainer:
    """Qwen3 30B-A3B pretrain: 8× VR200, FP8-MX (alias of GB300)."""
    return qwen3_30b_a3b_pretrain_8gpu_gb300_fp8mx_config()
