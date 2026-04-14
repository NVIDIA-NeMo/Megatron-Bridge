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

"""Parallelism presets for Wan 14B performance configs.

Config naming convention:
    {MODEL}_{SIZE}_{TASK}_CONFIG_{GPU}_{PRECISION}_{VERSION}

All configs use bf16 precision (diffusion training does not use fp8).
Parallelism settings are sourced from the per-GPU YAML perf configs in
examples/diffusion/recipes/wan/conf/.
"""

from dataclasses import replace

from utils.utils import WorkloadBaseConfig


BASE_WAN_14B_CONFIG = WorkloadBaseConfig(
    num_gpus=8,
    global_batch_size=64,
    micro_batch_size=1,
)

# =============================================================================
# Wan 14B pretrain presets
# =============================================================================

# GB200: 16 GPUs (4 nodes), TP=1, CP=4, DP=4, GBS=64
WAN_14B_PRETRAIN_CONFIG_GB200_BF16_V1 = replace(
    BASE_WAN_14B_CONFIG,
    num_gpus=16,
    tensor_model_parallel_size=1,
    context_parallel_size=4,
)

# H100: 32 GPUs (4 nodes), TP=2, CP=4, DP=4, GBS=64, activation recompute (block/8 layers)
WAN_14B_PRETRAIN_CONFIG_H100_BF16_V1 = replace(
    BASE_WAN_14B_CONFIG,
    num_gpus=32,
    tensor_model_parallel_size=2,
    context_parallel_size=4,
    recompute_num_layers=8,
)
