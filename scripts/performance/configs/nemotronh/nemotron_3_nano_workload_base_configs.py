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

"""Parallelism presets for Nemotron 3 Nano performance configs.

Config naming convention:
    {MODEL}_{SIZE}_{TASK}_CONFIG_{GPU}_{PRECISION}_{VERSION}

V1: 30B_a3b

Use --config_variant to select a variant.
Use --list_config_variants to see available variants interactively.
"""

from dataclasses import replace

from utils.utils import WorkloadBaseConfig


BASE_NEMOTRON_3_NANO_CONFIG = WorkloadBaseConfig(
    num_gpus=8,
    global_batch_size=512,
    micro_batch_size=2,
    tensor_model_parallel_size=1,
    expert_tensor_parallel_size=1,
    expert_model_parallel_size=8,
    moe_flex_dispatcher_backend="hybridep",
)

NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB300_BF16_V1 = replace(
    BASE_NEMOTRON_3_NANO_CONFIG,
    tensor_model_parallel_size=1,
)
NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB300_FP8_MX_V1 = NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB300_BF16_V1
NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB300_NVFP4_V1 = NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB300_BF16_V1

NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB200_BF16_V1 = replace(
    BASE_NEMOTRON_3_NANO_CONFIG,
    tensor_model_parallel_size=1,
)
NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB200_FP8_MX_V1 = NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB200_BF16_V1
NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB200_NVFP4_V1 = NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB200_BF16_V1

NEMOTRON_3_NANO_PRETRAIN_CONFIG_B300_BF16_V1 = replace(
    BASE_NEMOTRON_3_NANO_CONFIG,
    tensor_model_parallel_size=1,
)
NEMOTRON_3_NANO_PRETRAIN_CONFIG_B300_FP8_MX_V1 = NEMOTRON_3_NANO_PRETRAIN_CONFIG_B300_BF16_V1
NEMOTRON_3_NANO_PRETRAIN_CONFIG_B300_NVFP4_V1 = NEMOTRON_3_NANO_PRETRAIN_CONFIG_B300_BF16_V1

NEMOTRON_3_NANO_PRETRAIN_CONFIG_B200_BF16_V1 = replace(
    BASE_NEMOTRON_3_NANO_CONFIG,
    tensor_model_parallel_size=1,
)
NEMOTRON_3_NANO_PRETRAIN_CONFIG_B200_FP8_MX_V1 = NEMOTRON_3_NANO_PRETRAIN_CONFIG_B200_BF16_V1
NEMOTRON_3_NANO_PRETRAIN_CONFIG_B200_NVFP4_V1 = NEMOTRON_3_NANO_PRETRAIN_CONFIG_B200_BF16_V1

_NEMOTRON_3_NANO_PRETRAIN_CONFIG_H100 = replace(
    BASE_NEMOTRON_3_NANO_CONFIG,
    num_gpus=16,
    global_batch_size=1024,
    micro_batch_size=1,
    recompute_modules=["moe", "layernorm"],
)

NEMOTRON_3_NANO_PRETRAIN_CONFIG_H100_BF16_V1 = BASE_NEMOTRON_3_NANO_CONFIG
NEMOTRON_3_NANO_PRETRAIN_CONFIG_H100_FP8_CS_V1 = BASE_NEMOTRON_3_NANO_CONFIG

__all__ = [
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB300_BF16_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB300_FP8_MX_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB300_NVFP4_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB200_BF16_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB200_FP8_MX_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB200_NVFP4_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_B300_BF16_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_B300_FP8_MX_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_B300_NVFP4_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_B200_BF16_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_B200_FP8_MX_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_B200_NVFP4_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_H100_BF16_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_H100_FP8_CS_V1",
]
