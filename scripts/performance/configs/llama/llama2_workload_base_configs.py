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

"""Parallelism presets for Llama3 performance configs.

Config naming convention:
    {MODEL}_{SIZE}_{TASK}_CONFIG_{GPU}_{PRECISION}_{VERSION}

Use --config_variant to select a variant.
Use --list_config_variants to see available variants interactively.
"""

from dataclasses import replace

from utils.utils import WorkloadBaseConfig

# =============================================================================
# Llama2 70B finetune (LoRA) presets - MLPerf
# =============================================================================

BASE_LLAMA2_70B_CONFIG_MLPERF = WorkloadBaseConfig(
    num_gpus=8,
    peft="lora",
    global_batch_size=8,
    micro_batch_size=1,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
    cpu_offloading_num_layers=20,
)

LLAMA2_70B_LORA_CONFIG_GB200_FP8_DS_V1 = replace(
    BASE_LLAMA2_70B_CONFIG_MLPERF,
    num_gpus=4,
    cuda_graph_impl=None,
    cuda_graph_scope=None,
)

LLAMA2_70B_LORA_CONFIG_GB200_NVFP4_V1 = replace(
    BASE_LLAMA2_70B_CONFIG_MLPERF,
    context_parallel_size=2,
)

LLAMA2_70B_LORA_CONFIG_GB200_FP8_DS_V2 = replace(
    BASE_LLAMA2_70B_CONFIG_MLPERF,
    cpu_offloading_num_layers=11,
)

LLAMA2_70B_LORA_CONFIG_GB200_FP8_DS_V3 = replace(
    BASE_LLAMA2_70B_CONFIG_MLPERF,
    num_gpus=72,
    context_parallel_size=8,
    global_batch_size=9,
)

LLAMA2_70B_LORA_CONFIG_GB200_FP8_DS_V4 = replace(
    BASE_LLAMA2_70B_CONFIG_MLPERF,
    num_gpus=512,
    context_parallel_size=8,
    global_batch_size=64,
)

LLAMA2_70B_LORA_CONFIG_GB300_FP8_DS_V1 = replace(
    BASE_LLAMA2_70B_CONFIG_MLPERF,
    num_gpus=4,
)

LLAMA2_70B_LORA_CONFIG_GB300_FP8_DS_V2 = BASE_LLAMA2_70B_CONFIG_MLPERF
LLAMA2_70B_LORA_CONFIG_GB300_FP8_DS_V3 = LLAMA2_70B_LORA_CONFIG_GB200_FP8_DS_V3
LLAMA2_70B_LORA_CONFIG_GB300_FP8_DS_V4 = LLAMA2_70B_LORA_CONFIG_GB200_FP8_DS_V4

__all__ = [
    # 70B LoRA MLPerf
    "LLAMA2_70B_LORA_CONFIG_GB200_FP8_DS_V1",
    "LLAMA2_70B_LORA_CONFIG_GB200_NVFP4_V1",
    "LLAMA2_70B_LORA_CONFIG_GB200_FP8_DS_V2",
    "LLAMA2_70B_LORA_CONFIG_GB200_FP8_DS_V3",
    "LLAMA2_70B_LORA_CONFIG_GB200_FP8_DS_V4",
    "LLAMA2_70B_LORA_CONFIG_GB300_FP8_DS_V1",
    "LLAMA2_70B_LORA_CONFIG_GB300_FP8_DS_V2",
    "LLAMA2_70B_LORA_CONFIG_GB300_FP8_DS_V3",
    "LLAMA2_70B_LORA_CONFIG_GB300_FP8_DS_V4",
]