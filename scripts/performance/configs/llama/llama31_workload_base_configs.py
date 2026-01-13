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

"""Parallelism presets for Llama3.1 performance configs.

Config naming convention:
    {MODEL}_{SIZE}_{TASK}_CONFIG_{GPU}_{PRECISION}_{VERSION}

Use --config_variant to select a variant (default: v1).
Use --list_config_variants to see available variants interactively.
"""

from dataclasses import replace

from utils.utils import WorkloadBaseConfig


BASE_LLAMA31_405B_CONFIG = WorkloadBaseConfig()


# Llama3.1 405B presets - V2 (original) ---------------------------------------------------------

LLAMA31_405B_PRETRAIN_CONFIG_GB300_BF16_V2 = replace(
    BASE_LLAMA31_405B_CONFIG,
    num_gpus=128,
    tensor_model_parallel_size=2,
    global_batch_size=64,
    use_megatron_fsdp=True,
    cpu_offloading_num_layers=40,
)


LLAMA31_405B_PRETRAIN_CONFIG_GB300_FP8_CS_V2 = replace(
    BASE_LLAMA31_405B_CONFIG,
    num_gpus=128,
    tensor_model_parallel_size=2,
    global_batch_size=64,
    use_megatron_fsdp=True,
    cpu_offloading_num_layers=10,
)


LLAMA31_405B_PRETRAIN_CONFIG_GB300_FP8_MX_V2 = replace(
    BASE_LLAMA31_405B_CONFIG,
    num_gpus=128,
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=8,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=4,
    global_batch_size=64,
)

LLAMA31_405B_PRETRAIN_CONFIG_GB300_NVFP4_V2 = replace(
    BASE_LLAMA31_405B_CONFIG,
    num_gpus=128,
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=8,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=4,
    global_batch_size=64,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)


LLAMA31_405B_PRETRAIN_CONFIG_GB200_BF16_V2 = replace(
    BASE_LLAMA31_405B_CONFIG,
    num_gpus=128,
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=8,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=8,
    global_batch_size=64,
)


LLAMA31_405B_PRETRAIN_CONFIG_GB200_FP8_CS_V2 = replace(
    BASE_LLAMA31_405B_CONFIG,
    num_gpus=128,
    tensor_model_parallel_size=2,
    global_batch_size=64,
    use_megatron_fsdp=True,
    cpu_offloading_num_layers=92,
)


LLAMA31_405B_PRETRAIN_CONFIG_GB200_FP8_MX_V2 = replace(
    BASE_LLAMA31_405B_CONFIG,
    num_gpus=128,
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=8,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=8,
    global_batch_size=64,
)

LLAMA31_405B_PRETRAIN_CONFIG_GB200_NVFP4_V2 = replace(
    BASE_LLAMA31_405B_CONFIG,
    num_gpus=128,
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=8,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=4,
    global_batch_size=64,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)


LLAMA31_405B_PRETRAIN_CONFIG_B300_BF16_V1 = replace(
    BASE_LLAMA31_405B_CONFIG,
    num_gpus=128,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=8,
    context_parallel_size=1,
    virtual_pipeline_model_parallel_size=8,
    global_batch_size=64,
)


LLAMA31_405B_PRETRAIN_CONFIG_B300_FP8_CS_V1 = replace(
    BASE_LLAMA31_405B_CONFIG,
    num_gpus=128,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=8,
    context_parallel_size=1,
    virtual_pipeline_model_parallel_size=8,
    global_batch_size=64,
)


LLAMA31_405B_PRETRAIN_CONFIG_B300_FP8_MX_V1 = LLAMA31_405B_PRETRAIN_CONFIG_B300_FP8_CS_V1

LLAMA31_405B_PRETRAIN_CONFIG_B300_NVFP4_V1 = replace(
    BASE_LLAMA31_405B_CONFIG,
    num_gpus=128,
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=8,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=4,
    global_batch_size=64,
)


LLAMA31_405B_PRETRAIN_CONFIG_B200_BF16_V1 = replace(
    BASE_LLAMA31_405B_CONFIG,
    num_gpus=128,
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=8,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=8,
    global_batch_size=64,
)


LLAMA31_405B_PRETRAIN_CONFIG_B200_FP8_CS_V1 = replace(
    BASE_LLAMA31_405B_CONFIG,
    num_gpus=128,
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=8,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=8,
    global_batch_size=64,
)


LLAMA31_405B_PRETRAIN_CONFIG_B200_FP8_MX_V1 = LLAMA31_405B_PRETRAIN_CONFIG_B200_FP8_CS_V1

LLAMA31_405B_PRETRAIN_CONFIG_B200_NVFP4_V1 = replace(
    BASE_LLAMA31_405B_CONFIG,
    num_gpus=128,
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=8,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=4,
    global_batch_size=64,
)

LLAMA31_405B_PRETRAIN_CONFIG_H100_BF16_V2 = replace(
    BASE_LLAMA31_405B_CONFIG,
    num_gpus=1024,
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=8,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=8,
    global_batch_size=512,
)


LLAMA31_405B_PRETRAIN_CONFIG_H100_FP8_CS_V2 = replace(
    BASE_LLAMA31_405B_CONFIG,
    num_gpus=1024,
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=8,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=8,
    global_batch_size=512,
)


# =============================================================================
# Llama3.1 405B presets - V1 (default, GB300/GB200: num_gpus=256, GBS=1536; H100: GBS=1536)
# =============================================================================

LLAMA31_405B_PRETRAIN_CONFIG_GB300_BF16_V1 = replace(
    LLAMA31_405B_PRETRAIN_CONFIG_GB300_BF16_V2,
    num_gpus=256,
    global_batch_size=1536,
)


LLAMA31_405B_PRETRAIN_CONFIG_GB300_FP8_CS_V1 = replace(
    LLAMA31_405B_PRETRAIN_CONFIG_GB300_FP8_CS_V2,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=8,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=4,
    num_gpus=256,
    global_batch_size=1536,
    use_megatron_fsdp=False,
    cpu_offloading_num_layers=None,
)


LLAMA31_405B_PRETRAIN_CONFIG_GB300_FP8_MX_V1 = replace(
    LLAMA31_405B_PRETRAIN_CONFIG_GB300_FP8_MX_V2,
    tensor_model_parallel_size=2,
    num_gpus=256,
    global_batch_size=1536,
)

LLAMA31_405B_PRETRAIN_CONFIG_GB300_NVFP4_V1 = replace(
    LLAMA31_405B_PRETRAIN_CONFIG_GB300_NVFP4_V2,
    tensor_model_parallel_size=2,
    num_gpus=256,
    global_batch_size=1536,
)


LLAMA31_405B_PRETRAIN_CONFIG_GB200_BF16_V1 = replace(
    LLAMA31_405B_PRETRAIN_CONFIG_GB200_BF16_V2,
    num_gpus=256,
    pipeline_model_parallel_size=16,
    context_parallel_size=1,
    global_batch_size=1536,
)


LLAMA31_405B_PRETRAIN_CONFIG_GB200_FP8_CS_V1 = replace(
    LLAMA31_405B_PRETRAIN_CONFIG_GB200_FP8_CS_V2,
    num_gpus=256,
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=16,
    context_parallel_size=1,
    virtual_pipeline_model_parallel_size=4,
    global_batch_size=1536,
    use_megatron_fsdp=False,
    cpu_offloading_num_layers=None,
)


LLAMA31_405B_PRETRAIN_CONFIG_GB200_FP8_MX_V1 = replace(
    LLAMA31_405B_PRETRAIN_CONFIG_GB200_FP8_MX_V2,
    num_gpus=256,
    pipeline_model_parallel_size=16,
    context_parallel_size=1,
    global_batch_size=1536,
)

LLAMA31_405B_PRETRAIN_CONFIG_GB200_NVFP4_V1 = replace(
    LLAMA31_405B_PRETRAIN_CONFIG_GB200_NVFP4_V2,
    num_gpus=256,
    pipeline_model_parallel_size=16,
    context_parallel_size=1,
    global_batch_size=1536,
)


LLAMA31_405B_PRETRAIN_CONFIG_H100_BF16_V1 = replace(
    LLAMA31_405B_PRETRAIN_CONFIG_H100_BF16_V2,
    global_batch_size=1536,
)


LLAMA31_405B_PRETRAIN_CONFIG_H100_FP8_CS_V1 = replace(
    LLAMA31_405B_PRETRAIN_CONFIG_H100_FP8_CS_V2,
    global_batch_size=1536,
)


__all__ = [
    # V1 (GB300/GB200: num_gpus=256, GBS=1536; H100: GBS=1536)
    "LLAMA31_405B_PRETRAIN_CONFIG_GB300_BF16_V1",
    "LLAMA31_405B_PRETRAIN_CONFIG_GB300_FP8_CS_V1",
    "LLAMA31_405B_PRETRAIN_CONFIG_GB300_FP8_MX_V1",
    "LLAMA31_405B_PRETRAIN_CONFIG_GB300_NVFP4_V1",
    "LLAMA31_405B_PRETRAIN_CONFIG_GB200_BF16_V1",
    "LLAMA31_405B_PRETRAIN_CONFIG_GB200_FP8_CS_V1",
    "LLAMA31_405B_PRETRAIN_CONFIG_GB200_FP8_MX_V1",
    "LLAMA31_405B_PRETRAIN_CONFIG_GB200_NVFP4_V1",
    "LLAMA31_405B_PRETRAIN_CONFIG_H100_BF16_V1",
    "LLAMA31_405B_PRETRAIN_CONFIG_H100_FP8_CS_V1",
    # V2 (original)
    "LLAMA31_405B_PRETRAIN_CONFIG_GB300_BF16_V2",
    "LLAMA31_405B_PRETRAIN_CONFIG_GB300_FP8_CS_V2",
    "LLAMA31_405B_PRETRAIN_CONFIG_GB300_FP8_MX_V2",
    "LLAMA31_405B_PRETRAIN_CONFIG_GB300_NVFP4_V2",
    "LLAMA31_405B_PRETRAIN_CONFIG_GB200_BF16_V2",
    "LLAMA31_405B_PRETRAIN_CONFIG_GB200_FP8_CS_V2",
    "LLAMA31_405B_PRETRAIN_CONFIG_GB200_FP8_MX_V2",
    "LLAMA31_405B_PRETRAIN_CONFIG_GB200_NVFP4_V2",
    "LLAMA31_405B_PRETRAIN_CONFIG_B300_BF16_V1",
    "LLAMA31_405B_PRETRAIN_CONFIG_B300_FP8_CS_V1",
    "LLAMA31_405B_PRETRAIN_CONFIG_B300_FP8_MX_V1",
    "LLAMA31_405B_PRETRAIN_CONFIG_B300_NVFP4_V1",
    "LLAMA31_405B_PRETRAIN_CONFIG_B200_BF16_V1",
    "LLAMA31_405B_PRETRAIN_CONFIG_B200_FP8_CS_V1",
    "LLAMA31_405B_PRETRAIN_CONFIG_B200_FP8_MX_V1",
    "LLAMA31_405B_PRETRAIN_CONFIG_B200_NVFP4_V1",
    "LLAMA31_405B_PRETRAIN_CONFIG_H100_BF16_V2",
    "LLAMA31_405B_PRETRAIN_CONFIG_H100_FP8_CS_V2",
]
