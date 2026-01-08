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

"""Parallelism presets for Llama3.1 performance configs."""

from dataclasses import replace

from utils.utils import WorkloadBaseConfig


BASE_LLAMA31_405B_CONFIG = WorkloadBaseConfig()


# Llama3.1 405B presets ---------------------------------------------------------

LLAMA31_405B_PRETRAIN_CONFIG_GB300_BF16 = replace(
    BASE_LLAMA31_405B_CONFIG,
    num_gpus=128,
    tensor_model_parallel_size=2,
    global_batch_size=64,
    use_megatron_fsdp=True,
    cpu_offloading_num_layers=40,
)


LLAMA31_405B_PRETRAIN_CONFIG_GB300_FP8_CS = replace(
    BASE_LLAMA31_405B_CONFIG,
    num_gpus=128,
    tensor_model_parallel_size=2,
    global_batch_size=64,
    use_megatron_fsdp=True,
    cpu_offloading_num_layers=10,
)


LLAMA31_405B_PRETRAIN_CONFIG_GB300_FP8_MX = replace(
    BASE_LLAMA31_405B_CONFIG,
    num_gpus=128,
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=8,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=4,
    global_batch_size=64,
)

LLAMA31_405B_PRETRAIN_CONFIG_GB300_NVFP4 = replace(
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


LLAMA31_405B_PRETRAIN_CONFIG_GB200_BF16 = replace(
    BASE_LLAMA31_405B_CONFIG,
    num_gpus=128,
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=8,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=8,
    global_batch_size=64,
)


LLAMA31_405B_PRETRAIN_CONFIG_GB200_FP8_CS = replace(
    BASE_LLAMA31_405B_CONFIG,
    num_gpus=128,
    tensor_model_parallel_size=2,
    global_batch_size=64,
    use_megatron_fsdp=True,
    cpu_offloading_num_layers=92,
)


LLAMA31_405B_PRETRAIN_CONFIG_GB200_FP8_MX = replace(
    BASE_LLAMA31_405B_CONFIG,
    num_gpus=128,
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=8,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=8,
    global_batch_size=64,
)

LLAMA31_405B_PRETRAIN_CONFIG_GB200_NVFP4 = replace(
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


LLAMA31_405B_PRETRAIN_CONFIG_B300_BF16 = replace(
    BASE_LLAMA31_405B_CONFIG,
    num_gpus=128,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=8,
    context_parallel_size=1,
    virtual_pipeline_model_parallel_size=8,
    global_batch_size=64,
)


LLAMA31_405B_PRETRAIN_CONFIG_B300_FP8_CS = replace(
    BASE_LLAMA31_405B_CONFIG,
    num_gpus=128,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=8,
    context_parallel_size=1,
    virtual_pipeline_model_parallel_size=8,
    global_batch_size=64,
)


LLAMA31_405B_PRETRAIN_CONFIG_B300_FP8_MX = LLAMA31_405B_PRETRAIN_CONFIG_B300_FP8_CS

LLAMA31_405B_PRETRAIN_CONFIG_B300_NVFP4 = replace(
    BASE_LLAMA31_405B_CONFIG,
    num_gpus=128,
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=8,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=4,
    global_batch_size=64,
)


LLAMA31_405B_PRETRAIN_CONFIG_B200_BF16 = replace(
    BASE_LLAMA31_405B_CONFIG,
    num_gpus=128,
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=8,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=8,
    global_batch_size=64,
)


LLAMA31_405B_PRETRAIN_CONFIG_B200_FP8_CS = replace(
    BASE_LLAMA31_405B_CONFIG,
    num_gpus=128,
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=8,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=8,
    global_batch_size=64,
)


LLAMA31_405B_PRETRAIN_CONFIG_B200_FP8_MX = LLAMA31_405B_PRETRAIN_CONFIG_B200_FP8_CS

LLAMA31_405B_PRETRAIN_CONFIG_B200_NVFP4 = replace(
    BASE_LLAMA31_405B_CONFIG,
    num_gpus=128,
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=8,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=4,
    global_batch_size=64,
)

LLAMA31_405B_PRETRAIN_CONFIG_H100_BF16 = replace(
    BASE_LLAMA31_405B_CONFIG,
    num_gpus=1024,
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=8,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=8,
    global_batch_size=512,
)


LLAMA31_405B_PRETRAIN_CONFIG_H100_FP8_CS = replace(
    BASE_LLAMA31_405B_CONFIG,
    num_gpus=1024,
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=8,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=8,
    global_batch_size=512,
)


__all__ = [
    "LLAMA31_405B_PRETRAIN_CONFIG_GB300_BF16",
    "LLAMA31_405B_PRETRAIN_CONFIG_GB300_FP8_CS",
    "LLAMA31_405B_PRETRAIN_CONFIG_GB300_FP8_MX",
    "LLAMA31_405B_PRETRAIN_CONFIG_GB300_NVFP4",
    "LLAMA31_405B_PRETRAIN_CONFIG_GB200_BF16",
    "LLAMA31_405B_PRETRAIN_CONFIG_GB200_FP8_CS",
    "LLAMA31_405B_PRETRAIN_CONFIG_GB200_FP8_MX",
    "LLAMA31_405B_PRETRAIN_CONFIG_GB200_NVFP4",
    "LLAMA31_405B_PRETRAIN_CONFIG_B300_BF16",
    "LLAMA31_405B_PRETRAIN_CONFIG_B300_FP8_CS",
    "LLAMA31_405B_PRETRAIN_CONFIG_B300_FP8_MX",
    "LLAMA31_405B_PRETRAIN_CONFIG_B300_NVFP4",
    "LLAMA31_405B_PRETRAIN_CONFIG_B200_BF16",
    "LLAMA31_405B_PRETRAIN_CONFIG_B200_FP8_CS",
    "LLAMA31_405B_PRETRAIN_CONFIG_B200_FP8_MX",
    "LLAMA31_405B_PRETRAIN_CONFIG_B200_NVFP4",
    "LLAMA31_405B_PRETRAIN_CONFIG_H100_BF16",
    "LLAMA31_405B_PRETRAIN_CONFIG_H100_FP8_CS",
]
