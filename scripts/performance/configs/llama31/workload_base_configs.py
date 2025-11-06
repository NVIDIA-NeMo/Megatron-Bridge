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

from utils.utils import WorkloadBaseConfig


# Llama3.1 405B presets ---------------------------------------------------------

LLAMA31_405B_GB300_128GPUS_BF16_PARALLEL_CONFIG = WorkloadBaseConfig(
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None,
    global_batch_size=64,
    micro_batch_size=1,
    use_megatron_fsdp=True,
    cpu_offloading_num_layers=40,
)


LLAMA31_405B_GB300_128GPUS_FP8_CS_PARALLEL_CONFIG = WorkloadBaseConfig(
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None,
    global_batch_size=64,
    micro_batch_size=1,
    use_megatron_fsdp=True,
    cpu_offloading_num_layers=10,
)


LLAMA31_405B_GB300_128GPUS_FP8_MX_PARALLEL_CONFIG = WorkloadBaseConfig(
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=8,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None,
    global_batch_size=64,
    micro_batch_size=1,
)


LLAMA31_405B_GB200_128GPUS_BF16_PARALLEL_CONFIG = WorkloadBaseConfig(
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=8,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=8,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None,
    global_batch_size=64,
    micro_batch_size=1,
)


LLAMA31_405B_GB200_128GPUS_FP8_CS_PARALLEL_CONFIG = WorkloadBaseConfig(
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None,
    global_batch_size=64,
    micro_batch_size=1,
    use_megatron_fsdp=True,
    cpu_offloading_num_layers=95,
)


LLAMA31_405B_GB200_128GPUS_FP8_MX_PARALLEL_CONFIG = WorkloadBaseConfig(
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=8,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=8,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None,
    global_batch_size=64,
    micro_batch_size=1,
)


LLAMA31_405B_B200_128GPUS_BF16_PARALLEL_CONFIG = WorkloadBaseConfig(
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=8,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=8,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None,
    global_batch_size=64,
    micro_batch_size=1,
)


LLAMA31_405B_B200_128GPUS_FP8_CS_PARALLEL_CONFIG = WorkloadBaseConfig(
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=8,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=8,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None,
    global_batch_size=64,
    micro_batch_size=1,
)


LLAMA31_405B_B200_128GPUS_FP8_MX_PARALLEL_CONFIG = LLAMA31_405B_B200_128GPUS_FP8_CS_PARALLEL_CONFIG

LLAMA31_405B_H100_1024GPUS_BF16_PARALLEL_CONFIG = WorkloadBaseConfig(
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=8,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=8,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None,
    global_batch_size=512,
    micro_batch_size=1,
)


LLAMA31_405B_H100_1024GPUS_FP8_CS_PARALLEL_CONFIG = WorkloadBaseConfig(
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=8,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=8,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None,
    global_batch_size=512,
    micro_batch_size=1,
)


__all__ = [
    "LLAMA31_405B_GB300_128GPUS_BF16_PARALLEL_CONFIG",
    "LLAMA31_405B_GB300_128GPUS_FP8_CS_PARALLEL_CONFIG",
    "LLAMA31_405B_GB300_128GPUS_FP8_MX_PARALLEL_CONFIG",
    "LLAMA31_405B_GB200_128GPUS_BF16_PARALLEL_CONFIG",
    "LLAMA31_405B_GB200_128GPUS_FP8_CS_PARALLEL_CONFIG",
    "LLAMA31_405B_GB200_128GPUS_FP8_MX_PARALLEL_CONFIG",
    "LLAMA31_405B_B200_128GPUS_BF16_PARALLEL_CONFIG",
    "LLAMA31_405B_B200_128GPUS_FP8_CS_PARALLEL_CONFIG",
    "LLAMA31_405B_B200_128GPUS_FP8_MX_PARALLEL_CONFIG",
    "LLAMA31_405B_H100_1024GPUS_BF16_PARALLEL_CONFIG",
    "LLAMA31_405B_H100_1024GPUS_FP8_CS_PARALLEL_CONFIG",
]
