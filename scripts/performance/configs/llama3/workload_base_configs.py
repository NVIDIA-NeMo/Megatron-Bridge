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

"""Parallelism presets for Llama3 performance configs."""

from utils.utils import WorkloadBaseConfig


# Llama3 70B presets ---------------------------------------------------------

LLAMA3_70B_GB300_64GPUS_BF16_PARALLEL_CONFIG = WorkloadBaseConfig(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None,
    global_batch_size=128,
    micro_batch_size=2,
    use_megatron_fsdp=True,
    cpu_offloading_num_layers=30,
)


LLAMA3_70B_GB300_64GPUS_FP8_CS_PARALLEL_CONFIG = WorkloadBaseConfig(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None,
    global_batch_size=128,
    micro_batch_size=2,
    use_megatron_fsdp=True,
    cpu_offloading_num_layers=20,
)


LLAMA3_70B_GB300_64GPUS_FP8_MX_PARALLEL_CONFIG = WorkloadBaseConfig(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=4,
    context_parallel_size=1,
    virtual_pipeline_model_parallel_size=5,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None,
    global_batch_size=128,
    micro_batch_size=1,
)


LLAMA3_70B_GB200_64GPUS_BF16_PARALLEL_CONFIG = WorkloadBaseConfig(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None,
    global_batch_size=128,
    micro_batch_size=1,
    use_megatron_fsdp=True,
    cpu_offloading_num_layers=20,
)


LLAMA3_70B_GB200_64GPUS_FP8_CS_PARALLEL_CONFIG = WorkloadBaseConfig(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None,
    global_batch_size=128,
    micro_batch_size=2,
    use_megatron_fsdp=True,
    cpu_offloading_num_layers=40,
)


LLAMA3_70B_GB200_64GPUS_FP8_MX_PARALLEL_CONFIG = WorkloadBaseConfig(
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=4,
    context_parallel_size=1,
    virtual_pipeline_model_parallel_size=5,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None,
    global_batch_size=128,
    micro_batch_size=1,
)


LLAMA3_70B_B200_64GPUS_BF16_PARALLEL_CONFIG = WorkloadBaseConfig(
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=4,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=5,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None,
    global_batch_size=128,
    micro_batch_size=1,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)


LLAMA3_70B_B200_64GPUS_FP8_CS_PARALLEL_CONFIG = WorkloadBaseConfig(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None,
    global_batch_size=128,
    micro_batch_size=1,
    use_megatron_fsdp=True,
    cpu_offloading_num_layers=5,
)


LLAMA3_70B_B200_64GPUS_FP8_MX_PARALLEL_CONFIG = WorkloadBaseConfig(
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=4,
    context_parallel_size=1,
    virtual_pipeline_model_parallel_size=5,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None,
    global_batch_size=128,
    micro_batch_size=1,
)


LLAMA3_70B_H100_64GPUS_BF16_PARALLEL_CONFIG = WorkloadBaseConfig(
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=4,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=5,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None,
    global_batch_size=128,
    micro_batch_size=1,
)


LLAMA3_70B_H100_64GPUS_FP8_CS_PARALLEL_CONFIG = WorkloadBaseConfig(
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=8,
    context_parallel_size=1,
    virtual_pipeline_model_parallel_size=5,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None,
    global_batch_size=128,
    micro_batch_size=1,
)

# Llama3 8B presets ---------------------------------------------------------


LLAMA3_8B_GB300_8GPUS_BF16_PARALLEL_CONFIG = WorkloadBaseConfig(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None,
    global_batch_size=128,
    micro_batch_size=4,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)


LLAMA3_8B_GB300_8GPUS_FP8_CS_PARALLEL_CONFIG = WorkloadBaseConfig(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None,
    global_batch_size=128,
    micro_batch_size=4,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)

LLAMA3_8B_GB300_8GPUS_FP8_MX_PARALLEL_CONFIG = LLAMA3_8B_GB300_8GPUS_FP8_CS_PARALLEL_CONFIG


LLAMA3_8B_GB200_8GPUS_BF16_PARALLEL_CONFIG = WorkloadBaseConfig(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None,
    global_batch_size=128,
    micro_batch_size=2,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)


LLAMA3_8B_GB200_8GPUS_FP8_CS_PARALLEL_CONFIG = WorkloadBaseConfig(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None,
    global_batch_size=128,
    micro_batch_size=2,
)

LLAMA3_8B_GB200_8GPUS_FP8_MX_PARALLEL_CONFIG = WorkloadBaseConfig(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None,
    global_batch_size=128,
    micro_batch_size=2,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)


LLAMA3_8B_B200_8GPUS_BF16_PARALLEL_CONFIG = WorkloadBaseConfig(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None,
    global_batch_size=128,
    micro_batch_size=2,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)


LLAMA3_8B_B200_8GPUS_FP8_CS_PARALLEL_CONFIG = WorkloadBaseConfig(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None,
    global_batch_size=128,
    micro_batch_size=2,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)


LLAMA3_8B_B200_8GPUS_FP8_MX_PARALLEL_CONFIG = LLAMA3_8B_B200_8GPUS_FP8_CS_PARALLEL_CONFIG

LLAMA3_8B_H100_8GPUS_BF16_PARALLEL_CONFIG = WorkloadBaseConfig(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=None,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None,
    global_batch_size=128,
    micro_batch_size=1,
)


LLAMA3_8B_H100_8GPUS_FP8_CS_PARALLEL_CONFIG = WorkloadBaseConfig(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None,
    global_batch_size=128,
    micro_batch_size=1,
    use_megatron_fsdp=True,
)


__all__ = [
    "LLAMA3_70B_GB300_64GPUS_BF16_PARALLEL_CONFIG",
    "LLAMA3_70B_GB300_64GPUS_FP8_CS_PARALLEL_CONFIG",
    "LLAMA3_70B_GB300_64GPUS_FP8_MX_PARALLEL_CONFIG",
    "LLAMA3_70B_GB200_64GPUS_BF16_PARALLEL_CONFIG",
    "LLAMA3_70B_GB200_64GPUS_FP8_CS_PARALLEL_CONFIG",
    "LLAMA3_70B_GB200_64GPUS_FP8_MX_PARALLEL_CONFIG",
    "LLAMA3_70B_B200_64GPUS_BF16_PARALLEL_CONFIG",
    "LLAMA3_70B_B200_64GPUS_FP8_CS_PARALLEL_CONFIG",
    "LLAMA3_70B_B200_64GPUS_FP8_MX_PARALLEL_CONFIG",
    "LLAMA3_70B_H100_64GPUS_BF16_PARALLEL_CONFIG",
    "LLAMA3_70B_H100_64GPUS_FP8_CS_PARALLEL_CONFIG",
    "LLAMA3_8B_GB300_8GPUS_BF16_PARALLEL_CONFIG",
    "LLAMA3_8B_GB300_8GPUS_FP8_CS_PARALLEL_CONFIG",
    "LLAMA3_8B_GB300_8GPUS_FP8_MX_PARALLEL_CONFIG",
    "LLAMA3_8B_GB200_8GPUS_BF16_PARALLEL_CONFIG",
    "LLAMA3_8B_GB200_8GPUS_FP8_CS_PARALLEL_CONFIG",
    "LLAMA3_8B_GB200_8GPUS_FP8_MX_PARALLEL_CONFIG",
    "LLAMA3_8B_B200_8GPUS_BF16_PARALLEL_CONFIG",
    "LLAMA3_8B_B200_8GPUS_FP8_CS_PARALLEL_CONFIG",
    "LLAMA3_8B_B200_8GPUS_FP8_MX_PARALLEL_CONFIG",
    "LLAMA3_8B_H100_8GPUS_BF16_PARALLEL_CONFIG",
    "LLAMA3_8B_H100_8GPUS_FP8_CS_PARALLEL_CONFIG",
]
