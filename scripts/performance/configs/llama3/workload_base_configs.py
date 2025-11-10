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

from dataclasses import replace

from utils.utils import WorkloadBaseConfig


BASE_LLAMA3_8B_CONFIG = WorkloadBaseConfig(
    num_gpus=8,
    global_batch_size=128,
)


BASE_LLAMA3_70B_CONFIG = WorkloadBaseConfig(
    num_gpus=64,
    global_batch_size=128,
)

# Llama3 70B presets ---------------------------------------------------------

LLAMA3_70B_GB300_BF16_BASE_CONFIG = replace(
    BASE_LLAMA3_70B_CONFIG,
    micro_batch_size=2,
    use_megatron_fsdp=True,
    cpu_offloading_num_layers=30,
)


LLAMA3_70B_GB300_FP8_CS_BASE_CONFIG = replace(
    BASE_LLAMA3_70B_CONFIG,
    micro_batch_size=2,
    use_megatron_fsdp=True,
    cpu_offloading_num_layers=20,
)


LLAMA3_70B_GB300_FP8_MX_BASE_CONFIG = replace(
    BASE_LLAMA3_70B_CONFIG,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=5,
)


LLAMA3_70B_GB200_BF16_BASE_CONFIG = replace(
    BASE_LLAMA3_70B_CONFIG,
    use_megatron_fsdp=True,
    cpu_offloading_num_layers=20,
)


LLAMA3_70B_GB200_FP8_CS_BASE_CONFIG = replace(
    BASE_LLAMA3_70B_CONFIG,
    micro_batch_size=2,
    use_megatron_fsdp=True,
    cpu_offloading_num_layers=40,
)


LLAMA3_70B_GB200_FP8_MX_BASE_CONFIG = replace(
    BASE_LLAMA3_70B_CONFIG,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=5,
)


LLAMA3_70B_B200_BF16_BASE_CONFIG = replace(
    BASE_LLAMA3_70B_CONFIG,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=4,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=5,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)


LLAMA3_70B_B200_FP8_CS_BASE_CONFIG = replace(
    BASE_LLAMA3_70B_CONFIG,
    use_megatron_fsdp=True,
    cpu_offloading_num_layers=5,
)


LLAMA3_70B_B200_FP8_MX_BASE_CONFIG = replace(
    BASE_LLAMA3_70B_CONFIG,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=5,
)


LLAMA3_70B_H100_BF16_BASE_CONFIG = replace(
    BASE_LLAMA3_70B_CONFIG,
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=4,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=5,
)


LLAMA3_70B_H100_FP8_CS_BASE_CONFIG = replace(
    BASE_LLAMA3_70B_CONFIG,
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=5,
)

# Llama3 8B presets ---------------------------------------------------------


LLAMA3_8B_GB300_BF16_BASE_CONFIG = replace(
    BASE_LLAMA3_8B_CONFIG,
    micro_batch_size=4,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)


LLAMA3_8B_GB300_FP8_CS_BASE_CONFIG = replace(
    BASE_LLAMA3_8B_CONFIG,
    micro_batch_size=4,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)

LLAMA3_8B_GB300_FP8_MX_BASE_CONFIG = LLAMA3_8B_GB300_FP8_CS_BASE_CONFIG


LLAMA3_8B_GB200_BF16_BASE_CONFIG = replace(
    BASE_LLAMA3_8B_CONFIG,
    micro_batch_size=2,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)


LLAMA3_8B_GB200_FP8_CS_BASE_CONFIG = replace(
    BASE_LLAMA3_8B_CONFIG,
    micro_batch_size=2,
)

LLAMA3_8B_GB200_FP8_MX_BASE_CONFIG = replace(
    BASE_LLAMA3_8B_CONFIG,
    micro_batch_size=2,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)


LLAMA3_8B_B200_BF16_BASE_CONFIG = replace(
    BASE_LLAMA3_8B_CONFIG,
    micro_batch_size=2,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)


LLAMA3_8B_B200_FP8_CS_BASE_CONFIG = replace(
    BASE_LLAMA3_8B_CONFIG,
    micro_batch_size=2,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)


LLAMA3_8B_B200_FP8_MX_BASE_CONFIG = LLAMA3_8B_B200_FP8_CS_BASE_CONFIG

LLAMA3_8B_H100_BF16_BASE_CONFIG = replace(
    BASE_LLAMA3_8B_CONFIG,
    context_parallel_size=2,
)


LLAMA3_8B_H100_FP8_CS_BASE_CONFIG = replace(
    BASE_LLAMA3_8B_CONFIG,
    use_megatron_fsdp=True,
)


__all__ = [
    "LLAMA3_70B_GB300_BF16_BASE_CONFIG",
    "LLAMA3_70B_GB300_FP8_CS_BASE_CONFIG",
    "LLAMA3_70B_GB300_FP8_MX_BASE_CONFIG",
    "LLAMA3_70B_GB200_BF16_BASE_CONFIG",
    "LLAMA3_70B_GB200_FP8_CS_BASE_CONFIG",
    "LLAMA3_70B_GB200_FP8_MX_BASE_CONFIG",
    "LLAMA3_70B_B200_BF16_BASE_CONFIG",
    "LLAMA3_70B_B200_FP8_CS_BASE_CONFIG",
    "LLAMA3_70B_B200_FP8_MX_BASE_CONFIG",
    "LLAMA3_70B_H100_BF16_BASE_CONFIG",
    "LLAMA3_70B_H100_FP8_CS_BASE_CONFIG",
    "LLAMA3_8B_GB300_BF16_BASE_CONFIG",
    "LLAMA3_8B_GB300_FP8_CS_BASE_CONFIG",
    "LLAMA3_8B_GB300_FP8_MX_BASE_CONFIG",
    "LLAMA3_8B_GB200_BF16_BASE_CONFIG",
    "LLAMA3_8B_GB200_FP8_CS_BASE_CONFIG",
    "LLAMA3_8B_GB200_FP8_MX_BASE_CONFIG",
    "LLAMA3_8B_B200_BF16_BASE_CONFIG",
    "LLAMA3_8B_B200_FP8_CS_BASE_CONFIG",
    "LLAMA3_8B_B200_FP8_MX_BASE_CONFIG",
    "LLAMA3_8B_H100_BF16_BASE_CONFIG",
    "LLAMA3_8B_H100_FP8_CS_BASE_CONFIG",
]
