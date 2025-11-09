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

"""Parallelism presets for Qwen3 performance configs."""

from dataclasses import replace

from utils.utils import WorkloadBaseConfig


BASE_QWEN3_235B_A22B_CONFIG = WorkloadBaseConfig(
    expert_tensor_parallel_size=1,
)


BASE_QWEN3_30B_A3B_CONFIG = WorkloadBaseConfig(
    expert_model_parallel_size=8,
    expert_tensor_parallel_size=1,
    global_batch_size=512,
)


# Qwen3 235B A22B presets ----------------------------------------------------


QWEN3_235B_A22B_GB300_BF16_BASE_CONFIG = replace(
    BASE_QWEN3_235B_A22B_CONFIG,
    num_gpus=64,
    tensor_model_parallel_size=2,
    expert_model_parallel_size=64,
    global_batch_size=1024,
    micro_batch_size=4,
    cuda_graph_impl="none",
    cuda_graph_scope="full_iteration",
)


QWEN3_235B_A22B_GB300_FP8_CS_BASE_CONFIG = replace(
    BASE_QWEN3_235B_A22B_CONFIG,
    num_gpus=64,
    tensor_model_parallel_size=2,
    expert_model_parallel_size=64,
    global_batch_size=1024,
    micro_batch_size=4,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)


QWEN3_235B_A22B_GB300_FP8_MX_BASE_CONFIG = QWEN3_235B_A22B_GB300_FP8_CS_BASE_CONFIG


QWEN3_235B_A22B_GB200_BF16_BASE_CONFIG = replace(
    BASE_QWEN3_235B_A22B_CONFIG,
    num_gpus=64,
    pipeline_model_parallel_size=8,
    expert_model_parallel_size=8,
    global_batch_size=1024,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)


QWEN3_235B_A22B_GB200_FP8_CS_BASE_CONFIG = replace(
    BASE_QWEN3_235B_A22B_CONFIG,
    num_gpus=64,
    pipeline_model_parallel_size=8,
    expert_model_parallel_size=8,
    global_batch_size=1024,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)


QWEN3_235B_A22B_GB200_FP8_MX_BASE_CONFIG = QWEN3_235B_A22B_GB200_FP8_CS_BASE_CONFIG


QWEN3_235B_A22B_B200_BF16_BASE_CONFIG = replace(
    BASE_QWEN3_235B_A22B_CONFIG,
    num_gpus=64,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=2,
    expert_model_parallel_size=8,
    global_batch_size=1024,
)


QWEN3_235B_A22B_B200_FP8_CS_BASE_CONFIG = replace(
    BASE_QWEN3_235B_A22B_CONFIG,
    num_gpus=64,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=2,
    expert_model_parallel_size=8,
    global_batch_size=1024,
)


QWEN3_235B_A22B_B200_FP8_MX_BASE_CONFIG = QWEN3_235B_A22B_B200_FP8_CS_BASE_CONFIG


QWEN3_235B_A22B_H100_BF16_BASE_CONFIG = replace(
    BASE_QWEN3_235B_A22B_CONFIG,
    num_gpus=256,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=32,
    global_batch_size=2048,
)


QWEN3_235B_A22B_H100_FP8_CS_BASE_CONFIG = replace(
    BASE_QWEN3_235B_A22B_CONFIG,
    num_gpus=256,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=32,
    global_batch_size=2048,
)


# Qwen3 30B A3B presets ------------------------------------------------------


QWEN3_30B_A3B_GB300_BF16_BASE_CONFIG = replace(
    BASE_QWEN3_30B_A3B_CONFIG,
    num_gpus=8,
    micro_batch_size=8,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)


QWEN3_30B_A3B_GB300_FP8_CS_BASE_CONFIG = replace(
    BASE_QWEN3_30B_A3B_CONFIG,
    num_gpus=8,
    micro_batch_size=8,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)


QWEN3_30B_A3B_GB300_FP8_MX_BASE_CONFIG = QWEN3_30B_A3B_GB300_FP8_CS_BASE_CONFIG


QWEN3_30B_A3B_GB200_BF16_BASE_CONFIG = replace(
    BASE_QWEN3_30B_A3B_CONFIG,
    num_gpus=8,
    micro_batch_size=4,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)


QWEN3_30B_A3B_GB200_FP8_CS_BASE_CONFIG = replace(
    BASE_QWEN3_30B_A3B_CONFIG,
    num_gpus=8,
    micro_batch_size=4,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)


QWEN3_30B_A3B_GB200_FP8_MX_BASE_CONFIG = QWEN3_30B_A3B_GB200_FP8_CS_BASE_CONFIG


QWEN3_30B_A3B_B200_BF16_BASE_CONFIG = replace(
    BASE_QWEN3_30B_A3B_CONFIG,
    num_gpus=8,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)


QWEN3_30B_A3B_B200_FP8_CS_BASE_CONFIG = replace(
    BASE_QWEN3_30B_A3B_CONFIG,
    num_gpus=8,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)


QWEN3_30B_A3B_B200_FP8_MX_BASE_CONFIG = QWEN3_30B_A3B_B200_FP8_CS_BASE_CONFIG


QWEN3_30B_A3B_H100_BF16_BASE_CONFIG = replace(
    BASE_QWEN3_30B_A3B_CONFIG,
    num_gpus=16,
    pipeline_model_parallel_size=2,
    virtual_pipeline_model_parallel_size=12,
)


QWEN3_30B_A3B_H100_FP8_CS_BASE_CONFIG = replace(
    BASE_QWEN3_30B_A3B_CONFIG,
    num_gpus=16,
    pipeline_model_parallel_size=2,
    virtual_pipeline_model_parallel_size=12,
    micro_batch_size=2,
)


__all__ = [
    "QWEN3_235B_A22B_GB300_BF16_BASE_CONFIG",
    "QWEN3_235B_A22B_GB300_FP8_CS_BASE_CONFIG",
    "QWEN3_235B_A22B_GB300_FP8_MX_BASE_CONFIG",
    "QWEN3_235B_A22B_GB200_BF16_BASE_CONFIG",
    "QWEN3_235B_A22B_GB200_FP8_CS_BASE_CONFIG",
    "QWEN3_235B_A22B_GB200_FP8_MX_BASE_CONFIG",
    "QWEN3_235B_A22B_B200_BF16_BASE_CONFIG",
    "QWEN3_235B_A22B_B200_FP8_CS_BASE_CONFIG",
    "QWEN3_235B_A22B_B200_FP8_MX_BASE_CONFIG",
    "QWEN3_235B_A22B_H100_BF16_BASE_CONFIG",
    "QWEN3_235B_A22B_H100_FP8_CS_BASE_CONFIG",
    "QWEN3_30B_A3B_GB300_BF16_BASE_CONFIG",
    "QWEN3_30B_A3B_GB300_FP8_CS_BASE_CONFIG",
    "QWEN3_30B_A3B_GB300_FP8_MX_BASE_CONFIG",
    "QWEN3_30B_A3B_GB200_BF16_BASE_CONFIG",
    "QWEN3_30B_A3B_GB200_FP8_CS_BASE_CONFIG",
    "QWEN3_30B_A3B_GB200_FP8_MX_BASE_CONFIG",
    "QWEN3_30B_A3B_B200_BF16_BASE_CONFIG",
    "QWEN3_30B_A3B_B200_FP8_CS_BASE_CONFIG",
    "QWEN3_30B_A3B_B200_FP8_MX_BASE_CONFIG",
    "QWEN3_30B_A3B_H100_BF16_BASE_CONFIG",
    "QWEN3_30B_A3B_H100_FP8_CS_BASE_CONFIG",
]
