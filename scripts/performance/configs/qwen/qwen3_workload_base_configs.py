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

BASE_QWEN3_NEXT_80B_A3B_CONFIG = WorkloadBaseConfig(
    expert_model_parallel_size=64,
    expert_tensor_parallel_size=1,
    global_batch_size=1024,
)

# Qwen3 235B A22B presets ----------------------------------------------------


QWEN3_235B_A22B_PRETRAIN_CONFIG_GB300_BF16 = replace(
    BASE_QWEN3_235B_A22B_CONFIG,
    num_gpus=64,
    tensor_model_parallel_size=1,
    expert_model_parallel_size=64,
    global_batch_size=1024,
    micro_batch_size=2,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)


QWEN3_235B_A22B_PRETRAIN_CONFIG_GB300_FP8_CS = replace(
    BASE_QWEN3_235B_A22B_CONFIG,
    num_gpus=64,
    tensor_model_parallel_size=1,
    expert_model_parallel_size=64,
    global_batch_size=1024,
    micro_batch_size=2,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)


QWEN3_235B_A22B_PRETRAIN_CONFIG_GB300_FP8_MX = QWEN3_235B_A22B_PRETRAIN_CONFIG_GB300_FP8_CS


QWEN3_235B_A22B_PRETRAIN_CONFIG_GB200_BF16 = replace(
    BASE_QWEN3_235B_A22B_CONFIG,
    num_gpus=64,
    pipeline_model_parallel_size=8,
    expert_model_parallel_size=8,
    global_batch_size=1024,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["attn", "moe_router", "moe_preprocess"],
)


QWEN3_235B_A22B_PRETRAIN_CONFIG_GB200_FP8_CS = replace(
    BASE_QWEN3_235B_A22B_CONFIG,
    num_gpus=64,
    pipeline_model_parallel_size=8,
    expert_model_parallel_size=8,
    global_batch_size=1024,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["attn", "moe_router", "moe_preprocess"],
)


QWEN3_235B_A22B_PRETRAIN_CONFIG_GB200_FP8_MX = QWEN3_235B_A22B_PRETRAIN_CONFIG_GB200_FP8_CS


QWEN3_235B_A22B_PRETRAIN_CONFIG_B300_BF16 = replace(
    BASE_QWEN3_235B_A22B_CONFIG,
    num_gpus=64,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=8,
    global_batch_size=1024,
    moe_a2a_overlap=True,
)


QWEN3_235B_A22B_PRETRAIN_CONFIG_B300_FP8_CS = replace(
    BASE_QWEN3_235B_A22B_CONFIG,
    num_gpus=64,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=8,
    global_batch_size=1024,
    moe_a2a_overlap=True,
)


QWEN3_235B_A22B_PRETRAIN_CONFIG_B300_FP8_MX = QWEN3_235B_A22B_PRETRAIN_CONFIG_B300_FP8_CS


QWEN3_235B_A22B_PRETRAIN_CONFIG_B200_BF16 = replace(
    BASE_QWEN3_235B_A22B_CONFIG,
    num_gpus=64,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=8,
    global_batch_size=1024,
    moe_a2a_overlap=True,
)


QWEN3_235B_A22B_PRETRAIN_CONFIG_B200_FP8_CS = replace(
    BASE_QWEN3_235B_A22B_CONFIG,
    num_gpus=64,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=8,
    global_batch_size=1024,
    moe_a2a_overlap=True,
)


QWEN3_235B_A22B_PRETRAIN_CONFIG_B200_FP8_MX = QWEN3_235B_A22B_PRETRAIN_CONFIG_B200_FP8_CS


QWEN3_235B_A22B_PRETRAIN_CONFIG_H100_BF16 = replace(
    BASE_QWEN3_235B_A22B_CONFIG,
    num_gpus=256,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=32,
    global_batch_size=2048,
    moe_a2a_overlap=True,
)


QWEN3_235B_A22B_PRETRAIN_CONFIG_H100_FP8_CS = replace(
    BASE_QWEN3_235B_A22B_CONFIG,
    num_gpus=256,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=32,
    global_batch_size=2048,
    moe_a2a_overlap=True,
)


# Qwen3 30B A3B presets ------------------------------------------------------


QWEN3_30B_A3B_PRETRAIN_CONFIG_GB300_BF16 = replace(
    BASE_QWEN3_30B_A3B_CONFIG,
    num_gpus=8,
    micro_batch_size=8,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)


QWEN3_30B_A3B_PRETRAIN_CONFIG_GB300_FP8_CS = replace(
    BASE_QWEN3_30B_A3B_CONFIG,
    num_gpus=8,
    micro_batch_size=8,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)


QWEN3_30B_A3B_PRETRAIN_CONFIG_GB300_FP8_MX = QWEN3_30B_A3B_PRETRAIN_CONFIG_GB300_FP8_CS


QWEN3_30B_A3B_PRETRAIN_CONFIG_GB200_BF16 = replace(
    BASE_QWEN3_30B_A3B_CONFIG,
    num_gpus=8,
    micro_batch_size=4,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["attn", "moe_router", "moe_preprocess"],
)


QWEN3_30B_A3B_PRETRAIN_CONFIG_GB200_FP8_CS = replace(
    BASE_QWEN3_30B_A3B_CONFIG,
    num_gpus=8,
    micro_batch_size=4,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["attn", "moe_router", "moe_preprocess"],
)


QWEN3_30B_A3B_PRETRAIN_CONFIG_GB200_FP8_MX = replace(
    BASE_QWEN3_30B_A3B_CONFIG,
    num_gpus=8,
    micro_batch_size=4,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)


QWEN3_30B_A3B_PRETRAIN_CONFIG_B300_BF16 = replace(
    BASE_QWEN3_30B_A3B_CONFIG,
    num_gpus=8,
    micro_batch_size=8,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)


QWEN3_30B_A3B_PRETRAIN_CONFIG_B300_FP8_CS = replace(
    BASE_QWEN3_30B_A3B_CONFIG,
    num_gpus=8,
    micro_batch_size=8,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)


QWEN3_30B_A3B_PRETRAIN_CONFIG_B300_FP8_MX = QWEN3_30B_A3B_PRETRAIN_CONFIG_B300_FP8_CS


QWEN3_30B_A3B_PRETRAIN_CONFIG_B200_BF16 = replace(
    BASE_QWEN3_30B_A3B_CONFIG,
    num_gpus=8,
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)


QWEN3_30B_A3B_PRETRAIN_CONFIG_B200_FP8_CS = replace(
    BASE_QWEN3_30B_A3B_CONFIG,
    num_gpus=8,
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)


QWEN3_30B_A3B_PRETRAIN_CONFIG_B200_FP8_MX = QWEN3_30B_A3B_PRETRAIN_CONFIG_B200_FP8_CS


QWEN3_30B_A3B_PRETRAIN_CONFIG_H100_BF16 = replace(
    BASE_QWEN3_30B_A3B_CONFIG,
    num_gpus=16,
    pipeline_model_parallel_size=2,
    virtual_pipeline_model_parallel_size=12,
    moe_a2a_overlap=True,
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)


QWEN3_30B_A3B_PRETRAIN_CONFIG_H100_FP8_CS = replace(
    BASE_QWEN3_30B_A3B_CONFIG,
    num_gpus=16,
    pipeline_model_parallel_size=2,
    virtual_pipeline_model_parallel_size=12,
    moe_a2a_overlap=True,
)


# Qwen3 Next Presets --------------------------------------

QWEN3_NEXT_80B_A3B_PRETRAIN_CONFIG_GB200_FP8_MX = replace(
    BASE_QWEN3_NEXT_80B_A3B_CONFIG,
    num_gpus=64,
    micro_batch_size=1,
)

QWEN3_NEXT_80B_A3B_PRETRAIN_CONFIG_GB200_BF16 = replace(
    BASE_QWEN3_NEXT_80B_A3B_CONFIG,
    num_gpus=64,
    micro_batch_size=1,
)

QWEN3_NEXT_80B_A3B_PRETRAIN_CONFIG_GB300_FP8_MX = replace(
    BASE_QWEN3_NEXT_80B_A3B_CONFIG,
    num_gpus=64,
    micro_batch_size=2,
)

QWEN3_NEXT_80B_A3B_PRETRAIN_CONFIG_GB300_BF16 = replace(
    BASE_QWEN3_NEXT_80B_A3B_CONFIG,
    num_gpus=64,
    micro_batch_size=1,
)

QWEN3_NEXT_80B_A3B_PRETRAIN_CONFIG_H100_FP8_CS = replace(
    BASE_QWEN3_NEXT_80B_A3B_CONFIG,
    num_gpus=128,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=12,
    expert_model_parallel_size=8,
    micro_batch_size=1,
)

QWEN3_NEXT_80B_A3B_PRETRAIN_CONFIG_H100_BF16 = replace(
    BASE_QWEN3_NEXT_80B_A3B_CONFIG,
    num_gpus=128,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=12,
    expert_model_parallel_size=8,
    micro_batch_size=1,
)

QWEN3_NEXT_80B_A3B_PRETRAIN_CONFIG_B300_FP8_MX = replace(
    BASE_QWEN3_NEXT_80B_A3B_CONFIG,
    num_gpus=64,
    micro_batch_size=2,
)

QWEN3_NEXT_80B_A3B_PRETRAIN_CONFIG_B300_BF16 = replace(
    BASE_QWEN3_NEXT_80B_A3B_CONFIG,
    num_gpus=64,
    micro_batch_size=1,
)

QWEN3_NEXT_80B_A3B_PRETRAIN_CONFIG_B200_FP8_MX = replace(
    BASE_QWEN3_NEXT_80B_A3B_CONFIG,
    num_gpus=64,
    micro_batch_size=1,
)

QWEN3_NEXT_80B_A3B_PRETRAIN_CONFIG_B200_BF16 = replace(
    BASE_QWEN3_NEXT_80B_A3B_CONFIG,
    num_gpus=64,
    micro_batch_size=1,
)


__all__ = [
    "QWEN3_235B_A22B_PRETRAIN_CONFIG_GB300_BF16",
    "QWEN3_235B_A22B_PRETRAIN_CONFIG_GB300_FP8_CS",
    "QWEN3_235B_A22B_PRETRAIN_CONFIG_GB300_FP8_MX",
    "QWEN3_235B_A22B_PRETRAIN_CONFIG_GB200_BF16",
    "QWEN3_235B_A22B_PRETRAIN_CONFIG_GB200_FP8_CS",
    "QWEN3_235B_A22B_PRETRAIN_CONFIG_GB200_FP8_MX",
    "QWEN3_235B_A22B_PRETRAIN_CONFIG_B300_BF16",
    "QWEN3_235B_A22B_PRETRAIN_CONFIG_B300_FP8_CS",
    "QWEN3_235B_A22B_PRETRAIN_CONFIG_B300_FP8_MX",
    "QWEN3_235B_A22B_PRETRAIN_CONFIG_B200_BF16",
    "QWEN3_235B_A22B_PRETRAIN_CONFIG_B200_FP8_CS",
    "QWEN3_235B_A22B_PRETRAIN_CONFIG_B200_FP8_MX",
    "QWEN3_235B_A22B_PRETRAIN_CONFIG_H100_BF16",
    "QWEN3_235B_A22B_PRETRAIN_CONFIG_H100_FP8_CS",
    "QWEN3_30B_A3B_PRETRAIN_CONFIG_GB300_BF16",
    "QWEN3_30B_A3B_PRETRAIN_CONFIG_GB300_FP8_CS",
    "QWEN3_30B_A3B_PRETRAIN_CONFIG_GB300_FP8_MX",
    "QWEN3_30B_A3B_PRETRAIN_CONFIG_GB200_BF16",
    "QWEN3_30B_A3B_PRETRAIN_CONFIG_GB200_FP8_CS",
    "QWEN3_30B_A3B_PRETRAIN_CONFIG_GB200_FP8_MX",
    "QWEN3_30B_A3B_PRETRAIN_CONFIG_B300_BF16",
    "QWEN3_30B_A3B_PRETRAIN_CONFIG_B300_FP8_CS",
    "QWEN3_30B_A3B_PRETRAIN_CONFIG_B300_FP8_MX",
    "QWEN3_30B_A3B_PRETRAIN_CONFIG_B200_BF16",
    "QWEN3_30B_A3B_PRETRAIN_CONFIG_B200_FP8_CS",
    "QWEN3_30B_A3B_PRETRAIN_CONFIG_B200_FP8_MX",
    "QWEN3_30B_A3B_PRETRAIN_CONFIG_H100_BF16",
    "QWEN3_30B_A3B_PRETRAIN_CONFIG_H100_FP8_CS",
    "QWEN3_NEXT_80B_A3B_PRETRAIN_CONFIG_GB200_BF16",
    "QWEN3_NEXT_80B_A3B_PRETRAIN_CONFIG_GB200_FP8_MX",
    "QWEN3_NEXT_80B_A3B_PRETRAIN_CONFIG_GB300_FP8_MX",
    "QWEN3_NEXT_80B_A3B_PRETRAIN_CONFIG_GB300_BF16",
    "QWEN3_NEXT_80B_A3B_PRETRAIN_CONFIG_H100_FP8_CS",
    "QWEN3_NEXT_80B_A3B_PRETRAIN_CONFIG_H100_BF16",
    "QWEN3_NEXT_80B_A3B_PRETRAIN_CONFIG_B300_FP8_MX",
    "QWEN3_NEXT_80B_A3B_PRETRAIN_CONFIG_B300_BF16",
    "QWEN3_NEXT_80B_A3B_PRETRAIN_CONFIG_B200_FP8_MX",
    "QWEN3_NEXT_80B_A3B_PRETRAIN_CONFIG_B200_BF16",
]
