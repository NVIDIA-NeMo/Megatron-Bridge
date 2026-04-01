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

"""Parallelism presets for Qwen3.5-VL MoE performance configs."""

from dataclasses import replace

from utils.utils import WorkloadBaseConfig


# =============================================================================
# Qwen3.5-VL 35B-A3B base config
# =============================================================================

BASE_QWEN35_VL_35B_A3B_CONFIG = WorkloadBaseConfig(
    expert_model_parallel_size=8,
    expert_tensor_parallel_size=1,
    global_batch_size=512,
)


# Qwen3.5-VL 35B-A3B presets --------------------------------------------------

QWEN35_VL_35B_A3B_PRETRAIN_CONFIG_GB300_BF16 = replace(
    BASE_QWEN35_VL_35B_A3B_CONFIG,
    num_gpus=8,
    micro_batch_size=8,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)


QWEN35_VL_35B_A3B_PRETRAIN_CONFIG_GB300_FP8_CS = replace(
    BASE_QWEN35_VL_35B_A3B_CONFIG,
    num_gpus=8,
    micro_batch_size=8,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)


QWEN35_VL_35B_A3B_PRETRAIN_CONFIG_GB300_FP8_MX = QWEN35_VL_35B_A3B_PRETRAIN_CONFIG_GB300_FP8_CS


QWEN35_VL_35B_A3B_PRETRAIN_CONFIG_GB200_BF16 = replace(
    BASE_QWEN35_VL_35B_A3B_CONFIG,
    num_gpus=8,
    micro_batch_size=4,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["attn", "moe_router", "moe_preprocess"],
)


QWEN35_VL_35B_A3B_PRETRAIN_CONFIG_GB200_FP8_CS = replace(
    BASE_QWEN35_VL_35B_A3B_CONFIG,
    num_gpus=8,
    micro_batch_size=4,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["attn", "moe_router", "moe_preprocess"],
)


QWEN35_VL_35B_A3B_PRETRAIN_CONFIG_GB200_FP8_MX = replace(
    BASE_QWEN35_VL_35B_A3B_CONFIG,
    num_gpus=8,
    micro_batch_size=4,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)


QWEN35_VL_35B_A3B_PRETRAIN_CONFIG_B200_BF16 = replace(
    BASE_QWEN35_VL_35B_A3B_CONFIG,
    num_gpus=8,
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)


QWEN35_VL_35B_A3B_PRETRAIN_CONFIG_B200_FP8_CS = replace(
    BASE_QWEN35_VL_35B_A3B_CONFIG,
    num_gpus=8,
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)


QWEN35_VL_35B_A3B_PRETRAIN_CONFIG_B200_FP8_MX = QWEN35_VL_35B_A3B_PRETRAIN_CONFIG_B200_FP8_CS


QWEN35_VL_35B_A3B_PRETRAIN_CONFIG_H100_BF16 = replace(
    BASE_QWEN35_VL_35B_A3B_CONFIG,
    num_gpus=16,
    pipeline_model_parallel_size=2,
    virtual_pipeline_model_parallel_size=12,
    moe_a2a_overlap=True,
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)


QWEN35_VL_35B_A3B_PRETRAIN_CONFIG_H100_FP8_CS = replace(
    BASE_QWEN35_VL_35B_A3B_CONFIG,
    num_gpus=16,
    pipeline_model_parallel_size=2,
    virtual_pipeline_model_parallel_size=12,
    moe_a2a_overlap=True,
)


# =============================================================================
# Qwen3.5-VL 122B-A10B base config
# =============================================================================

BASE_QWEN35_VL_122B_A10B_CONFIG = WorkloadBaseConfig(
    expert_model_parallel_size=8,
    expert_tensor_parallel_size=1,
    global_batch_size=1024,
)


# Qwen3.5-VL 122B-A10B presets ------------------------------------------------

QWEN35_VL_122B_A10B_PRETRAIN_CONFIG_GB300_BF16 = replace(
    BASE_QWEN35_VL_122B_A10B_CONFIG,
    num_gpus=32,
    tensor_model_parallel_size=1,
    expert_model_parallel_size=32,
    micro_batch_size=2,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)


QWEN35_VL_122B_A10B_PRETRAIN_CONFIG_GB300_FP8_CS = replace(
    BASE_QWEN35_VL_122B_A10B_CONFIG,
    num_gpus=32,
    tensor_model_parallel_size=1,
    expert_model_parallel_size=32,
    micro_batch_size=2,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)


QWEN35_VL_122B_A10B_PRETRAIN_CONFIG_GB300_FP8_MX = QWEN35_VL_122B_A10B_PRETRAIN_CONFIG_GB300_FP8_CS


QWEN35_VL_122B_A10B_PRETRAIN_CONFIG_GB200_BF16 = replace(
    BASE_QWEN35_VL_122B_A10B_CONFIG,
    num_gpus=32,
    pipeline_model_parallel_size=4,
    expert_model_parallel_size=8,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["attn", "moe_router", "moe_preprocess"],
)


QWEN35_VL_122B_A10B_PRETRAIN_CONFIG_GB200_FP8_CS = replace(
    BASE_QWEN35_VL_122B_A10B_CONFIG,
    num_gpus=32,
    pipeline_model_parallel_size=4,
    expert_model_parallel_size=8,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["attn", "moe_router", "moe_preprocess"],
)


QWEN35_VL_122B_A10B_PRETRAIN_CONFIG_GB200_FP8_MX = QWEN35_VL_122B_A10B_PRETRAIN_CONFIG_GB200_FP8_CS


QWEN35_VL_122B_A10B_PRETRAIN_CONFIG_B200_BF16 = replace(
    BASE_QWEN35_VL_122B_A10B_CONFIG,
    num_gpus=32,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=8,
    moe_a2a_overlap=True,
)


QWEN35_VL_122B_A10B_PRETRAIN_CONFIG_B200_FP8_CS = replace(
    BASE_QWEN35_VL_122B_A10B_CONFIG,
    num_gpus=32,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=8,
    moe_a2a_overlap=True,
)


QWEN35_VL_122B_A10B_PRETRAIN_CONFIG_B200_FP8_MX = QWEN35_VL_122B_A10B_PRETRAIN_CONFIG_B200_FP8_CS


QWEN35_VL_122B_A10B_PRETRAIN_CONFIG_H100_BF16 = replace(
    BASE_QWEN35_VL_122B_A10B_CONFIG,
    num_gpus=128,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=16,
    moe_a2a_overlap=True,
)


QWEN35_VL_122B_A10B_PRETRAIN_CONFIG_H100_FP8_CS = replace(
    BASE_QWEN35_VL_122B_A10B_CONFIG,
    num_gpus=128,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=16,
    moe_a2a_overlap=True,
)


# =============================================================================
# Qwen3.5-VL 397B-A17B base config
# =============================================================================

BASE_QWEN35_VL_397B_A17B_CONFIG = WorkloadBaseConfig(
    expert_model_parallel_size=8,
    expert_tensor_parallel_size=1,
    global_batch_size=1024,
)


# Qwen3.5-VL 397B-A17B presets ------------------------------------------------

QWEN35_VL_397B_A17B_PRETRAIN_CONFIG_GB300_BF16 = replace(
    BASE_QWEN35_VL_397B_A17B_CONFIG,
    num_gpus=64,
    tensor_model_parallel_size=1,
    expert_model_parallel_size=64,
    micro_batch_size=1,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)


QWEN35_VL_397B_A17B_PRETRAIN_CONFIG_GB300_FP8_CS = replace(
    BASE_QWEN35_VL_397B_A17B_CONFIG,
    num_gpus=64,
    tensor_model_parallel_size=1,
    expert_model_parallel_size=64,
    micro_batch_size=1,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)


QWEN35_VL_397B_A17B_PRETRAIN_CONFIG_GB300_FP8_MX = QWEN35_VL_397B_A17B_PRETRAIN_CONFIG_GB300_FP8_CS


QWEN35_VL_397B_A17B_PRETRAIN_CONFIG_GB200_BF16 = replace(
    BASE_QWEN35_VL_397B_A17B_CONFIG,
    num_gpus=64,
    pipeline_model_parallel_size=8,
    expert_model_parallel_size=8,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["attn", "moe_router", "moe_preprocess"],
)


QWEN35_VL_397B_A17B_PRETRAIN_CONFIG_GB200_FP8_CS = replace(
    BASE_QWEN35_VL_397B_A17B_CONFIG,
    num_gpus=64,
    pipeline_model_parallel_size=8,
    expert_model_parallel_size=8,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["attn", "moe_router", "moe_preprocess"],
)


QWEN35_VL_397B_A17B_PRETRAIN_CONFIG_GB200_FP8_MX = QWEN35_VL_397B_A17B_PRETRAIN_CONFIG_GB200_FP8_CS


QWEN35_VL_397B_A17B_PRETRAIN_CONFIG_B200_BF16 = replace(
    BASE_QWEN35_VL_397B_A17B_CONFIG,
    num_gpus=64,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=8,
    moe_a2a_overlap=True,
)


QWEN35_VL_397B_A17B_PRETRAIN_CONFIG_B200_FP8_CS = replace(
    BASE_QWEN35_VL_397B_A17B_CONFIG,
    num_gpus=64,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=8,
    moe_a2a_overlap=True,
)


QWEN35_VL_397B_A17B_PRETRAIN_CONFIG_B200_FP8_MX = QWEN35_VL_397B_A17B_PRETRAIN_CONFIG_B200_FP8_CS


QWEN35_VL_397B_A17B_PRETRAIN_CONFIG_H100_BF16 = replace(
    BASE_QWEN35_VL_397B_A17B_CONFIG,
    num_gpus=256,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=32,
    moe_a2a_overlap=True,
)


QWEN35_VL_397B_A17B_PRETRAIN_CONFIG_H100_FP8_CS = replace(
    BASE_QWEN35_VL_397B_A17B_CONFIG,
    num_gpus=256,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=32,
    moe_a2a_overlap=True,
)


__all__ = [
    # 35B-A3B
    "QWEN35_VL_35B_A3B_PRETRAIN_CONFIG_GB300_BF16",
    "QWEN35_VL_35B_A3B_PRETRAIN_CONFIG_GB300_FP8_CS",
    "QWEN35_VL_35B_A3B_PRETRAIN_CONFIG_GB300_FP8_MX",
    "QWEN35_VL_35B_A3B_PRETRAIN_CONFIG_GB200_BF16",
    "QWEN35_VL_35B_A3B_PRETRAIN_CONFIG_GB200_FP8_CS",
    "QWEN35_VL_35B_A3B_PRETRAIN_CONFIG_GB200_FP8_MX",
    "QWEN35_VL_35B_A3B_PRETRAIN_CONFIG_B200_BF16",
    "QWEN35_VL_35B_A3B_PRETRAIN_CONFIG_B200_FP8_CS",
    "QWEN35_VL_35B_A3B_PRETRAIN_CONFIG_B200_FP8_MX",
    "QWEN35_VL_35B_A3B_PRETRAIN_CONFIG_H100_BF16",
    "QWEN35_VL_35B_A3B_PRETRAIN_CONFIG_H100_FP8_CS",
    # 122B-A10B
    "QWEN35_VL_122B_A10B_PRETRAIN_CONFIG_GB300_BF16",
    "QWEN35_VL_122B_A10B_PRETRAIN_CONFIG_GB300_FP8_CS",
    "QWEN35_VL_122B_A10B_PRETRAIN_CONFIG_GB300_FP8_MX",
    "QWEN35_VL_122B_A10B_PRETRAIN_CONFIG_GB200_BF16",
    "QWEN35_VL_122B_A10B_PRETRAIN_CONFIG_GB200_FP8_CS",
    "QWEN35_VL_122B_A10B_PRETRAIN_CONFIG_GB200_FP8_MX",
    "QWEN35_VL_122B_A10B_PRETRAIN_CONFIG_B200_BF16",
    "QWEN35_VL_122B_A10B_PRETRAIN_CONFIG_B200_FP8_CS",
    "QWEN35_VL_122B_A10B_PRETRAIN_CONFIG_B200_FP8_MX",
    "QWEN35_VL_122B_A10B_PRETRAIN_CONFIG_H100_BF16",
    "QWEN35_VL_122B_A10B_PRETRAIN_CONFIG_H100_FP8_CS",
    # 397B-A17B
    "QWEN35_VL_397B_A17B_PRETRAIN_CONFIG_GB300_BF16",
    "QWEN35_VL_397B_A17B_PRETRAIN_CONFIG_GB300_FP8_CS",
    "QWEN35_VL_397B_A17B_PRETRAIN_CONFIG_GB300_FP8_MX",
    "QWEN35_VL_397B_A17B_PRETRAIN_CONFIG_GB200_BF16",
    "QWEN35_VL_397B_A17B_PRETRAIN_CONFIG_GB200_FP8_CS",
    "QWEN35_VL_397B_A17B_PRETRAIN_CONFIG_GB200_FP8_MX",
    "QWEN35_VL_397B_A17B_PRETRAIN_CONFIG_B200_BF16",
    "QWEN35_VL_397B_A17B_PRETRAIN_CONFIG_B200_FP8_CS",
    "QWEN35_VL_397B_A17B_PRETRAIN_CONFIG_B200_FP8_MX",
    "QWEN35_VL_397B_A17B_PRETRAIN_CONFIG_H100_BF16",
    "QWEN35_VL_397B_A17B_PRETRAIN_CONFIG_H100_FP8_CS",
]
