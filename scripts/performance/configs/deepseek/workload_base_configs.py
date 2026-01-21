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

"""Workload base presets for DeepSeek-V3 performance configs."""

from dataclasses import replace

from utils.utils import WorkloadBaseConfig


BASE_DEEPSEEK_V3_CONFIG = WorkloadBaseConfig(
    expert_tensor_parallel_size=1,
)


DEEPSEEK_V3_GB300_BASE_CONFIG = replace(
    BASE_DEEPSEEK_V3_CONFIG,
    num_gpus=256,
    global_batch_size=2048,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=64,
    moe_flex_dispatcher_backend="hybridep",
    moe_a2a_overlap=False,
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["attn", "moe_router", "moe_preprocess"],
    recompute_modules=["moe_act"],
)
DEEPSEEK_V3_GB300_BF16_BASE_CONFIG = DEEPSEEK_V3_GB300_BASE_CONFIG
DEEPSEEK_V3_GB300_FP8_CS_BASE_CONFIG = DEEPSEEK_V3_GB300_BASE_CONFIG
DEEPSEEK_V3_GB300_FP8_MX_BASE_CONFIG = DEEPSEEK_V3_GB300_BASE_CONFIG


DEEPSEEK_V3_GB200_BASE_CONFIG = replace(
    BASE_DEEPSEEK_V3_CONFIG,
    num_gpus=256,
    global_batch_size=2048,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=64,
    moe_flex_dispatcher_backend="hybridep",
    moe_a2a_overlap=False,
    recompute_modules=["mla_up_proj"],
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)
DEEPSEEK_V3_GB200_BF16_BASE_CONFIG = DEEPSEEK_V3_GB200_BASE_CONFIG
DEEPSEEK_V3_GB200_FP8_CS_BASE_CONFIG = DEEPSEEK_V3_GB200_BASE_CONFIG
DEEPSEEK_V3_GB200_FP8_MX_BASE_CONFIG = DEEPSEEK_V3_GB200_BASE_CONFIG


DEEPSEEK_V3_B200_BASE_CONFIG = replace(
    BASE_DEEPSEEK_V3_CONFIG,
    num_gpus=256,
    pipeline_model_parallel_size=16,
    expert_model_parallel_size=8,
    global_batch_size=2048,
    recompute_modules=["mla_up_proj"],
    moe_a2a_overlap=False,
)
DEEPSEEK_V3_B200_BF16_BASE_CONFIG = DEEPSEEK_V3_B200_BASE_CONFIG
DEEPSEEK_V3_B200_FP8_CS_BASE_CONFIG = DEEPSEEK_V3_B200_BASE_CONFIG
DEEPSEEK_V3_B200_FP8_MX_BASE_CONFIG = DEEPSEEK_V3_B200_FP8_CS_BASE_CONFIG


DEEPSEEK_V3_H100_BASE_CONFIG = replace(
    BASE_DEEPSEEK_V3_CONFIG,
    num_gpus=1024,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=64,
    global_batch_size=8192,
    recompute_modules=["mla_up_proj", "mlp"],
    moe_flex_dispatcher_backend="deepep",
    moe_a2a_overlap=True,
)
DEEPSEEK_V3_H100_BF16_BASE_CONFIG = DEEPSEEK_V3_H100_BASE_CONFIG
DEEPSEEK_V3_H100_FP8_CS_BASE_CONFIG = DEEPSEEK_V3_H100_BASE_CONFIG
DEEPSEEK_V3_H100_FP8_SC_BASE_CONFIG = DEEPSEEK_V3_H100_FP8_CS_BASE_CONFIG


# -----------------------------------------------------------------------------
# Moonlight-16B configs (based on DeepSeek architecture)
# -----------------------------------------------------------------------------

BASE_MOONLIGHT_CONFIG = WorkloadBaseConfig(
    expert_tensor_parallel_size=1,
)

# GB200 single node (4 GPUs): TP=1, PP=1, EP=4
DEEPSEEK_MOONLIGHT_16B_GB200_BASE_CONFIG = replace(
    BASE_MOONLIGHT_CONFIG,
    num_gpus=4,
    global_batch_size=64,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    expert_model_parallel_size=4,
    moe_flex_dispatcher_backend="hybridep",
)
DEEPSEEK_MOONLIGHT_16B_GB200_BF16_BASE_CONFIG = DEEPSEEK_MOONLIGHT_16B_GB200_BASE_CONFIG
DEEPSEEK_MOONLIGHT_16B_GB200_FP8_CS_BASE_CONFIG = DEEPSEEK_MOONLIGHT_16B_GB200_BASE_CONFIG

# H100 (16 GPUs): TP=2, PP=1, EP=8
DEEPSEEK_MOONLIGHT_16B_H100_BASE_CONFIG = replace(
    BASE_MOONLIGHT_CONFIG,
    num_gpus=16,
    global_batch_size=2048,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=1,
    expert_model_parallel_size=8,
    moe_flex_dispatcher_backend="deepep",
)
DEEPSEEK_MOONLIGHT_16B_H100_BF16_BASE_CONFIG = DEEPSEEK_MOONLIGHT_16B_H100_BASE_CONFIG
DEEPSEEK_MOONLIGHT_16B_H100_FP8_CS_BASE_CONFIG = DEEPSEEK_MOONLIGHT_16B_H100_BASE_CONFIG


# -----------------------------------------------------------------------------
# DeepSeek-V3 Lite configs (close to Moonlight-16B)
# -----------------------------------------------------------------------------

# GB200 single node (4 GPUs): TP=1, PP=1, EP=4
DEEPSEEK_V3_LITE_GB200_BASE_CONFIG = replace(
    BASE_DEEPSEEK_V3_CONFIG,
    num_gpus=4,
    global_batch_size=64,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    expert_model_parallel_size=4,
    moe_flex_dispatcher_backend="hybridep",
)
DEEPSEEK_V3_LITE_GB200_BF16_BASE_CONFIG = DEEPSEEK_V3_LITE_GB200_BASE_CONFIG
DEEPSEEK_V3_LITE_GB200_FP8_CS_BASE_CONFIG = DEEPSEEK_V3_LITE_GB200_BASE_CONFIG

# H100 (8 GPUs for lite): TP=1, PP=1, EP=8
DEEPSEEK_V3_LITE_H100_BASE_CONFIG = replace(
    BASE_DEEPSEEK_V3_CONFIG,
    num_gpus=8,
    global_batch_size=1024,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    expert_model_parallel_size=8,
    moe_flex_dispatcher_backend="deepep",
)
DEEPSEEK_V3_LITE_H100_BF16_BASE_CONFIG = DEEPSEEK_V3_LITE_H100_BASE_CONFIG
DEEPSEEK_V3_LITE_H100_FP8_CS_BASE_CONFIG = DEEPSEEK_V3_LITE_H100_BASE_CONFIG


__all__ = [
    "DEEPSEEK_V3_GB300_BF16_BASE_CONFIG",
    "DEEPSEEK_V3_GB300_FP8_CS_BASE_CONFIG",
    "DEEPSEEK_V3_GB300_FP8_MX_BASE_CONFIG",
    "DEEPSEEK_V3_GB200_BF16_BASE_CONFIG",
    "DEEPSEEK_V3_GB200_FP8_CS_BASE_CONFIG",
    "DEEPSEEK_V3_GB200_FP8_MX_BASE_CONFIG",
    "DEEPSEEK_V3_B200_BF16_BASE_CONFIG",
    "DEEPSEEK_V3_B200_FP8_CS_BASE_CONFIG",
    "DEEPSEEK_V3_B200_FP8_MX_BASE_CONFIG",
    "DEEPSEEK_V3_H100_BF16_BASE_CONFIG",
    "DEEPSEEK_V3_H100_FP8_CS_BASE_CONFIG",
    "DEEPSEEK_V3_H100_FP8_SC_BASE_CONFIG",
    # Moonlight-16B
    "DEEPSEEK_MOONLIGHT_16B_GB200_BF16_BASE_CONFIG",
    "DEEPSEEK_MOONLIGHT_16B_GB200_FP8_CS_BASE_CONFIG",
    "DEEPSEEK_MOONLIGHT_16B_H100_BF16_BASE_CONFIG",
    "DEEPSEEK_MOONLIGHT_16B_H100_FP8_CS_BASE_CONFIG",
    # DeepSeek-V3 Lite
    "DEEPSEEK_V3_LITE_GB200_BF16_BASE_CONFIG",
    "DEEPSEEK_V3_LITE_GB200_FP8_CS_BASE_CONFIG",
    "DEEPSEEK_V3_LITE_H100_BF16_BASE_CONFIG",
    "DEEPSEEK_V3_LITE_H100_FP8_CS_BASE_CONFIG",
]
