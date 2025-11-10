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


DEEPSEEK_V3_GB300_256GPUS_BF16_BASE_CONFIG = replace(
    BASE_DEEPSEEK_V3_CONFIG,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=64,
    global_batch_size=2048,
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope="attn",
    recompute_modules=["mlp", "moe_act"],
)


DEEPSEEK_V3_GB300_256GPUS_FP8_CS_BASE_CONFIG = replace(
    BASE_DEEPSEEK_V3_CONFIG,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=64,
    global_batch_size=2048,
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope="attn",
    recompute_modules=["mlp", "moe_act"],
)


DEEPSEEK_V3_GB300_256GPUS_FP8_MX_BASE_CONFIG = replace(
    BASE_DEEPSEEK_V3_CONFIG,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=64,
    global_batch_size=2048,
    recompute_modules=["mla_up_proj"],
)


DEEPSEEK_V3_GB200_256GPUS_BF16_BASE_CONFIG = replace(
    BASE_DEEPSEEK_V3_CONFIG,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=64,
    global_batch_size=2048,
    recompute_modules=["mla_up_proj"],
)


DEEPSEEK_V3_GB200_256GPUS_FP8_CS_BASE_CONFIG = replace(
    BASE_DEEPSEEK_V3_CONFIG,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=64,
    global_batch_size=2048,
    recompute_modules=["mla_up_proj"],
)


DEEPSEEK_V3_GB200_256GPUS_FP8_MX_BASE_CONFIG = replace(
    BASE_DEEPSEEK_V3_CONFIG,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=64,
    global_batch_size=2048,
    recompute_modules=["mla_up_proj", "mlp"],
)

DEEPSEEK_V3_B200_256GPUS_BF16_BASE_CONFIG = replace(
    BASE_DEEPSEEK_V3_CONFIG,
    pipeline_model_parallel_size=16,
    expert_model_parallel_size=8,
    global_batch_size=2048,
    recompute_modules=["mla_up_proj"],
)


DEEPSEEK_V3_B200_256GPUS_FP8_CS_BASE_CONFIG = replace(
    BASE_DEEPSEEK_V3_CONFIG,
    pipeline_model_parallel_size=16,
    expert_model_parallel_size=8,
    global_batch_size=2048,
    recompute_modules=["mla_up_proj"],
)


DEEPSEEK_V3_B200_256GPUS_FP8_MX_BASE_CONFIG = DEEPSEEK_V3_B200_256GPUS_FP8_CS_BASE_CONFIG


DEEPSEEK_V3_H100_1024GPUS_BF16_BASE_CONFIG = replace(
    BASE_DEEPSEEK_V3_CONFIG,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=64,
    global_batch_size=8192,
    recompute_modules=["mla_up_proj", "mlp"],
)


DEEPSEEK_V3_H100_1024GPUS_FP8_CS_BASE_CONFIG = replace(
    BASE_DEEPSEEK_V3_CONFIG,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=64,
    global_batch_size=8192,
    recompute_modules=["mla_up_proj", "mlp"],
)


DEEPSEEK_V3_H100_1024GPUS_FP8_SC_BASE_CONFIG = DEEPSEEK_V3_H100_1024GPUS_FP8_CS_BASE_CONFIG


__all__ = [
    "DEEPSEEK_V3_GB200_256GPUS_BF16_BASE_CONFIG",
    "DEEPSEEK_V3_GB200_256GPUS_FP8_CS_BASE_CONFIG",
    "DEEPSEEK_V3_GB200_256GPUS_FP8_MX_BASE_CONFIG",
    "DEEPSEEK_V3_B200_256GPUS_BF16_BASE_CONFIG",
    "DEEPSEEK_V3_B200_256GPUS_FP8_CS_BASE_CONFIG",
    "DEEPSEEK_V3_B200_256GPUS_FP8_MX_BASE_CONFIG",
    "DEEPSEEK_V3_H100_1024GPUS_BF16_BASE_CONFIG",
    "DEEPSEEK_V3_H100_1024GPUS_FP8_CS_BASE_CONFIG",
    "DEEPSEEK_V3_H100_1024GPUS_FP8_SC_BASE_CONFIG",
]
