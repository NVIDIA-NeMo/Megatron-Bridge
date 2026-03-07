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

"""Workload base presets for DeepSeek-V3 performance configs.

Config naming convention:
    {MODEL}_{SIZE}_{TASK}_CONFIG_{GPU}_{PRECISION}_{VERSION}

V1: GBS=2048 for Blackwell variants, GBS=8192 for H100
V2: GBS=4096 for Blackwell variants, GBS=16384 for H100

Use --config_variant to select a variant.
Use --list_config_variants to see available variants interactively.
"""

from dataclasses import replace

from utils.utils import WorkloadBaseConfig


BASE_DEEPSEEK_V3_CONFIG = WorkloadBaseConfig(
    expert_tensor_parallel_size=1,
)


# =============================================================================
# DeepSeek V3 Pretrain - V1 (original GBS settings)
# =============================================================================

DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_V1 = replace(
    BASE_DEEPSEEK_V3_CONFIG,
    num_gpus=256,
    global_batch_size=2048,
    micro_batch_size=2,
    pipeline_model_parallel_size=2,
    virtual_pipeline_model_parallel_size=8,
    pp_layout="Et*4|(t*4|)*14tmL",
    expert_model_parallel_size=32,
    moe_flex_dispatcher_backend="hybridep",
    moe_a2a_overlap=False,
    cuda_graph_scope=[],
    recompute_modules=["mla_up_proj"],
)
DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_BF16_V1 = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_V1,
    micro_batch_size=1,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=64,
    moe_flex_dispatcher_backend="hybridep",
    moe_a2a_overlap=False,
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["attn", "moe_router", "moe_preprocess"],
    recompute_modules=["moe_act"],
)
DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_FP8_CS_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_V1
DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_FP8_MX_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_V1
DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_NVFP4_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_V1


DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_V1 = replace(
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
    cuda_graph_scope=["attn", "moe_router", "moe_preprocess"],
)
DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_BF16_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_V1
DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_FP8_CS_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_V1
DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_FP8_MX_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_V1
DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_NVFP4_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_V1


DEEPSEEK_V3_PRETRAIN_CONFIG_B300_V1 = replace(
    BASE_DEEPSEEK_V3_CONFIG,
    num_gpus=256,
    pipeline_model_parallel_size=16,
    expert_model_parallel_size=8,
    global_batch_size=2048,
    recompute_modules=["mla_up_proj"],
    moe_flex_dispatcher_backend="hybridep",
    moe_a2a_overlap=False,
)
DEEPSEEK_V3_PRETRAIN_CONFIG_B300_BF16_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_B300_V1
DEEPSEEK_V3_PRETRAIN_CONFIG_B300_FP8_CS_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_B300_V1
DEEPSEEK_V3_PRETRAIN_CONFIG_B300_FP8_MX_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_B300_FP8_CS_V1
DEEPSEEK_V3_PRETRAIN_CONFIG_B300_NVFP4_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_B300_V1


DEEPSEEK_V3_PRETRAIN_CONFIG_B200_V1 = replace(
    BASE_DEEPSEEK_V3_CONFIG,
    num_gpus=256,
    pipeline_model_parallel_size=16,
    expert_model_parallel_size=8,
    global_batch_size=2048,
    recompute_modules=["mla_up_proj"],
    moe_flex_dispatcher_backend="hybridep",
    moe_a2a_overlap=False,
)
DEEPSEEK_V3_PRETRAIN_CONFIG_B200_BF16_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_B200_V1
DEEPSEEK_V3_PRETRAIN_CONFIG_B200_FP8_CS_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_B200_V1
DEEPSEEK_V3_PRETRAIN_CONFIG_B200_FP8_MX_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_B200_FP8_CS_V1
DEEPSEEK_V3_PRETRAIN_CONFIG_B200_NVFP4_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_B200_V1


DEEPSEEK_V3_PRETRAIN_CONFIG_H100_V1 = replace(
    BASE_DEEPSEEK_V3_CONFIG,
    num_gpus=1024,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=64,
    global_batch_size=8192,
    recompute_modules=["mla_up_proj", "mlp"],
    moe_flex_dispatcher_backend="hybridep",
    moe_a2a_overlap=False,
    pp_layout="Et|(tt|)*30mL",
)
DEEPSEEK_V3_PRETRAIN_CONFIG_H100_BF16_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_H100_V1
DEEPSEEK_V3_PRETRAIN_CONFIG_H100_FP8_CS_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_H100_V1
DEEPSEEK_V3_PRETRAIN_CONFIG_H100_FP8_SC_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_H100_FP8_CS_V1


# =============================================================================
# DeepSeek V3 Pretrain - V2 (GBS=4096 for Blackwell, GBS=16384 for H100)
# =============================================================================

DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_V2 = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_V1,
    global_batch_size=4096,
)
DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_BF16_V2 = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_BF16_V1,
    global_batch_size=4096,
)
DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_FP8_CS_V2 = DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_V2
DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_FP8_MX_V2 = DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_V2
DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_NVFP4_V2 = DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_V2


DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_V2 = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_V1,
    global_batch_size=4096,
)
DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_BF16_V2 = DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_V2
DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_FP8_CS_V2 = DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_V2
DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_FP8_MX_V2 = DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_V2
DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_NVFP4_V2 = DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_V2


DEEPSEEK_V3_PRETRAIN_CONFIG_B300_V2 = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_B300_V1,
    global_batch_size=4096,
)
DEEPSEEK_V3_PRETRAIN_CONFIG_B300_BF16_V2 = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_B300_V2,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=2,
    recompute_modules=["mla_up_proj", "moe_act", "layernorm"],
)
DEEPSEEK_V3_PRETRAIN_CONFIG_B300_FP8_CS_V2 = DEEPSEEK_V3_PRETRAIN_CONFIG_B300_V2
DEEPSEEK_V3_PRETRAIN_CONFIG_B300_FP8_MX_V2 = DEEPSEEK_V3_PRETRAIN_CONFIG_B300_FP8_CS_V2
DEEPSEEK_V3_PRETRAIN_CONFIG_B300_NVFP4_V2 = DEEPSEEK_V3_PRETRAIN_CONFIG_B300_V2


DEEPSEEK_V3_PRETRAIN_CONFIG_B200_V2 = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_B200_V1,
    global_batch_size=4096,
)
DEEPSEEK_V3_PRETRAIN_CONFIG_B200_BF16_V2 = DEEPSEEK_V3_PRETRAIN_CONFIG_B200_V2
DEEPSEEK_V3_PRETRAIN_CONFIG_B200_FP8_CS_V2 = DEEPSEEK_V3_PRETRAIN_CONFIG_B200_V2
DEEPSEEK_V3_PRETRAIN_CONFIG_B200_FP8_MX_V2 = DEEPSEEK_V3_PRETRAIN_CONFIG_B200_FP8_CS_V2
DEEPSEEK_V3_PRETRAIN_CONFIG_B200_NVFP4_V2 = DEEPSEEK_V3_PRETRAIN_CONFIG_B200_V2


DEEPSEEK_V3_PRETRAIN_CONFIG_H100_V2 = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_H100_V1,
    global_batch_size=16384,
)
DEEPSEEK_V3_PRETRAIN_CONFIG_H100_BF16_V2 = DEEPSEEK_V3_PRETRAIN_CONFIG_H100_V2
DEEPSEEK_V3_PRETRAIN_CONFIG_H100_FP8_CS_V2 = DEEPSEEK_V3_PRETRAIN_CONFIG_H100_V2
DEEPSEEK_V3_PRETRAIN_CONFIG_H100_FP8_SC_V2 = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_H100_FP8_CS_V2,
    virtual_pipeline_model_parallel_size=2,
    pp_layout=None,
)


# =============================================================================
# DeepSeek V3 Pretrain - Large Scale Proxy
# =============================================================================

DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_FP8_MX_LARGE_SCALE = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_BF16_V1,
    global_batch_size=256,
)


DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_FP8_MX_LARGE_SCALE = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_FP8_CS_V1,
    global_batch_size=256,
)


DEEPSEEK_V3_PRETRAIN_CONFIG_B300_FP8_MX_LARGE_SCALE = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_B300_FP8_MX_V1,
    global_batch_size=256,
)


DEEPSEEK_V3_PRETRAIN_CONFIG_B200_FP8_MX_LARGE_SCALE = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_B200_FP8_MX_V1,
    global_batch_size=256,
)


DEEPSEEK_V3_PRETRAIN_CONFIG_H100_FP8_SC_LARGE_SCALE = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_H100_FP8_SC_V1,
    global_batch_size=1024,
    virtual_pipeline_model_parallel_size=2,
    pp_layout=None,
)


__all__ = [
    # V1 (original GBS settings)
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_BF16_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_FP8_CS_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_FP8_MX_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_NVFP4_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_BF16_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_FP8_CS_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_FP8_MX_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_NVFP4_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B300_BF16_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B300_FP8_CS_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B300_FP8_MX_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B300_NVFP4_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B200_BF16_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B200_FP8_CS_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B200_FP8_MX_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B200_NVFP4_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_H100_BF16_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_H100_FP8_CS_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_H100_FP8_SC_V1",
    # V2 (GBS=4096 for Blackwell, GBS=16384 for H100)
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_BF16_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_FP8_CS_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_FP8_MX_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_NVFP4_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_BF16_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_FP8_CS_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_FP8_MX_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_NVFP4_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B300_BF16_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B300_FP8_CS_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B300_FP8_MX_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B300_NVFP4_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B200_BF16_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B200_FP8_CS_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B200_FP8_MX_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B200_NVFP4_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_H100_BF16_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_H100_FP8_CS_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_H100_FP8_SC_V2",
    # Large Scale Proxy
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_FP8_MX_LARGE_SCALE",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_FP8_MX_LARGE_SCALE",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B300_FP8_MX_LARGE_SCALE",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B200_FP8_MX_LARGE_SCALE",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_H100_FP8_SC_LARGE_SCALE",
]
