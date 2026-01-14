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

"""Parallelism presets for Llama3 performance configs.

Config naming convention:
    {MODEL}_{SIZE}_{TASK}_CONFIG_{GPU}_{PRECISION}_{VERSION}

V1: Default configs (GBS=256 for 70B pretrain)
V2: Alternate configs (GBS=128 for 70B pretrain, original settings for 8B/SFT/LoRA)

Use --config_variant to select a variant (default: v1).
Use --list_config_variants to see available variants interactively.
"""

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

# For V1 configs with GBS=256
BASE_LLAMA3_70B_CONFIG_GBS256 = WorkloadBaseConfig(
    num_gpus=64,
    global_batch_size=256,
)

# =============================================================================
# Llama3 70B pretrain presets - V2 (GBS=128)
# =============================================================================

LLAMA3_70B_PRETRAIN_CONFIG_GB300_BF16_V2 = replace(
    BASE_LLAMA3_70B_CONFIG,
    micro_batch_size=2,
    use_megatron_fsdp=True,
    cpu_offloading_num_layers=30,
    nccl_ub=True,
)


LLAMA3_70B_PRETRAIN_CONFIG_GB300_FP8_CS_V2 = replace(
    BASE_LLAMA3_70B_CONFIG,
    micro_batch_size=2,
    use_megatron_fsdp=True,
    cpu_offloading_num_layers=20,
)


LLAMA3_70B_PRETRAIN_CONFIG_GB300_FP8_MX_V2 = replace(
    BASE_LLAMA3_70B_CONFIG,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=5,
)

LLAMA3_70B_PRETRAIN_CONFIG_GB300_NVFP4_V2 = replace(
    BASE_LLAMA3_70B_CONFIG,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=5,
    cuda_graph_impl="none",
    cuda_graph_scope="full_iteration",
)


LLAMA3_70B_PRETRAIN_CONFIG_GB200_BF16_V2 = replace(
    BASE_LLAMA3_70B_CONFIG,
    use_megatron_fsdp=True,
    cpu_offloading_num_layers=20,
)


LLAMA3_70B_PRETRAIN_CONFIG_GB200_FP8_CS_V2 = replace(
    BASE_LLAMA3_70B_CONFIG,
    micro_batch_size=2,
    use_megatron_fsdp=True,
    cpu_offloading_num_layers=40,
)


LLAMA3_70B_PRETRAIN_CONFIG_GB200_FP8_MX_V2 = replace(
    BASE_LLAMA3_70B_CONFIG,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=5,
)

LLAMA3_70B_PRETRAIN_CONFIG_GB200_NVFP4_V2 = replace(
    BASE_LLAMA3_70B_CONFIG,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=5,
    context_parallel_size=1,
    cuda_graph_impl="none",
    cuda_graph_scope="full_iteration",
)


LLAMA3_70B_PRETRAIN_CONFIG_B300_BF16_V2 = replace(
    BASE_LLAMA3_70B_CONFIG,
    micro_batch_size=1,
    use_megatron_fsdp=True,
)


LLAMA3_70B_PRETRAIN_CONFIG_B300_FP8_CS_V2 = replace(
    BASE_LLAMA3_70B_CONFIG,
    micro_batch_size=1,
    use_megatron_fsdp=True,
)


LLAMA3_70B_PRETRAIN_CONFIG_B300_FP8_MX_V2 = replace(
    BASE_LLAMA3_70B_CONFIG,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=5,
)

LLAMA3_70B_PRETRAIN_CONFIG_B300_NVFP4_V2 = replace(
    BASE_LLAMA3_70B_CONFIG,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=5,
    cuda_graph_impl="none",
    cuda_graph_scope="full_iteration",
)


LLAMA3_70B_PRETRAIN_CONFIG_B200_BF16_V2 = replace(
    BASE_LLAMA3_70B_CONFIG,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=4,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=5,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)


LLAMA3_70B_PRETRAIN_CONFIG_B200_FP8_CS_V2 = replace(
    BASE_LLAMA3_70B_CONFIG,
    use_megatron_fsdp=True,
    cpu_offloading_num_layers=5,
)


LLAMA3_70B_PRETRAIN_CONFIG_B200_FP8_MX_V2 = replace(
    BASE_LLAMA3_70B_CONFIG,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=5,
)

LLAMA3_70B_PRETRAIN_CONFIG_B200_NVFP4_V2 = replace(
    BASE_LLAMA3_70B_CONFIG,
    tensor_model_parallel_size=2,
    context_parallel_size=1,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=5,
)


LLAMA3_70B_PRETRAIN_CONFIG_H100_BF16_V2 = replace(
    BASE_LLAMA3_70B_CONFIG,
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=4,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=5,
)


LLAMA3_70B_PRETRAIN_CONFIG_H100_FP8_CS_V2 = replace(
    BASE_LLAMA3_70B_CONFIG,
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=5,
)


# =============================================================================
# Llama3 70B pretrain presets - V1 (default, GBS=256)
# =============================================================================

LLAMA3_70B_PRETRAIN_CONFIG_GB300_BF16_V1 = replace(
    BASE_LLAMA3_70B_CONFIG_GBS256,
    micro_batch_size=2,
    use_megatron_fsdp=True,
    cpu_offloading_num_layers=30,
    nccl_ub=True,
)


LLAMA3_70B_PRETRAIN_CONFIG_GB300_FP8_CS_V1 = replace(
    BASE_LLAMA3_70B_CONFIG_GBS256,
    micro_batch_size=2,
    use_megatron_fsdp=True,
    cpu_offloading_num_layers=20,
)


LLAMA3_70B_PRETRAIN_CONFIG_GB300_FP8_MX_V1 = replace(
    BASE_LLAMA3_70B_CONFIG_GBS256,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=5,
)

LLAMA3_70B_PRETRAIN_CONFIG_GB300_NVFP4_V1 = replace(
    BASE_LLAMA3_70B_CONFIG_GBS256,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=5,
    cuda_graph_impl="none",
    cuda_graph_scope="full_iteration",
)


LLAMA3_70B_PRETRAIN_CONFIG_GB200_BF16_V1 = replace(
    BASE_LLAMA3_70B_CONFIG_GBS256,
    use_megatron_fsdp=True,
    cpu_offloading_num_layers=20,
)


LLAMA3_70B_PRETRAIN_CONFIG_GB200_FP8_CS_V1 = replace(
    BASE_LLAMA3_70B_CONFIG_GBS256,
    micro_batch_size=2,
    use_megatron_fsdp=True,
    cpu_offloading_num_layers=40,
)


LLAMA3_70B_PRETRAIN_CONFIG_GB200_FP8_MX_V1 = replace(
    BASE_LLAMA3_70B_CONFIG_GBS256,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=5,
)

LLAMA3_70B_PRETRAIN_CONFIG_GB200_NVFP4_V1 = replace(
    BASE_LLAMA3_70B_CONFIG_GBS256,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=5,
    context_parallel_size=1,
    cuda_graph_impl="none",
    cuda_graph_scope="full_iteration",
)


LLAMA3_70B_PRETRAIN_CONFIG_B300_BF16_V1 = replace(
    BASE_LLAMA3_70B_CONFIG_GBS256,
    micro_batch_size=1,
    use_megatron_fsdp=True,
)


LLAMA3_70B_PRETRAIN_CONFIG_B300_FP8_CS_V1 = replace(
    BASE_LLAMA3_70B_CONFIG_GBS256,
    micro_batch_size=1,
    use_megatron_fsdp=True,
)


LLAMA3_70B_PRETRAIN_CONFIG_B300_FP8_MX_V1 = replace(
    BASE_LLAMA3_70B_CONFIG_GBS256,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=5,
)

LLAMA3_70B_PRETRAIN_CONFIG_B300_NVFP4_V1 = replace(
    BASE_LLAMA3_70B_CONFIG_GBS256,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=5,
    cuda_graph_impl="none",
    cuda_graph_scope="full_iteration",
)


LLAMA3_70B_PRETRAIN_CONFIG_B200_BF16_V1 = replace(
    BASE_LLAMA3_70B_CONFIG_GBS256,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=4,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=5,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)


LLAMA3_70B_PRETRAIN_CONFIG_B200_FP8_CS_V1 = replace(
    BASE_LLAMA3_70B_CONFIG_GBS256,
    use_megatron_fsdp=True,
    cpu_offloading_num_layers=5,
)


LLAMA3_70B_PRETRAIN_CONFIG_B200_FP8_MX_V1 = replace(
    BASE_LLAMA3_70B_CONFIG_GBS256,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=5,
)

LLAMA3_70B_PRETRAIN_CONFIG_B200_NVFP4_V1 = replace(
    BASE_LLAMA3_70B_CONFIG_GBS256,
    tensor_model_parallel_size=2,
    context_parallel_size=1,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=5,
)


LLAMA3_70B_PRETRAIN_CONFIG_H100_BF16_V1 = replace(
    BASE_LLAMA3_70B_CONFIG_GBS256,
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=4,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=5,
)


LLAMA3_70B_PRETRAIN_CONFIG_H100_FP8_CS_V1 = replace(
    BASE_LLAMA3_70B_CONFIG_GBS256,
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=5,
)


# =============================================================================
# Llama3 8B pretrain presets - V1 (only version)
# =============================================================================

LLAMA3_8B_PRETRAIN_CONFIG_GB300_BF16_V1 = replace(
    BASE_LLAMA3_8B_CONFIG,
    micro_batch_size=4,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)


LLAMA3_8B_PRETRAIN_CONFIG_GB300_FP8_CS_V1 = replace(
    BASE_LLAMA3_8B_CONFIG,
    micro_batch_size=4,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)

LLAMA3_8B_PRETRAIN_CONFIG_GB300_FP8_MX_V1 = LLAMA3_8B_PRETRAIN_CONFIG_GB300_FP8_CS_V1

LLAMA3_8B_PRETRAIN_CONFIG_GB300_NVFP4_V1 = replace(
    BASE_LLAMA3_8B_CONFIG,
    micro_batch_size=4,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)

LLAMA3_8B_PRETRAIN_CONFIG_GB200_BF16_V1 = replace(
    BASE_LLAMA3_8B_CONFIG,
    micro_batch_size=2,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)


LLAMA3_8B_PRETRAIN_CONFIG_GB200_FP8_CS_V1 = replace(
    BASE_LLAMA3_8B_CONFIG,
    micro_batch_size=2,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)

LLAMA3_8B_PRETRAIN_CONFIG_GB200_FP8_MX_V1 = LLAMA3_8B_PRETRAIN_CONFIG_GB200_FP8_CS_V1

LLAMA3_8B_PRETRAIN_CONFIG_GB200_NVFP4_V1 = replace(
    BASE_LLAMA3_8B_CONFIG,
    micro_batch_size=4,
    cuda_graph_impl="none",
    cuda_graph_scope="full_iteration",
)


LLAMA3_8B_PRETRAIN_CONFIG_B300_BF16_V1 = replace(
    BASE_LLAMA3_8B_CONFIG,
    micro_batch_size=4,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)


LLAMA3_8B_PRETRAIN_CONFIG_B300_FP8_CS_V1 = replace(
    BASE_LLAMA3_8B_CONFIG,
    micro_batch_size=4,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)


LLAMA3_8B_PRETRAIN_CONFIG_B300_FP8_MX_V1 = LLAMA3_8B_PRETRAIN_CONFIG_B300_FP8_CS_V1

LLAMA3_8B_PRETRAIN_CONFIG_B300_NVFP4_V1 = replace(
    BASE_LLAMA3_8B_CONFIG,
    micro_batch_size=4,
)


LLAMA3_8B_PRETRAIN_CONFIG_B200_BF16_V1 = replace(
    BASE_LLAMA3_8B_CONFIG,
    micro_batch_size=2,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)


LLAMA3_8B_PRETRAIN_CONFIG_B200_FP8_CS_V1 = replace(
    BASE_LLAMA3_8B_CONFIG,
    micro_batch_size=2,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)


LLAMA3_8B_PRETRAIN_CONFIG_B200_FP8_MX_V1 = LLAMA3_8B_PRETRAIN_CONFIG_B200_FP8_CS_V1

LLAMA3_8B_PRETRAIN_CONFIG_B200_NVFP4_V1 = replace(
    BASE_LLAMA3_8B_CONFIG,
    micro_batch_size=4,
)


LLAMA3_8B_PRETRAIN_CONFIG_H100_BF16_V1 = replace(
    BASE_LLAMA3_8B_CONFIG,
    context_parallel_size=2,
)


LLAMA3_8B_PRETRAIN_CONFIG_H100_FP8_CS_V1 = replace(
    BASE_LLAMA3_8B_CONFIG,
    context_parallel_size=1,
    recompute_num_layers=5,
)


# =============================================================================
# Llama3 8B finetune presets - V1 (only version)
# =============================================================================

_LLAMA3_8B_SFT_CONFIG_GB200 = replace(
    BASE_LLAMA3_8B_CONFIG,
    peft="none",
    micro_batch_size=1,
    global_batch_size=8,
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope="mlp",
)

LLAMA3_8B_SFT_CONFIG_GB200_BF16_V1 = _LLAMA3_8B_SFT_CONFIG_GB200
LLAMA3_8B_SFT_CONFIG_GB200_FP8_CS_V1 = _LLAMA3_8B_SFT_CONFIG_GB200
LLAMA3_8B_SFT_CONFIG_GB200_FP8_MX_V1 = _LLAMA3_8B_SFT_CONFIG_GB200


_LLAMA3_8B_SFT_CONFIG_H100 = replace(
    BASE_LLAMA3_8B_CONFIG,
    peft="none",
    micro_batch_size=1,
    global_batch_size=32,
)

LLAMA3_8B_SFT_CONFIG_H100_BF16_V1 = _LLAMA3_8B_SFT_CONFIG_H100
LLAMA3_8B_SFT_CONFIG_H100_FP8_CS_V1 = replace(
    _LLAMA3_8B_SFT_CONFIG_H100,
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope="mlp",
)
LLAMA3_8B_SFT_CONFIG_H100_FP8_MX_V1 = LLAMA3_8B_SFT_CONFIG_H100_FP8_CS_V1


# =============================================================================
# Llama3 70B finetune (SFT) presets - V1 (only version)
# =============================================================================

_LLAMA3_70B_SFT_CONFIG_GB300 = replace(
    BASE_LLAMA3_70B_CONFIG,
    num_gpus=32,
    peft="none",
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=5,
    micro_batch_size=1,
    global_batch_size=32,
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope="mlp",
)

LLAMA3_70B_SFT_CONFIG_GB300_BF16_V1 = _LLAMA3_70B_SFT_CONFIG_GB300
LLAMA3_70B_SFT_CONFIG_GB300_FP8_CS_V1 = _LLAMA3_70B_SFT_CONFIG_GB300
LLAMA3_70B_SFT_CONFIG_GB300_FP8_MX_V1 = _LLAMA3_70B_SFT_CONFIG_GB300


_LLAMA3_70B_SFT_CONFIG_GB200 = replace(
    BASE_LLAMA3_70B_CONFIG,
    num_gpus=32,
    peft="none",
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=5,
    micro_batch_size=1,
    global_batch_size=32,
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope="mlp",
)

LLAMA3_70B_SFT_CONFIG_GB200_BF16_V1 = _LLAMA3_70B_SFT_CONFIG_GB200
LLAMA3_70B_SFT_CONFIG_GB200_FP8_CS_V1 = _LLAMA3_70B_SFT_CONFIG_GB200
LLAMA3_70B_SFT_CONFIG_GB200_FP8_MX_V1 = _LLAMA3_70B_SFT_CONFIG_GB200


_LLAMA3_70B_SFT_CONFIG_H100 = replace(
    BASE_LLAMA3_70B_CONFIG,
    num_gpus=32,
    peft="none",
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=5,
    micro_batch_size=1,
    global_batch_size=32,
)

LLAMA3_70B_SFT_CONFIG_H100_BF16_V1 = _LLAMA3_70B_SFT_CONFIG_H100
LLAMA3_70B_SFT_CONFIG_H100_FP8_CS_V1 = _LLAMA3_70B_SFT_CONFIG_H100
LLAMA3_70B_SFT_CONFIG_H100_FP8_MX_V1 = _LLAMA3_70B_SFT_CONFIG_H100


# =============================================================================
# Llama3 70B finetune (LoRA) presets - V1 (only version)
# =============================================================================

_LLAMA3_70B_LORA_CONFIG_GB300 = replace(
    BASE_LLAMA3_70B_CONFIG,
    num_gpus=8,
    peft="lora",
    # pipeline_model_parallel_size=4,
    # virtual_pipeline_model_parallel_size=20,
    micro_batch_size=1,
    global_batch_size=64,
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope="mlp",
)

LLAMA3_70B_LORA_CONFIG_GB300_BF16_V1 = _LLAMA3_70B_LORA_CONFIG_GB300
LLAMA3_70B_LORA_CONFIG_GB300_FP8_CS_V1 = _LLAMA3_70B_LORA_CONFIG_GB300
LLAMA3_70B_LORA_CONFIG_GB300_FP8_MX_V1 = LLAMA3_70B_LORA_CONFIG_GB300_FP8_CS_V1


_LLAMA3_70B_LORA_CONFIG_GB200 = replace(
    BASE_LLAMA3_70B_CONFIG,
    num_gpus=8,
    peft="lora",
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=20,
    micro_batch_size=1,
    global_batch_size=64,
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope="mlp",
)

LLAMA3_70B_LORA_CONFIG_GB200_BF16_V1 = _LLAMA3_70B_LORA_CONFIG_GB200
LLAMA3_70B_LORA_CONFIG_GB200_FP8_CS_V1 = _LLAMA3_70B_LORA_CONFIG_GB200
LLAMA3_70B_LORA_CONFIG_GB200_FP8_MX_V1 = LLAMA3_70B_LORA_CONFIG_GB200_FP8_CS_V1


_LLAMA3_70B_LORA_CONFIG_H100 = replace(
    BASE_LLAMA3_70B_CONFIG,
    num_gpus=8,
    peft="lora",
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=20,
    micro_batch_size=1,
    global_batch_size=32,
)

LLAMA3_70B_LORA_CONFIG_H100_BF16_V1 = _LLAMA3_70B_LORA_CONFIG_H100
LLAMA3_70B_LORA_CONFIG_H100_FP8_CS_V1 = replace(
    LLAMA3_70B_LORA_CONFIG_H100_BF16_V1,
    recompute_num_layers=1,
)
LLAMA3_70B_LORA_CONFIG_H100_FP8_MX_V1 = LLAMA3_70B_LORA_CONFIG_H100_FP8_CS_V1


__all__ = [
    # 70B Pretrain V1 (GBS=256)
    "LLAMA3_70B_PRETRAIN_CONFIG_GB300_BF16_V1",
    "LLAMA3_70B_PRETRAIN_CONFIG_GB300_FP8_CS_V1",
    "LLAMA3_70B_PRETRAIN_CONFIG_GB300_FP8_MX_V1",
    "LLAMA3_70B_PRETRAIN_CONFIG_GB300_NVFP4_V1",
    "LLAMA3_70B_PRETRAIN_CONFIG_GB200_BF16_V1",
    "LLAMA3_70B_PRETRAIN_CONFIG_GB200_FP8_CS_V1",
    "LLAMA3_70B_PRETRAIN_CONFIG_GB200_FP8_MX_V1",
    "LLAMA3_70B_PRETRAIN_CONFIG_GB200_NVFP4_V1",
    "LLAMA3_70B_PRETRAIN_CONFIG_B300_BF16_V1",
    "LLAMA3_70B_PRETRAIN_CONFIG_B300_FP8_CS_V1",
    "LLAMA3_70B_PRETRAIN_CONFIG_B300_FP8_MX_V1",
    "LLAMA3_70B_PRETRAIN_CONFIG_B300_NVFP4_V1",
    "LLAMA3_70B_PRETRAIN_CONFIG_B200_BF16_V1",
    "LLAMA3_70B_PRETRAIN_CONFIG_B200_FP8_CS_V1",
    "LLAMA3_70B_PRETRAIN_CONFIG_B200_FP8_MX_V1",
    "LLAMA3_70B_PRETRAIN_CONFIG_B200_NVFP4_V1",
    "LLAMA3_70B_PRETRAIN_CONFIG_H100_BF16_V1",
    "LLAMA3_70B_PRETRAIN_CONFIG_H100_FP8_CS_V1",
    # 70B Pretrain V2 (GBS=128)
    "LLAMA3_70B_PRETRAIN_CONFIG_GB300_BF16_V2",
    "LLAMA3_70B_PRETRAIN_CONFIG_GB300_FP8_CS_V2",
    "LLAMA3_70B_PRETRAIN_CONFIG_GB300_FP8_MX_V2",
    "LLAMA3_70B_PRETRAIN_CONFIG_GB300_NVFP4_V2",
    "LLAMA3_70B_PRETRAIN_CONFIG_GB200_BF16_V2",
    "LLAMA3_70B_PRETRAIN_CONFIG_GB200_FP8_CS_V2",
    "LLAMA3_70B_PRETRAIN_CONFIG_GB200_FP8_MX_V2",
    "LLAMA3_70B_PRETRAIN_CONFIG_GB200_NVFP4_V2",
    "LLAMA3_70B_PRETRAIN_CONFIG_B300_BF16_V2",
    "LLAMA3_70B_PRETRAIN_CONFIG_B300_FP8_CS_V2",
    "LLAMA3_70B_PRETRAIN_CONFIG_B300_FP8_MX_V2",
    "LLAMA3_70B_PRETRAIN_CONFIG_B300_NVFP4_V2",
    "LLAMA3_70B_PRETRAIN_CONFIG_B200_BF16_V2",
    "LLAMA3_70B_PRETRAIN_CONFIG_B200_FP8_CS_V2",
    "LLAMA3_70B_PRETRAIN_CONFIG_B200_FP8_MX_V2",
    "LLAMA3_70B_PRETRAIN_CONFIG_B200_NVFP4_V2",
    "LLAMA3_70B_PRETRAIN_CONFIG_H100_BF16_V2",
    "LLAMA3_70B_PRETRAIN_CONFIG_H100_FP8_CS_V2",
    # 8B Pretrain V1 (only version)
    "LLAMA3_8B_PRETRAIN_CONFIG_GB300_BF16_V1",
    "LLAMA3_8B_PRETRAIN_CONFIG_GB300_FP8_CS_V1",
    "LLAMA3_8B_PRETRAIN_CONFIG_GB300_FP8_MX_V1",
    "LLAMA3_8B_PRETRAIN_CONFIG_GB300_NVFP4_V1",
    "LLAMA3_8B_PRETRAIN_CONFIG_GB200_BF16_V1",
    "LLAMA3_8B_PRETRAIN_CONFIG_GB200_FP8_CS_V1",
    "LLAMA3_8B_PRETRAIN_CONFIG_GB200_FP8_MX_V1",
    "LLAMA3_8B_PRETRAIN_CONFIG_GB200_NVFP4_V1",
    "LLAMA3_8B_PRETRAIN_CONFIG_B300_BF16_V1",
    "LLAMA3_8B_PRETRAIN_CONFIG_B300_FP8_CS_V1",
    "LLAMA3_8B_PRETRAIN_CONFIG_B300_FP8_MX_V1",
    "LLAMA3_8B_PRETRAIN_CONFIG_B300_NVFP4_V1",
    "LLAMA3_8B_PRETRAIN_CONFIG_B200_BF16_V1",
    "LLAMA3_8B_PRETRAIN_CONFIG_B200_FP8_CS_V1",
    "LLAMA3_8B_PRETRAIN_CONFIG_B200_FP8_MX_V1",
    "LLAMA3_8B_PRETRAIN_CONFIG_B200_NVFP4_V1",
    "LLAMA3_8B_PRETRAIN_CONFIG_H100_BF16_V1",
    "LLAMA3_8B_PRETRAIN_CONFIG_H100_FP8_CS_V1",
    # 8B SFT V1 (only version)
    "LLAMA3_8B_SFT_CONFIG_GB200_BF16_V1",
    "LLAMA3_8B_SFT_CONFIG_GB200_FP8_CS_V1",
    "LLAMA3_8B_SFT_CONFIG_GB200_FP8_MX_V1",
    "LLAMA3_8B_SFT_CONFIG_H100_BF16_V1",
    "LLAMA3_8B_SFT_CONFIG_H100_FP8_CS_V1",
    "LLAMA3_8B_SFT_CONFIG_H100_FP8_MX_V1",
    # 70B SFT V1 (only version)
    "LLAMA3_70B_SFT_CONFIG_GB200_BF16_V1",
    "LLAMA3_70B_SFT_CONFIG_GB200_FP8_CS_V1",
    "LLAMA3_70B_SFT_CONFIG_GB200_FP8_MX_V1",
    "LLAMA3_70B_SFT_CONFIG_H100_BF16_V1",
    "LLAMA3_70B_SFT_CONFIG_H100_FP8_CS_V1",
    "LLAMA3_70B_SFT_CONFIG_H100_FP8_MX_V1",
    "LLAMA3_70B_SFT_CONFIG_GB300_BF16_V1",
    "LLAMA3_70B_SFT_CONFIG_GB300_FP8_CS_V1",
    "LLAMA3_70B_SFT_CONFIG_GB300_FP8_MX_V1",
    # 70B LoRA V1 (only version)
    "LLAMA3_70B_LORA_CONFIG_GB200_BF16_V1",
    "LLAMA3_70B_LORA_CONFIG_GB200_FP8_CS_V1",
    "LLAMA3_70B_LORA_CONFIG_GB200_FP8_MX_V1",
    "LLAMA3_70B_LORA_CONFIG_H100_BF16_V1",
    "LLAMA3_70B_LORA_CONFIG_H100_FP8_CS_V1",
    "LLAMA3_70B_LORA_CONFIG_H100_FP8_MX_V1",
    "LLAMA3_70B_LORA_CONFIG_GB300_BF16_V1",
    "LLAMA3_70B_LORA_CONFIG_GB300_FP8_CS_V1",
    "LLAMA3_70B_LORA_CONFIG_GB300_FP8_MX_V1",
]
