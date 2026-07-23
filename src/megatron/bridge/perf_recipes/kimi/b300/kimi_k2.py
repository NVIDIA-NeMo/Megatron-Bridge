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
"""B300 performance recipes for Kimi K2."""

from megatron.bridge.perf_recipes.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.perf_recipes.kimi.b200.kimi_k2 import (
    kimi_k2_pretrain_256gpu_b200_bf16_config,
    kimi_k2_pretrain_256gpu_b200_fp8cs_config,
    kimi_k2_pretrain_256gpu_b200_fp8mx_config,
)
from megatron.bridge.perf_recipes.kimi.common import (
    ConfigContainer,
)


def kimi_k2_pretrain_256gpu_b300_bf16_config() -> ConfigContainer:
    """Kimi K2 pretrain: 256× B300, BF16."""
    cfg = kimi_k2_pretrain_256gpu_b200_bf16_config()
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 2
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # B300 CPU-affinity behavior.
        "NCCL_IGNORE_CPU_AFFINITY": 1,
    }
    return cfg


def kimi_k2_pretrain_256gpu_b300_fp8cs_config() -> ConfigContainer:
    """Kimi K2 pretrain: 256× B300, FP8 current-scaling."""
    cfg = kimi_k2_pretrain_256gpu_b200_fp8cs_config()
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 2
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # B300 CPU-affinity behavior.
        "NCCL_IGNORE_CPU_AFFINITY": 1,
    }
    return cfg


def kimi_k2_pretrain_256gpu_b300_fp8mx_config() -> ConfigContainer:
    """Kimi K2 pretrain: 256× B300, MXFP8."""
    cfg = kimi_k2_pretrain_256gpu_b200_fp8mx_config()
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 2
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # B300 CPU-affinity behavior.
        "NCCL_IGNORE_CPU_AFFINITY": 1,
    }
    return cfg
