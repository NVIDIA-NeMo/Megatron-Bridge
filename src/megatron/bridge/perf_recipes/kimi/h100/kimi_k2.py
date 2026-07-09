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
"""H100 performance recipes for Kimi K2."""

from megatron.bridge.perf_recipes.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.perf_recipes.kimi.common import (
    ConfigContainer,
    _benchmark_common,
    _perf_precision,
    kimi_k2_pretrain_config,
)


def kimi_k2_pretrain_1024gpu_h100_bf16_config() -> ConfigContainer:
    """Kimi K2 pretrain: 1024× H100, BF16."""
    cfg = kimi_k2_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = "selective"
    cfg.dist.enable_megatron_core_experimental = True
    cfg.model.moe_router_force_load_balancing = True
    cfg.model.qk_clip = False

    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 16
    cfg.model.virtual_pipeline_model_parallel_size = 2
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 8192
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = ["mla_up_proj", "mlp"]
    cfg.model.pipeline_model_parallel_layout = "Et|(tt|)*30L"
    cfg.model.moe_shared_expert_overlap = False

    cfg.comm_overlap.overlap_grad_reduce = False

    _benchmark_common(cfg)
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


def kimi_k2_pretrain_1024gpu_h100_fp8cs_config() -> ConfigContainer:
    """Kimi K2 pretrain: 1024× H100, FP8 current-scaling."""
    cfg = kimi_k2_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = "selective"
    cfg.dist.enable_megatron_core_experimental = True
    cfg.model.moe_router_force_load_balancing = True
    cfg.model.qk_clip = False

    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 16
    cfg.model.virtual_pipeline_model_parallel_size = 2
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 8192
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = ["mla_up_proj", "mlp"]
    cfg.model.pipeline_model_parallel_layout = "Et|(tt|)*30L"
    cfg.model.moe_shared_expert_overlap = False

    cfg.comm_overlap.overlap_grad_reduce = False

    _benchmark_common(cfg)
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


def kimi_k2_pretrain_1024gpu_h100_fp8sc_config() -> ConfigContainer:
    """Kimi K2 pretrain: 1024× H100, FP8-SC."""
    cfg = kimi_k2_pretrain_1024gpu_h100_fp8cs_config()
    cfg.mixed_precision.fp8_recipe = "blockwise"
    cfg.mixed_precision.fp8_param = False
    cfg.mixed_precision.fp8_param_gather = False
    cfg.mixed_precision.num_layers_at_start_in_bf16 = 0
    cfg.mixed_precision.num_layers_at_end_in_bf16 = 0
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg
