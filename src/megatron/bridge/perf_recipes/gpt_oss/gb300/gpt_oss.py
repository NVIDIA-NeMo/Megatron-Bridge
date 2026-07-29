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
"""GB300 performance recipes for GPT-OSS."""

from megatron.bridge.perf_recipes.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.perf_recipes.gpt_oss.common import (
    ConfigContainer,
    _apply_gpt_oss_20b_common_configs,
    _apply_gpt_oss_20b_local_graph_configs,
    _apply_gpt_oss_20b_transformer_engine_graph_configs,
    _apply_gpt_oss_120b_full_iter_fp8mx_configs,
    _benchmark_common,
    _gpt_oss_20b_fp8mx_precision,
    _gpt_oss_20b_nvfp4_precision,
    _perf_precision,
    gpt_oss_20b_pretrain_config,
    gpt_oss_120b_pretrain_config,
)


def gpt_oss_20b_pretrain_8gpu_gb300_nvfp4_config() -> ConfigContainer:
    """GPT-OSS 20B pretrain: 8× GB300, NVFP4."""
    cfg = gpt_oss_20b_pretrain_config()
    cfg.mixed_precision = _gpt_oss_20b_nvfp4_precision()

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.context_parallel_size = 2
    cfg.model.expert_model_parallel_size = 4
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 4
    cfg.train.micro_batch_size = 1

    _benchmark_common(cfg)
    _apply_gpt_oss_20b_common_configs(cfg)
    _apply_gpt_oss_20b_transformer_engine_graph_configs(cfg)

    cfg.optimizer.lr = 0.0006
    cfg.optimizer.min_lr = 0.0006
    cfg.validation.eval_interval = 341
    cfg.validation.eval_iters = 29
    cfg.scheduler.lr_warmup_iters = 64
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        # CUDA graph and allocator behavior for this recipe.
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 1,
        "NCCL_CTA_POLICY": 1,
        # HybridEP topology for the target system.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 4,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # Use cuDNN LayerNorm for this measured baseline.
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        # NVFP4 fast-math path.
        "NVTE_USE_FAST_MATH": 1,
    }
    return cfg


def gpt_oss_20b_pretrain_72gpu_gb300_nvfp4_config() -> ConfigContainer:
    """GPT-OSS 20B pretrain: 72× GB300, NVFP4."""
    cfg = gpt_oss_20b_pretrain_config()
    cfg.mixed_precision = _gpt_oss_20b_nvfp4_precision()

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.context_parallel_size = 2
    cfg.model.expert_model_parallel_size = 4
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 36
    cfg.train.micro_batch_size = 1

    _benchmark_common(cfg)
    _apply_gpt_oss_20b_common_configs(cfg)
    _apply_gpt_oss_20b_transformer_engine_graph_configs(cfg)

    cfg.optimizer.lr = 0.0006
    cfg.optimizer.min_lr = 0.0006
    cfg.validation.eval_interval = 341
    cfg.validation.eval_iters = 29
    cfg.scheduler.lr_warmup_iters = 64
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        # CUDA graph and allocator behavior for this recipe.
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 1,
        "NCCL_CTA_POLICY": 1,
        # HybridEP topology for the target system.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 4,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # Use cuDNN LayerNorm for this measured baseline.
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        # NVFP4 fast-math path.
        "NVTE_USE_FAST_MATH": 1,
    }
    return cfg


def gpt_oss_20b_pretrain_512gpu_gb300_fp8mx_config() -> ConfigContainer:
    """GPT-OSS 20B pretrain: 512× GB300, MXFP8."""
    cfg = gpt_oss_20b_pretrain_config()
    cfg.mixed_precision = _gpt_oss_20b_fp8mx_precision()

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.context_parallel_size = 4
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 64
    cfg.train.micro_batch_size = 1

    _benchmark_common(cfg)
    _apply_gpt_oss_20b_common_configs(cfg)
    _apply_gpt_oss_20b_local_graph_configs(cfg)

    cfg.model.moe_expert_rank_capacity_factor = 7
    cfg.model.sequence_parallel = True
    cfg.optimizer.lr = 0.00052
    cfg.optimizer.min_lr = 0.00052
    cfg.validation.eval_interval = 192
    cfg.validation.eval_iters = 16
    cfg.scheduler.lr_warmup_iters = 32
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        # CUDA graph and allocator behavior for this recipe.
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 1,
        "NCCL_CTA_POLICY": 1,
        # HybridEP topology for the target system.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # Use cuDNN LayerNorm for this measured baseline.
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
    }
    return cfg


def gpt_oss_120b_pretrain_64gpu_gb300_bf16_config() -> ConfigContainer:
    """GPT-OSS 120B pretrain: 64× GB300, BF16, GBS=1280."""
    cfg = gpt_oss_120b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.model.moe_router_fusion = True
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 1280
    cfg.train.micro_batch_size = 4

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]

    _benchmark_common(cfg)
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
    }
    return cfg


def gpt_oss_120b_pretrain_64gpu_gb300_fp8mx_config() -> ConfigContainer:
    """GPT-OSS 120B pretrain: 64× GB300, FP8-MX."""
    cfg = gpt_oss_120b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    cfg.model.moe_router_fusion = True
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.expert_model_parallel_size = 16
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 1280
    cfg.train.micro_batch_size = 4

    _benchmark_common(cfg)
    _apply_gpt_oss_120b_full_iter_fp8mx_configs(cfg)
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # HybridEP topology for the target system.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 16,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        # Transformer Engine overlap settings for this model.
        "CUDNNFE_CLUSTER_OVERLAP_MARGIN": 8,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg
