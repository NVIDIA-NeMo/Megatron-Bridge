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
"""VRNVL72 performance recipes for Qwen3 MoE."""

import torch

from megatron.bridge.perf_recipes.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.perf_recipes.qwen.common import ConfigContainer
from megatron.bridge.perf_recipes.qwen.gb300.qwen3_moe import (
    _build_qwen3_30b_a3b_gb300_bf16,
    _build_qwen3_30b_a3b_gb300_fp8mx,
    _build_qwen3_235b_a22b_gb300_bf16,
    _build_qwen3_235b_a22b_gb300_fp8mx,
    _build_qwen3_235b_a22b_gb300_nvfp4,
    qwen3_30b_a3b_pretrain_8gpu_gb300_nvfp4_config,
)


def qwen3_235b_a22b_pretrain_256gpu_vrnvl72_bf16_config() -> ConfigContainer:
    """Qwen3 235B A22B pretrain: 256× VRNVL72, BF16 (alias of GB300)."""
    cfg = _build_qwen3_235b_a22b_gb300_bf16()
    cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]

    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # HybridEP topology for the target system.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


def _build_qwen3_235b_a22b_vrnvl72_bf16_benchmark(
    *,
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    expert_model_parallel_size: int,
    micro_batch_size: int,
    global_batch_size: int,
    optimizer_moments_dtype: torch.dtype,
) -> ConfigContainer:
    """Build one measured VRNVL72 BF16 benchmark point with a shared execution contract."""
    cfg = qwen3_235b_a22b_pretrain_256gpu_vrnvl72_bf16_config()

    cfg.model.tensor_model_parallel_size = tensor_model_parallel_size
    cfg.model.pipeline_model_parallel_size = pipeline_model_parallel_size
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = expert_model_parallel_size
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = tensor_model_parallel_size > 1

    cfg.train.micro_batch_size = micro_batch_size
    cfg.train.global_batch_size = global_batch_size

    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = []

    # Main parameters and main gradients remain FP32. Moments are BF16 on the
    # capacity-constrained 16-GPU cases and FP32 on the 32-GPU cases.
    cfg.optimizer.exp_avg_dtype = optimizer_moments_dtype
    cfg.optimizer.exp_avg_sq_dtype = optimizer_moments_dtype

    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.dist.use_megatron_fsdp = False
    return cfg


def qwen3_235b_a22b_pretrain_16gpu_vrnvl72_bf16_gbs16_config() -> ConfigContainer:
    """Qwen3 235B-A22B benchmark: 16× VRNVL72, BF16, GBS16."""
    return _build_qwen3_235b_a22b_vrnvl72_bf16_benchmark(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        expert_model_parallel_size=8,
        micro_batch_size=1,
        global_batch_size=16,
        optimizer_moments_dtype=torch.bfloat16,
    )


def qwen3_235b_a22b_pretrain_16gpu_vrnvl72_bf16_gbs256_config() -> ConfigContainer:
    """Qwen3 235B-A22B benchmark: 16× VRNVL72, BF16, GBS256."""
    return _build_qwen3_235b_a22b_vrnvl72_bf16_benchmark(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        expert_model_parallel_size=8,
        micro_batch_size=1,
        global_batch_size=256,
        optimizer_moments_dtype=torch.bfloat16,
    )


def qwen3_235b_a22b_pretrain_16gpu_vrnvl72_bf16_gbs1024_config() -> ConfigContainer:
    """Qwen3 235B-A22B benchmark: 16× VRNVL72, BF16, GBS1024."""
    return _build_qwen3_235b_a22b_vrnvl72_bf16_benchmark(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        expert_model_parallel_size=8,
        micro_batch_size=1,
        global_batch_size=1024,
        optimizer_moments_dtype=torch.bfloat16,
    )


def qwen3_235b_a22b_pretrain_32gpu_vrnvl72_bf16_gbs16_config() -> ConfigContainer:
    """Qwen3 235B-A22B benchmark: 32× VRNVL72, BF16, GBS16."""
    return _build_qwen3_235b_a22b_vrnvl72_bf16_benchmark(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=32,
        micro_batch_size=1,
        global_batch_size=16,
        optimizer_moments_dtype=torch.float32,
    )


def qwen3_235b_a22b_pretrain_32gpu_vrnvl72_bf16_gbs256_config() -> ConfigContainer:
    """Qwen3 235B-A22B benchmark: 32× VRNVL72, BF16, GBS256."""
    return _build_qwen3_235b_a22b_vrnvl72_bf16_benchmark(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=32,
        micro_batch_size=2,
        global_batch_size=256,
        optimizer_moments_dtype=torch.float32,
    )


def qwen3_235b_a22b_pretrain_32gpu_vrnvl72_bf16_gbs1024_config() -> ConfigContainer:
    """Qwen3 235B-A22B benchmark: 32× VRNVL72, BF16, GBS1024."""
    return _build_qwen3_235b_a22b_vrnvl72_bf16_benchmark(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=32,
        micro_batch_size=2,
        global_batch_size=1024,
        optimizer_moments_dtype=torch.float32,
    )


def qwen3_235b_a22b_pretrain_256gpu_vrnvl72_fp8mx_config() -> ConfigContainer:
    """Qwen3 235B A22B pretrain: 256× VRNVL72, FP8-MX (alias of GB300)."""
    cfg = _build_qwen3_235b_a22b_gb300_fp8mx()
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
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
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


def qwen3_235b_a22b_pretrain_256gpu_vrnvl72_nvfp4_config() -> ConfigContainer:
    """Qwen3 235B A22B pretrain: 256× VRNVL72, NVFP4, full-iteration CG and A2A overlap."""
    cfg = _build_qwen3_235b_a22b_gb300_nvfp4()

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
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        # Transformer Engine overlap settings for this model.
        "CUDNNFE_CLUSTER_OVERLAP_MARGIN": 8,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # NVFP4 fast-math path.
        "NVTE_USE_FAST_MATH": 1,
    }
    return cfg


def qwen3_30b_a3b_pretrain_8gpu_vrnvl72_bf16_config() -> ConfigContainer:
    """Qwen3 30B-A3B pretrain: 8× VRNVL72, BF16 (alias of GB300)."""
    cfg = _build_qwen3_30b_a3b_gb300_bf16()
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
        # HybridEP topology for the target system.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


def qwen3_30b_a3b_pretrain_8gpu_vrnvl72_fp8mx_config() -> ConfigContainer:
    """Qwen3 30B-A3B pretrain: 8× VRNVL72, FP8-MX (alias of GB300)."""
    cfg = _build_qwen3_30b_a3b_gb300_fp8mx()

    cfg.train.micro_batch_size = 4

    from megatron.core.transformer.enums import AttnBackend

    cfg.model.attention_backend = AttnBackend.auto

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
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
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


def qwen3_30b_a3b_pretrain_8gpu_vrnvl72_nvfp4_config() -> ConfigContainer:
    """Qwen3 30B-A3B pretrain: 8× VRNVL72, NVFP4 (same layout as GB300).

    NVFP4's fp4_param_gather path is incompatible with TP comm overlap, so it
    is disabled here.
    """
    cfg = qwen3_30b_a3b_pretrain_8gpu_gb300_nvfp4_config()
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_USE_FAST_MATH": 1,
    }
    return cfg
