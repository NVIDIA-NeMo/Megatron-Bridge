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
"""H100 performance recipes for Qwen3 MoE."""

import torch

from megatron.bridge.perf_recipes.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.perf_recipes.qwen.common import (
    CommOverlapConfig,
    ConfigContainer,
    _benchmark_common,
    _enable_overlap_param_gather_with_optimizer_step,
    _perf_precision,
    _with_global_batch_size,
    qwen3_30b_a3b_pretrain_config,
    qwen3_235b_a22b_pretrain_config,
    qwen3_next_80b_a3b_pretrain_config,
)
from megatron.bridge.recipes.qwen.h100.qwen3_moe import (
    qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config as _qwen3_30b_a3b_default_pretrain_config,
)


def qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config() -> ConfigContainer:
    """Qwen3 30B-A3B pretrain: 16× H100, BF16, EP=16."""
    cfg = _qwen3_30b_a3b_default_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.model.moe_router_force_load_balancing = True

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
        # HybridEP topology for the target system.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


def qwen3_30b_a3b_pretrain_16gpu_h100_fp8cs_config() -> ConfigContainer:
    """Qwen3 30B-A3B pretrain: 16× H100, FP8 current-scaling, EP=16."""
    cfg = qwen3_30b_a3b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.model.bias_activation_fusion = True
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.moe_router_fusion = True
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 16
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_a2a_overlap = False

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

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
        # HybridEP topology for the target system.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
    }
    return cfg


def qwen3_235b_a22b_pretrain_256gpu_h100_bf16_config() -> ConfigContainer:
    """Qwen3 235B-A22B pretrain: 256× H100, BF16, TP=2 PP=8 VP=4 EP=32."""
    cfg = qwen3_235b_a22b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.model.bias_activation_fusion = True
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.moe_router_fusion = True
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.expert_model_parallel_size = 32
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 8192
    cfg.train.micro_batch_size = 1

    cfg.model.moe_flex_dispatcher_backend = None
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_a2a_overlap = True
    cfg.model.moe_shared_expert_overlap = False

    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=False,
        overlap_moe_expert_parallel_comm=True,
        delay_wgrad_compute=True,
    )

    _benchmark_common(cfg)
    _enable_overlap_param_gather_with_optimizer_step(cfg)
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 8,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 16,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 16,
    }
    return cfg


def qwen3_235b_a22b_pretrain_256gpu_h100_fp8cs_config() -> ConfigContainer:
    """Qwen3 235B-A22B pretrain: 256× H100, FP8 current-scaling, TP=2 PP=8 VP=4 EP=32."""
    cfg = qwen3_235b_a22b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.model.bias_activation_fusion = True
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.moe_router_fusion = True
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.expert_model_parallel_size = 32
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 8192
    cfg.train.micro_batch_size = 1

    cfg.model.moe_flex_dispatcher_backend = None
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_a2a_overlap = True
    cfg.model.moe_shared_expert_overlap = False

    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=False,
        overlap_moe_expert_parallel_comm=True,
        delay_wgrad_compute=True,
    )

    _benchmark_common(cfg)
    _enable_overlap_param_gather_with_optimizer_step(cfg)
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 8,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 16,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 16,
    }
    return cfg


def qwen3_235b_a22b_pretrain_256gpu_h100_fp8cs_large_scale_config() -> ConfigContainer:
    """Qwen3 235B A22B pretrain: 256× H100, FP8-CS, large-scale proxy (GBS=512)."""
    cfg = qwen3_235b_a22b_pretrain_256gpu_h100_fp8cs_config()
    cfg.train.global_batch_size = 512
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 8,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 16,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 16,
    }
    return cfg


def qwen3_30b_a3b_pretrain_64gpu_h100_bf16_config() -> ConfigContainer:
    """Qwen3 30B-A3B pretrain: 64× H100, BF16, legacy-scaled GBS."""
    cfg = _with_global_batch_size(qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config(), 4096)
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
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


def qwen3_30b_a3b_pretrain_64gpu_h100_fp8cs_config() -> ConfigContainer:
    """Qwen3 30B-A3B pretrain: 64× H100, FP8 current-scaling, legacy-scaled GBS."""
    cfg = _with_global_batch_size(qwen3_30b_a3b_pretrain_16gpu_h100_fp8cs_config(), 4096)
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
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


def qwen3_next_80b_a3b_pretrain_128gpu_h100_bf16_config() -> ConfigContainer:
    """Qwen3 Next 80B-A3B pretrain: 128× H100, BF16, EP=128, MBS=1."""
    cfg = qwen3_next_80b_a3b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.model.bias_activation_fusion = True
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.moe_router_fusion = True
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 128
    cfg.model.expert_tensor_parallel_size = 1
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1

    cfg.model.moe_token_dispatcher_type = "alltoall"

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

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


def qwen3_next_80b_a3b_pretrain_128gpu_h100_fp8cs_config() -> ConfigContainer:
    """Qwen3 Next 80B-A3B pretrain: 128× H100, FP8-CS (same layout as BF16)."""
    cfg = qwen3_next_80b_a3b_pretrain_128gpu_h100_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
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


def qwen3_30b_a3b_pretrain_32gpu_h100_bf16_dev_config() -> ConfigContainer:
    """Qwen3 30B-A3B pretrain: 32 × H100, BF16, HybridEP."""
    cfg = qwen3_30b_a3b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    _benchmark_common(cfg)

    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.yarn_original_max_position_embeddings = 40960
    cfg.model.make_vocab_size_divisible_by = 1187
    cfg.model.moe_router_force_load_balancing = True
    cfg.model.moe_router_fusion = True
    cfg.model.moe_router_dtype = torch.float32
    cfg.model.moe_token_dispatcher_type = "flex"

    cfg.dataset.seq_length = cfg.model.seq_length
    cfg.train.manual_gc_interval = 5

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.moe_flex_dispatcher_backend = "hybridep"

    cfg.train.micro_batch_size = 1
    cfg.train.global_batch_size = 256
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
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


def qwen3_30b_a3b_pretrain_32gpu_h100_fp8sc_dev_config() -> ConfigContainer:
    """Qwen3 30B-A3B pretrain: 32 × H100, FP8 subchannel scaling and HybridEP."""
    cfg = qwen3_30b_a3b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.mixed_precision.fp8_recipe = "blockwise"
    _benchmark_common(cfg)

    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.yarn_original_max_position_embeddings = 40960
    cfg.model.make_vocab_size_divisible_by = 1187
    cfg.model.moe_router_force_load_balancing = True
    cfg.model.moe_router_fusion = True
    cfg.model.moe_router_dtype = torch.float32
    cfg.model.moe_token_dispatcher_type = "flex"

    cfg.dataset.seq_length = cfg.model.seq_length
    cfg.train.manual_gc_interval = 5

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
    cfg.model.use_te_rng_tracker = True

    cfg.train.micro_batch_size = 1
    cfg.train.global_batch_size = 256

    cfg.rng.te_rng_tracker = True
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16
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
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


def qwen3_235b_a22b_pretrain_256gpu_h100_bf16_hybridep_dev_config() -> ConfigContainer:
    """Qwen3 235B-A22B pretrain: 256× H100, BF16, HybridEP and scoped CUDA graphs."""
    cfg = qwen3_235b_a22b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    _benchmark_common(cfg)

    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.yarn_original_max_position_embeddings = 4096
    cfg.model.make_vocab_size_divisible_by = 1187
    cfg.model.moe_router_force_load_balancing = True
    cfg.model.moe_router_fusion = True
    cfg.model.moe_router_dtype = torch.float32
    cfg.model.moe_token_dispatcher_type = "flex"

    cfg.dataset.seq_length = cfg.model.seq_length
    cfg.train.manual_gc_interval = 5

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.virtual_pipeline_model_parallel_size = 3
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["moe_act", "layernorm"]
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]
    cfg.model.use_te_rng_tracker = True

    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=False,
        overlap_moe_expert_parallel_comm=True,
        delay_wgrad_compute=True,
    )
    cfg.train.micro_batch_size = 1
    cfg.train.global_batch_size = 2048
    cfg.checkpoint.dist_ckpt_strictness = "log_all"
    cfg.rng.te_rng_tracker = True
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
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg
