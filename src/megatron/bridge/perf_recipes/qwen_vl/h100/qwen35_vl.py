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
"""H100 performance recipes for Qwen3.5-VL."""

import torch

from megatron.bridge.perf_recipes.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.perf_recipes.qwen_vl.common import (
    CommOverlapConfig,
    ConfigContainer,
    _benchmark_common,
    _perf_precision,
    _qwen35_vl_common,
    _qwen35_vl_post_clear_scope_with_overlap,
    qwen35_vl_35b_a3b_pretrain_mock_config,
    qwen35_vl_122b_a10b_pretrain_mock_config,
    qwen35_vl_397b_a17b_pretrain_mock_config,
)
from megatron.bridge.utils.cuda_graph import set_cuda_graph_modules


def qwen35_vl_35b_a3b_pretrain_16gpu_h100_bf16_config() -> ConfigContainer:
    """Qwen3.5-VL 35B-A3B pretrain: 16× H100, BF16, PP=2 EP=8."""
    cfg = qwen35_vl_35b_a3b_pretrain_mock_config()
    cfg.mixed_precision = _perf_precision("bf16")
    _qwen35_vl_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.num_layers_in_first_pipeline_stage = 17
    cfg.model.num_layers_in_last_pipeline_stage = 23
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1

    cfg.model.moe_shared_expert_overlap = False
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=False,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        overlap_param_gather_with_optimizer_step=False,
        overlap_moe_expert_parallel_comm=False,
        delay_wgrad_compute=False,
    )

    _benchmark_common(cfg)

    # Measured 16-H100 execution policy. Keep these settings after
    # _benchmark_common because that helper owns generic benchmark defaults.
    cfg.model.apply_rope_fusion = False
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_hybridep_num_sms = None
    cfg.model.moe_flex_dispatcher_num_sms = 16
    cfg.model.moe_hybridep_num_sms_preprocessing = 16
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_permute_fusion_into_hybridep = True
    cfg.model.overlap_dispatch_backward_with_experts_wgrad = True
    cfg.model.batch_p2p_sync = False

    cfg.model.cuda_graph_impl = "transformer_engine"
    set_cuda_graph_modules(cfg.model, ["attn", "moe_router", "moe_preprocess"])
    cfg.model.vision_cuda_graph_impl = "transformer_engine"
    cfg.model.vision_cuda_graph_scope = ["attn", "mlp"]
    cfg.model.max_vision_cuda_graph_seq_length = 784
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True

    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16
    cfg.optimizer.overlap_param_gather = False
    cfg.optimizer.overlap_param_gather_with_optimizer_step = False

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
        # HybridEP topology and chunking for two 8-GPU NVLink domains.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 64,
        "NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API": 64,
        "NUM_OF_TOKENS_PER_CHUNK_PREPROCESSING_API": 64,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 0,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 0,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
    }
    return cfg


def qwen35_vl_35b_a3b_pretrain_16gpu_h100_fp8cs_config() -> ConfigContainer:
    """Qwen3.5-VL 35B-A3B pretrain: 16× H100, FP8 current-scaling, PP=2 VP=12."""
    cfg = qwen35_vl_35b_a3b_pretrain_mock_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    _qwen35_vl_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 12
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1

    cfg.model.moe_shared_expert_overlap = False

    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=True,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        overlap_moe_expert_parallel_comm=True,
        delay_wgrad_compute=True,
    )

    _benchmark_common(cfg)
    _qwen35_vl_post_clear_scope_with_overlap(cfg)
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


def qwen35_vl_122b_a10b_pretrain_128gpu_h100_bf16_config() -> ConfigContainer:
    """Qwen3.5-VL 122B-A10B pretrain: 128× H100, BF16, TP=2 PP=8 VP=2 EP=16."""
    cfg = qwen35_vl_122b_a10b_pretrain_mock_config()
    cfg.mixed_precision = _perf_precision("bf16")
    _qwen35_vl_common(cfg)

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 2
    cfg.model.expert_model_parallel_size = 16
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1

    cfg.model.moe_shared_expert_overlap = False

    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=False,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        overlap_moe_expert_parallel_comm=True,
        delay_wgrad_compute=True,
    )

    _benchmark_common(cfg)
    _qwen35_vl_post_clear_scope_with_overlap(cfg)
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


def qwen35_vl_122b_a10b_pretrain_128gpu_h100_fp8cs_config() -> ConfigContainer:
    """Qwen3.5-VL 122B-A10B pretrain: 128× H100, FP8 current-scaling."""
    cfg = qwen35_vl_122b_a10b_pretrain_128gpu_h100_bf16_config()
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


def qwen35_vl_397b_a17b_pretrain_256gpu_h100_bf16_config() -> ConfigContainer:
    """Qwen3.5-VL 397B-A17B pretrain: 256× H100, BF16, TP=2 PP=8 VP=4 EP=32."""
    cfg = qwen35_vl_397b_a17b_pretrain_mock_config()
    cfg.mixed_precision = _perf_precision("bf16")
    _qwen35_vl_common(cfg)

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1

    cfg.model.moe_shared_expert_overlap = False

    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=False,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        overlap_moe_expert_parallel_comm=True,
        delay_wgrad_compute=True,
    )

    _benchmark_common(cfg)
    _qwen35_vl_post_clear_scope_with_overlap(cfg)
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


def qwen35_vl_397b_a17b_pretrain_256gpu_h100_fp8cs_config() -> ConfigContainer:
    """Qwen3.5-VL 397B-A17B pretrain: 256× H100, FP8 current-scaling."""
    cfg = qwen35_vl_397b_a17b_pretrain_256gpu_h100_bf16_config()
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
