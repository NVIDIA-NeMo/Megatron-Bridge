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
"""H100 performance recipes for Qwen3-VL."""

from megatron.bridge.perf_recipes.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.perf_recipes.qwen_vl.common import (
    CommOverlapConfig,
    ConfigContainer,
    _benchmark_common,
    _finalize_qwen3_vl_with_moe_a2a_and_overlap,
    _perf_precision,
    qwen3_vl_30b_a3b_pretrain_mock_config,
    qwen3_vl_235b_a22b_pretrain_mock_config,
)


def qwen3_vl_235b_a22b_pretrain_256gpu_h100_bf16_config() -> ConfigContainer:
    """Qwen3-VL 235B-A22B pretrain: 256× H100, BF16, TP=2 PP=8 VP=4 EP=32."""
    cfg = qwen3_vl_235b_a22b_pretrain_mock_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.model.bias_activation_fusion = True
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.moe_router_fusion = True
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.model.moe_router_force_load_balancing = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.optimizer.overlap_param_gather = False

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.expert_model_parallel_size = 32
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1

    cfg.model.moe_a2a_overlap = True

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    _benchmark_common(cfg)
    _finalize_qwen3_vl_with_moe_a2a_and_overlap(cfg)
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


def qwen3_vl_235b_a22b_pretrain_256gpu_h100_fp8cs_config() -> ConfigContainer:
    """Qwen3-VL 235B-A22B pretrain: 256× H100, FP8 current-scaling, TP=2 PP=8 VP=4 EP=32."""
    cfg = qwen3_vl_235b_a22b_pretrain_mock_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.model.bias_activation_fusion = True
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.moe_router_fusion = True
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.model.moe_router_force_load_balancing = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.optimizer.overlap_param_gather = False

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.expert_model_parallel_size = 32
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1

    cfg.model.moe_a2a_overlap = True

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    _benchmark_common(cfg)
    _finalize_qwen3_vl_with_moe_a2a_and_overlap(cfg)
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


def qwen3_vl_30b_a3b_pretrain_16gpu_h100_bf16_config() -> ConfigContainer:
    """Qwen3-VL 30B-A3B pretrain: 16× H100, BF16, PP=2 VP=12 EP=8."""
    cfg = qwen3_vl_30b_a3b_pretrain_mock_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.model.bias_activation_fusion = True
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.moe_router_fusion = True
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.model.moe_router_force_load_balancing = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.optimizer.overlap_param_gather = False

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 12
    cfg.model.expert_model_parallel_size = 8
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]

    cfg.model.moe_a2a_overlap = True

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    _finalize_qwen3_vl_with_moe_a2a_and_overlap(cfg)
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


def qwen3_vl_30b_a3b_pretrain_16gpu_h100_fp8cs_config() -> ConfigContainer:
    """Qwen3-VL 30B-A3B pretrain: 16× H100, FP8 current-scaling, PP=2 VP=12 EP=8."""
    cfg = qwen3_vl_30b_a3b_pretrain_mock_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.model.bias_activation_fusion = True
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.moe_router_fusion = True
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.model.moe_router_force_load_balancing = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.optimizer.overlap_param_gather = False

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 12
    cfg.model.expert_model_parallel_size = 8
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1

    cfg.model.moe_a2a_overlap = True

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    _finalize_qwen3_vl_with_moe_a2a_and_overlap(cfg)
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
