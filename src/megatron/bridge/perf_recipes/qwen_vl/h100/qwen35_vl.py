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

from megatron.bridge.perf_recipes.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.perf_recipes.qwen_vl.common import (
    CommOverlapConfig,
    ConfigContainer,
    _benchmark_common,
    _perf_precision,
    _qwen35_vl_common,
    _qwen35_vl_post_clear_scope_with_overlap,
    _qwen35_vl_post_with_overlap,
    qwen35_vl_35b_a3b_pretrain_mock_config,
    qwen35_vl_122b_a10b_pretrain_mock_config,
    qwen35_vl_397b_a17b_pretrain_mock_config,
)


def qwen35_vl_35b_a3b_pretrain_16gpu_h100_bf16_config() -> ConfigContainer:
    """Qwen3.5-VL 35B-A3B pretrain: 16× H100, BF16, PP=2 VP=12 EP=8."""
    cfg = qwen35_vl_35b_a3b_pretrain_mock_config()
    cfg.mixed_precision = _perf_precision("bf16")
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

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]

    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=True,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        overlap_moe_expert_parallel_comm=True,
        delay_wgrad_compute=True,
    )

    _benchmark_common(cfg)
    _qwen35_vl_post_with_overlap(cfg)
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
