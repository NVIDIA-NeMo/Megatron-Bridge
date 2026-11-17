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

"""qwen benchmark overrides applied to library recipes.

The literal names preserve existing CLI selectors. Common and model overrides
retain their ordering because graph, RNG and precision policies are dependent.
"""

import torch

from megatron.bridge.recipes.qwen.gb200.qwen3_moe import (
    qwen3_30b_a3b_pretrain_8gpu_gb200_fp8mx_config as _megatron_bridge_recipes_qwen_gb200_qwen3_moe_qwen3_30b_a3b_pretrain_8gpu_gb200_fp8mx_config,
)
from megatron.bridge.recipes.qwen.h100.qwen3_moe import qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config
from megatron.bridge.recipes.qwen.h100.qwen3_moe import (
    qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config as _megatron_bridge_recipes_qwen_h100_qwen3_moe_qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config,
)
from megatron.bridge.recipes.qwen.qwen3_moe import qwen3_235b_a22b_pretrain_config
from megatron.bridge.recipes.qwen.qwen3_next import qwen3_next_80b_a3b_pretrain_config
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.perf_benchmark_overrides._common import (
    _benchmark_common,
    _enable_ncclep,
    _perf_precision,
)
from megatron.bridge.training.perf_benchmark_overrides.environment import COMMON_PERF_ENV_VARS


BENCHMARK_ENVIRONMENTS = {
    "qwen3_235b_a22b_pretrain_128gpu_b200_fp8mx_deepep_dev": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    },
    "qwen3_235b_a22b_pretrain_256gpu_b200_bf16": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NCCL_EP_HT_EM_PULL_PUSH": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 16,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 16,
    },
    "qwen3_235b_a22b_pretrain_256gpu_b200_fp8cs": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 16,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 16,
    },
    "qwen3_235b_a22b_pretrain_256gpu_b200_fp8mx": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
        "NCCL_NVLS_ENABLE": 0,
        "NCCL_EP_HT_EM_PULL_PUSH": 1,
        "CUDNNFE_CLUSTER_OVERLAP_MARGIN": 8,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    },
    "qwen3_30b_a3b_pretrain_64gpu_b200_bf16": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    },
    "qwen3_30b_a3b_pretrain_8gpu_b200_bf16": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NCCL_EP_HT_EM_PULL_PUSH": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    },
    "qwen3_235b_a22b_pretrain_128gpu_b300_fp8mx_hybridep_dev": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NCCL_IGNORE_CPU_AFFINITY": 1,
    },
    "qwen3_235b_a22b_pretrain_256gpu_b300_bf16": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NCCL_EP_HT_EM_PULL_PUSH": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NCCL_IGNORE_CPU_AFFINITY": 1,
    },
    "qwen3_235b_a22b_pretrain_256gpu_b300_fp8mx": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
        "NCCL_NVLS_ENABLE": 0,
        "NCCL_EP_HT_EM_PULL_PUSH": 1,
        "CUDNNFE_CLUSTER_OVERLAP_MARGIN": 8,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NCCL_IGNORE_CPU_AFFINITY": 1,
    },
    "qwen3_next_80b_a3b_pretrain_64gpu_b300_bf16": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NCCL_IGNORE_CPU_AFFINITY": 1,
    },
    "qwen3_235b_a22b_pretrain_128gpu_gb200_fp8mx_paged_stash_dev": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
        "NCCL_NVLS_ENABLE": 0,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 64,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        "CUDNNFE_CLUSTER_OVERLAP_MARGIN": 8,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    },
    "qwen3_235b_a22b_pretrain_256gpu_gb200_fp8cs": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    },
    "qwen3_235b_a22b_pretrain_256gpu_gb200_fp8mx": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
        "NCCL_NVLS_ENABLE": 0,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        "CUDNNFE_CLUSTER_OVERLAP_MARGIN": 8,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    },
    "qwen3_235b_a22b_pretrain_256gpu_gb200_nvfp4": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
        "NCCL_NVLS_ENABLE": 0,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        "CUDNNFE_CLUSTER_OVERLAP_MARGIN": 8,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_USE_FAST_MATH": 1,
    },
    "qwen3_235b_a22b_pretrain_64gpu_gb200_bf16": {
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
    },
    "qwen3_235b_a22b_pretrain_64gpu_gb200_nvfp4": {
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
    },
    "qwen3_30b_a3b_pretrain_16gpu_gb200_bf16_dev": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 16,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    },
    "qwen3_235b_a22b_pretrain_256gpu_h100_bf16": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 8,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 16,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 16,
    },
}


def _with_global_batch_size(cfg: ConfigContainer, global_batch_size: int) -> ConfigContainer:
    cfg.train.global_batch_size = global_batch_size
    return cfg


def _enable_hybridep_full_iteration_mxfp8(cfg: ConfigContainer) -> None:
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_hybridep_num_sms = 32
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = None

    cfg.model.cuda_graph_impl = "full_iteration"
    cfg.model.cuda_graph_scope = []
    cfg.rng.te_rng_tracker = True
    cfg.model.use_te_rng_tracker = True

    cfg.model.offload_modules = []
    cfg.model.moe_pad_experts_for_cuda_graph_inference = True
    cfg.model.moe_paged_stash = True
    cfg.model.moe_expert_rank_capacity_factor = 1.5
    cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.2
    cfg.model.moe_paged_stash_buffer_size_factor_cpu = 1.0

    cfg.model.moe_shared_expert_overlap = False
    cfg.model.high_priority_a2a_comm_stream = True
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.model.moe_mlp_glu_interleave_size = 32
    cfg.model.moe_hybridep_num_sms_preprocessing = 32

    cfg.mixed_precision.fp8_dot_product_attention = True
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=True,
        overlap_moe_expert_parallel_comm=True,
        delay_wgrad_compute=True,
    )


def _enable_hybridep_full_iteration_nvfp4(cfg: ConfigContainer) -> None:
    """Enable the full-iteration HybridEP stack for an NVFP4 benchmark."""
    _enable_hybridep_full_iteration_mxfp8(cfg)
    cfg.mixed_precision.fp8_dot_product_attention = False
    cfg.comm_overlap.tp_comm_overlap = False
    cfg.model.high_priority_a2a_comm_stream = False


def _apply_qwen3_30b_a3b_gb300_bf16(cfg: ConfigContainer) -> ConfigContainer:
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.model.bias_activation_fusion = True
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = []
    cfg.model.moe_router_fusion = True
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.model.moe_router_force_load_balancing = True
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 8
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 8
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
    _benchmark_common(cfg)
    return cfg


def _apply_qwen3_30b_a3b_gb300_fp8mx(cfg: ConfigContainer) -> ConfigContainer:
    cfg.mixed_precision = _perf_precision("fp8_mx")
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
    cfg.model.expert_model_parallel_size = 8
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 8
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    _benchmark_common(cfg)
    _enable_hybridep_full_iteration_mxfp8(cfg)
    cfg.model.recompute_modules = []
    return cfg


def _apply_qwen3_235b_a22b_gb300_bf16(cfg: ConfigContainer) -> ConfigContainer:
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.model.bias_activation_fusion = True
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = []
    cfg.model.moe_router_fusion = True
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.model.moe_router_force_load_balancing = True
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 32
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 8192
    cfg.train.micro_batch_size = 2
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
    _benchmark_common(cfg)
    return cfg


def _apply_qwen3_235b_a22b_gb300_fp8mx(cfg: ConfigContainer) -> ConfigContainer:
    cfg.mixed_precision = _perf_precision("fp8_mx")
    cfg.model.bias_activation_fusion = True
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.moe_router_fusion = True
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.model.moe_router_force_load_balancing = True
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 12
    cfg.model.expert_model_parallel_size = 32
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 8192
    cfg.train.micro_batch_size = 2
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    _benchmark_common(cfg)
    _enable_hybridep_full_iteration_mxfp8(cfg)
    cfg.model.recompute_modules = []
    return cfg


def _apply_qwen3_235b_a22b_gb300_nvfp4(cfg: ConfigContainer) -> ConfigContainer:
    apply_overrides(cfg, "qwen3_235b_a22b_pretrain_256gpu_gb300_fp8cs_config")
    cfg.mixed_precision = _perf_precision("nvfp4")
    cfg.comm_overlap.tp_comm_overlap = False
    cfg.model.virtual_pipeline_model_parallel_size = 12
    _enable_hybridep_full_iteration_nvfp4(cfg)
    cfg.model.recompute_modules = []
    return cfg


LIBRARY_RECIPES = {
    "qwen3_235b_a22b_pretrain_128gpu_b200_fp8mx_deepep_dev_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_256gpu_b200_bf16_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_256gpu_b200_fp8cs_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_256gpu_b200_fp8mx_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_256gpu_b200_fp8mx_large_scale_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_256gpu_b200_nvfp4_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_64gpu_b200_bf16_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_64gpu_b200_fp8cs_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_64gpu_b200_fp8mx_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_64gpu_b200_nvfp4_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_30b_a3b_pretrain_64gpu_b200_bf16_config": qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config,
    "qwen3_30b_a3b_pretrain_64gpu_b200_fp8cs_config": qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config,
    "qwen3_30b_a3b_pretrain_8gpu_b200_bf16_config": qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config,
    "qwen3_30b_a3b_pretrain_8gpu_b200_fp8cs_config": qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config,
    "qwen3_30b_a3b_pretrain_8gpu_b200_fp8mx_config": qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config,
    "qwen3_30b_a3b_pretrain_8gpu_b200_nvfp4_config": qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config,
    "qwen3_next_80b_a3b_pretrain_64gpu_b200_bf16_config": qwen3_next_80b_a3b_pretrain_config,
    "qwen3_next_80b_a3b_pretrain_64gpu_b200_fp8mx_config": qwen3_next_80b_a3b_pretrain_config,
    "qwen3_235b_a22b_pretrain_128gpu_b300_fp8mx_hybridep_dev_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_256gpu_b300_bf16_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_256gpu_b300_fp8cs_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_256gpu_b300_fp8mx_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_256gpu_b300_fp8mx_large_scale_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_256gpu_b300_nvfp4_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_64gpu_b300_bf16_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_64gpu_b300_fp8cs_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_64gpu_b300_fp8mx_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_64gpu_b300_nvfp4_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_30b_a3b_pretrain_8gpu_b300_bf16_config": qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config,
    "qwen3_30b_a3b_pretrain_8gpu_b300_fp8cs_config": qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config,
    "qwen3_30b_a3b_pretrain_8gpu_b300_fp8mx_config": qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config,
    "qwen3_30b_a3b_pretrain_8gpu_b300_nvfp4_config": qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config,
    "qwen3_next_80b_a3b_pretrain_64gpu_b300_bf16_config": qwen3_next_80b_a3b_pretrain_config,
    "qwen3_next_80b_a3b_pretrain_64gpu_b300_fp8mx_config": qwen3_next_80b_a3b_pretrain_config,
    "qwen3_235b_a22b_pretrain_128gpu_gb200_fp8mx_paged_stash_dev_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_128gpu_gb200_fp8mx_partial_cg_dev_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_256gpu_gb200_bf16_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_256gpu_gb200_fp8cs_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_256gpu_gb200_fp8mx_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_256gpu_gb200_fp8mx_large_scale_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_256gpu_gb200_nvfp4_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_64gpu_gb200_bf16_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_64gpu_gb200_fp8cs_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_64gpu_gb200_fp8mx_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_64gpu_gb200_nvfp4_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_64gpu_gb200_nvfp4_full_iteration_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_30b_a3b_pretrain_16gpu_gb200_bf16_dev_config": qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config,
    "qwen3_30b_a3b_pretrain_16gpu_gb200_fp8mx_paged_stash_dev_config": qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config,
    "qwen3_30b_a3b_pretrain_16gpu_gb200_fp8mx_partial_cg_dev_config": qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config,
    "qwen3_30b_a3b_pretrain_32gpu_gb200_bf16_config": qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config,
    "qwen3_30b_a3b_pretrain_32gpu_gb200_fp8cs_config": qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config,
    "qwen3_30b_a3b_pretrain_8gpu_gb200_bf16_config": qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config,
    "qwen3_30b_a3b_pretrain_8gpu_gb200_fp8cs_config": qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config,
    "qwen3_30b_a3b_pretrain_8gpu_gb200_fp8mx_config": _megatron_bridge_recipes_qwen_gb200_qwen3_moe_qwen3_30b_a3b_pretrain_8gpu_gb200_fp8mx_config,
    "qwen3_30b_a3b_pretrain_8gpu_gb200_nvfp4_config": qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config,
    "qwen3_next_80b_a3b_pretrain_64gpu_gb200_bf16_config": qwen3_next_80b_a3b_pretrain_config,
    "qwen3_next_80b_a3b_pretrain_64gpu_gb200_fp8mx_config": qwen3_next_80b_a3b_pretrain_config,
    "qwen3_235b_a22b_pretrain_128gpu_gb300_fp8mx_paged_stash_dev_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_256gpu_gb300_bf16_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_256gpu_gb300_fp8cs_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_256gpu_gb300_fp8mx_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_256gpu_gb300_fp8mx_large_scale_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_256gpu_gb300_nvfp4_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_64gpu_gb300_bf16_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_64gpu_gb300_fp8cs_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_64gpu_gb300_fp8mx_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_64gpu_gb300_nvfp4_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_64gpu_gb300_nvfp4_full_iteration_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_30b_a3b_pretrain_32gpu_gb300_bf16_config": qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config,
    "qwen3_30b_a3b_pretrain_32gpu_gb300_fp8cs_config": qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config,
    "qwen3_30b_a3b_pretrain_8gpu_gb300_bf16_config": qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config,
    "qwen3_30b_a3b_pretrain_8gpu_gb300_fp8cs_config": qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config,
    "qwen3_30b_a3b_pretrain_8gpu_gb300_fp8mx_config": qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config,
    "qwen3_30b_a3b_pretrain_8gpu_gb300_nvfp4_config": qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config,
    "qwen3_next_80b_a3b_pretrain_64gpu_gb300_bf16_config": qwen3_next_80b_a3b_pretrain_config,
    "qwen3_next_80b_a3b_pretrain_64gpu_gb300_fp8mx_config": qwen3_next_80b_a3b_pretrain_config,
    "qwen3_235b_a22b_pretrain_256gpu_h100_bf16_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_256gpu_h100_bf16_hybridep_dev_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_256gpu_h100_fp8cs_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_256gpu_h100_fp8cs_large_scale_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config": _megatron_bridge_recipes_qwen_h100_qwen3_moe_qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config,
    "qwen3_30b_a3b_pretrain_16gpu_h100_fp8cs_config": qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config,
    "qwen3_30b_a3b_pretrain_16gpu_h100_fp8ds_config": qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config,
    "qwen3_30b_a3b_pretrain_32gpu_h100_bf16_dev_config": qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config,
    "qwen3_30b_a3b_pretrain_32gpu_h100_fp8sc_dev_config": qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config,
    "qwen3_30b_a3b_pretrain_64gpu_h100_bf16_config": _megatron_bridge_recipes_qwen_h100_qwen3_moe_qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config,
    "qwen3_30b_a3b_pretrain_64gpu_h100_fp8cs_config": qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config,
    "qwen3_next_80b_a3b_pretrain_128gpu_h100_bf16_config": qwen3_next_80b_a3b_pretrain_config,
    "qwen3_next_80b_a3b_pretrain_128gpu_h100_fp8cs_config": qwen3_next_80b_a3b_pretrain_config,
    "qwen3_235b_a22b_pretrain_256gpu_vrnvl72_bf16_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_256gpu_vrnvl72_fp8mx_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_235b_a22b_pretrain_256gpu_vrnvl72_nvfp4_config": qwen3_235b_a22b_pretrain_config,
    "qwen3_30b_a3b_pretrain_8gpu_vrnvl72_bf16_config": qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config,
    "qwen3_30b_a3b_pretrain_8gpu_vrnvl72_fp8mx_config": qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config,
    "qwen3_30b_a3b_pretrain_8gpu_vrnvl72_nvfp4_config": qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config,
}
ALLOCATION_AWARE = ()


def apply_overrides(cfg: ConfigContainer, name: str) -> ConfigContainer:
    """Apply model-specific benchmark overrides to an existing library config."""
    match name:
        case "qwen3_235b_a22b_pretrain_128gpu_b200_fp8mx_deepep_dev_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
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
            cfg.model.pipeline_model_parallel_size = 4
            cfg.model.virtual_pipeline_model_parallel_size = 2
            cfg.model.expert_model_parallel_size = 16
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.sequence_parallel = True
            cfg.model.moe_flex_dispatcher_backend = "deepep"
            cfg.model.moe_router_padding_for_quantization = True
            cfg.model.moe_deepep_num_sms = 20
            cfg.model.overlap_p2p_comm = True
            cfg.model.batch_p2p_comm = False
            cfg.comm_overlap = CommOverlapConfig(
                tp_comm_overlap=False, overlap_moe_expert_parallel_comm=True, delay_wgrad_compute=True
            )
            cfg.train.micro_batch_size = 3
            cfg.train.global_batch_size = 3072
            cfg.ddp.reuse_grad_buf_for_mxfp8_param_ag = True
            cfg.checkpoint.dist_ckpt_strictness = "log_all"
            cfg.optimizer.use_precision_aware_optimizer = True
            cfg.optimizer.exp_avg_dtype = torch.bfloat16
            cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_128gpu_b200_fp8mx_deepep_dev"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_256gpu_b200_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 4
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 8192
            cfg.train.micro_batch_size = 1
            cfg.model.moe_flex_dispatcher_backend = None
            cfg.model.moe_token_dispatcher_type = "alltoall"
            cfg.comm_overlap = CommOverlapConfig(
                tp_comm_overlap=True, overlap_param_gather=False, align_param_gather=False
            )
            cfg.ddp.overlap_param_gather = False
            cfg.ddp.align_param_gather = False
            cfg.optimizer.overlap_param_gather = False
            _benchmark_common(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_b200_bf16"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_256gpu_b200_fp8cs_config":
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
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 8192
            cfg.train.micro_batch_size = 1
            cfg.model.moe_flex_dispatcher_backend = None
            cfg.model.moe_token_dispatcher_type = "alltoall"
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_b200_fp8cs"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_256gpu_b200_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 3
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 8192
            cfg.train.micro_batch_size = 1
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            _enable_hybridep_full_iteration_mxfp8(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_b200_fp8mx"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_256gpu_b200_fp8mx_large_scale_config":
            apply_overrides(cfg, "qwen3_235b_a22b_pretrain_256gpu_b200_fp8mx_config")
            cfg.train.global_batch_size = 512
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_b200_fp8mx"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_256gpu_b200_nvfp4_config":
            apply_overrides(cfg, "qwen3_235b_a22b_pretrain_256gpu_b200_fp8cs_config")
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.model.virtual_pipeline_model_parallel_size = 2
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            _enable_hybridep_full_iteration_nvfp4(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
                "NCCL_NVLS_ENABLE": 0,
                "NCCL_EP_HT_EM_PULL_PUSH": 1,
                "CUDNNFE_CLUSTER_OVERLAP_MARGIN": 8,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 16,
                "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 16,
                "NVTE_USE_FAST_MATH": 1,
            }
            return cfg
        case "qwen3_235b_a22b_pretrain_64gpu_b200_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 4
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 2048
            cfg.train.micro_batch_size = 1
            cfg.model.moe_flex_dispatcher_backend = None
            cfg.model.moe_token_dispatcher_type = "alltoall"
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_b200_bf16"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_64gpu_b200_fp8cs_config":
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
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 2048
            cfg.train.micro_batch_size = 1
            cfg.model.moe_flex_dispatcher_backend = None
            cfg.model.moe_token_dispatcher_type = "alltoall"
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_b200_fp8cs"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_64gpu_b200_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 2048
            cfg.train.micro_batch_size = 1
            cfg.model.moe_flex_dispatcher_backend = None
            cfg.model.moe_token_dispatcher_type = "alltoall"
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_b200_bf16"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_64gpu_b200_nvfp4_config":
            apply_overrides(cfg, "qwen3_235b_a22b_pretrain_64gpu_b200_fp8cs_config")
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.comm_overlap.tp_comm_overlap = False
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
                "NCCL_NVLS_ENABLE": 0,
                "NCCL_EP_HT_EM_PULL_PUSH": 1,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 16,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 16,
                "NVTE_USE_FAST_MATH": 1,
            }
            return cfg
        case "qwen3_30b_a3b_pretrain_64gpu_b200_bf16_config":
            _with_global_batch_size(apply_overrides(cfg, "qwen3_30b_a3b_pretrain_8gpu_b200_bf16_config"), 4096)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_30b_a3b_pretrain_64gpu_b200_bf16"].copy()
            return cfg
        case "qwen3_30b_a3b_pretrain_64gpu_b200_fp8cs_config":
            _with_global_batch_size(apply_overrides(cfg, "qwen3_30b_a3b_pretrain_8gpu_b200_fp8cs_config"), 4096)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_30b_a3b_pretrain_64gpu_b200_bf16"].copy()
            return cfg
        case "qwen3_30b_a3b_pretrain_8gpu_b200_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
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
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 4
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_30b_a3b_pretrain_8gpu_b200_bf16"].copy()
            return cfg
        case "qwen3_30b_a3b_pretrain_8gpu_b200_fp8cs_config":
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
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 4
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_30b_a3b_pretrain_64gpu_b200_bf16"].copy()
            return cfg
        case "qwen3_30b_a3b_pretrain_8gpu_b200_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
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
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 4
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(
                tp_comm_overlap=True, overlap_moe_expert_parallel_comm=True, delay_wgrad_compute=True
            )
            _benchmark_common(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_flex_dispatcher_num_sms = 32
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
                "NCCL_NVLS_ENABLE": 0,
                "NCCL_EP_HT_EM_PULL_PUSH": 1,
                "NVTE_NORM_FWD_USE_CUDNN": 1,
                "NVTE_NORM_BWD_USE_CUDNN": 1,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
            }
            return cfg
        case "qwen3_30b_a3b_pretrain_8gpu_b200_nvfp4_config":
            apply_overrides(cfg, "qwen3_30b_a3b_pretrain_8gpu_b200_fp8cs_config")
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.comm_overlap.tp_comm_overlap = False
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
                "NCCL_NVLS_ENABLE": 0,
                "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
                "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
                "NVLINK_DOMAIN_SIZE": 8,
                "USE_MNNVL": 0,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_USE_FAST_MATH": 1,
            }
            return cfg
        case "qwen3_next_80b_a3b_pretrain_64gpu_b200_bf16_config":
            apply_overrides(cfg, "qwen3_next_80b_a3b_pretrain_64gpu_b300_bf16_config")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_128gpu_b200_fp8mx_deepep_dev"].copy()
            return cfg
        case "qwen3_next_80b_a3b_pretrain_64gpu_b200_fp8mx_config":
            apply_overrides(cfg, "qwen3_next_80b_a3b_pretrain_64gpu_b300_bf16_config")
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_128gpu_b200_fp8mx_deepep_dev"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_128gpu_b300_fp8mx_hybridep_dev_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
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
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.expert_model_parallel_size = 16
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.sequence_parallel = True
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_router_padding_for_quantization = True
            cfg.model.moe_hybridep_num_sms = 32
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router"]
            cfg.model.cuda_graph_warmup_steps = 2
            cfg.model.use_te_rng_tracker = True
            cfg.comm_overlap = CommOverlapConfig(
                tp_comm_overlap=False, overlap_moe_expert_parallel_comm=True, delay_wgrad_compute=True
            )
            cfg.train.micro_batch_size = 3
            cfg.train.global_batch_size = 9216
            cfg.ddp.reuse_grad_buf_for_mxfp8_param_ag = True
            cfg.checkpoint.dist_ckpt_strictness = "log_all"
            cfg.rng.te_rng_tracker = True
            cfg.optimizer.use_precision_aware_optimizer = True
            cfg.optimizer.exp_avg_dtype = torch.bfloat16
            cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_128gpu_b300_fp8mx_hybridep_dev"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_256gpu_b300_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 8192
            cfg.train.micro_batch_size = 2
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_b300_bf16"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_256gpu_b300_fp8cs_config":
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
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 8192
            cfg.train.micro_batch_size = 2
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_128gpu_b300_fp8mx_hybridep_dev"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_256gpu_b300_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 3
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 8192
            cfg.train.micro_batch_size = 2
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            _enable_hybridep_full_iteration_mxfp8(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_b300_fp8mx"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_256gpu_b300_fp8mx_large_scale_config":
            apply_overrides(cfg, "qwen3_235b_a22b_pretrain_256gpu_b300_fp8mx_config")
            cfg.train.global_batch_size = 512
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_b300_fp8mx"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_256gpu_b300_nvfp4_config":
            apply_overrides(cfg, "qwen3_235b_a22b_pretrain_256gpu_b300_fp8cs_config")
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.model.virtual_pipeline_model_parallel_size = 3
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            _enable_hybridep_full_iteration_nvfp4(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
                "NCCL_NVLS_ENABLE": 0,
                "NCCL_EP_HT_EM_PULL_PUSH": 1,
                "CUDNNFE_CLUSTER_OVERLAP_MARGIN": 8,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
                "NCCL_IGNORE_CPU_AFFINITY": 1,
                "NVTE_USE_FAST_MATH": 1,
            }
            return cfg
        case "qwen3_235b_a22b_pretrain_64gpu_b300_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 2048
            cfg.train.micro_batch_size = 2
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_b300_bf16"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_64gpu_b300_fp8cs_config":
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
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 2048
            cfg.train.micro_batch_size = 2
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_128gpu_b300_fp8mx_hybridep_dev"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_64gpu_b300_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 2048
            cfg.train.micro_batch_size = 2
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_b300_bf16"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_64gpu_b300_nvfp4_config":
            apply_overrides(cfg, "qwen3_235b_a22b_pretrain_64gpu_b300_fp8cs_config")
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.comm_overlap.tp_comm_overlap = False
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
                "NCCL_NVLS_ENABLE": 0,
                "NCCL_EP_HT_EM_PULL_PUSH": 1,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
                "NCCL_IGNORE_CPU_AFFINITY": 1,
                "NVTE_USE_FAST_MATH": 1,
            }
            return cfg
        case "qwen3_30b_a3b_pretrain_8gpu_b300_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
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
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 8
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_b300_bf16"].copy()
            return cfg
        case "qwen3_30b_a3b_pretrain_8gpu_b300_fp8cs_config":
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
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 8
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_128gpu_b300_fp8mx_hybridep_dev"].copy()
            return cfg
        case "qwen3_30b_a3b_pretrain_8gpu_b300_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
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
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 8
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            _benchmark_common(cfg)
            _enable_hybridep_full_iteration_mxfp8(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_b300_fp8mx"].copy()
            return cfg
        case "qwen3_30b_a3b_pretrain_8gpu_b300_nvfp4_config":
            apply_overrides(cfg, "qwen3_30b_a3b_pretrain_8gpu_b300_fp8cs_config")
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.comm_overlap.tp_comm_overlap = False
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
                "NCCL_NVLS_ENABLE": 0,
                "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
                "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
                "NVLINK_DOMAIN_SIZE": 8,
                "USE_MNNVL": 0,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
                "NCCL_IGNORE_CPU_AFFINITY": 1,
                "NVTE_USE_FAST_MATH": 1,
            }
            return cfg
        case "qwen3_next_80b_a3b_pretrain_64gpu_b300_bf16_config":
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
            cfg.model.expert_model_parallel_size = 64
            cfg.model.expert_tensor_parallel_size = 1
            cfg.train.global_batch_size = 1024
            cfg.train.micro_batch_size = 1
            cfg.model.moe_flex_dispatcher_backend = "deepep"
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_next_80b_a3b_pretrain_64gpu_b300_bf16"].copy()
            return cfg
        case "qwen3_next_80b_a3b_pretrain_64gpu_b300_fp8mx_config":
            apply_overrides(cfg, "qwen3_next_80b_a3b_pretrain_64gpu_b300_bf16_config")
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.train.micro_batch_size = 2
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_next_80b_a3b_pretrain_64gpu_b300_bf16"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_128gpu_gb200_fp8mx_paged_stash_dev_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
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
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.expert_model_parallel_size = 64
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.sequence_parallel = False
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_router_padding_for_quantization = True
            cfg.model.moe_hybridep_num_sms = 32
            cfg.model.moe_paged_stash = True
            cfg.model.moe_expert_rank_capacity_factor = 1.2
            cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.0
            cfg.model.moe_paged_stash_buffer_size_factor_cpu = 0.8
            cfg.model.moe_pad_experts_for_cuda_graph_inference = True
            cfg.model.moe_mlp_glu_interleave_size = 32
            cfg.model.use_transformer_engine_op_fuser = True
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_scope = "full_iteration"
            cfg.model.cuda_graph_warmup_steps = 2
            cfg.model.use_te_rng_tracker = True
            cfg.model.offload_modules = []
            cfg.comm_overlap = CommOverlapConfig(
                tp_comm_overlap=False, overlap_moe_expert_parallel_comm=True, delay_wgrad_compute=True
            )
            cfg.train.micro_batch_size = 1
            cfg.train.global_batch_size = 8192
            cfg.ddp.reuse_grad_buf_for_mxfp8_param_ag = True
            cfg.ddp.check_for_nan_in_grad = False
            cfg.checkpoint.dist_ckpt_strictness = "log_all"
            cfg.rng.te_rng_tracker = True
            cfg.dist.distributed_timeout_minutes = 220
            cfg.optimizer.use_precision_aware_optimizer = True
            cfg.optimizer.exp_avg_dtype = torch.bfloat16
            cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_128gpu_gb200_fp8mx_paged_stash_dev"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_128gpu_gb200_fp8mx_partial_cg_dev_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
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
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.expert_model_parallel_size = 64
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.sequence_parallel = False
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_router_padding_for_quantization = True
            cfg.model.moe_hybridep_num_sms = 32
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.model.cuda_graph_warmup_steps = 2
            cfg.model.use_te_rng_tracker = True
            cfg.comm_overlap = CommOverlapConfig(
                tp_comm_overlap=False, overlap_moe_expert_parallel_comm=True, delay_wgrad_compute=True
            )
            cfg.train.micro_batch_size = 1
            cfg.train.global_batch_size = 8192
            cfg.ddp.reuse_grad_buf_for_mxfp8_param_ag = True
            cfg.checkpoint.dist_ckpt_strictness = "log_all"
            cfg.rng.te_rng_tracker = True
            cfg.dist.distributed_timeout_minutes = 220
            cfg.optimizer.use_precision_aware_optimizer = True
            cfg.optimizer.exp_avg_dtype = torch.bfloat16
            cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
                "NCCL_NVLS_ENABLE": 0,
                "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 64,
                "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
                "NVLINK_DOMAIN_SIZE": 72,
                "USE_MNNVL": 1,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
            }
            return cfg
        case "qwen3_235b_a22b_pretrain_256gpu_gb200_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.recompute_modules = []
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 8192
            cfg.train.micro_batch_size = 1
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
                "TORCH_CPP_LOG_LEVEL": "ERROR",
                "NCCL_NVLS_ENABLE": 0,
                "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
                "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
                "NVLINK_DOMAIN_SIZE": 72,
                "USE_MNNVL": 1,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
            }
            return cfg
        case "qwen3_235b_a22b_pretrain_256gpu_gb200_fp8cs_config":
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
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 32
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 8192
            cfg.train.micro_batch_size = 1
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_gb200_fp8cs"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_256gpu_gb200_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 3
            cfg.model.expert_model_parallel_size = 32
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 8192
            cfg.train.micro_batch_size = 1
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            _benchmark_common(cfg)
            _enable_hybridep_full_iteration_mxfp8(cfg)
            cfg.model.recompute_modules = []
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_gb200_fp8mx"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_256gpu_gb200_fp8mx_large_scale_config":
            apply_overrides(cfg, "qwen3_235b_a22b_pretrain_256gpu_gb200_fp8mx_config")
            cfg.train.global_batch_size = 512
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_gb200_fp8mx"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_256gpu_gb200_nvfp4_config":
            apply_overrides(cfg, "qwen3_235b_a22b_pretrain_256gpu_gb200_fp8cs_config")
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.comm_overlap.tp_comm_overlap = False
            cfg.model.virtual_pipeline_model_parallel_size = 3
            _enable_hybridep_full_iteration_nvfp4(cfg)
            cfg.model.recompute_modules = []
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_gb200_nvfp4"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_64gpu_gb200_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 2048
            cfg.train.micro_batch_size = 1
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_64gpu_gb200_bf16"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_64gpu_gb200_fp8cs_config":
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
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 2048
            cfg.train.micro_batch_size = 1
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_64gpu_gb200_bf16"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_64gpu_gb200_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 3
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 2048
            cfg.train.micro_batch_size = 1
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            _benchmark_common(cfg)
            _enable_hybridep_full_iteration_mxfp8(cfg)
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
                "NCCL_NVLS_ENABLE": 0,
                "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
                "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
                "NVLINK_DOMAIN_SIZE": 72,
                "USE_MNNVL": 1,
                "CUDNNFE_CLUSTER_OVERLAP_MARGIN": 8,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
            }
            return cfg
        case "qwen3_235b_a22b_pretrain_64gpu_gb200_nvfp4_config":
            apply_overrides(cfg, "qwen3_235b_a22b_pretrain_64gpu_gb200_fp8cs_config")
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.comm_overlap.tp_comm_overlap = False
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_64gpu_gb200_nvfp4"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_64gpu_gb200_nvfp4_full_iteration_config":
            apply_overrides(cfg, "qwen3_235b_a22b_pretrain_64gpu_gb200_nvfp4_config")
            cfg.model.virtual_pipeline_model_parallel_size = 3
            _enable_hybridep_full_iteration_nvfp4(cfg)
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
                "NCCL_NVLS_ENABLE": 0,
                "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
                "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
                "NVLINK_DOMAIN_SIZE": 72,
                "USE_MNNVL": 1,
                "CUDNNFE_CLUSTER_OVERLAP_MARGIN": 8,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_USE_FAST_MATH": 1,
            }
            return cfg
        case "qwen3_30b_a3b_pretrain_16gpu_gb200_bf16_dev_config":
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
            cfg.model.expert_model_parallel_size = 16
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.sequence_parallel = False
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.train.micro_batch_size = 4
            cfg.train.global_batch_size = 512
            cfg.checkpoint.load_rng = False
            cfg.checkpoint.load_optim = False
            cfg.ddp.overlap_grad_reduce = False
            cfg.ddp.overlap_param_gather = False
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_30b_a3b_pretrain_16gpu_gb200_bf16_dev"].copy()
            return cfg
        case "qwen3_30b_a3b_pretrain_16gpu_gb200_fp8mx_paged_stash_dev_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.mixed_precision.fp8_param_gather = False
            cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
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
            cfg.model.expert_model_parallel_size = 16
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.sequence_parallel = False
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_scope = "full_iteration"
            cfg.model.use_transformer_engine_op_fuser = True
            cfg.model.moe_paged_stash = True
            cfg.model.moe_expert_rank_capacity_factor = 1.5
            cfg.model.moe_paged_stash_page_size = 64
            cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.1
            cfg.model.moe_pad_experts_for_cuda_graph_inference = True
            cfg.model.moe_mlp_glu_interleave_size = 32
            cfg.model.use_te_rng_tracker = True
            cfg.model.offload_modules = []
            cfg.train.micro_batch_size = 4
            cfg.train.global_batch_size = 512
            cfg.checkpoint.load_rng = False
            cfg.checkpoint.load_optim = False
            cfg.ddp.overlap_grad_reduce = False
            cfg.ddp.overlap_param_gather = False
            cfg.rng.te_rng_tracker = True
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
                "NCCL_NVLS_ENABLE": 0,
                "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 16,
                "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
                "NVLINK_DOMAIN_SIZE": 72,
                "USE_MNNVL": 1,
                "CUDNNFE_CLUSTER_OVERLAP_MARGIN": 8,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
            }
            return cfg
        case "qwen3_30b_a3b_pretrain_16gpu_gb200_fp8mx_partial_cg_dev_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.mixed_precision.fp8_param_gather = False
            cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
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
            cfg.model.expert_model_parallel_size = 16
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.sequence_parallel = False
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.external_cuda_graph = True
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.model.use_te_rng_tracker = True
            cfg.train.micro_batch_size = 4
            cfg.train.global_batch_size = 512
            cfg.checkpoint.load_rng = False
            cfg.checkpoint.load_optim = False
            cfg.ddp.overlap_grad_reduce = False
            cfg.ddp.overlap_param_gather = False
            cfg.rng.te_rng_tracker = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_30b_a3b_pretrain_16gpu_gb200_bf16_dev"].copy()
            return cfg
        case "qwen3_30b_a3b_pretrain_32gpu_gb200_bf16_config":
            _with_global_batch_size(apply_overrides(cfg, "qwen3_30b_a3b_pretrain_8gpu_gb200_bf16_config"), 2048)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_64gpu_gb200_bf16"].copy()
            return cfg
        case "qwen3_30b_a3b_pretrain_32gpu_gb200_fp8cs_config":
            _with_global_batch_size(apply_overrides(cfg, "qwen3_30b_a3b_pretrain_8gpu_gb200_fp8cs_config"), 2048)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_64gpu_gb200_bf16"].copy()
            return cfg
        case "qwen3_30b_a3b_pretrain_8gpu_gb200_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.recompute_modules = []
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 4
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_64gpu_gb200_bf16"].copy()
            return cfg
        case "qwen3_30b_a3b_pretrain_8gpu_gb200_fp8cs_config":
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
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 4
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_64gpu_gb200_bf16"].copy()
            return cfg
        case "qwen3_30b_a3b_pretrain_8gpu_gb200_fp8mx_config":
            _benchmark_common(cfg)
            _enable_hybridep_full_iteration_mxfp8(cfg)
            _enable_ncclep(cfg)
            cfg.model.recompute_modules = []
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_b200_fp8mx"].copy()
            return cfg
        case "qwen3_30b_a3b_pretrain_8gpu_gb200_nvfp4_config":
            apply_overrides(cfg, "qwen3_30b_a3b_pretrain_8gpu_gb200_fp8cs_config")
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.comm_overlap.tp_comm_overlap = False
            cfg.optimizer.use_precision_aware_optimizer = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_64gpu_gb200_nvfp4"].copy()
            return cfg
        case "qwen3_next_80b_a3b_pretrain_64gpu_gb200_bf16_config":
            apply_overrides(cfg, "qwen3_next_80b_a3b_pretrain_64gpu_gb300_bf16_config")
            cfg.train.micro_batch_size = 2
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_gb200_fp8cs"].copy()
            return cfg
        case "qwen3_next_80b_a3b_pretrain_64gpu_gb200_fp8mx_config":
            apply_overrides(cfg, "qwen3_next_80b_a3b_pretrain_64gpu_gb200_bf16_config")
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.optimizer.overlap_param_gather_with_optimizer_step = False
            cfg.comm_overlap.overlap_param_gather_with_optimizer_step = None
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_gb200_fp8cs"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_128gpu_gb300_fp8mx_paged_stash_dev_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
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
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.expert_model_parallel_size = 64
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.sequence_parallel = False
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_router_padding_for_quantization = True
            cfg.model.moe_hybridep_num_sms = 32
            cfg.model.moe_paged_stash = True
            cfg.model.moe_expert_rank_capacity_factor = 1.2
            cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.0
            cfg.model.moe_pad_experts_for_cuda_graph_inference = True
            cfg.model.moe_mlp_glu_interleave_size = 32
            cfg.model.use_transformer_engine_op_fuser = True
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_scope = "full_iteration"
            cfg.model.cuda_graph_warmup_steps = 2
            cfg.model.use_te_rng_tracker = True
            cfg.model.offload_modules = []
            cfg.comm_overlap = CommOverlapConfig(
                tp_comm_overlap=False, overlap_moe_expert_parallel_comm=True, delay_wgrad_compute=True
            )
            cfg.train.micro_batch_size = 1
            cfg.train.global_batch_size = 8192
            cfg.ddp.reuse_grad_buf_for_mxfp8_param_ag = True
            cfg.ddp.check_for_nan_in_grad = False
            cfg.checkpoint.dist_ckpt_strictness = "log_all"
            cfg.rng.te_rng_tracker = True
            cfg.dist.distributed_timeout_minutes = 220
            cfg.optimizer.use_precision_aware_optimizer = True
            cfg.optimizer.exp_avg_dtype = torch.bfloat16
            cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_128gpu_gb200_fp8mx_paged_stash_dev"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_256gpu_gb300_bf16_config":
            _apply_qwen3_235b_a22b_gb300_bf16(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
                "TORCH_CPP_LOG_LEVEL": "ERROR",
                "NCCL_NVLS_ENABLE": 0,
                "NCCL_EP_HT_EM_PULL_PUSH": 1,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
            }
            return cfg
        case "qwen3_235b_a22b_pretrain_256gpu_gb300_fp8cs_config":
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
            cfg.model.pipeline_model_parallel_size = 4
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 32
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 8192
            cfg.train.micro_batch_size = 2
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_gb200_fp8cs"].copy()
            return cfg
        case (
            "qwen3_235b_a22b_pretrain_256gpu_gb300_fp8mx_config"
            | "qwen3_235b_a22b_pretrain_256gpu_vrnvl72_fp8mx_config"
        ):
            _apply_qwen3_235b_a22b_gb300_fp8mx(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_b200_fp8mx"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_256gpu_gb300_fp8mx_large_scale_config":
            apply_overrides(cfg, "qwen3_235b_a22b_pretrain_256gpu_gb300_fp8mx_config")
            cfg.train.global_batch_size = 512
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_b200_fp8mx"].copy()
            return cfg
        case (
            "qwen3_235b_a22b_pretrain_256gpu_gb300_nvfp4_config"
            | "qwen3_235b_a22b_pretrain_256gpu_vrnvl72_nvfp4_config"
        ):
            _apply_qwen3_235b_a22b_gb300_nvfp4(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
                "NCCL_NVLS_ENABLE": 0,
                "NCCL_EP_HT_EM_PULL_PUSH": 1,
                "CUDNNFE_CLUSTER_OVERLAP_MARGIN": 8,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_USE_FAST_MATH": 1,
            }
            return cfg
        case "qwen3_235b_a22b_pretrain_64gpu_gb300_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 2
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 32
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 2048
            cfg.train.micro_batch_size = 2
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_gb200_fp8cs"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_64gpu_gb300_fp8cs_config":
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
            cfg.model.pipeline_model_parallel_size = 2
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 32
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 2048
            cfg.train.micro_batch_size = 2
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_gb200_fp8cs"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_64gpu_gb300_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 2
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 12
            cfg.model.expert_model_parallel_size = 32
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 2048
            cfg.train.micro_batch_size = 2
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            _benchmark_common(cfg)
            _enable_hybridep_full_iteration_mxfp8(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_gb200_fp8mx"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_64gpu_gb300_nvfp4_config":
            apply_overrides(cfg, "qwen3_235b_a22b_pretrain_64gpu_gb300_fp8cs_config")
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.comm_overlap.tp_comm_overlap = False
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
                "NCCL_NVLS_ENABLE": 0,
                "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
                "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
                "NVLINK_DOMAIN_SIZE": 72,
                "USE_MNNVL": 1,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_USE_FAST_MATH": 1,
            }
            return cfg
        case "qwen3_235b_a22b_pretrain_64gpu_gb300_nvfp4_full_iteration_config":
            apply_overrides(cfg, "qwen3_235b_a22b_pretrain_64gpu_gb300_nvfp4_config")
            cfg.model.virtual_pipeline_model_parallel_size = 12
            _enable_hybridep_full_iteration_nvfp4(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_gb200_nvfp4"].copy()
            return cfg
        case "qwen3_30b_a3b_pretrain_32gpu_gb300_bf16_config":
            _with_global_batch_size(apply_overrides(cfg, "qwen3_30b_a3b_pretrain_8gpu_gb300_bf16_config"), 2048)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_30b_a3b_pretrain_8gpu_b200_bf16"].copy()
            return cfg
        case "qwen3_30b_a3b_pretrain_32gpu_gb300_fp8cs_config":
            _with_global_batch_size(apply_overrides(cfg, "qwen3_30b_a3b_pretrain_8gpu_gb300_fp8cs_config"), 2048)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_64gpu_gb200_bf16"].copy()
            return cfg
        case "qwen3_30b_a3b_pretrain_8gpu_gb300_bf16_config" | "qwen3_30b_a3b_pretrain_8gpu_vrnvl72_bf16_config":
            _apply_qwen3_30b_a3b_gb300_bf16(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_30b_a3b_pretrain_8gpu_b200_bf16"].copy()
            return cfg
        case "qwen3_30b_a3b_pretrain_8gpu_gb300_fp8cs_config":
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
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 8
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_64gpu_gb200_bf16"].copy()
            return cfg
        case "qwen3_30b_a3b_pretrain_8gpu_gb300_fp8mx_config":
            _apply_qwen3_30b_a3b_gb300_fp8mx(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_b200_fp8mx"].copy()
            return cfg
        case "qwen3_30b_a3b_pretrain_8gpu_gb300_nvfp4_config":
            apply_overrides(cfg, "qwen3_30b_a3b_pretrain_8gpu_gb300_fp8cs_config")
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.comm_overlap.tp_comm_overlap = False
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_64gpu_gb200_nvfp4"].copy()
            return cfg
        case "qwen3_next_80b_a3b_pretrain_64gpu_gb300_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 2
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 4
            cfg.model.expert_model_parallel_size = 32
            cfg.model.expert_tensor_parallel_size = 1
            cfg.train.global_batch_size = 1024
            cfg.train.micro_batch_size = 4
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.model.pipeline_model_parallel_layout = "Et*4|(t*7|)*5t*8|tmL"
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            cfg.model.moe_hybridep_num_sms = 16
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_gb200_fp8cs"].copy()
            return cfg
        case "qwen3_next_80b_a3b_pretrain_64gpu_gb300_fp8mx_config":
            apply_overrides(cfg, "qwen3_next_80b_a3b_pretrain_64gpu_gb300_bf16_config")
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_gb200_fp8cs"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_256gpu_h100_bf16_config":
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
            cfg.model.moe_shared_expert_overlap = False
            cfg.comm_overlap = CommOverlapConfig(
                tp_comm_overlap=False, overlap_moe_expert_parallel_comm=True, delay_wgrad_compute=True
            )
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_h100_bf16"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_256gpu_h100_bf16_hybridep_dev_config":
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
                tp_comm_overlap=False, overlap_moe_expert_parallel_comm=True, delay_wgrad_compute=True
            )
            cfg.train.micro_batch_size = 1
            cfg.train.global_batch_size = 2048
            cfg.checkpoint.dist_ckpt_strictness = "log_all"
            cfg.rng.te_rng_tracker = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_30b_a3b_pretrain_64gpu_b200_bf16"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_256gpu_h100_fp8cs_config":
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
            cfg.model.moe_shared_expert_overlap = False
            cfg.comm_overlap = CommOverlapConfig(
                tp_comm_overlap=False, overlap_moe_expert_parallel_comm=True, delay_wgrad_compute=True
            )
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_h100_bf16"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_256gpu_h100_fp8cs_large_scale_config":
            apply_overrides(cfg, "qwen3_235b_a22b_pretrain_256gpu_h100_fp8cs_config")
            cfg.train.global_batch_size = 512
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_h100_bf16"].copy()
            return cfg
        case "qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.moe_permute_fusion_into_hybridep = True
            cfg.comm_overlap.overlap_moe_expert_parallel_comm = True
            cfg.comm_overlap.delay_wgrad_compute = False
            _benchmark_common(cfg)
            cfg.env_vars = {
                **cfg.env_vars,
                "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 64,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 10,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 10,
            }
            return cfg
        case "qwen3_30b_a3b_pretrain_16gpu_h100_fp8cs_config":
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
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            cfg.env_vars = {
                **cfg.env_vars,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
                "NVLINK_DOMAIN_SIZE": 8,
                "USE_MNNVL": 0,
                "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_NORM_FWD_USE_CUDNN": 1,
                "NVTE_NORM_BWD_USE_CUDNN": 1,
            }
            return cfg
        case "qwen3_30b_a3b_pretrain_16gpu_h100_fp8ds_config":
            apply_overrides(cfg, "qwen3_30b_a3b_pretrain_16gpu_h100_fp8cs_config")
            cfg.mixed_precision = _perf_precision("fp8_ds")
            return cfg
        case "qwen3_30b_a3b_pretrain_32gpu_h100_bf16_dev_config":
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
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_30b_a3b_pretrain_64gpu_b200_bf16"].copy()
            return cfg
        case "qwen3_30b_a3b_pretrain_32gpu_h100_fp8sc_dev_config":
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
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_30b_a3b_pretrain_64gpu_b200_bf16"].copy()
            return cfg
        case "qwen3_30b_a3b_pretrain_64gpu_h100_bf16_config":
            _with_global_batch_size(apply_overrides(cfg, "qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config"), 4096)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_30b_a3b_pretrain_64gpu_b200_bf16"].copy()
            return cfg
        case "qwen3_30b_a3b_pretrain_64gpu_h100_fp8cs_config":
            _with_global_batch_size(apply_overrides(cfg, "qwen3_30b_a3b_pretrain_16gpu_h100_fp8cs_config"), 4096)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_30b_a3b_pretrain_64gpu_b200_bf16"].copy()
            return cfg
        case "qwen3_next_80b_a3b_pretrain_128gpu_h100_bf16_config":
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
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_128gpu_b200_fp8mx_deepep_dev"].copy()
            return cfg
        case "qwen3_next_80b_a3b_pretrain_128gpu_h100_fp8cs_config":
            apply_overrides(cfg, "qwen3_next_80b_a3b_pretrain_128gpu_h100_bf16_config")
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_128gpu_b200_fp8mx_deepep_dev"].copy()
            return cfg
        case "qwen3_235b_a22b_pretrain_256gpu_vrnvl72_bf16_config":
            _apply_qwen3_235b_a22b_gb300_bf16(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
                "NCCL_NVLS_ENABLE": 0,
                "NCCL_EP_HT_EM_PULL_PUSH": 1,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
            }
            return cfg
        case "qwen3_30b_a3b_pretrain_8gpu_vrnvl72_fp8mx_config":
            _apply_qwen3_30b_a3b_gb300_fp8mx(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.train.micro_batch_size = 4
            from megatron.core.transformer.enums import AttnBackend

            cfg.model.attention_backend = AttnBackend.auto
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_235b_a22b_pretrain_256gpu_b200_fp8mx"].copy()
            return cfg
        case "qwen3_30b_a3b_pretrain_8gpu_vrnvl72_nvfp4_config":
            apply_overrides(cfg, "qwen3_30b_a3b_pretrain_8gpu_gb300_nvfp4_config")
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
                "NCCL_NVLS_ENABLE": 0,
                "NCCL_EP_HT_EM_PULL_PUSH": 1,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_USE_FAST_MATH": 1,
            }
            return cfg
        case _:
            raise ValueError(f"Unknown benchmark overrides: {name}")
