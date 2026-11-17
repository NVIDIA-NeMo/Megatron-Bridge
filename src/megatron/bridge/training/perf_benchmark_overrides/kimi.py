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

"""kimi benchmark overrides applied to library recipes.

The literal names preserve existing CLI selectors. Common and model overrides
retain their ordering because graph, RNG and precision policies are dependent.
"""

from megatron.bridge.recipes.kimi.kimi_k2 import _get_kimi_k2_pipeline_layout, kimi_k2_pretrain_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.perf_benchmark_overrides._common import (
    _benchmark_common,
    _enable_ncclep,
    _perf_precision,
)
from megatron.bridge.training.perf_benchmark_overrides.environment import COMMON_PERF_ENV_VARS


BENCHMARK_ENVIRONMENTS = {
    "kimi_k2_pretrain_256gpu_b200_bf16": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    },
    "kimi_k2_pretrain_256gpu_b300_bf16": {
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
    "kimi_k2_pretrain_256gpu_gb200_bf16": {
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
    },
    "kimi_k2_pretrain_256gpu_gb200_fp8mx": {
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
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
    },
    "kimi_k2_pretrain_1024gpu_h100_bf16": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    },
}


def _apply_kimi_k2_gb300_fp8mx(cfg: ConfigContainer) -> ConfigContainer:
    cfg.mixed_precision = _perf_precision("fp8_mx")
    cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.mixed_precision.fp8_param_gather = False
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = "selective"
    cfg.dist.enable_megatron_core_experimental = True
    cfg.model.moe_router_force_load_balancing = True
    cfg.model.qk_clip = False
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 2
    cfg.model.recompute_modules = ["mla_up_proj"]
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.pipeline_model_parallel_layout = _get_kimi_k2_pipeline_layout(4, 4)
    cfg.ddp.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_grad_reduce = True
    cfg.model.cuda_graph_scope = []
    _benchmark_common(cfg)
    cfg.rng.te_rng_tracker = True
    return cfg


LIBRARY_RECIPES = {
    "kimi_k2_pretrain_256gpu_b200_bf16_config": kimi_k2_pretrain_config,
    "kimi_k2_pretrain_256gpu_b200_fp8cs_config": kimi_k2_pretrain_config,
    "kimi_k2_pretrain_256gpu_b200_fp8mx_config": kimi_k2_pretrain_config,
    "kimi_k2_pretrain_256gpu_b300_bf16_config": kimi_k2_pretrain_config,
    "kimi_k2_pretrain_256gpu_b300_fp8cs_config": kimi_k2_pretrain_config,
    "kimi_k2_pretrain_256gpu_b300_fp8mx_config": kimi_k2_pretrain_config,
    "kimi_k2_pretrain_256gpu_gb200_bf16_config": kimi_k2_pretrain_config,
    "kimi_k2_pretrain_256gpu_gb200_fp8cs_config": kimi_k2_pretrain_config,
    "kimi_k2_pretrain_256gpu_gb200_fp8mx_config": kimi_k2_pretrain_config,
    "kimi_k2_pretrain_256gpu_gb300_bf16_config": kimi_k2_pretrain_config,
    "kimi_k2_pretrain_256gpu_gb300_fp8cs_config": kimi_k2_pretrain_config,
    "kimi_k2_pretrain_256gpu_gb300_fp8mx_config": kimi_k2_pretrain_config,
    "kimi_k2_pretrain_256gpu_gb300_nvfp4_config": kimi_k2_pretrain_config,
    "kimi_k2_pretrain_1024gpu_h100_bf16_config": kimi_k2_pretrain_config,
    "kimi_k2_pretrain_1024gpu_h100_fp8cs_config": kimi_k2_pretrain_config,
    "kimi_k2_pretrain_1024gpu_h100_fp8sc_config": kimi_k2_pretrain_config,
    "kimi_k2_pretrain_256gpu_vrnvl72_bf16_config": kimi_k2_pretrain_config,
    "kimi_k2_pretrain_256gpu_vrnvl72_fp8mx_config": kimi_k2_pretrain_config,
}
ALLOCATION_AWARE = ()


def apply_overrides(cfg: ConfigContainer, name: str) -> ConfigContainer:
    """Apply model-specific benchmark overrides to an existing library config."""
    match name:
        case "kimi_k2_pretrain_256gpu_b200_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_fusion = True
            cfg.model.recompute_granularity = "selective"
            cfg.dist.enable_megatron_core_experimental = True
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.qk_clip = False
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 16
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.context_parallel_size = 1
            cfg.model.expert_model_parallel_size = 16
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 2048
            cfg.train.micro_batch_size = 1
            cfg.model.recompute_modules = ["mla_up_proj"]
            cfg.model.pipeline_model_parallel_layout = _get_kimi_k2_pipeline_layout(16, 1)
            cfg.model.moe_shared_expert_overlap = False
            cfg.ddp.overlap_grad_reduce = True
            cfg.comm_overlap.overlap_grad_reduce = True
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["kimi_k2_pretrain_256gpu_b200_bf16"].copy()
            return cfg
        case "kimi_k2_pretrain_256gpu_b200_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_fusion = True
            cfg.model.recompute_granularity = "selective"
            cfg.dist.enable_megatron_core_experimental = True
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.qk_clip = False
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 16
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.context_parallel_size = 1
            cfg.model.expert_model_parallel_size = 16
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 2048
            cfg.train.micro_batch_size = 1
            cfg.model.recompute_modules = ["mla_up_proj"]
            cfg.model.pipeline_model_parallel_layout = _get_kimi_k2_pipeline_layout(16, 1)
            cfg.model.moe_shared_expert_overlap = False
            cfg.ddp.overlap_grad_reduce = True
            cfg.comm_overlap.overlap_grad_reduce = True
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["kimi_k2_pretrain_256gpu_b200_bf16"].copy()
            return cfg
        case "kimi_k2_pretrain_256gpu_b200_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
            cfg.mixed_precision.fp8_param_gather = False
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_fusion = True
            cfg.model.recompute_granularity = "selective"
            cfg.dist.enable_megatron_core_experimental = True
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.qk_clip = False
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 16
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.context_parallel_size = 1
            cfg.model.expert_model_parallel_size = 16
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 2048
            cfg.train.micro_batch_size = 1
            cfg.model.recompute_modules = ["mla_up_proj"]
            cfg.model.pipeline_model_parallel_layout = _get_kimi_k2_pipeline_layout(16, 1)
            cfg.model.moe_shared_expert_overlap = False
            cfg.ddp.overlap_grad_reduce = True
            cfg.comm_overlap.overlap_grad_reduce = True
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["kimi_k2_pretrain_256gpu_b200_bf16"].copy()
            return cfg
        case "kimi_k2_pretrain_256gpu_b300_bf16_config":
            apply_overrides(cfg, "kimi_k2_pretrain_256gpu_b200_bf16_config")
            cfg.train.global_batch_size = 4096
            cfg.train.micro_batch_size = 2
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["kimi_k2_pretrain_256gpu_b300_bf16"].copy()
            return cfg
        case "kimi_k2_pretrain_256gpu_b300_fp8cs_config":
            apply_overrides(cfg, "kimi_k2_pretrain_256gpu_b200_fp8cs_config")
            cfg.train.global_batch_size = 4096
            cfg.train.micro_batch_size = 2
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["kimi_k2_pretrain_256gpu_b300_bf16"].copy()
            return cfg
        case "kimi_k2_pretrain_256gpu_b300_fp8mx_config":
            apply_overrides(cfg, "kimi_k2_pretrain_256gpu_b200_fp8mx_config")
            cfg.train.global_batch_size = 4096
            cfg.train.micro_batch_size = 2
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["kimi_k2_pretrain_256gpu_b300_bf16"].copy()
            return cfg
        case "kimi_k2_pretrain_256gpu_gb200_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_fusion = True
            cfg.model.recompute_granularity = "selective"
            cfg.dist.enable_megatron_core_experimental = True
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.qk_clip = False
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 4
            cfg.model.virtual_pipeline_model_parallel_size = 4
            cfg.model.context_parallel_size = 1
            cfg.model.expert_model_parallel_size = 64
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 2048
            cfg.train.micro_batch_size = 1
            cfg.model.recompute_modules = ["mla_up_proj"]
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.moe_shared_expert_overlap = False
            cfg.model.pipeline_model_parallel_layout = _get_kimi_k2_pipeline_layout(4, 4)
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]
            cfg.ddp.overlap_grad_reduce = True
            cfg.comm_overlap.overlap_grad_reduce = True
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["kimi_k2_pretrain_256gpu_gb200_bf16"].copy()
            return cfg
        case "kimi_k2_pretrain_256gpu_gb200_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_fusion = True
            cfg.model.recompute_granularity = "selective"
            cfg.dist.enable_megatron_core_experimental = True
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.qk_clip = False
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 4
            cfg.model.virtual_pipeline_model_parallel_size = 4
            cfg.model.context_parallel_size = 1
            cfg.model.expert_model_parallel_size = 64
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 2048
            cfg.train.micro_batch_size = 1
            cfg.model.recompute_modules = ["mla_up_proj"]
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.moe_shared_expert_overlap = False
            cfg.model.pipeline_model_parallel_layout = _get_kimi_k2_pipeline_layout(4, 4)
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]
            cfg.ddp.overlap_grad_reduce = True
            cfg.comm_overlap.overlap_grad_reduce = True
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["kimi_k2_pretrain_256gpu_gb200_bf16"].copy()
            return cfg
        case "kimi_k2_pretrain_256gpu_gb200_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
            cfg.mixed_precision.fp8_param_gather = False
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_fusion = True
            cfg.model.recompute_granularity = "selective"
            cfg.dist.enable_megatron_core_experimental = True
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.qk_clip = False
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 4
            cfg.model.virtual_pipeline_model_parallel_size = 4
            cfg.model.context_parallel_size = 1
            cfg.model.expert_model_parallel_size = 64
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 2048
            cfg.train.micro_batch_size = 1
            cfg.model.recompute_modules = ["mla_up_proj"]
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.moe_shared_expert_overlap = False
            cfg.model.pipeline_model_parallel_layout = _get_kimi_k2_pipeline_layout(4, 4)
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]
            cfg.ddp.overlap_grad_reduce = True
            cfg.comm_overlap.overlap_grad_reduce = True
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["kimi_k2_pretrain_256gpu_gb200_fp8mx"].copy()
            return cfg
        case "kimi_k2_pretrain_256gpu_gb300_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_fusion = True
            cfg.model.recompute_granularity = "selective"
            cfg.dist.enable_megatron_core_experimental = True
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.qk_clip = False
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 4
            cfg.model.virtual_pipeline_model_parallel_size = 4
            cfg.model.context_parallel_size = 1
            cfg.model.expert_model_parallel_size = 64
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 4096
            cfg.train.micro_batch_size = 2
            cfg.model.recompute_modules = ["mla_up_proj"]
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.moe_shared_expert_overlap = False
            cfg.model.pipeline_model_parallel_layout = _get_kimi_k2_pipeline_layout(4, 4)
            cfg.ddp.overlap_grad_reduce = True
            cfg.comm_overlap.overlap_grad_reduce = True
            cfg.model.cuda_graph_scope = []
            _benchmark_common(cfg)
            cfg.rng.te_rng_tracker = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["kimi_k2_pretrain_256gpu_gb200_bf16"].copy()
            return cfg
        case "kimi_k2_pretrain_256gpu_gb300_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_fusion = True
            cfg.model.recompute_granularity = "selective"
            cfg.dist.enable_megatron_core_experimental = True
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.qk_clip = False
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 4
            cfg.model.virtual_pipeline_model_parallel_size = 4
            cfg.model.context_parallel_size = 1
            cfg.model.expert_model_parallel_size = 64
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 4096
            cfg.train.micro_batch_size = 2
            cfg.model.recompute_modules = ["mla_up_proj"]
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.moe_shared_expert_overlap = False
            cfg.model.pipeline_model_parallel_layout = _get_kimi_k2_pipeline_layout(4, 4)
            cfg.ddp.overlap_grad_reduce = True
            cfg.comm_overlap.overlap_grad_reduce = True
            cfg.model.cuda_graph_scope = []
            _benchmark_common(cfg)
            cfg.rng.te_rng_tracker = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["kimi_k2_pretrain_256gpu_gb200_bf16"].copy()
            return cfg
        case "kimi_k2_pretrain_256gpu_gb300_fp8mx_config":
            _apply_kimi_k2_gb300_fp8mx(cfg)
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
                "NVTE_NORM_BWD_USE_CUDNN": 1,
                "NVTE_NORM_FWD_USE_CUDNN": 1,
            }
            return cfg
        case "kimi_k2_pretrain_256gpu_gb300_nvfp4_config":
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_fusion = True
            cfg.model.recompute_granularity = "selective"
            cfg.dist.enable_megatron_core_experimental = True
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.qk_clip = False
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 4
            cfg.model.virtual_pipeline_model_parallel_size = 4
            cfg.model.context_parallel_size = 1
            cfg.model.expert_model_parallel_size = 64
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 4096
            cfg.train.micro_batch_size = 2
            cfg.model.recompute_modules = ["mla_up_proj"]
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.moe_shared_expert_overlap = False
            cfg.model.pipeline_model_parallel_layout = _get_kimi_k2_pipeline_layout(4, 4)
            cfg.ddp.overlap_grad_reduce = True
            cfg.comm_overlap.overlap_grad_reduce = True
            cfg.model.cuda_graph_scope = []
            _benchmark_common(cfg)
            cfg.rng.te_rng_tracker = True
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
                "NVTE_USE_FAST_MATH": 1,
            }
            return cfg
        case "kimi_k2_pretrain_1024gpu_h100_bf16_config":
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
            cfg.comm_overlap.overlap_grad_reduce = True
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["kimi_k2_pretrain_1024gpu_h100_bf16"].copy()
            return cfg
        case "kimi_k2_pretrain_1024gpu_h100_fp8cs_config":
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
            cfg.comm_overlap.overlap_grad_reduce = True
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["kimi_k2_pretrain_1024gpu_h100_bf16"].copy()
            return cfg
        case "kimi_k2_pretrain_1024gpu_h100_fp8sc_config":
            apply_overrides(cfg, "kimi_k2_pretrain_1024gpu_h100_fp8cs_config")
            cfg.mixed_precision.fp8_recipe = "blockwise"
            cfg.mixed_precision.fp8_param = False
            cfg.mixed_precision.fp8_param_gather = False
            cfg.mixed_precision.num_layers_at_start_in_bf16 = 0
            cfg.mixed_precision.num_layers_at_end_in_bf16 = 0
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["kimi_k2_pretrain_1024gpu_h100_bf16"].copy()
            return cfg
        case "kimi_k2_pretrain_256gpu_vrnvl72_bf16_config":
            apply_overrides(cfg, "kimi_k2_pretrain_256gpu_gb300_bf16_config")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["kimi_k2_pretrain_256gpu_gb200_bf16"].copy()
            return cfg
        case "kimi_k2_pretrain_256gpu_vrnvl72_fp8mx_config":
            _apply_kimi_k2_gb300_fp8mx(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["kimi_k2_pretrain_256gpu_gb200_fp8mx"].copy()
            return cfg
        case _:
            raise ValueError(f"Unknown benchmark overrides: {name}")
