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

"""deepseek benchmark overrides applied to library recipes.

The literal names preserve existing CLI selectors. Common and model overrides
retain their ordering because graph, RNG and precision policies are dependent.
"""

import torch

from megatron.bridge.models.deepseek.deepseek_v4_bridge import set_deepseek_v4_pipeline_model_parallel_layout
from megatron.bridge.recipes.deepseek.deepseek_v3 import (
    deepseek_v3_pretrain_config,
    set_deepseek_v3_pipeline_model_parallel_layout,
)
from megatron.bridge.recipes.deepseek.gb200.deepseek_v4 import deepseek_v4_flash_pretrain_64gpu_gb200_fp8mx_config
from megatron.bridge.recipes.deepseek.gb300.deepseek_v4 import deepseek_v4_pro_pretrain_32gpu_gb300_fp8mx_config
from megatron.bridge.recipes.utils.determinism_utils import apply_determinism_overrides
from megatron.bridge.recipes.utils.environment_utils import DETERMINISM_ENV_VARS
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.perf_benchmark_overrides._common import (
    _benchmark_common,
    _enable_ncclep,
    _perf_precision,
)
from megatron.bridge.training.perf_benchmark_overrides.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.utils.cuda_graph import set_full_iteration_cuda_graph


BENCHMARK_ENVIRONMENTS = {
    "deepseek_v3_pretrain_256gpu_b200_bf16": {
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
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
    },
    "deepseek_v3_pretrain_256gpu_b300_bf16": {
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
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
        "NCCL_IGNORE_CPU_AFFINITY": 1,
    },
    "deepseek_v3_pretrain_256gpu_b300_fp8mx": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
        "CUDNNFE_CLUSTER_OVERLAP_MARGIN": 8,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
        "NCCL_IGNORE_CPU_AFFINITY": 1,
    },
    "deepseek_v3_pretrain_256gpu_gb200_fp8cs": {
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
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
    },
    "deepseek_v3_pretrain_256gpu_gb200_fp8mx": {
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
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
    },
    "deepseek_v4_flash_pretrain_128gpu_gb200_fp8mx": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
        "NCCL_NVLS_ENABLE": 0,
        "NCCL_EP_HT_EM_PULL_PUSH": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
    },
    "deepseek_v3_pretrain_256gpu_gb300_fp8mx": {
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
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
    },
    "deepseek_v3_pretrain_1024gpu_h100_bf16": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
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
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
    },
    "deepseek_v3_pretrain_128gpu_vrnvl72_bf16": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NCCL_EP_HT_EM_PULL_PUSH": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
    },
    "deepseek_v3_pretrain_256gpu_vrnvl72_fp8mx": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "TORCH_DIST_USE_NCCL2": 0,
        "NCCL_NVLS_ENABLE": 0,
        "NCCL_EP_HT_EM_PULL_PUSH": 1,
        "NCCL_EP_COMBINE_SMS": 24,
        "NCCL_EP_DISPATCH_SMS": 32,
        "NCCL_EP_HT_UNFUSED_SYNC": 1,
        "NCCL_EP_PREPROCESS_NUM_SMS": 80,
        "NCCL_EP_PROLOG_EPILOG_SMS": 148,
        "NCCL_EP_SHUFFLE_SMS": 128,
        "NVTE_EP_FUSED_PREPARE_DISPATCH": 1,
        "CUDNNFE_CLUSTER_OVERLAP_MARGIN": 8,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 0,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_EXT_MARGIN_SM": 16,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 0,
        "NVTE_CUDNN_MXFP8_NORM_OUTPUT_IN_INPUT_DTYPE": 1,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        "NVTE_MXFP8_PER_TILE_SWIZZLE": 1,
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 1,
    },
    "deepseek_v3_pretrain_256gpu_vrnvl72_nvfp4": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NCCL_EP_HT_EM_PULL_PUSH": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
        "NVTE_USE_FAST_MATH": 1,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_DPA_FP8_RECIPE": "MXFP8BlockScaling",
        "NVTE_DPA_FP8_FORMAT": "E4M3",
    },
}


def _deepseek_v3_common(cfg: ConfigContainer) -> None:
    """Apply DeepSeek V3 perf defaults shared by the legacy workload configs."""
    cfg.dataset.seq_length = cfg.model.seq_length
    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = "selective"
    cfg.dist.enable_megatron_core_experimental = True
    cfg.model.moe_router_force_load_balancing = True


def _enable_deepseek_precision_aware_optimizer(cfg: ConfigContainer) -> None:
    """Enable the optimizer precision settings used by tuned DeepSeek recipes."""
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16


def _enable_deepseek_full_iteration(cfg: ConfigContainer) -> None:
    """Apply legacy DeepSeek V3 HybridEP full-iteration settings."""
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_hybridep_num_sms = 32
    cfg.model.cuda_graph_impl = "full_iteration"
    cfg.model.cuda_graph_scope = []
    cfg.model.high_priority_a2a_comm_stream = True
    cfg.model.moe_expert_rank_capacity_factor = 1.5
    cfg.model.moe_hybridep_num_sms_preprocessing = 32
    cfg.model.moe_mlp_glu_interleave_size = 32
    cfg.model.moe_pad_experts_for_cuda_graph_inference = True
    cfg.model.moe_paged_stash = True
    cfg.model.moe_paged_stash_buffer_size_factor_cpu = 1.0
    cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.2
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True

    cfg.comm_overlap.delay_wgrad_compute = True
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = True


def _enable_deepseek_transformer_engine_graph(cfg: ConfigContainer) -> None:
    """Apply legacy DeepSeek V3 Transformer Engine graph capture settings."""
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True


def _apply_deepseek_v3_64gpu_gb300_fsdp_configs(cfg: ConfigContainer) -> None:
    """Apply shared DeepSeek V3 64-GPU GB300 Megatron FSDP settings."""
    _deepseek_v3_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 256
    cfg.train.micro_batch_size = 2

    cfg.ddp.use_megatron_fsdp = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.keep_fp8_transpose_cache = False
    cfg.ddp.average_in_collective = False
    cfg.model.init_model_with_meta_device = True
    cfg.model.gradient_accumulation_fusion = True
    cfg.checkpoint.load = None

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.cuda_graph_scope = []
    cfg.model.recompute_modules = ["layernorm", "mla_up_proj"]
    cfg.model.fine_grained_activation_offloading = True
    cfg.model.offload_modules = ["core_attn", "attn_proj"]
    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    cfg.comm_overlap.overlap_grad_reduce = True

    _benchmark_common(cfg)


def _apply_deepseek_v4_b200_overrides(cfg: ConfigContainer) -> None:
    """Apply the lower-memory B200 topology and communication settings."""
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 8
    cfg.train.micro_batch_size = 1
    set_deepseek_v4_pipeline_model_parallel_layout(cfg.model)
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
    cfg.model.moe_pad_experts_for_cuda_graph_inference = False
    cfg.model.moe_paged_stash = False
    recompute_modules = list(cfg.model.recompute_modules or [])
    if "mlp" not in recompute_modules:
        recompute_modules.append("mlp")
    cfg.model.recompute_modules = recompute_modules
    cfg.comm_overlap.overlap_grad_reduce = True
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.dist.distributed_timeout_minutes = 30
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False


def _apply_deepseek_v4_flash_gb200_fp8mx(cfg: ConfigContainer) -> ConfigContainer:
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    set_deepseek_v4_pipeline_model_parallel_layout(cfg.model)
    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 1
    cfg.model.attention_backend = "auto"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_hybridep_num_sms_preprocessing = 108
    cfg.model.moe_router_fusion = True
    cfg.model.moe_router_force_load_balancing = True
    cfg.model.moe_router_load_balancing_type = "seq_aux_loss"
    cfg.model.moe_aux_loss_coeff = 0.0001
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["mla_up_proj"]
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = []
    cfg.model.fine_grained_offloading_max_inflight_offloads = None
    _benchmark_common(cfg, cross_entropy_impl="native")
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_hybridep_num_sms = None
    set_full_iteration_cuda_graph(cfg.model)
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True
    cfg.train.manual_gc_interval = 10
    cfg.model.moe_expert_rank_capacity_factor = 1.5
    cfg.model.moe_pad_experts_for_cuda_graph_inference = True
    cfg.model.moe_paged_stash = True
    cfg.model.moe_paged_stash_page_size = 64
    cfg.model.moe_paged_stash_buffer_size_factor_cpu = 0.0
    cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.2
    cfg.model.moe_mlp_glu_interleave_size = 32
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.model.csa_compress_rotary_base = 40000
    cfg.model.rotary_scaling_factor = 4
    cfg.model.dsa_kernel_backend = "cudnn"
    cfg.model.dsa_indexer_loss_coeff = 0.01
    cfg.model.dsa_indexer_use_sparse_loss = True
    cfg.model.quant_recipe = None
    cfg.model.moe_router_padding_for_fp8 = False
    cfg.model.moe_router_padding_for_quantization = True
    cfg.mixed_precision.fp8_param_gather = True
    cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16
    cfg.optimizer.optimizer_offload_fraction = 1.0
    cfg.optimizer.offload_optimizer_states = False
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.average_in_collective = False
    cfg.comm_overlap.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.checkpoint.load_optim = False
    cfg.checkpoint.load_rng = False
    cfg.checkpoint.save_optim = False
    return cfg


def _apply_deepseek_v3_gb200_bf16(cfg: ConfigContainer) -> ConfigContainer:
    cfg.mixed_precision = _perf_precision("bf16")
    _deepseek_v3_common(cfg)
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 1
    cfg.model.recompute_modules = ["mla_up_proj"]
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
    cfg.ddp.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_grad_reduce = True
    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)
    _benchmark_common(cfg)
    return cfg


def _apply_deepseek_v3_4gpu_vrnvl72_proxy(cfg: ConfigContainer) -> None:
    """Shrink a 256-GPU VRNVL72 DeepSeek V3 recipe onto one 4-GPU VRNVL72 node.

    Both 4-GPU proxies go through this helper so that every model, parallelism and batch
    setting matches and only the quantization recipe differs between their profiles.
    """
    cfg.model.num_layers = 13
    cfg.model.moe_layer_freq = [0] * 3 + [1] * (cfg.model.num_layers - 3)
    cfg.model.num_moe_experts = 32
    cfg.model.expert_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)
    cfg.train.global_batch_size = 128
    cfg.model.fp8_output_proj = False
    cfg.model.mla_down_proj_fusion = True


def _apply_deepseek_v4_b300_overrides(cfg: ConfigContainer) -> None:
    """Apply a B300 topology scaled down from the larger DeepSeek V3 model."""
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.train.micro_batch_size = 2
    cfg.model.moe_hybridep_num_sms_preprocessing = 32
    cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.5
    set_deepseek_v4_pipeline_model_parallel_layout(cfg.model)


def _apply_deepseek_v4_pro_pretrain_256gpu_gb300_common(cfg: ConfigContainer) -> ConfigContainer:
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 1
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_shared_expert_overlap = False
    set_deepseek_v4_pipeline_model_parallel_layout(cfg.model, logical_layers_per_stage=[4] * 15 + [1])
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["mla_up_proj", "mhc"]
    _benchmark_common(cfg, cross_entropy_impl="native")
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_hybridep_num_sms = None
    cfg.model.cuda_graph_impl = "full_iteration"
    cfg.model.cuda_graph_scope = []
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True
    cfg.model.moe_pad_experts_for_cuda_graph_inference = True
    cfg.model.moe_paged_stash = True
    cfg.model.moe_expert_rank_capacity_factor = 1.5
    cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.2
    cfg.model.moe_paged_stash_buffer_size_factor_cpu = 0.0
    cfg.model.moe_router_force_load_balancing = True
    cfg.model.dsa_kernel_backend = "cudnn"
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.moe_mlp_glu_interleave_size = 32
    cfg.model.quant_recipe = None
    cfg.mixed_precision.fp8_param_gather = True
    cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.model.dsa_indexer_loss_coeff = 0.01
    cfg.model.dsa_indexer_use_sparse_loss = True
    cfg.optimizer.main_grads_dtype = torch.bfloat16
    cfg.dist.enable_megatron_core_experimental = True
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.model.fine_grained_activation_offloading = True
    cfg.model.offload_modules = ["core_attn", "attn_proj"]
    cfg.model.fine_grained_offloading_max_inflight_offloads = 2
    cfg.comm_overlap.overlap_grad_reduce = True
    return cfg


def _apply_deepseek_v3_gb300_bf16(cfg: ConfigContainer) -> ConfigContainer:
    cfg.mixed_precision = _perf_precision("bf16")
    _deepseek_v3_common(cfg)
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 1
    cfg.model.recompute_modules = ["moe_act"]
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
    cfg.ddp.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_grad_reduce = True
    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model, "Et*4|(t*4|)*14tmL")
    _benchmark_common(cfg)
    return cfg


def _apply_deepseek_v3_gb300_fp8mx(cfg: ConfigContainer) -> ConfigContainer:
    cfg.mixed_precision = _perf_precision("fp8_mx")
    _deepseek_v3_common(cfg)
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 32
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 1
    cfg.model.recompute_modules = []
    cfg.model.cuda_graph_scope = []
    cfg.ddp.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_grad_reduce = True
    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model, "Et*4|(t*4|)*14tmL")
    _benchmark_common(cfg)
    _enable_deepseek_full_iteration(cfg)
    cfg.model.fp8_output_proj = True
    cfg.mixed_precision.fp8_dot_product_attention = True
    return cfg


def _apply_deepseek_v3_gb300_nvfp4(cfg: ConfigContainer) -> ConfigContainer:
    cfg.mixed_precision = _perf_precision("nvfp4")
    _deepseek_v3_common(cfg)
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 32
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 1
    cfg.model.cuda_graph_scope = []
    cfg.ddp.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_grad_reduce = True
    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model, "Et*5|(t*4|)*14mL")
    _benchmark_common(cfg)
    _enable_deepseek_full_iteration(cfg)
    cfg.model.fp8_output_proj = False
    cfg.mixed_precision.fp8_dot_product_attention = True
    cfg.model.mla_down_proj_fusion = True
    cfg.model.recompute_modules = []
    return cfg


LIBRARY_RECIPES = {
    "deepseek_v3_pretrain_256gpu_b200_bf16_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_256gpu_b200_fp8cs_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_256gpu_b200_fp8mx_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_256gpu_b200_fp8mx_large_scale_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_256gpu_b200_nvfp4_config": deepseek_v3_pretrain_config,
    "deepseek_v4_flash_pretrain_128gpu_b200_fp8mx_config": deepseek_v4_flash_pretrain_64gpu_gb200_fp8mx_config,
    "deepseek_v3_pretrain_256gpu_b300_bf16_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_256gpu_b300_fp8cs_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_256gpu_b300_fp8mx_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_256gpu_b300_fp8mx_large_scale_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_256gpu_b300_nvfp4_config": deepseek_v3_pretrain_config,
    "deepseek_v4_flash_pretrain_128gpu_b300_fp8mx_config": deepseek_v4_flash_pretrain_64gpu_gb200_fp8mx_config,
    "deepseek_v3_pretrain_256gpu_gb200_bf16_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_256gpu_gb200_fp8cs_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_256gpu_gb200_fp8mx_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_256gpu_gb200_fp8mx_large_scale_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_256gpu_gb200_fp8mx_partial_cg_dev_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_256gpu_gb200_nvfp4_config": deepseek_v3_pretrain_config,
    "deepseek_v4_flash_pretrain_128gpu_gb200_fp8mx_config": deepseek_v4_flash_pretrain_64gpu_gb200_fp8mx_config,
    "deepseek_v3_pretrain_256gpu_gb300_bf16_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_256gpu_gb300_fp8cs_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_256gpu_gb300_fp8mx_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_256gpu_gb300_fp8mx_deterministic_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_256gpu_gb300_fp8mx_hsdp_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_256gpu_gb300_fp8mx_large_scale_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_256gpu_gb300_fp8mx_partial_cg_dev_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_256gpu_gb300_nvfp4_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_64gpu_gb300_bf16_fsdp_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_64gpu_gb300_fp8mx_fsdp_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_64gpu_gb300_fp8mx_mfsdpv2_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_64gpu_gb300_fp8mx_proxy_config": deepseek_v3_pretrain_config,
    "deepseek_v4_flash_pretrain_128gpu_gb300_fp8mx_config": deepseek_v4_flash_pretrain_64gpu_gb200_fp8mx_config,
    "deepseek_v4_pro_pretrain_256gpu_gb300_bf16_config": deepseek_v4_pro_pretrain_32gpu_gb300_fp8mx_config,
    "deepseek_v4_pro_pretrain_256gpu_gb300_fp8mx_config": deepseek_v4_pro_pretrain_32gpu_gb300_fp8mx_config,
    "deepseek_v3_pretrain_1024gpu_h100_bf16_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_1024gpu_h100_fp8cs_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_1024gpu_h100_fp8sc_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_1024gpu_h100_fp8sc_large_scale_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_64gpu_h100_bf16_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_64gpu_h100_fp8cs_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_128gpu_vrnvl72_bf16_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_128gpu_vrnvl72_fp8cs_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_128gpu_vrnvl72_fp8mx_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_128gpu_vrnvl72_nvfp4_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_256gpu_vrnvl72_bf16_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_256gpu_vrnvl72_fp8cs_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_256gpu_vrnvl72_fp8mx_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_256gpu_vrnvl72_nvfp4_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_4gpu_vrnvl72_fp8mx_proxy_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_4gpu_vrnvl72_nvfp4_proxy_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_64gpu_vrnvl72_fp8mx_proxy_config": deepseek_v3_pretrain_config,
    "deepseek_v3_pretrain_64gpu_vrnvl72_nvfp4_proxy_config": deepseek_v3_pretrain_config,
    "deepseek_v4_flash_pretrain_128gpu_vrnvl72_fp8mx_config": deepseek_v4_flash_pretrain_64gpu_gb200_fp8mx_config,
}
ALLOCATION_AWARE = ()


def apply_overrides(cfg: ConfigContainer, name: str) -> ConfigContainer:
    """Apply model-specific benchmark overrides to an existing library config."""
    match name:
        case "deepseek_v3_pretrain_256gpu_b200_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            _deepseek_v3_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.context_parallel_size = 1
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 4096
            cfg.train.micro_batch_size = 1
            cfg.model.recompute_modules = ["mla_up_proj"]
            cfg.ddp.overlap_grad_reduce = True
            cfg.comm_overlap.overlap_grad_reduce = True
            set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["deepseek_v3_pretrain_256gpu_b200_bf16"].copy()
            return cfg
        case "deepseek_v3_pretrain_256gpu_b200_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            _deepseek_v3_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.context_parallel_size = 1
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 4096
            cfg.train.micro_batch_size = 1
            cfg.model.recompute_modules = ["mla_up_proj"]
            cfg.ddp.overlap_grad_reduce = True
            cfg.comm_overlap.overlap_grad_reduce = True
            set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["deepseek_v3_pretrain_256gpu_b200_bf16"].copy()
            return cfg
        case "deepseek_v3_pretrain_256gpu_b200_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            _deepseek_v3_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.virtual_pipeline_model_parallel_size = 2
            cfg.model.context_parallel_size = 1
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 4096
            cfg.train.micro_batch_size = 1
            cfg.model.recompute_modules = ["mla_up_proj"]
            cfg.ddp.overlap_grad_reduce = True
            cfg.comm_overlap.overlap_grad_reduce = True
            set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)
            _benchmark_common(cfg)
            _enable_deepseek_transformer_engine_graph(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["deepseek_v3_pretrain_256gpu_b200_bf16"].copy()
            return cfg
        case "deepseek_v3_pretrain_256gpu_b200_fp8mx_large_scale_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            _deepseek_v3_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 16
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.context_parallel_size = 1
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 256
            cfg.train.micro_batch_size = 1
            cfg.model.recompute_modules = ["mla_up_proj"]
            cfg.ddp.overlap_grad_reduce = True
            cfg.comm_overlap.overlap_grad_reduce = True
            set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["deepseek_v3_pretrain_256gpu_b200_bf16"].copy()
            return cfg
        case "deepseek_v3_pretrain_256gpu_b200_nvfp4_config":
            apply_overrides(cfg, "deepseek_v3_pretrain_256gpu_b200_bf16_config")
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.mixed_precision.fp4_param = False
            cfg.mixed_precision.fp4_param_gather = False
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.virtual_pipeline_model_parallel_size = 2
            cfg.model.recompute_modules = ["mla_up_proj", "layernorm", "moe_act"]
            set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)
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
                "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
                "NVTE_USE_FAST_MATH": 1,
            }
            return cfg
        case "deepseek_v4_flash_pretrain_128gpu_b200_fp8mx_config":
            apply_overrides(cfg, "deepseek_v4_flash_pretrain_128gpu_b300_fp8mx_config")
            _apply_deepseek_v4_b200_overrides(cfg)
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
                "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
            }
            return cfg
        case "deepseek_v3_pretrain_256gpu_b300_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            _deepseek_v3_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.virtual_pipeline_model_parallel_size = 2
            cfg.model.context_parallel_size = 1
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 4096
            cfg.train.micro_batch_size = 1
            cfg.model.recompute_modules = ["mla_up_proj"]
            cfg.ddp.overlap_grad_reduce = True
            cfg.comm_overlap.overlap_grad_reduce = True
            set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)
            _benchmark_common(cfg)
            _enable_deepseek_transformer_engine_graph(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["deepseek_v3_pretrain_256gpu_b300_bf16"].copy()
            return cfg
        case "deepseek_v3_pretrain_256gpu_b300_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            _deepseek_v3_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.virtual_pipeline_model_parallel_size = 2
            cfg.model.context_parallel_size = 1
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 4096
            cfg.train.micro_batch_size = 1
            cfg.model.recompute_modules = ["mla_up_proj"]
            cfg.ddp.overlap_grad_reduce = True
            cfg.comm_overlap.overlap_grad_reduce = True
            set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)
            _benchmark_common(cfg)
            _enable_deepseek_transformer_engine_graph(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["deepseek_v3_pretrain_256gpu_b300_bf16"].copy()
            return cfg
        case "deepseek_v3_pretrain_256gpu_b300_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            _deepseek_v3_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.virtual_pipeline_model_parallel_size = 2
            cfg.model.context_parallel_size = 1
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 4096
            cfg.train.micro_batch_size = 1
            cfg.model.recompute_modules = ["mla_up_proj"]
            cfg.ddp.overlap_grad_reduce = True
            cfg.comm_overlap.overlap_grad_reduce = True
            set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)
            _benchmark_common(cfg)
            _enable_deepseek_full_iteration(cfg)
            cfg.model.fp8_output_proj = True
            cfg.mixed_precision.fp8_dot_product_attention = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["deepseek_v3_pretrain_256gpu_b300_fp8mx"].copy()
            return cfg
        case "deepseek_v3_pretrain_256gpu_b300_fp8mx_large_scale_config":
            apply_overrides(cfg, "deepseek_v3_pretrain_256gpu_b300_fp8mx_config")
            cfg.train.global_batch_size = 256
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["deepseek_v3_pretrain_256gpu_b300_fp8mx"].copy()
            return cfg
        case "deepseek_v3_pretrain_256gpu_b300_nvfp4_config":
            apply_overrides(cfg, "deepseek_v3_pretrain_256gpu_b300_bf16_config")
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.train.micro_batch_size = 2
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
                "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
                "NCCL_IGNORE_CPU_AFFINITY": 1,
                "NVTE_USE_FAST_MATH": 1,
            }
            return cfg
        case "deepseek_v4_flash_pretrain_128gpu_b300_fp8mx_config":
            apply_overrides(cfg, "deepseek_v4_flash_pretrain_128gpu_gb300_fp8mx_config")
            _apply_deepseek_v4_b300_overrides(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["deepseek_v3_pretrain_256gpu_b300_fp8mx"].copy()
            return cfg
        case "deepseek_v3_pretrain_256gpu_gb200_bf16_config":
            _apply_deepseek_v3_gb200_bf16(cfg)
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
                "NVTE_NORM_BWD_USE_CUDNN": 1,
                "NVTE_NORM_FWD_USE_CUDNN": 1,
                "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
            }
            return cfg
        case "deepseek_v3_pretrain_256gpu_gb200_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            _deepseek_v3_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 4
            cfg.model.virtual_pipeline_model_parallel_size = 4
            cfg.model.context_parallel_size = 1
            cfg.model.expert_model_parallel_size = 64
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 4096
            cfg.train.micro_batch_size = 1
            cfg.model.recompute_modules = ["mlp"]
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.ddp.overlap_grad_reduce = True
            cfg.comm_overlap.overlap_grad_reduce = True
            set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["deepseek_v3_pretrain_256gpu_gb200_fp8cs"].copy()
            return cfg
        case "deepseek_v3_pretrain_256gpu_gb200_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            _deepseek_v3_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 4
            cfg.model.virtual_pipeline_model_parallel_size = 4
            cfg.model.context_parallel_size = 1
            cfg.model.expert_model_parallel_size = 64
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 4096
            cfg.train.micro_batch_size = 1
            cfg.model.recompute_modules = ["mla_up_proj"]
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.ddp.overlap_grad_reduce = True
            cfg.comm_overlap.overlap_grad_reduce = True
            set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)
            _benchmark_common(cfg)
            _enable_deepseek_full_iteration(cfg)
            cfg.model.fp8_output_proj = True
            cfg.mixed_precision.fp8_dot_product_attention = False
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["deepseek_v3_pretrain_256gpu_gb200_fp8mx"].copy()
            return cfg
        case "deepseek_v3_pretrain_256gpu_gb200_fp8mx_large_scale_config":
            apply_overrides(cfg, "deepseek_v3_pretrain_256gpu_gb200_fp8mx_config")
            cfg.train.global_batch_size = 256
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["deepseek_v3_pretrain_256gpu_gb200_fp8mx"].copy()
            return cfg
        case "deepseek_v3_pretrain_256gpu_gb200_fp8mx_partial_cg_dev_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            _benchmark_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 4
            cfg.model.virtual_pipeline_model_parallel_size = 2
            cfg.model.expert_model_parallel_size = 64
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.kv_channels = 128
            cfg.model.make_vocab_size_divisible_by = 1280
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.moe_router_fusion = True
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.moe_router_padding_for_quantization = True
            cfg.model.moe_hybridep_num_sms = 32
            cfg.model.recompute_granularity = "selective"
            cfg.model.recompute_modules = ["mla_up_proj"]
            cfg.model.fine_grained_activation_offloading = True
            cfg.model.offload_modules = ["expert_fc1"]
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.model.cuda_graph_warmup_steps = 1
            cfg.model.use_te_rng_tracker = True
            cfg.comm_overlap.delay_wgrad_compute = True
            cfg.comm_overlap.overlap_moe_expert_parallel_comm = True
            cfg.ddp.reuse_grad_buf_for_mxfp8_param_ag = True
            cfg.ddp.overlap_grad_reduce = True
            cfg.ddp.overlap_param_gather = True
            cfg.train.micro_batch_size = 1
            cfg.train.global_batch_size = 8192
            cfg.train.exit_duration_in_mins = 220
            cfg.train.manual_gc_interval = 10
            cfg.rng.te_rng_tracker = True
            set_deepseek_v3_pipeline_model_parallel_layout(cfg.model, "Et*4|(tttt|)*14tmL")
            _enable_deepseek_precision_aware_optimizer(cfg)
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
                "NVTE_CPU_OFFLOAD_V1": 1,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
            }
            return cfg
        case "deepseek_v3_pretrain_256gpu_gb200_nvfp4_config":
            _apply_deepseek_v3_gb200_bf16(cfg)
            cfg.mixed_precision = _perf_precision("nvfp4")
            _enable_deepseek_full_iteration(cfg)
            cfg.model.fp8_output_proj = False
            cfg.mixed_precision.fp8_dot_product_attention = True
            cfg.model.cuda_graph_use_single_mempool = True
            cfg.optimizer.optimizer_cuda_graph = True
            cfg.optimizer.store_param_remainders = False
            cfg.model.mla_down_proj_fusion = True
            cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.1
            cfg.model.recompute_modules = ["mla_up_proj", "mlp"]
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
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_NORM_BWD_USE_CUDNN": 1,
                "NVTE_NORM_FWD_USE_CUDNN": 1,
                "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
                "NVTE_USE_FAST_MATH": 1,
                "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
                "NVTE_DPA_FP8_RECIPE": "MXFP8BlockScaling",
                "NVTE_DPA_FP8_FORMAT": "E4M3",
            }
            return cfg
        case "deepseek_v4_flash_pretrain_128gpu_gb200_fp8mx_config":
            _apply_deepseek_v4_flash_gb200_fp8mx(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["deepseek_v4_flash_pretrain_128gpu_gb200_fp8mx"].copy()
            return cfg
        case "deepseek_v3_pretrain_256gpu_gb300_bf16_config":
            _apply_deepseek_v3_gb300_bf16(cfg)
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
                "NVTE_NORM_BWD_USE_CUDNN": 1,
                "NVTE_NORM_FWD_USE_CUDNN": 1,
            }
            return cfg
        case "deepseek_v3_pretrain_256gpu_gb300_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            _deepseek_v3_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 2
            cfg.model.virtual_pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.expert_model_parallel_size = 32
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 4096
            cfg.train.micro_batch_size = 2
            cfg.model.recompute_modules = ["mla_up_proj"]
            cfg.model.cuda_graph_scope = []
            cfg.ddp.overlap_grad_reduce = True
            cfg.comm_overlap.overlap_grad_reduce = True
            set_deepseek_v3_pipeline_model_parallel_layout(cfg.model, "Et*4|(t*4|)*14tmL")
            _benchmark_common(cfg)
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
            }
            return cfg
        case "deepseek_v3_pretrain_256gpu_gb300_fp8mx_config":
            _apply_deepseek_v3_gb300_fp8mx(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["deepseek_v3_pretrain_256gpu_gb300_fp8mx"].copy()
            return cfg
        case "deepseek_v3_pretrain_256gpu_gb300_fp8mx_deterministic_config":
            apply_overrides(cfg, "deepseek_v3_pretrain_256gpu_gb300_fp8mx_config")
            apply_determinism_overrides(cfg)
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                **DETERMINISM_ENV_VARS,
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
                "NVTE_NORM_BWD_USE_CUDNN": 1,
                "NVTE_NORM_FWD_USE_CUDNN": 1,
            }
            return cfg
        case "deepseek_v3_pretrain_256gpu_gb300_fp8mx_hsdp_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.model.fp8_output_proj = True
            _apply_deepseek_v3_64gpu_gb300_fsdp_configs(cfg)
            cfg.model.expert_model_parallel_size = 64
            cfg.ddp.data_parallel_sharding_strategy = "optim_grads"
            cfg.ddp.expert_data_parallel_sharding_strategy = "optim_grads_params"
            cfg.model.gradient_accumulation_fusion = False
            cfg.train.micro_batch_size = 1
            cfg.train.global_batch_size = 1024
            cfg.ddp.outer_dp_sharding_strategy = "optim"
            cfg.ddp.expert_outer_dp_sharding_strategy = "optim"
            cfg.ddp.num_distributed_optimizer_instances = 4
            cfg.model.fp8_param_gather = True
            cfg.model.fp8_param = True
            cfg.model.moe_router_dtype = "bf16"
            cfg.model.average_in_collective = cfg.ddp.average_in_collective = True
            cfg.model.cuda_graph_impl = "full_iteration"
            cfg.model.overlap_dispatch_backward_with_experts_wgrad = False
            cfg.ddp.megatron_fsdp_cuda_graph_mode = True
            cfg.ddp.fsdp_all_gather_in_start_param_sync = False
            cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True
            cfg.model.moe_pad_experts_for_cuda_graph_inference = True
            cfg.model.moe_paged_stash = True
            cfg.model.moe_expert_rank_capacity_factor = 5
            cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.2
            cfg.model.moe_paged_stash_buffer_size_factor_cpu = 1.0
            cfg.model.fine_grained_offloading_max_inflight_offloads = 1
            cfg.model.recompute_granularity = None
            cfg.model.recompute_modules = []
            cfg.model.fine_grained_activation_offloading = True
            cfg.model.offload_modules = ["core_attn", "attn_proj"]
            cfg.model.high_priority_a2a_comm_stream = True
            cfg.model.fused_residual_rmsnorm = True
            cfg.model.moe_hybridep_num_sms_preprocessing = 32
            cfg.comm_overlap.overlap_moe_expert_parallel_comm = True
            cfg.comm_overlap.delay_wgrad_compute = True
            cfg.comm_overlap.align_param_gather = True
            cfg.model.use_transformer_engine_op_fuser = True
            cfg.model.moe_mlp_glu_interleave_size = 32
            cfg.model.mla_down_proj_fusion = True
            cfg.mixed_precision.fp8_dot_product_attention = False
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NVTE_CUDNN_MXFP8_NORM_OUTPUT_IN_INPUT_DTYPE": 0,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
                "NCCL_NVLS_ENABLE": 0,
                "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 64,
                "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
                "NVLINK_DOMAIN_SIZE": 72,
                "USE_MNNVL": 1,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_CPU_OFFLOAD_V1": 1,
                "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_NORM_BWD_USE_CUDNN": 1,
                "NVTE_NORM_FWD_USE_CUDNN": 1,
            }
            return cfg
        case "deepseek_v3_pretrain_256gpu_gb300_fp8mx_large_scale_config":
            apply_overrides(cfg, "deepseek_v3_pretrain_256gpu_gb300_fp8mx_config")
            cfg.train.global_batch_size = 256
            cfg.model.pipeline_model_parallel_size = 4
            cfg.model.virtual_pipeline_model_parallel_size = 4
            cfg.model.expert_model_parallel_size = 64
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["deepseek_v3_pretrain_256gpu_gb300_fp8mx"].copy()
            return cfg
        case "deepseek_v3_pretrain_256gpu_gb300_fp8mx_partial_cg_dev_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            _benchmark_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 4
            cfg.model.virtual_pipeline_model_parallel_size = 2
            cfg.model.expert_model_parallel_size = 64
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.kv_channels = 128
            cfg.model.make_vocab_size_divisible_by = 1280
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.moe_router_fusion = True
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.moe_router_padding_for_quantization = True
            cfg.model.moe_hybridep_num_sms = 32
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.model.cuda_graph_warmup_steps = 1
            cfg.model.use_te_rng_tracker = True
            cfg.comm_overlap.delay_wgrad_compute = True
            cfg.comm_overlap.overlap_moe_expert_parallel_comm = True
            cfg.ddp.reuse_grad_buf_for_mxfp8_param_ag = True
            cfg.ddp.overlap_grad_reduce = True
            cfg.ddp.overlap_param_gather = True
            cfg.ddp.check_for_nan_in_grad = False
            cfg.train.micro_batch_size = 1
            cfg.train.global_batch_size = 8192
            cfg.train.exit_duration_in_mins = 220
            cfg.train.manual_gc_interval = 10
            cfg.rng.te_rng_tracker = True
            set_deepseek_v3_pipeline_model_parallel_layout(cfg.model, "Et*4|(tttt|)*14tmL")
            _enable_deepseek_precision_aware_optimizer(cfg)
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
        case "deepseek_v3_pretrain_256gpu_gb300_nvfp4_config":
            _apply_deepseek_v3_gb300_nvfp4(cfg)
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
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_NORM_BWD_USE_CUDNN": 1,
                "NVTE_NORM_FWD_USE_CUDNN": 1,
                "NVTE_USE_FAST_MATH": 1,
                "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
                "NVTE_DPA_FP8_RECIPE": "MXFP8BlockScaling",
                "NVTE_DPA_FP8_FORMAT": "E4M3",
            }
            return cfg
        case "deepseek_v3_pretrain_64gpu_gb300_bf16_fsdp_config":
            cfg.mixed_precision = _perf_precision("bf16")
            _apply_deepseek_v3_64gpu_gb300_fsdp_configs(cfg)
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
                "NVTE_CPU_OFFLOAD_V1": 1,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
            }
            return cfg
        case "deepseek_v3_pretrain_64gpu_gb300_fp8mx_fsdp_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.model.fp8_output_proj = True
            _apply_deepseek_v3_64gpu_gb300_fsdp_configs(cfg)
            cfg.ddp.outer_dp_sharding_strategy = "no_shard"
            cfg.ddp.num_distributed_optimizer_instances = 1
            cfg.model.fp8_param_gather = True
            cfg.model.fp8_param = True
            cfg.model.moe_router_dtype = "bf16"
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
                "NVTE_CPU_OFFLOAD_V1": 1,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_NORM_BWD_USE_CUDNN": 1,
                "NVTE_NORM_FWD_USE_CUDNN": 1,
            }
            return cfg
        case "deepseek_v3_pretrain_64gpu_gb300_fp8mx_mfsdpv2_config":
            apply_overrides(cfg, "deepseek_v3_pretrain_64gpu_gb300_fp8mx_fsdp_config")
            cfg.ddp.megatron_fsdp_version = 2
            _enable_deepseek_full_iteration(cfg)
            _enable_ncclep(cfg)
            cfg.model.fine_grained_offloading_max_inflight_offloads = 1
            cfg.train.micro_batch_size = 1
            cfg.train.global_batch_size = 64
            cfg.comm_overlap.overlap_moe_expert_parallel_comm = False
            cfg.comm_overlap.delay_wgrad_compute = False
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
                "NCCL_NVLS_ENABLE": 0,
                "NCCL_EP_HT_EM_PULL_PUSH": 1,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_CPU_OFFLOAD_V1": 1,
                "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_NORM_BWD_USE_CUDNN": 1,
                "NVTE_NORM_FWD_USE_CUDNN": 1,
            }
            return cfg
        case "deepseek_v3_pretrain_64gpu_gb300_fp8mx_proxy_config":
            apply_overrides(cfg, "deepseek_v3_pretrain_256gpu_gb300_fp8mx_config")
            cfg.model.num_layers = 13
            cfg.model.moe_layer_freq = [0] * 3 + [1] * (cfg.model.num_layers - 3)
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)
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
                "NVTE_NORM_BWD_USE_CUDNN": 1,
                "NVTE_NORM_FWD_USE_CUDNN": 1,
                "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 1,
            }
            return cfg
        case "deepseek_v4_flash_pretrain_128gpu_gb300_fp8mx_config":
            _apply_deepseek_v4_flash_gb200_fp8mx(cfg)
            cfg.model.expert_model_parallel_size = 32
            cfg.train.micro_batch_size = 2
            cfg.env_vars = {
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
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_NORM_BWD_USE_CUDNN": 1,
                "NVTE_NORM_FWD_USE_CUDNN": 1,
                "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
            }
            return cfg
        case "deepseek_v4_pro_pretrain_256gpu_gb300_bf16_config":
            _apply_deepseek_v4_pro_pretrain_256gpu_gb300_common(cfg)
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.quant_recipe = None
            cfg.model.moe_router_padding_for_fp8 = False
            cfg.model.moe_router_padding_for_quantization = False
            cfg.model.recompute_modules = ["mla_up_proj", "mhc", "moe"]
            cfg.model.cuda_graph_impl = "none"
            cfg.model.cuda_graph_scope = []
            cfg.model.use_te_rng_tracker = False
            cfg.rng.te_rng_tracker = False
            cfg.model.moe_pad_experts_for_cuda_graph_inference = False
            cfg.model.moe_paged_stash = False
            cfg.model.fine_grained_activation_offloading = False
            cfg.model.offload_modules = []
            cfg.model.fine_grained_offloading_max_inflight_offloads = None
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
                "NCCL_NVLS_ENABLE": 0,
                "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 64,
                "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
                "NVLINK_DOMAIN_SIZE": 72,
                "USE_MNNVL": 1,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_NORM_BWD_USE_CUDNN": 1,
                "NVTE_NORM_FWD_USE_CUDNN": 1,
                "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
            }
            return cfg
        case "deepseek_v4_pro_pretrain_256gpu_gb300_fp8mx_config":
            _apply_deepseek_v4_pro_pretrain_256gpu_gb300_common(cfg)
            cfg.env_vars = {
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
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_CPU_OFFLOAD_V1": 1,
                "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_NORM_BWD_USE_CUDNN": 1,
                "NVTE_NORM_FWD_USE_CUDNN": 1,
                "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
            }
            return cfg
        case "deepseek_v3_pretrain_1024gpu_h100_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            _deepseek_v3_common(cfg)
            cfg.model.tensor_model_parallel_size = 2
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.virtual_pipeline_model_parallel_size = 4
            cfg.model.context_parallel_size = 1
            cfg.model.expert_model_parallel_size = 64
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 16384
            cfg.train.micro_batch_size = 1
            cfg.model.recompute_modules = ["mla_up_proj", "mlp"]
            cfg.ddp.overlap_grad_reduce = False
            cfg.comm_overlap.overlap_grad_reduce = False
            set_deepseek_v3_pipeline_model_parallel_layout(cfg.model, "Et|(tt|)*30mL")
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["deepseek_v3_pretrain_1024gpu_h100_bf16"].copy()
            return cfg
        case "deepseek_v3_pretrain_1024gpu_h100_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            _deepseek_v3_common(cfg)
            cfg.model.tensor_model_parallel_size = 2
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.virtual_pipeline_model_parallel_size = 4
            cfg.model.context_parallel_size = 1
            cfg.model.expert_model_parallel_size = 64
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 16384
            cfg.train.micro_batch_size = 1
            cfg.model.recompute_modules = ["mla_up_proj", "mlp"]
            cfg.ddp.overlap_grad_reduce = False
            cfg.comm_overlap.overlap_grad_reduce = False
            set_deepseek_v3_pipeline_model_parallel_layout(cfg.model, "Et|(tt|)*30mL")
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["deepseek_v3_pretrain_1024gpu_h100_bf16"].copy()
            return cfg
        case "deepseek_v3_pretrain_1024gpu_h100_fp8sc_config":
            apply_overrides(cfg, "deepseek_v3_pretrain_1024gpu_h100_fp8cs_config")
            cfg.mixed_precision.fp8_recipe = "blockwise"
            cfg.mixed_precision.fp8_param = False
            cfg.mixed_precision.fp8_param_gather = False
            cfg.mixed_precision.num_layers_at_start_in_bf16 = 0
            cfg.mixed_precision.num_layers_at_end_in_bf16 = 0
            cfg.model.virtual_pipeline_model_parallel_size = 2
            set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["deepseek_v3_pretrain_1024gpu_h100_bf16"].copy()
            return cfg
        case "deepseek_v3_pretrain_1024gpu_h100_fp8sc_large_scale_config":
            apply_overrides(cfg, "deepseek_v3_pretrain_1024gpu_h100_fp8sc_config")
            cfg.train.global_batch_size = 1024
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["deepseek_v3_pretrain_1024gpu_h100_bf16"].copy()
            return cfg
        case "deepseek_v3_pretrain_64gpu_h100_bf16_config":
            apply_overrides(cfg, "deepseek_v3_pretrain_1024gpu_h100_bf16_config")
            cfg.train.global_batch_size = 1024
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["deepseek_v3_pretrain_1024gpu_h100_bf16"].copy()
            return cfg
        case "deepseek_v3_pretrain_64gpu_h100_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            _deepseek_v3_common(cfg)
            cfg.model.tensor_model_parallel_size = 2
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.virtual_pipeline_model_parallel_size = 4
            cfg.model.context_parallel_size = 1
            cfg.model.expert_model_parallel_size = 64
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 16384
            cfg.train.micro_batch_size = 1
            cfg.model.recompute_modules = ["mla_up_proj", "mlp"]
            cfg.ddp.overlap_grad_reduce = False
            cfg.comm_overlap.overlap_grad_reduce = False
            set_deepseek_v3_pipeline_model_parallel_layout(cfg.model, "Et|(tt|)*30mL")
            _benchmark_common(cfg)
            cfg.train.global_batch_size = 1024
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["deepseek_v3_pretrain_1024gpu_h100_bf16"].copy()
            return cfg
        case "deepseek_v3_pretrain_128gpu_vrnvl72_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            _deepseek_v3_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 2
            cfg.model.virtual_pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.expert_model_parallel_size = 64
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 2048
            cfg.train.micro_batch_size = 1
            cfg.model.recompute_modules = ["mla_up_proj"]
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.ddp.overlap_grad_reduce = True
            cfg.comm_overlap.overlap_grad_reduce = True
            set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)
            _benchmark_common(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["deepseek_v3_pretrain_128gpu_vrnvl72_bf16"].copy()
            return cfg
        case "deepseek_v3_pretrain_128gpu_vrnvl72_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            _deepseek_v3_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 2
            cfg.model.virtual_pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.expert_model_parallel_size = 64
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 2048
            cfg.train.micro_batch_size = 1
            cfg.model.recompute_modules = ["mlp"]
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.ddp.overlap_grad_reduce = True
            cfg.comm_overlap.overlap_grad_reduce = True
            set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["deepseek_v3_pretrain_256gpu_gb200_fp8cs"].copy()
            return cfg
        case "deepseek_v3_pretrain_128gpu_vrnvl72_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            _deepseek_v3_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 2
            cfg.model.virtual_pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.expert_model_parallel_size = 64
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 2048
            cfg.train.micro_batch_size = 1
            cfg.model.recompute_modules = ["mla_up_proj"]
            cfg.ddp.overlap_grad_reduce = True
            cfg.comm_overlap.overlap_grad_reduce = True
            set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)
            _benchmark_common(cfg)
            _enable_deepseek_full_iteration(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.model.fp8_output_proj = False
            cfg.mixed_precision.fp8_dot_product_attention = False
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["deepseek_v3_pretrain_256gpu_gb200_fp8mx"].copy()
            return cfg
        case "deepseek_v3_pretrain_128gpu_vrnvl72_nvfp4_config":
            cfg.mixed_precision = _perf_precision("nvfp4")
            _deepseek_v3_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 2
            cfg.model.virtual_pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.expert_model_parallel_size = 64
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 2048
            cfg.train.micro_batch_size = 1
            cfg.model.recompute_modules = ["mlp"]
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.ddp.overlap_grad_reduce = True
            cfg.comm_overlap.overlap_grad_reduce = True
            set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)
            _benchmark_common(cfg)
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
                "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
                "NVTE_USE_FAST_MATH": 1,
            }
            return cfg
        case "deepseek_v3_pretrain_256gpu_vrnvl72_bf16_config":
            _apply_deepseek_v3_gb300_bf16(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.model.cuda_graph_scope = ["attn"]
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["deepseek_v3_pretrain_128gpu_vrnvl72_bf16"].copy()
            return cfg
        case "deepseek_v3_pretrain_256gpu_vrnvl72_fp8cs_config":
            apply_overrides(cfg, "deepseek_v3_pretrain_256gpu_gb300_fp8cs_config")
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
                "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
            }
            return cfg
        case "deepseek_v3_pretrain_256gpu_vrnvl72_fp8mx_config":
            _apply_deepseek_v3_gb300_fp8mx(cfg)
            cfg.model.pipeline_model_parallel_size = 2
            cfg.model.virtual_pipeline_model_parallel_size = 8
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.model.moe_flex_dispatcher_num_sms = 64
            cfg.model.moe_combine_bwd_dtype = "mxfp8"
            cfg.model.moe_expert_rank_capacity_factor = 1.5
            cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.2
            cfg.model.moe_router_dtype = "bf16"
            cfg.model.moe_router_padding_for_quantization = True
            cfg.model.fused_residual_rmsnorm = True
            cfg.model.mla_down_proj_fusion = True
            cfg.model.use_fused_mla_kv_rope_quant = True
            cfg.model.use_grouped_gemm_for_shared_expert = True
            cfg.model.moe_shared_expert_glu_interleave_size = 32
            cfg.model.cuda_graph_warmup_steps = 2
            cfg.optimizer.optimizer_cuda_graph = True
            cfg.dist.use_gloo_process_groups = False
            cfg.train.manual_gc_interval = 500
            cfg.train.manual_gc_eval = False
            cfg.train.check_optimizer_step_success = False
            cfg.dataset.create_attention_mask = False
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["deepseek_v3_pretrain_256gpu_vrnvl72_fp8mx"].copy()
            return cfg
        case "deepseek_v3_pretrain_256gpu_vrnvl72_nvfp4_config":
            _apply_deepseek_v3_gb300_nvfp4(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["deepseek_v3_pretrain_256gpu_vrnvl72_nvfp4"].copy()
            return cfg
        case "deepseek_v3_pretrain_4gpu_vrnvl72_fp8mx_proxy_config":
            _apply_deepseek_v3_gb300_fp8mx(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            _apply_deepseek_v3_4gpu_vrnvl72_proxy(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["deepseek_v3_pretrain_256gpu_gb200_fp8mx"].copy()
            return cfg
        case "deepseek_v3_pretrain_4gpu_vrnvl72_nvfp4_proxy_config":
            apply_overrides(cfg, "deepseek_v3_pretrain_256gpu_vrnvl72_nvfp4_config")
            _apply_deepseek_v3_4gpu_vrnvl72_proxy(cfg)
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
                "NVTE_NORM_BWD_USE_CUDNN": 1,
                "NVTE_NORM_FWD_USE_CUDNN": 1,
                "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
                "NVTE_USE_FAST_MATH": 1,
                "NVTE_DPA_FP8_RECIPE": "MXFP8BlockScaling",
                "NVTE_DPA_FP8_FORMAT": "E4M3",
            }
            return cfg
        case "deepseek_v3_pretrain_64gpu_vrnvl72_fp8mx_proxy_config":
            apply_overrides(cfg, "deepseek_v3_pretrain_256gpu_vrnvl72_fp8mx_config")
            cfg.model.moe_dispatch_fwd_dtype = "mxfp8"
            cfg.model.moe_expert_rank_capacity_factor = 1.5
            cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.2
            cfg.model.num_layers = 13
            cfg.model.moe_layer_freq = [0] * 3 + [1] * (cfg.model.num_layers - 3)
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["deepseek_v3_pretrain_256gpu_vrnvl72_fp8mx"].copy()
            return cfg
        case "deepseek_v3_pretrain_64gpu_vrnvl72_nvfp4_proxy_config":
            apply_overrides(cfg, "deepseek_v3_pretrain_256gpu_vrnvl72_nvfp4_config")
            cfg.model.num_layers = 13
            cfg.model.moe_layer_freq = [0] * 3 + [1] * (cfg.model.num_layers - 3)
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["deepseek_v3_pretrain_256gpu_vrnvl72_nvfp4"].copy()
            return cfg
        case "deepseek_v4_flash_pretrain_128gpu_vrnvl72_fp8mx_config":
            apply_overrides(cfg, "deepseek_v4_flash_pretrain_128gpu_gb300_fp8mx_config")
            _enable_ncclep(cfg)
            cfg.model.activation_func_clamp_value = None
            cfg.model.use_transformer_engine_op_fuser = True
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["deepseek_v4_flash_pretrain_128gpu_gb200_fp8mx"].copy()
            return cfg
        case _:
            raise ValueError(f"Unknown benchmark overrides: {name}")
