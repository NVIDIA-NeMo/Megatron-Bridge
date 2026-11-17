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

"""nemotronh benchmark overrides applied to library recipes.

The literal names preserve existing CLI selectors. Common and model overrides
retain their ordering because graph, RNG and precision policies are dependent.
"""

from pathlib import Path

import torch
from megatron.core.quantization.utils import load_quantization_recipe

from megatron.bridge.recipes.nemotronh.h100.nemotron_3_super import nemotron_3_super_pretrain_16gpu_h100_bf16_config
from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_pretrain_config
from megatron.bridge.recipes.nemotronh.nemotron_3_super import nemotron_3_super_pretrain_config
from megatron.bridge.recipes.nemotronh.nemotron_3_ultra import nemotron_3_ultra_pretrain_config
from megatron.bridge.recipes.utils.determinism_utils import apply_determinism_overrides
from megatron.bridge.recipes.utils.environment_utils import DETERMINISM_ENV_VARS
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import (
    MixedPrecisionConfig,
    nemotron_3_super_bf16_with_nvfp4_mixed,
    nemotron_3_ultra_bf16_with_nvfp4_mixed,
)
from megatron.bridge.training.perf_benchmark_overrides._common import (
    _benchmark_common,
    _enable_ncclep,
    _perf_precision,
)
from megatron.bridge.training.perf_benchmark_overrides.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.utils.cuda_graph import set_cuda_graph_modules, set_full_iteration_cuda_graph


BENCHMARK_ENVIRONMENTS = {
    "nemotron_3_5_lightning_pretrain_8gpu_b200_bf16": {
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
    },
    "nemotron_3_5_lightning_pretrain_8gpu_b200_nvfp4": {
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
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        "NVTE_USE_FAST_MATH": 1,
    },
    "nemotron_3_nano_pretrain_8gpu_b200_bf16": {
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
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
    },
    "nemotron_3_super_pretrain_64gpu_b200_bf16": {
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
    "nemotron_3_5_lightning_pretrain_8gpu_b300_nvfp4": {
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
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        "NCCL_IGNORE_CPU_AFFINITY": 1,
        "NVTE_USE_FAST_MATH": 1,
    },
    "nemotron_3_nano_pretrain_8gpu_b300_bf16": {
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
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        "NCCL_IGNORE_CPU_AFFINITY": 1,
    },
    "nemotron_3_super_pretrain_64gpu_b300_bf16": {
        **COMMON_PERF_ENV_VARS,
        "NVTE_TENSOR_HANDLE_POOL_SIZE_MB": 128,
        "TORCHINDUCTOR_COMPILE_THREADS": 1,
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
    "nemotron_3_5_lightning_pretrain_8gpu_gb200_fp8mx": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NCCL_EP_HT_EM_PULL_PUSH": 1,
        "CUDNNFE_CLUSTER_OVERLAP_MARGIN": 8,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
    },
    "nemotron_3_5_lightning_pretrain_8gpu_gb200_fp8mx_fsdp": {
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
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
    },
    "nemotron_3_5_lightning_pretrain_8gpu_gb200_nvfp4": {
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
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        "NVTE_USE_FAST_MATH": 1,
    },
    "nemotron_3_nano_pretrain_8gpu_gb200_fp8mx": {
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
        "CUDNNFE_CLUSTER_OVERLAP_MARGIN": 8,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
    },
    "nemotron_3_super_pretrain_64gpu_gb200_fp8mx": {
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
    "nemotron_3_super_pretrain_64gpu_gb200_nvfp4": {
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
    },
    "nemotron_3_5_lightning_pretrain_8gpu_gb300_fp8mx": {
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
}
_TE_QUANT_CFG_PATH = Path(__file__).with_name("te_quant.cfg")
_GB300_NVLINK_DOMAIN_GPUS = 64


def _apply_nemotron_3_nano_perf_defaults(cfg: ConfigContainer) -> None:
    """Apply the canonical Nemotron 3 Nano performance workload defaults."""
    cfg.model.seq_length = 8192
    cfg.dataset.seq_length = 8192
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    # Preserve the optimizer-state precision used by the measured performance
    # recipes even if the source library recipe later adopts different
    # optimizer-state storage for a constrained hardware target.
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32


def _nemotron_3_super_nvfp4_precision() -> MixedPrecisionConfig:
    """Return the NVFP4 precision config used by Nemotron 3 Super perf recipes."""
    cfg = nemotron_3_super_bf16_with_nvfp4_mixed()
    # Although MCore PR 4358 is merged,
    # Megatron-FSDP's ParamAndGradBuffer has no NVFP4 packed-storage support, unlike the
    # legacy DDP buffer, so FP4 primary weights fault during buffer init.
    cfg.fp4_param_gather = False
    return cfg


def _nemotron_3_ultra_nvfp4_precision() -> MixedPrecisionConfig:
    """Return the NVFP4 precision config used by Nemotron 3 Ultra perf recipes."""
    cfg = nemotron_3_ultra_bf16_with_nvfp4_mixed()
    return cfg


def _apply_nemotron_3_super_perf_defaults(cfg: ConfigContainer) -> None:
    """Apply shared Nemotron 3 Super perf defaults after recipe-specific overrides."""
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    # The Nemotron 3 Super base recipe is memory-bounded for 16-GPU support
    # runs. Benchmarks measure throughput on larger systems, so restore
    # overlapped collectives and full-precision optimizer state.
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    cfg.model.moe_router_force_load_balancing = True
    cfg.checkpoint.async_save = False

    _benchmark_common(cfg)


def _enable_nemotron_3_super_full_iteration(cfg: ConfigContainer) -> None:
    """Enable the HybridEP full-iteration stack for Nemotron 3 Super."""
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
    cfg.model.high_priority_a2a_comm_stream = False
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.model.moe_mlp_glu_interleave_size = 32
    cfg.model.moe_hybridep_num_sms_preprocessing = 32

    # Nemotron 3 Super uses two MTP layers. MCore only supports expert-parallel
    # A2A overlap and delayed wgrad with at most one MTP layer.
    cfg.comm_overlap = None
    cfg.model.overlap_moe_expert_parallel_comm = False
    cfg.model.delay_wgrad_compute = False


def _enable_nemotron_3_5_lightning_full_iteration(cfg: ConfigContainer) -> None:
    """Enable full-iteration capture with static MoE buffers for Lightning."""
    set_full_iteration_cuda_graph(cfg.model)
    cfg.rng.te_rng_tracker = True
    cfg.model.use_te_rng_tracker = True

    # Static receive capacity avoids dispatcher CPU synchronization during capture.
    # The training loop retries capacity overflows without padding.
    cfg.model.moe_expert_rank_capacity_factor = 1.5
    cfg.model.moe_use_grouped_tensor = True
    cfg.model.moe_paged_stash = False
    cfg.model.offload_modules = []


def _apply_nemotron_3_ultra_perf_defaults(cfg: ConfigContainer) -> None:
    """Apply shared Nemotron 3 Ultra perf defaults after recipe-specific overrides."""

    # Native cross-entropy fusion
    # TE fusion has known stability issues and is rejected by Megatron-LM arg validation.
    _benchmark_common(cfg, cross_entropy_impl="native")

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.model.moe_router_force_load_balancing = True
    cfg.checkpoint.async_save = False

    # MoE token dispatcher + grouped-GEMM / router fusions
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_shared_expert_overlap = False  # unsupported by MCore during training
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_grouped_gemm = True
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_router_fusion = True

    # CuteDSL fused grouped MLP + TE op fuser
    cfg.model.use_transformer_engine_op_fuser = True

    # Kernel / graph selections.
    cfg.model.attention_backend = "fused"
    cfg.model.use_fused_weighted_squared_relu = True
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = []
    cfg.model.init_method_std = 0.0099

    # Batch sizing with manual GC.
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    # High priority NCCL stream for the EP communicator + longer init timeout
    cfg.dist.high_priority_stream_groups = ["ep"]
    cfg.dist.distributed_timeout_minutes = 30

    # Optimizer / scheduler
    cfg.optimizer.lr = 8.0e-4
    cfg.optimizer.min_lr = 8.0e-6
    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1e-8
    cfg.scheduler.start_weight_decay = 0.1
    cfg.scheduler.end_weight_decay = 0.1
    cfg.scheduler.lr_decay_style = "WSD"

    # DDP bucketing
    cfg.ddp.num_buckets = 48


def _apply_nemotron_3_ultra_gtp(cfg: ConfigContainer) -> None:
    """Apply rack-local dense GTP settings for Nemotron 3 Ultra on GB300.

    Dense weights are sharded across one 64-GPU NVLink domain. Routed-expert
    weights remain unsharded by GTP because EP64 already spans that domain;
    this avoids expert-weight all-gathers across racks.
    """
    cfg.ddp.use_megatron_fsdp = False

    # GTP is controlled through the user-facing total weight-shard counts.
    # TP1 x GTP64 gives one dense-weight shard per GPU within a GB300 rack.
    cfg.model.tensor_parallel_num_weight_shards = _GB300_NVLINK_DOMAIN_GPUS
    cfg.model.expert_tensor_parallel_num_weight_shards = cfg.model.expert_tensor_parallel_size

    # GTP currently supports only one distributed-optimizer instance.
    cfg.ddp.num_distributed_optimizer_instances = 1
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"
    cfg.ddp.outer_dp_sharding_strategy = "no_shard"


def _apply_nemotron_3_ultra_fsdp_hsdp(cfg: ConfigContainer, num_gpus: int) -> None:
    """Apply Megatron-FSDP (HSDP) settings for Nemotron 3 Ultra on GB300.

    Shards params/grads/optimizer
    within each NVLink domain and replicate (optimizer-sharded) across domains, with
    BF16 gradient comm, FP32 main params, and BF16 main grads. Applied last so it
    wins over the generic perf defaults.
    """
    # Base Megatron-FSDP enablement
    cfg.ddp.use_megatron_fsdp = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.keep_fp8_transpose_cache = False

    # average_in_collective is not supported with Megatron-FSDP.
    cfg.ddp.average_in_collective = False
    cfg.model.init_model_with_meta_device = True
    cfg.checkpoint.load = None

    # HSDP: shard within an NVLink domain, replicate (optim-sharded) across domains.
    num_optim_instances = max(1, num_gpus // _GB300_NVLINK_DOMAIN_GPUS)
    cfg.ddp.num_distributed_optimizer_instances = num_optim_instances

    # HSDP across NVLink domains. Megatron-FSDP
    # only enables HSDP when num_distributed_optimizer_instances > 1; with a single
    # NVLink domain HSDP is off, so the outer strategy must be "no_shard" (otherwise
    # the first param all-gather hits a None HSDP helper buffer).
    cfg.ddp.outer_dp_sharding_strategy = "optim" if num_optim_instances > 1 else "no_shard"

    cfg.ddp.megatron_fsdp_grad_comm_dtype = torch.bfloat16
    cfg.ddp.megatron_fsdp_main_params_dtype = torch.float32
    cfg.ddp.megatron_fsdp_main_grads_dtype = torch.bfloat16

    # incompatible with BF16 FSDP main grads
    cfg.model.gradient_accumulation_fusion = False

    cfg.checkpoint.ckpt_format = "fsdp_dtensor"


_b200_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
_b200_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_REVISION = "b3caaabed0263651a17dc1f2d4ce97e794f76c44"


def _b200_nemotronh__apply_nemotron_3_5_lightning_defaults(cfg: ConfigContainer) -> ConfigContainer:
    """Apply the Nemotron 3.5 Lightning workload contract to a B200 execution config."""
    cfg.model.mtp_num_layers = 2
    cfg.model.mtp_hybrid_override_pattern = "*E"
    cfg.model.mtp_use_repeated_layer = True
    cfg.model.mtp_loss_scaling_factor = 0.3
    cfg.model.extra_checkpoint_metadata.update(
        hf_model_id=_b200_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_ID,
        hf_model_revision=_b200_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_REVISION,
    )
    cfg.tokenizer.tokenizer_model = _b200_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_ID
    cfg.tokenizer.hf_tokenizer_kwargs = {"revision": _b200_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_REVISION}
    return cfg


_gb200_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
_gb200_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_REVISION = "b3caaabed0263651a17dc1f2d4ce97e794f76c44"


def _apply_nemotron_3_nano_gb200_mxfp8(cfg: ConfigContainer) -> ConfigContainer:
    _apply_nemotron_3_nano_perf_defaults(cfg)
    cfg.mixed_precision = _perf_precision("fp8_mx")
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 2
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_router_force_load_balancing = True
    cfg.model.cuda_graph_impl = "transformer_engine"
    set_cuda_graph_modules(cfg.model, ["attn", "mamba", "moe_router", "moe_preprocess"])
    cfg.comm_overlap.tp_comm_overlap = True
    _benchmark_common(cfg)
    cfg.model.moe_hybridep_num_sms = 32
    return cfg


def _apply_nemotron_3_5_lightning_gb200_mxfp8(cfg: ConfigContainer) -> ConfigContainer:
    _apply_nemotron_3_nano_gb200_mxfp8(cfg)
    cfg.model.moe_hybridep_num_sms = 16
    cfg.model.mtp_num_layers = 2
    cfg.model.mtp_hybrid_override_pattern = "*E"
    cfg.model.mtp_use_repeated_layer = True
    cfg.model.mtp_loss_scaling_factor = 0.3
    cfg.model.extra_checkpoint_metadata.update(
        hf_model_id=_gb200_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_ID,
        hf_model_revision=_gb200_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_REVISION,
    )
    cfg.tokenizer.tokenizer_model = _gb200_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_ID
    cfg.tokenizer.hf_tokenizer_kwargs = {"revision": _gb200_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_REVISION}
    return cfg


_h100_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
_h100_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_REVISION = "b3caaabed0263651a17dc1f2d4ce97e794f76c44"
_b300_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
_b300_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_REVISION = "b3caaabed0263651a17dc1f2d4ce97e794f76c44"


def _b300_nemotronh__apply_nemotron_3_5_lightning_defaults(cfg: ConfigContainer) -> ConfigContainer:
    """Apply the Nemotron 3.5 Lightning workload contract to a B300 execution config."""
    cfg.model.mtp_num_layers = 2
    cfg.model.mtp_hybrid_override_pattern = "*E"
    cfg.model.mtp_use_repeated_layer = True
    cfg.model.mtp_loss_scaling_factor = 0.3
    cfg.model.extra_checkpoint_metadata.update(
        hf_model_id=_b300_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_ID,
        hf_model_revision=_b300_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_REVISION,
    )
    cfg.tokenizer.tokenizer_model = _b300_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_ID
    cfg.tokenizer.hf_tokenizer_kwargs = {"revision": _b300_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_REVISION}
    return cfg


_gb300_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
_gb300_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_REVISION = "b3caaabed0263651a17dc1f2d4ce97e794f76c44"


def _apply_nemotron_3_super_gb300_bf16(cfg: ConfigContainer) -> ConfigContainer:
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]
    _apply_nemotron_3_super_perf_defaults(cfg)
    return cfg


def _apply_nemotron_3_super_gb300_mxfp8(cfg: ConfigContainer) -> ConfigContainer:
    cfg.mixed_precision = _perf_precision("fp8_mx")
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_padding_for_quantization = True
    _apply_nemotron_3_super_perf_defaults(cfg)
    _enable_nemotron_3_super_full_iteration(cfg)
    return cfg


def _apply_nemotron_3_super_gb300_nvfp4(cfg: ConfigContainer) -> ConfigContainer:
    cfg.mixed_precision = _nemotron_3_super_nvfp4_precision()
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 16
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_padding_for_quantization = True
    cfg.model.quant_recipe = load_quantization_recipe(str(_TE_QUANT_CFG_PATH))
    _apply_nemotron_3_super_perf_defaults(cfg)
    _enable_nemotron_3_super_full_iteration(cfg)
    cfg.mixed_precision.fp8_dot_product_attention = False
    return cfg


def _apply_nemotron_3_ultra_gb300_fp8mx_config(
    cfg: ConfigContainer, *, num_gpus: int, expert_model_parallel_size: int, global_batch_size: int
) -> ConfigContainer:
    cfg.mixed_precision = _perf_precision("fp8_mx")
    _apply_nemotron_3_ultra_perf_defaults(cfg)
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.seq_length = 8192
    cfg.model.min_offloaded_tensor_size = 500000000
    cfg.model.moe_router_padding_for_quantization = True
    cfg.model.expert_model_parallel_size = expert_model_parallel_size
    cfg.train.global_batch_size = global_batch_size
    cfg.model.fine_grained_activation_offloading = True
    cfg.model.offload_modules = ["fused_group_mlp"]
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["moe_act"]
    _apply_nemotron_3_ultra_gtp(cfg)
    cfg.model.cuda_graph_impl = "local"
    set_cuda_graph_modules(cfg.model, ["attn", "mamba", "moe_preprocess", "moe_router"])
    cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True
    _enable_ncclep(cfg)
    cfg.model.moe_use_grouped_tensor = True
    return cfg


def _apply_nemotron_3_nano_gb300_bf16(cfg: ConfigContainer) -> ConfigContainer:
    _apply_nemotron_3_nano_perf_defaults(cfg)
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 4
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_router_force_load_balancing = True
    cfg.model.cuda_graph_impl = "transformer_engine"
    set_cuda_graph_modules(cfg.model, ["attn", "mamba", "moe_router", "moe_preprocess"])
    cfg.comm_overlap.tp_comm_overlap = True
    _benchmark_common(cfg)
    cfg.model.moe_hybridep_num_sms = 16
    return cfg


def _apply_nemotron_3_nano_gb300_mxfp8(cfg: ConfigContainer) -> ConfigContainer:
    _apply_nemotron_3_nano_perf_defaults(cfg)
    cfg.mixed_precision = _perf_precision("fp8_mx")
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 4
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_router_force_load_balancing = True
    cfg.model.cuda_graph_impl = "transformer_engine"
    set_cuda_graph_modules(cfg.model, ["attn", "mamba", "moe_router", "moe_preprocess"])
    cfg.comm_overlap.tp_comm_overlap = True
    _benchmark_common(cfg)
    cfg.model.moe_hybridep_num_sms = 16
    return cfg


def _apply_nemotron_3_5_lightning_gb300_bf16(cfg: ConfigContainer) -> ConfigContainer:
    _apply_nemotron_3_nano_gb300_bf16(cfg)
    cfg.model.mtp_num_layers = 2
    cfg.model.mtp_hybrid_override_pattern = "*E"
    cfg.model.mtp_use_repeated_layer = True
    cfg.model.mtp_loss_scaling_factor = 0.3
    cfg.model.extra_checkpoint_metadata.update(
        hf_model_id=_gb300_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_ID,
        hf_model_revision=_gb300_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_REVISION,
    )
    cfg.tokenizer.tokenizer_model = _gb300_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_ID
    cfg.tokenizer.hf_tokenizer_kwargs = {"revision": _gb300_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_REVISION}
    return cfg


def _apply_nemotron_3_5_lightning_gb300_mxfp8(cfg: ConfigContainer) -> ConfigContainer:
    _apply_nemotron_3_nano_gb300_mxfp8(cfg)
    cfg.model.moe_hybridep_num_sms = 16
    cfg.model.mtp_num_layers = 2
    cfg.model.mtp_hybrid_override_pattern = "*E"
    cfg.model.mtp_use_repeated_layer = True
    cfg.model.mtp_loss_scaling_factor = 0.3
    cfg.model.extra_checkpoint_metadata.update(
        hf_model_id=_gb300_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_ID,
        hf_model_revision=_gb300_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_REVISION,
    )
    cfg.tokenizer.tokenizer_model = _gb300_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_ID
    cfg.tokenizer.hf_tokenizer_kwargs = {"revision": _gb300_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_REVISION}
    return cfg


LIBRARY_RECIPES = {
    "nemotron_3_5_lightning_pretrain_8gpu_b200_bf16_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_5_lightning_pretrain_8gpu_b200_fp8mx_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_5_lightning_pretrain_8gpu_b200_nvfp4_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_nano_pretrain_8gpu_b200_bf16_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_nano_pretrain_8gpu_b200_fp8mx_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_nano_pretrain_8gpu_b200_nvfp4_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_super_pretrain_64gpu_b200_bf16_config": nemotron_3_super_pretrain_16gpu_h100_bf16_config,
    "nemotron_3_super_pretrain_64gpu_b200_fp8mx_config": nemotron_3_super_pretrain_16gpu_h100_bf16_config,
    "nemotron_3_super_pretrain_64gpu_b200_nvfp4_config": nemotron_3_super_pretrain_16gpu_h100_bf16_config,
    "nemotron_3_5_lightning_pretrain_8gpu_b300_bf16_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_5_lightning_pretrain_8gpu_b300_fp8mx_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_5_lightning_pretrain_8gpu_b300_nvfp4_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_nano_pretrain_8gpu_b300_bf16_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_nano_pretrain_8gpu_b300_fp8mx_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_nano_pretrain_8gpu_b300_nvfp4_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_super_pretrain_64gpu_b300_bf16_config": nemotron_3_super_pretrain_16gpu_h100_bf16_config,
    "nemotron_3_super_pretrain_64gpu_b300_fp8mx_config": nemotron_3_super_pretrain_16gpu_h100_bf16_config,
    "nemotron_3_super_pretrain_64gpu_b300_nvfp4_config": nemotron_3_super_pretrain_16gpu_h100_bf16_config,
    "nemotron_3_5_lightning_pretrain_8gpu_gb200_bf16_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_5_lightning_pretrain_8gpu_gb200_fp8mx_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_5_lightning_pretrain_8gpu_gb200_fp8mx_fsdp_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_5_lightning_pretrain_8gpu_gb200_nvfp4_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_nano_pretrain_8gpu_gb200_bf16_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_nano_pretrain_8gpu_gb200_fp8mx_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_nano_pretrain_8gpu_gb200_nvfp4_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_super_pretrain_64gpu_gb200_bf16_config": nemotron_3_super_pretrain_16gpu_h100_bf16_config,
    "nemotron_3_super_pretrain_64gpu_gb200_fp8mx_config": nemotron_3_super_pretrain_16gpu_h100_bf16_config,
    "nemotron_3_super_pretrain_64gpu_gb200_nvfp4_config": nemotron_3_super_pretrain_16gpu_h100_bf16_config,
    "nemotron_3_ultra_pretrain_256gpu_gb200_fp8mx_config": nemotron_3_ultra_pretrain_config,
    "nemotron_3_ultra_pretrain_256gpu_gb200_fp8mx_deterministic_config": nemotron_3_ultra_pretrain_config,
    "nemotron_3_ultra_pretrain_256gpu_gb200_nvfp4_config": nemotron_3_ultra_pretrain_config,
    "nemotron_3_5_lightning_pretrain_8gpu_gb300_bf16_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_5_lightning_pretrain_8gpu_gb300_fp8mx_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_5_lightning_pretrain_8gpu_gb300_fp8mx_fsdp_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_5_lightning_pretrain_8gpu_gb300_nvfp4_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_nano_pretrain_8gpu_gb300_bf16_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_nano_pretrain_8gpu_gb300_fp8mx_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_nano_pretrain_8gpu_gb300_nvfp4_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_super_pretrain_64gpu_gb300_bf16_config": nemotron_3_super_pretrain_16gpu_h100_bf16_config,
    "nemotron_3_super_pretrain_64gpu_gb300_fp8mx_config": nemotron_3_super_pretrain_16gpu_h100_bf16_config,
    "nemotron_3_super_pretrain_64gpu_gb300_nvfp4_config": nemotron_3_super_pretrain_16gpu_h100_bf16_config,
    "nemotron_3_ultra_pretrain_256gpu_gb300_fp8mx_config": nemotron_3_ultra_pretrain_config,
    "nemotron_3_ultra_pretrain_256gpu_gb300_nvfp4_config": nemotron_3_ultra_pretrain_config,
    "nemotron_3_5_lightning_pretrain_16gpu_h100_bf16_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_5_lightning_pretrain_16gpu_h100_fp8cs_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_nano_pretrain_16gpu_h100_bf16_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_nano_pretrain_16gpu_h100_fp8cs_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_super_pretrain_64gpu_h100_bf16_config": nemotron_3_super_pretrain_config,
    "nemotron_3_5_lightning_pretrain_8gpu_vrnvl72_bf16_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_5_lightning_pretrain_8gpu_vrnvl72_fp8mx_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_nano_pretrain_8gpu_vrnvl72_bf16_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_nano_pretrain_8gpu_vrnvl72_fp8mx_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_nano_pretrain_8gpu_vrnvl72_nvfp4_config": nemotron_3_nano_pretrain_config,
    "nemotron_3_super_pretrain_64gpu_vrnvl72_bf16_config": nemotron_3_super_pretrain_16gpu_h100_bf16_config,
    "nemotron_3_super_pretrain_64gpu_vrnvl72_fp8mx_config": nemotron_3_super_pretrain_16gpu_h100_bf16_config,
    "nemotron_3_super_pretrain_64gpu_vrnvl72_nvfp4_config": nemotron_3_super_pretrain_16gpu_h100_bf16_config,
    "nemotron_3_ultra_pretrain_256gpu_vrnvl72_fp8mx_config": nemotron_3_ultra_pretrain_config,
    "nemotron_3_ultra_pretrain_256gpu_vrnvl72_nvfp4_config": nemotron_3_ultra_pretrain_config,
}
ALLOCATION_AWARE = (
    "nemotron_3_ultra_pretrain_256gpu_gb200_fp8mx_config",
    "nemotron_3_ultra_pretrain_256gpu_gb200_fp8mx_deterministic_config",
    "nemotron_3_ultra_pretrain_256gpu_gb200_nvfp4_config",
    "nemotron_3_ultra_pretrain_256gpu_gb300_fp8mx_config",
    "nemotron_3_ultra_pretrain_256gpu_gb300_nvfp4_config",
    "nemotron_3_ultra_pretrain_256gpu_vrnvl72_fp8mx_config",
    "nemotron_3_ultra_pretrain_256gpu_vrnvl72_nvfp4_config",
)


def apply_overrides(cfg: ConfigContainer, name: str, *, num_gpus: int = 256) -> ConfigContainer:
    """Apply model-specific benchmark overrides to an existing library config."""
    match name:
        case "nemotron_3_5_lightning_pretrain_8gpu_b200_bf16_config":
            _b200_nemotronh__apply_nemotron_3_5_lightning_defaults(
                apply_overrides(cfg, "nemotron_3_nano_pretrain_8gpu_b200_bf16_config")
            )
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_5_lightning_pretrain_8gpu_b200_bf16"].copy()
            return cfg
        case "nemotron_3_5_lightning_pretrain_8gpu_b200_fp8mx_config":
            _b200_nemotronh__apply_nemotron_3_5_lightning_defaults(
                apply_overrides(cfg, "nemotron_3_nano_pretrain_8gpu_b200_fp8mx_config")
            )
            cfg.model.use_transformer_engine_op_fuser = True
            cfg.mixed_precision.fp8_dot_product_attention = True
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
                "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
            }
            return cfg
        case "nemotron_3_5_lightning_pretrain_8gpu_b200_nvfp4_config":
            _b200_nemotronh__apply_nemotron_3_5_lightning_defaults(
                apply_overrides(cfg, "nemotron_3_nano_pretrain_8gpu_b200_nvfp4_config")
            )
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_5_lightning_pretrain_8gpu_b200_nvfp4"].copy()
            return cfg
        case "nemotron_3_nano_pretrain_8gpu_b200_bf16_config":
            _apply_nemotron_3_nano_perf_defaults(cfg)
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.model.expert_model_parallel_size = 8
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 2
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.cuda_graph_impl = "transformer_engine"
            set_cuda_graph_modules(cfg.model, ["attn", "mamba", "moe_router", "moe_preprocess"])
            cfg.comm_overlap.tp_comm_overlap = True
            _benchmark_common(cfg)
            cfg.model.moe_hybridep_num_sms = 16
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_nano_pretrain_8gpu_b200_bf16"].copy()
            return cfg
        case "nemotron_3_nano_pretrain_8gpu_b200_fp8mx_config":
            _apply_nemotron_3_nano_perf_defaults(cfg)
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.model.expert_model_parallel_size = 8
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 2
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.cuda_graph_impl = "transformer_engine"
            set_cuda_graph_modules(cfg.model, ["attn", "mamba", "moe_router", "moe_preprocess"])
            cfg.comm_overlap.tp_comm_overlap = True
            _benchmark_common(cfg)
            cfg.model.moe_hybridep_num_sms = 32
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_nano_pretrain_8gpu_b200_bf16"].copy()
            return cfg
        case "nemotron_3_nano_pretrain_8gpu_b200_nvfp4_config":
            _apply_nemotron_3_nano_perf_defaults(cfg)
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.model.expert_model_parallel_size = 8
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 2
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.cuda_graph_impl = "transformer_engine"
            set_cuda_graph_modules(cfg.model, ["attn", "mamba", "moe_router", "moe_preprocess"])
            cfg.comm_overlap.tp_comm_overlap = True
            _benchmark_common(cfg)
            cfg.model.moe_hybridep_num_sms = 16
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_5_lightning_pretrain_8gpu_b200_nvfp4"].copy()
            return cfg
        case "nemotron_3_super_pretrain_64gpu_b200_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.tensor_model_parallel_size = 2
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = True
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.expert_model_parallel_size = 8
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 1
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.moe_shared_expert_overlap = False
            cfg.model.recompute_modules = ["moe_act", "layernorm", "core_attn"]
            cfg.model.recompute_granularity = "selective"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["mamba", "attn", "moe_router", "moe_preprocess"]
            _apply_nemotron_3_super_perf_defaults(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_super_pretrain_64gpu_b200_bf16"].copy()
            return cfg
        case "nemotron_3_super_pretrain_64gpu_b200_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.model.tensor_model_parallel_size = 2
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = True
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.expert_model_parallel_size = 8
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 1
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.moe_shared_expert_overlap = False
            cfg.model.moe_router_padding_for_quantization = True
            cfg.model.recompute_modules = ["moe_act", "layernorm", "core_attn"]
            cfg.model.recompute_granularity = "selective"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["mamba", "attn", "moe_router", "moe_preprocess"]
            _apply_nemotron_3_super_perf_defaults(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_super_pretrain_64gpu_b200_bf16"].copy()
            return cfg
        case "nemotron_3_super_pretrain_64gpu_b200_nvfp4_config":
            cfg.mixed_precision = _nemotron_3_super_nvfp4_precision()
            cfg.model.tensor_model_parallel_size = 2
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = True
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.expert_model_parallel_size = 8
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 1
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.moe_shared_expert_overlap = False
            cfg.model.moe_router_padding_for_quantization = True
            cfg.model.recompute_modules = ["moe_act", "layernorm"]
            cfg.model.recompute_granularity = "selective"
            cfg.model.quant_recipe = load_quantization_recipe(str(_TE_QUANT_CFG_PATH))
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["mamba", "attn", "moe_router", "moe_preprocess"]
            _apply_nemotron_3_super_perf_defaults(cfg)
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
        case "nemotron_3_5_lightning_pretrain_8gpu_b300_bf16_config":
            _b300_nemotronh__apply_nemotron_3_5_lightning_defaults(
                apply_overrides(cfg, "nemotron_3_nano_pretrain_8gpu_b300_bf16_config")
            )
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
                "NCCL_IGNORE_CPU_AFFINITY": 1,
            }
            return cfg
        case "nemotron_3_5_lightning_pretrain_8gpu_b300_fp8mx_config":
            _b300_nemotronh__apply_nemotron_3_5_lightning_defaults(
                apply_overrides(cfg, "nemotron_3_nano_pretrain_8gpu_b300_fp8mx_config")
            )
            cfg.model.use_transformer_engine_op_fuser = True
            cfg.mixed_precision.fp8_dot_product_attention = True
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
                "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
                "NCCL_IGNORE_CPU_AFFINITY": 1,
            }
            return cfg
        case "nemotron_3_5_lightning_pretrain_8gpu_b300_nvfp4_config":
            _b300_nemotronh__apply_nemotron_3_5_lightning_defaults(
                apply_overrides(cfg, "nemotron_3_nano_pretrain_8gpu_b300_nvfp4_config")
            )
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_5_lightning_pretrain_8gpu_b300_nvfp4"].copy()
            return cfg
        case "nemotron_3_nano_pretrain_8gpu_b300_bf16_config":
            _apply_nemotron_3_nano_perf_defaults(cfg)
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.model.expert_model_parallel_size = 8
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 4
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.cuda_graph_impl = "transformer_engine"
            set_cuda_graph_modules(cfg.model, ["attn", "mamba", "moe_router", "moe_preprocess"])
            cfg.comm_overlap.tp_comm_overlap = True
            _benchmark_common(cfg)
            cfg.model.moe_hybridep_num_sms = 16
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_nano_pretrain_8gpu_b300_bf16"].copy()
            return cfg
        case "nemotron_3_nano_pretrain_8gpu_b300_fp8mx_config":
            _apply_nemotron_3_nano_perf_defaults(cfg)
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.model.expert_model_parallel_size = 8
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 4
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.cuda_graph_impl = "transformer_engine"
            set_cuda_graph_modules(cfg.model, ["attn", "mamba", "moe_router", "moe_preprocess"])
            cfg.comm_overlap.tp_comm_overlap = True
            _benchmark_common(cfg)
            cfg.model.moe_hybridep_num_sms = 16
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_nano_pretrain_8gpu_b300_bf16"].copy()
            return cfg
        case "nemotron_3_nano_pretrain_8gpu_b300_nvfp4_config":
            _apply_nemotron_3_nano_perf_defaults(cfg)
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.model.expert_model_parallel_size = 8
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 4
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.cuda_graph_impl = "transformer_engine"
            set_cuda_graph_modules(cfg.model, ["attn", "mamba", "moe_router", "moe_preprocess"])
            cfg.comm_overlap.tp_comm_overlap = True
            _benchmark_common(cfg)
            cfg.model.moe_hybridep_num_sms = 16
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_5_lightning_pretrain_8gpu_b300_nvfp4"].copy()
            return cfg
        case "nemotron_3_super_pretrain_64gpu_b300_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.expert_model_parallel_size = 8
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 1
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.moe_shared_expert_overlap = False
            cfg.model.recompute_modules = ["moe_act", "layernorm"]
            cfg.model.recompute_granularity = "selective"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]
            _apply_nemotron_3_super_perf_defaults(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_super_pretrain_64gpu_b300_bf16"].copy()
            return cfg
        case "nemotron_3_super_pretrain_64gpu_b300_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.expert_model_parallel_size = 8
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 1
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.moe_shared_expert_overlap = False
            cfg.model.moe_router_padding_for_quantization = True
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]
            _apply_nemotron_3_super_perf_defaults(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_super_pretrain_64gpu_b300_bf16"].copy()
            return cfg
        case "nemotron_3_super_pretrain_64gpu_b300_nvfp4_config":
            cfg.mixed_precision = _nemotron_3_super_nvfp4_precision()
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.expert_model_parallel_size = 8
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 1
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.moe_shared_expert_overlap = False
            cfg.model.moe_router_padding_for_quantization = True
            cfg.model.recompute_modules = ["moe_act", "layernorm"]
            cfg.model.recompute_granularity = "selective"
            cfg.model.quant_recipe = load_quantization_recipe(str(_TE_QUANT_CFG_PATH))
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]
            _apply_nemotron_3_super_perf_defaults(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "NVTE_TENSOR_HANDLE_POOL_SIZE_MB": 128,
                "TORCHINDUCTOR_COMPILE_THREADS": 1,
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
        case "nemotron_3_5_lightning_pretrain_8gpu_gb200_bf16_config":
            apply_overrides(cfg, "nemotron_3_nano_pretrain_8gpu_gb200_bf16_config")
            cfg.model.recompute_modules = []
            cfg.model.mtp_num_layers = 2
            cfg.model.mtp_hybrid_override_pattern = "*E"
            cfg.model.mtp_use_repeated_layer = True
            cfg.model.mtp_loss_scaling_factor = 0.3
            cfg.model.extra_checkpoint_metadata.update(
                hf_model_id=_gb200_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_ID,
                hf_model_revision=_gb200_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_REVISION,
            )
            cfg.tokenizer.tokenizer_model = _gb200_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_ID
            cfg.tokenizer.hf_tokenizer_kwargs = {"revision": _gb200_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_REVISION}
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_5_lightning_pretrain_8gpu_b200_bf16"].copy()
            return cfg
        case "nemotron_3_5_lightning_pretrain_8gpu_gb200_fp8mx_config":
            _apply_nemotron_3_5_lightning_gb200_mxfp8(cfg)
            cfg.model.recompute_modules = []
            cfg.model.use_transformer_engine_op_fuser = True
            cfg.mixed_precision.fp8_dot_product_attention = True
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_5_lightning_pretrain_8gpu_gb200_fp8mx"].copy()
            return cfg
        case "nemotron_3_5_lightning_pretrain_8gpu_gb200_fp8mx_fsdp_config":
            _apply_nemotron_3_5_lightning_gb200_mxfp8(cfg)
            cfg.train.global_batch_size = 384
            cfg.train.micro_batch_size = 3
            cfg.model.cuda_graph_impl = "none"
            set_cuda_graph_modules(cfg.model, [])
            cfg.model.init_model_with_meta_device = True
            cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
            cfg.ddp.use_megatron_fsdp = True
            cfg.ddp.num_distributed_optimizer_instances = 1
            cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
            cfg.ddp.outer_dp_sharding_strategy = "no_shard"
            cfg.ddp.average_in_collective = False
            cfg.ddp.keep_fp8_transpose_cache = False
            cfg.ddp.reuse_grad_buf_for_mxfp8_param_ag = False
            cfg.optimizer.reuse_grad_buf_for_mxfp8_param_ag = False
            cfg.checkpoint.load = None
            cfg.checkpoint.ckpt_format = "fsdp_dtensor"
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_5_lightning_pretrain_8gpu_gb200_fp8mx_fsdp"].copy()
            return cfg
        case "nemotron_3_5_lightning_pretrain_8gpu_gb200_nvfp4_config":
            apply_overrides(cfg, "nemotron_3_nano_pretrain_8gpu_gb200_nvfp4_config")
            cfg.model.mtp_num_layers = 2
            cfg.model.mtp_hybrid_override_pattern = "*E"
            cfg.model.mtp_use_repeated_layer = True
            cfg.model.mtp_loss_scaling_factor = 0.3
            cfg.model.extra_checkpoint_metadata.update(
                hf_model_id=_gb200_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_ID,
                hf_model_revision=_gb200_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_REVISION,
            )
            cfg.tokenizer.tokenizer_model = _gb200_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_ID
            cfg.tokenizer.hf_tokenizer_kwargs = {"revision": _gb200_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_REVISION}
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_5_lightning_pretrain_8gpu_gb200_nvfp4"].copy()
            return cfg
        case "nemotron_3_nano_pretrain_8gpu_gb200_bf16_config":
            _apply_nemotron_3_nano_perf_defaults(cfg)
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.model.expert_model_parallel_size = 8
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 2
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.cuda_graph_impl = "transformer_engine"
            set_cuda_graph_modules(cfg.model, ["attn", "mamba", "moe_router", "moe_preprocess"])
            cfg.comm_overlap.tp_comm_overlap = True
            _benchmark_common(cfg)
            cfg.model.moe_hybridep_num_sms = 16
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_5_lightning_pretrain_8gpu_gb200_fp8mx_fsdp"].copy()
            return cfg
        case "nemotron_3_nano_pretrain_8gpu_gb200_fp8mx_config":
            _apply_nemotron_3_nano_gb200_mxfp8(cfg)
            cfg.model.use_transformer_engine_op_fuser = True
            cfg.model.moe_mlp_glu_interleave_size = 32
            cfg.mixed_precision.fp8_dot_product_attention = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_nano_pretrain_8gpu_gb200_fp8mx"].copy()
            return cfg
        case "nemotron_3_nano_pretrain_8gpu_gb200_nvfp4_config":
            _apply_nemotron_3_nano_perf_defaults(cfg)
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.model.expert_model_parallel_size = 8
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 2
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.cuda_graph_impl = "transformer_engine"
            set_cuda_graph_modules(cfg.model, ["attn", "mamba", "moe_router", "moe_preprocess"])
            cfg.comm_overlap.tp_comm_overlap = True
            _benchmark_common(cfg)
            cfg.model.moe_hybridep_num_sms = 16
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_5_lightning_pretrain_8gpu_gb200_nvfp4"].copy()
            return cfg
        case "nemotron_3_super_pretrain_64gpu_gb200_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.tensor_model_parallel_size = 2
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = True
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.expert_model_parallel_size = 64
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 1
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.moe_shared_expert_overlap = False
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]
            _apply_nemotron_3_super_perf_defaults(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_super_pretrain_64gpu_b200_bf16"].copy()
            return cfg
        case "nemotron_3_super_pretrain_64gpu_gb200_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.model.tensor_model_parallel_size = 2
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = True
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.expert_model_parallel_size = 16
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 1
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.moe_shared_expert_overlap = False
            cfg.model.moe_router_padding_for_quantization = True
            _apply_nemotron_3_super_perf_defaults(cfg)
            _enable_nemotron_3_super_full_iteration(cfg)
            cfg.mixed_precision.fp8_dot_product_attention = True
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_super_pretrain_64gpu_gb200_fp8mx"].copy()
            return cfg
        case "nemotron_3_super_pretrain_64gpu_gb200_nvfp4_config":
            cfg.mixed_precision = _nemotron_3_super_nvfp4_precision()
            cfg.model.tensor_model_parallel_size = 2
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = True
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.expert_model_parallel_size = 16
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 1
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.moe_shared_expert_overlap = False
            cfg.model.moe_router_padding_for_quantization = True
            cfg.model.quant_recipe = load_quantization_recipe(str(_TE_QUANT_CFG_PATH))
            _apply_nemotron_3_super_perf_defaults(cfg)
            _enable_nemotron_3_super_full_iteration(cfg)
            cfg.mixed_precision.fp8_dot_product_attention = False
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_super_pretrain_64gpu_gb200_nvfp4"].copy()
            return cfg
        case "nemotron_3_ultra_pretrain_256gpu_gb200_fp8mx_config":
            expert_model_parallel_size = 64
            global_batch_size = 256
            cfg.mixed_precision = _perf_precision("fp8_mx")
            _apply_nemotron_3_ultra_perf_defaults(cfg)
            _apply_nemotron_3_ultra_gtp(cfg)
            cfg.model.gtp_remat_opt_in_modules = ["moe_latent_proj"]
            cfg.model.tensor_model_parallel_size = 2
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.context_parallel_size = 1
            cfg.model.sequence_parallel = True
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.pipeline_model_parallel_layout = None
            cfg.model.seq_length = 8192
            cfg.model.min_offloaded_tensor_size = 350000000
            cfg.model.moe_router_padding_for_quantization = True
            cfg.model.expert_model_parallel_size = expert_model_parallel_size
            cfg.train.global_batch_size = global_batch_size
            cfg.model.fine_grained_activation_offloading = True
            cfg.model.offload_modules = ["fused_group_mlp"]
            cfg.model.fine_grained_offloading_max_inflight_offloads = 1
            cfg.model.recompute_granularity = "selective"
            cfg.model.recompute_modules = ["moe_act"]
            cfg.model.gtp_remat_opt_in_modules = ["moe_latent_proj"]
            cfg.model.cuda_graph_impl = "local"
            set_cuda_graph_modules(cfg.model, ["attn", "mamba", "moe_preprocess", "moe_router"])
            cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True
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
                "NVTE_CPU_OFFLOAD_V1": 1,
                "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
            }
            return cfg
        case "nemotron_3_ultra_pretrain_256gpu_gb200_fp8mx_deterministic_config":
            apply_overrides(cfg, "nemotron_3_ultra_pretrain_256gpu_gb200_fp8mx_config", num_gpus=num_gpus)
            apply_determinism_overrides(cfg)
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                **DETERMINISM_ENV_VARS,
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
                "NVTE_CPU_OFFLOAD_V1": 1,
                "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
            }
            return cfg
        case "nemotron_3_ultra_pretrain_256gpu_gb200_nvfp4_config":
            expert_model_parallel_size = 64
            global_batch_size = 256
            cfg.mixed_precision = _nemotron_3_ultra_nvfp4_precision()
            _apply_nemotron_3_ultra_perf_defaults(cfg)
            cfg.model.tensor_model_parallel_size = 2
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.context_parallel_size = 1
            cfg.model.sequence_parallel = True
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.pipeline_model_parallel_layout = None
            cfg.model.seq_length = 8192
            cfg.model.min_offloaded_tensor_size = 350000000
            cfg.model.moe_router_padding_for_quantization = True
            cfg.model.expert_model_parallel_size = expert_model_parallel_size
            cfg.train.global_batch_size = global_batch_size
            cfg.model.fine_grained_activation_offloading = True
            cfg.model.offload_modules = ["fused_group_mlp"]
            cfg.model.fine_grained_offloading_max_inflight_offloads = 1
            cfg.model.recompute_granularity = "selective"
            cfg.model.recompute_modules = ["moe_act"]
            _apply_nemotron_3_ultra_gtp(cfg)
            cfg.model.cuda_graph_impl = "local"
            set_cuda_graph_modules(cfg.model, ["attn", "mamba", "moe_preprocess", "moe_router"])
            cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True
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
                "NVTE_CPU_OFFLOAD_V1": 1,
                "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
                "NVTE_USE_FAST_MATH": 1,
            }
            return cfg
        case "nemotron_3_5_lightning_pretrain_8gpu_gb300_bf16_config":
            _apply_nemotron_3_5_lightning_gb300_bf16(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_5_lightning_pretrain_8gpu_b200_bf16"].copy()
            return cfg
        case "nemotron_3_5_lightning_pretrain_8gpu_gb300_fp8mx_config":
            _apply_nemotron_3_5_lightning_gb300_mxfp8(cfg)
            cfg.model.recompute_modules = []
            cfg.model.use_transformer_engine_op_fuser = True
            cfg.mixed_precision.fp8_dot_product_attention = True
            _enable_ncclep(cfg)
            _enable_nemotron_3_5_lightning_full_iteration(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_5_lightning_pretrain_8gpu_gb300_fp8mx"].copy()
            return cfg
        case "nemotron_3_5_lightning_pretrain_8gpu_gb300_fp8mx_fsdp_config":
            _apply_nemotron_3_5_lightning_gb300_mxfp8(cfg)
            cfg.train.global_batch_size = 384
            cfg.train.micro_batch_size = 3
            cfg.model.cuda_graph_impl = "none"
            set_cuda_graph_modules(cfg.model, [])
            cfg.model.init_model_with_meta_device = True
            cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
            cfg.ddp.use_megatron_fsdp = True
            cfg.ddp.num_distributed_optimizer_instances = 1
            cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
            cfg.ddp.outer_dp_sharding_strategy = "no_shard"
            cfg.ddp.average_in_collective = False
            cfg.ddp.keep_fp8_transpose_cache = False
            cfg.ddp.reuse_grad_buf_for_mxfp8_param_ag = False
            cfg.optimizer.reuse_grad_buf_for_mxfp8_param_ag = False
            cfg.checkpoint.load = None
            cfg.checkpoint.ckpt_format = "fsdp_dtensor"
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_5_lightning_pretrain_8gpu_gb200_fp8mx_fsdp"].copy()
            return cfg
        case "nemotron_3_5_lightning_pretrain_8gpu_gb300_nvfp4_config":
            apply_overrides(cfg, "nemotron_3_nano_pretrain_8gpu_gb300_nvfp4_config")
            cfg.model.mtp_num_layers = 2
            cfg.model.mtp_hybrid_override_pattern = "*E"
            cfg.model.mtp_use_repeated_layer = True
            cfg.model.mtp_loss_scaling_factor = 0.3
            cfg.model.extra_checkpoint_metadata.update(
                hf_model_id=_gb300_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_ID,
                hf_model_revision=_gb300_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_REVISION,
            )
            cfg.tokenizer.tokenizer_model = _gb300_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_ID
            cfg.tokenizer.hf_tokenizer_kwargs = {"revision": _gb300_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_REVISION}
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_5_lightning_pretrain_8gpu_gb200_nvfp4"].copy()
            return cfg
        case "nemotron_3_nano_pretrain_8gpu_gb300_bf16_config":
            _apply_nemotron_3_nano_gb300_bf16(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_5_lightning_pretrain_8gpu_b200_bf16"].copy()
            return cfg
        case "nemotron_3_nano_pretrain_8gpu_gb300_fp8mx_config":
            _apply_nemotron_3_nano_gb300_mxfp8(cfg)
            cfg.model.use_transformer_engine_op_fuser = True
            cfg.model.moe_mlp_glu_interleave_size = 32
            cfg.mixed_precision.fp8_dot_product_attention = True
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_5_lightning_pretrain_8gpu_gb200_fp8mx"].copy()
            return cfg
        case "nemotron_3_nano_pretrain_8gpu_gb300_nvfp4_config":
            _apply_nemotron_3_nano_perf_defaults(cfg)
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.model.expert_model_parallel_size = 8
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 4
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.cuda_graph_impl = "transformer_engine"
            set_cuda_graph_modules(cfg.model, ["attn", "mamba", "moe_router", "moe_preprocess"])
            cfg.comm_overlap.tp_comm_overlap = True
            _benchmark_common(cfg)
            cfg.model.moe_hybridep_num_sms = 16
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_5_lightning_pretrain_8gpu_gb200_nvfp4"].copy()
            return cfg
        case (
            "nemotron_3_super_pretrain_64gpu_gb300_bf16_config" | "nemotron_3_super_pretrain_64gpu_vrnvl72_bf16_config"
        ):
            _apply_nemotron_3_super_gb300_bf16(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_super_pretrain_64gpu_b200_bf16"].copy()
            return cfg
        case "nemotron_3_super_pretrain_64gpu_gb300_fp8mx_config":
            _apply_nemotron_3_super_gb300_mxfp8(cfg)
            cfg.mixed_precision.fp8_dot_product_attention = True
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_super_pretrain_64gpu_gb200_fp8mx"].copy()
            return cfg
        case "nemotron_3_super_pretrain_64gpu_gb300_nvfp4_config":
            _apply_nemotron_3_super_gb300_nvfp4(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_super_pretrain_64gpu_gb200_nvfp4"].copy()
            return cfg
        case (
            "nemotron_3_ultra_pretrain_256gpu_gb300_fp8mx_config"
            | "nemotron_3_ultra_pretrain_256gpu_vrnvl72_fp8mx_config"
        ):
            _apply_nemotron_3_ultra_gb300_fp8mx_config(
                cfg, num_gpus=num_gpus, expert_model_parallel_size=64, global_batch_size=256
            )
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
                "NVTE_CPU_OFFLOAD_V1": 1,
                "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
            }
            return cfg
        case "nemotron_3_ultra_pretrain_256gpu_gb300_nvfp4_config":
            expert_model_parallel_size = 64
            global_batch_size = 256
            cfg.mixed_precision = _nemotron_3_ultra_nvfp4_precision()
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.context_parallel_size = 1
            cfg.model.sequence_parallel = False
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.pipeline_model_parallel_layout = None
            cfg.model.seq_length = 8192
            cfg.model.min_offloaded_tensor_size = 500000000
            cfg.model.moe_router_padding_for_quantization = True
            cfg.model.expert_model_parallel_size = expert_model_parallel_size
            cfg.train.global_batch_size = global_batch_size
            cfg.model.fine_grained_activation_offloading = True
            cfg.model.offload_modules = ["fused_group_mlp"]
            cfg.model.recompute_granularity = "selective"
            cfg.model.recompute_modules = ["moe_act"]
            _apply_nemotron_3_ultra_perf_defaults(cfg)
            _apply_nemotron_3_ultra_gtp(cfg)
            cfg.model.cuda_graph_impl = "local"
            set_cuda_graph_modules(cfg.model, ["attn", "mamba", "moe_preprocess", "moe_router"])
            cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True
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
                "NVTE_CPU_OFFLOAD_V1": 1,
                "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
                "NVTE_USE_FAST_MATH": 1,
            }
            return cfg
        case "nemotron_3_5_lightning_pretrain_16gpu_h100_bf16_config":
            apply_overrides(cfg, "nemotron_3_nano_pretrain_16gpu_h100_bf16_config")
            cfg.train.global_batch_size = 512
            cfg.model.mtp_num_layers = 2
            cfg.model.mtp_hybrid_override_pattern = "*E"
            cfg.model.mtp_use_repeated_layer = True
            cfg.model.mtp_loss_scaling_factor = 0.3
            cfg.model.extra_checkpoint_metadata.update(
                hf_model_id=_h100_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_ID,
                hf_model_revision=_h100_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_REVISION,
            )
            cfg.tokenizer.tokenizer_model = _h100_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_ID
            cfg.tokenizer.hf_tokenizer_kwargs = {"revision": _h100_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_REVISION}
            cfg.optimizer.use_precision_aware_optimizer = True
            cfg.optimizer.exp_avg_dtype = torch.bfloat16
            cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16
            cfg.model.recompute_modules = ["moe_act", "layernorm"]
            cfg.model.fine_grained_activation_offloading = True
            cfg.model.offload_modules = ["expert_fc1"]
            cfg.model.activation_offload_fraction = 1.0
            cfg.model.delay_offload_until_cuda_graph = True
            cfg.model.moe_router_fusion = True
            cfg.model.moe_permute_fusion_into_hybridep = True
            cfg.model.moe_hybridep_num_sms = None
            cfg.model.moe_flex_dispatcher_num_sms = 32
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
                "NCCL_NVLS_ENABLE": 0,
                "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
                "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 64,
                "NVLINK_DOMAIN_SIZE": 8,
                "USE_MNNVL": 0,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 10,
                "NVTE_CPU_OFFLOAD_V1": 1,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 10,
                "NVTE_NORM_BWD_USE_CUDNN": 1,
                "NVTE_NORM_FWD_USE_CUDNN": 1,
            }
            return cfg
        case "nemotron_3_5_lightning_pretrain_16gpu_h100_fp8cs_config":
            apply_overrides(cfg, "nemotron_3_nano_pretrain_16gpu_h100_fp8cs_config")
            cfg.model.expert_model_parallel_size = 8
            cfg.model.mtp_num_layers = 2
            cfg.model.mtp_hybrid_override_pattern = "*E"
            cfg.model.mtp_use_repeated_layer = True
            cfg.model.mtp_loss_scaling_factor = 0.3
            cfg.model.extra_checkpoint_metadata.update(
                hf_model_id=_h100_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_ID,
                hf_model_revision=_h100_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_REVISION,
            )
            cfg.tokenizer.tokenizer_model = _h100_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_ID
            cfg.tokenizer.hf_tokenizer_kwargs = {"revision": _h100_nemotronh__NEMOTRON_3_5_LIGHTNING_MODEL_REVISION}
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_nano_pretrain_8gpu_b200_bf16"].copy()
            return cfg
        case "nemotron_3_nano_pretrain_16gpu_h100_bf16_config":
            _apply_nemotron_3_nano_perf_defaults(cfg)
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.recompute_granularity = "selective"
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.model.expert_model_parallel_size = 8
            cfg.train.global_batch_size = 1024
            cfg.train.micro_batch_size = 1
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.cuda_graph_impl = "transformer_engine"
            set_cuda_graph_modules(cfg.model, ["attn", "mamba"])
            cfg.model.recompute_modules = ["moe", "layernorm"]
            cfg.comm_overlap.tp_comm_overlap = True
            _benchmark_common(cfg)
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_hybridep_num_sms = 16
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_nano_pretrain_8gpu_b200_bf16"].copy()
            return cfg
        case "nemotron_3_nano_pretrain_16gpu_h100_fp8cs_config":
            _apply_nemotron_3_nano_perf_defaults(cfg)
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.model.recompute_granularity = "selective"
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.model.expert_model_parallel_size = 16
            cfg.train.global_batch_size = 1024
            cfg.train.micro_batch_size = 1
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.cuda_graph_impl = "transformer_engine"
            set_cuda_graph_modules(cfg.model, ["attn", "mamba"])
            cfg.model.recompute_modules = ["layernorm"]
            cfg.comm_overlap.tp_comm_overlap = True
            _benchmark_common(cfg)
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_hybridep_num_sms = 16
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_nano_pretrain_8gpu_b200_bf16"].copy()
            return cfg
        case "nemotron_3_super_pretrain_64gpu_h100_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.moe_router_force_load_balancing = True
            _benchmark_common(cfg)
            cfg.ddp.overlap_param_gather = False
            cfg.model.moe_hybridep_num_sms = None
            cfg.checkpoint.async_save = False
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
                "NCCL_NVLS_ENABLE": 0,
                "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
                "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 64,
                "NVLINK_DOMAIN_SIZE": 8,
                "USE_MNNVL": 0,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
            }
            return cfg
        case "nemotron_3_5_lightning_pretrain_8gpu_vrnvl72_bf16_config":
            _apply_nemotron_3_5_lightning_gb300_bf16(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.model.moe_hybridep_num_sms = 32
            cfg.model.moe_router_fusion = True
            cfg.model.fine_grained_activation_offloading = True
            cfg.model.offload_modules = ["expert_fc1", "moe_act"]
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
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
        case "nemotron_3_5_lightning_pretrain_8gpu_vrnvl72_fp8mx_config":
            _apply_nemotron_3_5_lightning_gb300_mxfp8(cfg)
            cfg.model.recompute_modules = []
            cfg.model.use_transformer_engine_op_fuser = True
            cfg.mixed_precision.fp8_dot_product_attention = True
            _enable_ncclep(cfg)
            _enable_nemotron_3_5_lightning_full_iteration(cfg)
            cfg.model.moe_hybridep_num_sms = 32
            cfg.model.moe_router_fusion = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_5_lightning_pretrain_8gpu_gb300_fp8mx"].copy()
            return cfg
        case "nemotron_3_nano_pretrain_8gpu_vrnvl72_bf16_config":
            _apply_nemotron_3_nano_gb300_bf16(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_5_lightning_pretrain_8gpu_gb200_fp8mx_fsdp"].copy()
            return cfg
        case "nemotron_3_nano_pretrain_8gpu_vrnvl72_fp8mx_config":
            _apply_nemotron_3_nano_gb300_mxfp8(cfg)
            cfg.model.use_transformer_engine_op_fuser = True
            cfg.model.moe_mlp_glu_interleave_size = 32
            cfg.mixed_precision.fp8_dot_product_attention = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_nano_pretrain_8gpu_gb200_fp8mx"].copy()
            return cfg
        case "nemotron_3_nano_pretrain_8gpu_vrnvl72_nvfp4_config":
            apply_overrides(cfg, "nemotron_3_nano_pretrain_8gpu_gb300_nvfp4_config")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_5_lightning_pretrain_8gpu_gb200_nvfp4"].copy()
            return cfg
        case "nemotron_3_super_pretrain_64gpu_vrnvl72_fp8mx_config":
            _apply_nemotron_3_super_gb300_mxfp8(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.mixed_precision.fp8_dot_product_attention = True
            cfg.model.expert_model_parallel_size = 8
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["nemotron_3_super_pretrain_64gpu_gb200_fp8mx"].copy()
            return cfg
        case "nemotron_3_super_pretrain_64gpu_vrnvl72_nvfp4_config":
            _apply_nemotron_3_super_gb300_nvfp4(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.model.expert_model_parallel_size = 16
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
                "NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM": 1,
                "NVTE_USE_FAST_MATH": 1,
            }
            return cfg
        case "nemotron_3_ultra_pretrain_256gpu_vrnvl72_nvfp4_config":
            apply_overrides(cfg, "nemotron_3_ultra_pretrain_256gpu_gb300_nvfp4_config", num_gpus=num_gpus)
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
                "NVTE_CPU_OFFLOAD_V1": 1,
                "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
                "NVTE_USE_FAST_MATH": 1,
            }
            return cfg
        case _:
            raise ValueError(f"Unknown benchmark overrides: {name}")
