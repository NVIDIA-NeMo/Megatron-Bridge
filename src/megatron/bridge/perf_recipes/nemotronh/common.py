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
# ruff: noqa: F401
"""Common helpers for nemotronh performance recipes."""

from pathlib import Path

import torch
from megatron.core.quantization.utils import load_quantization_recipe

from megatron.bridge.perf_recipes._common import _benchmark_common, _perf_precision
from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_pretrain_config
from megatron.bridge.recipes.nemotronh.nemotron_3_super import nemotron_3_super_pretrain_config
from megatron.bridge.recipes.nemotronh.nemotron_3_ultra import nemotron_3_ultra_pretrain_config
from megatron.bridge.recipes.nemotronh.nemotronh import nemotronh_56b_pretrain_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig, nemotron_3_super_bf16_with_nvfp4_mixed


_TE_QUANT_CFG_PATH = Path(__file__).with_name("te_quant.cfg")


def _with_global_batch_size(cfg: ConfigContainer, global_batch_size: int) -> ConfigContainer:
    cfg.train.global_batch_size = global_batch_size
    return cfg


def _nemotron_3_super_nvfp4_precision() -> MixedPrecisionConfig:
    """Return the NVFP4 precision config used by Nemotron 3 Super perf recipes."""
    cfg = nemotron_3_super_bf16_with_nvfp4_mixed()
    # Disabled until MCore PR 4358 lands.
    cfg.fp4_param_gather = False
    return cfg


def _apply_nemotron_3_super_perf_defaults(cfg: ConfigContainer) -> None:
    """Apply shared Nemotron 3 Super perf defaults after recipe-specific overrides."""
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.model.moe_router_force_load_balancing = True
    cfg.checkpoint.async_save = False

    _benchmark_common(cfg)


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


def _apply_nemotron_3_ultra_fsdp_hsdp(
    cfg: ConfigContainer,
    *,
    num_gpus: int,
    fsdp_shard_group_gpus: int,
) -> None:
    """Apply topology-aware Megatron-FSDP (HSDP) settings for Nemotron 3 Ultra.

    Shards params/grads/optimizer
    within each FSDP shard group and replicate (optimizer-sharded) across groups, with
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

    # HSDP: shard within a hardware-appropriate group and replicate
    # (optimizer-sharded) across groups.
    num_optim_instances = max(1, num_gpus // fsdp_shard_group_gpus)
    cfg.ddp.num_distributed_optimizer_instances = num_optim_instances

    # HSDP across shard groups. Megatron-FSDP
    # only enables HSDP when num_distributed_optimizer_instances > 1; with a single
    # shard group HSDP is off, so the outer strategy must be "no_shard" (otherwise
    # the first param all-gather hits a None HSDP helper buffer).
    cfg.ddp.outer_dp_sharding_strategy = "optim" if num_optim_instances > 1 else "no_shard"

    cfg.ddp.megatron_fsdp_grad_comm_dtype = torch.bfloat16
    cfg.ddp.megatron_fsdp_main_params_dtype = torch.float32
    cfg.ddp.megatron_fsdp_main_grads_dtype = torch.bfloat16

    # incompatible with BF16 FSDP main grads
    cfg.model.gradient_accumulation_fusion = False

    cfg.checkpoint.ckpt_format = "fsdp_dtensor"


def _enable_nemotron_3_ultra_full_iteration_cuda_graphs(cfg: ConfigContainer) -> None:
    """Enable the full-iteration CUDA-graph path used by Blackwell Ultra recipes."""
    cfg.model.cuda_graph_impl = "full_iteration"
    cfg.model.cuda_graph_scope = []
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.rng.te_rng_tracker = True
    cfg.model.use_te_rng_tracker = True

    # Full-iteration graphs require fixed MoE buffer sizes. Keep the rank capacity
    # bounded while retaining fused-grouped-MLP activation offload; MoE paged stash
    # cannot be combined with that offload module.
    cfg.model.moe_pad_experts_for_cuda_graph_inference = True
    cfg.model.moe_expert_rank_capacity_factor = 1.5
    cfg.model.fine_grained_offloading_max_inflight_offloads = 1

    # Megatron-FSDP needs its CUDA-graph-aware synchronization path.
    cfg.ddp.megatron_fsdp_cuda_graph_mode = True
    cfg.ddp.fsdp_all_gather_in_start_param_sync = False


def _nemotron_3_ultra_fp8mx_config(
    *,
    num_gpus: int,
    expert_model_parallel_size: int,
    global_batch_size: int,
    fsdp_shard_group_gpus: int,
) -> ConfigContainer:
    """Build a Nemotron 3 Ultra MXFP8 Megatron-FSDP performance recipe."""
    cfg = nemotron_3_ultra_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")

    # Parallelism
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.seq_length = 8192

    # Only tensors larger than 500MB are offloaded, which approximates
    # offloading the moe_act input for sequence length 8192 and MBS 1.
    cfg.model.min_offloaded_tensor_size = 500_000_000

    # MXFP8 requires router padding for quantization.
    cfg.model.moe_router_padding_for_quantization = True

    cfg.model.expert_model_parallel_size = expert_model_parallel_size
    cfg.train.global_batch_size = global_batch_size

    # Fine-grained activation offloading. NVTE_CPU_OFFLOAD_V1 must be enabled
    # by the hardware-specific recipe environment.
    cfg.model.fine_grained_activation_offloading = True
    cfg.model.offload_modules = ["fused_group_mlp"]

    # Recompute the MoE expert activation output while the FC1 output is
    # retained and offloaded to CPU.
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["moe_act"]

    _apply_nemotron_3_ultra_perf_defaults(cfg)

    # Apply HSDP / FSDP dtype overrides last so they win over generic defaults.
    _apply_nemotron_3_ultra_fsdp_hsdp(
        cfg,
        num_gpus=num_gpus,
        fsdp_shard_group_gpus=fsdp_shard_group_gpus,
    )
    _enable_nemotron_3_ultra_full_iteration_cuda_graphs(cfg)

    return cfg
