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

# Number of GPUs in a single GB300 multi-node NVLink (MNNVL) domain
# (16 nodes x 4 GPUs). Each Megatron-FSDP optimizer instance is sharded within one
# NVLink domain (HSDP), matching ``--num-distributed-optimizer-instances $((nodes/16))``
# in the reference Megatron-LM launch script.
_GB300_NVLINK_DOMAIN_GPUS = 64


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
