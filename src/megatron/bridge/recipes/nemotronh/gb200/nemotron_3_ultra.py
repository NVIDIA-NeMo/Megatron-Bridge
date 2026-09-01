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

"""GB200 functional pretraining recipe for Nemotron 3 Ultra."""

import torch

from megatron.bridge.recipes.nemotronh.h100.nemotron_3_ultra import (
    NEMOTRON_3_ULTRA_PRETRAIN_SEQ_LENGTH,
    _nemotron_3_ultra_large_scale_bf16_config,
)
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_with_mxfp8_mixed


def _nemotron_3_ultra_pretrain_256gpu_gb200_bf16_config(
    *,
    expert_model_parallel_size: int = 64,
    hybrid_ep_num_sms: int = 32,
) -> ConfigContainer:
    """Build a Nemotron 3 Ultra BF16 tuning config for 256 GB200 GPUs."""
    cfg = _nemotron_3_ultra_large_scale_bf16_config()

    # TP4 retains enough activation headroom for naturally routed real-data
    # batches after Bridge enables safe uneven-input padding for eager HybridEP.
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = expert_model_parallel_size
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.seq_length = NEMOTRON_3_ULTRA_PRETRAIN_SEQ_LENGTH
    cfg.dataset.seq_length = NEMOTRON_3_ULTRA_PRETRAIN_SEQ_LENGTH
    cfg.train.global_batch_size = 256

    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    # Bridge finalization enables this safety path for eager HybridEP so
    # different local token counts cannot produce mismatched collectives.
    cfg.model.moe_hybridep_pad_uneven_dispatch_inputs = True
    cfg.model.moe_hybridep_num_sms = None
    cfg.model.moe_flex_dispatcher_num_sms = hybrid_ep_num_sms
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.model.fine_grained_activation_offloading = True
    cfg.model.min_offloaded_tensor_size = 350_000_000
    cfg.model.offload_modules = ["fused_group_mlp"]
    cfg.model.fine_grained_offloading_max_inflight_offloads = 1
    # Match the proven natural-routing BF16 baseline: Flash Attention already
    # recomputes its largest intermediates, while expert activation offload
    # carries the grouped-MLP memory policy. Transformer Engine's BF16
    # weighted-SReLU fallback cannot honor moe_act activation recompute.
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = ["core_attn"]

    cfg.dist.use_megatron_fsdp = False
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.num_distributed_optimizer_instances = 1
    cfg.ddp.num_buckets = 48
    cfg.ddp.average_in_collective = False
    # Keep the distributed optimizer's memory-efficient parameter handling;
    # optimizer/scheduler values continue to come from the library recipe.
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.overlap_param_gather = True
    cfg.checkpoint.ckpt_format = "torch_dist"

    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": expert_model_parallel_size,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_CPU_OFFLOAD_V1": 1,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


def nemotron_3_ultra_pretrain_256gpu_gb200_bf16_config() -> ConfigContainer:
    """Return a convergence-safe Nemotron 3 Ultra pretrain config for 256 GB200 GPUs.

    The recipe adopts the performance configuration's PP4/EP64 HybridEP
    execution and targeted expert-activation offload. It keeps BF16 compute,
    natural expert routing, numerical checks, and the library recipe's training
    objective instead of benchmark-only policy.

    Returns:
        GB200 BF16 distributed-optimizer pretraining configuration.
    """
    return _nemotron_3_ultra_pretrain_256gpu_gb200_bf16_config()


def nemotron_3_ultra_pretrain_256gpu_gb200_bf16_hybridep_sm16_config() -> ConfigContainer:
    """Return the GB200 BF16 recipe with 16 SMs reserved for HybridEP."""
    return _nemotron_3_ultra_pretrain_256gpu_gb200_bf16_config(hybrid_ep_num_sms=16)


def nemotron_3_ultra_pretrain_256gpu_gb200_bf16_ep16_config() -> ConfigContainer:
    """Return the GB200 BF16 recipe with four EP16 replicas per pipeline stage."""
    cfg = _nemotron_3_ultra_pretrain_256gpu_gb200_bf16_config(expert_model_parallel_size=16)
    # Natural-routing hotspots can leave less than one allocator page of HBM
    # headroom on expert ranks. Layernorm recompute supplies that margin with
    # negligible compute overhead while preserving the proven EP16 execution.
    cfg.model.recompute_modules = ["core_attn", "layernorm"]
    return cfg


def nemotron_3_ultra_pretrain_256gpu_gb200_bf16_ep32_config() -> ConfigContainer:
    """Return the GB200 BF16 recipe with two EP32 replicas per pipeline stage."""
    return _nemotron_3_ultra_pretrain_256gpu_gb200_bf16_config(expert_model_parallel_size=32)


def nemotron_3_ultra_pretrain_256gpu_gb200_fp8mx_fsdp_config() -> ConfigContainer:
    """Return an MXFP8 Nemotron 3 Ultra pretrain config for 256 GB200 GPUs.

    The execution policy adopts the canonical TP2/PP1/EP64 Megatron-FSDP
    configuration with HybridEP, targeted expert-activation offload, and
    selective routed-expert, shared-expert, and layernorm recompute. CP2 keeps
    naturally routed real-data batches within GB200 HBM. It retains natural
    routing, numerical checks, the library optimizer and scheduler, and the
    training objective instead of benchmark-only policy.

    Returns:
        GB200 MXFP8 Megatron-FSDP pretraining configuration.
    """
    cfg = _nemotron_3_ultra_large_scale_bf16_config()

    cfg.mixed_precision = bf16_with_mxfp8_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    # Unlike the force-balanced benchmark workload, natural routing can create
    # transient expert-token hotspots. CP2 reduces the per-rank 8K activation
    # footprint so those batches retain enough grouped-MLP backward workspace.
    cfg.model.context_parallel_size = 2
    cfg.model.sequence_parallel = True
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.seq_length = NEMOTRON_3_ULTRA_PRETRAIN_SEQ_LENGTH
    cfg.dataset.seq_length = NEMOTRON_3_ULTRA_PRETRAIN_SEQ_LENGTH
    cfg.train.global_batch_size = 256

    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_hybridep_pad_uneven_dispatch_inputs = True
    cfg.model.moe_router_padding_for_quantization = True
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.model.fine_grained_activation_offloading = True
    cfg.model.min_offloaded_tensor_size = 350_000_000
    cfg.model.offload_modules = ["fused_group_mlp"]
    cfg.model.fine_grained_offloading_max_inflight_offloads = 1
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    # The benchmark recipe force-balances expert routing. Natural routing can
    # produce hotter expert ranks, so recompute shared-expert and layernorm
    # activations as well to preserve HBM headroom without changing the routing
    # or training objective.
    cfg.model.recompute_modules = ["moe_act", "layernorm", "shared_experts"]

    cfg.dist.use_megatron_fsdp = True
    cfg.ddp.use_megatron_fsdp = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.keep_fp8_transpose_cache = False
    cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.ddp.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.optimizer.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.ddp.num_distributed_optimizer_instances = 4
    cfg.ddp.outer_dp_sharding_strategy = "optim"
    cfg.ddp.megatron_fsdp_grad_comm_dtype = torch.bfloat16
    cfg.ddp.megatron_fsdp_main_params_dtype = torch.float32
    cfg.ddp.megatron_fsdp_main_grads_dtype = torch.bfloat16
    cfg.ddp.average_in_collective = False
    cfg.ddp.num_buckets = 48
    cfg.model.init_model_with_meta_device = True
    cfg.model.gradient_accumulation_fusion = False
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.checkpoint.load = None
    cfg.checkpoint.ckpt_format = "fsdp_dtensor"

    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 64,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_CPU_OFFLOAD_V1": 1,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


__all__ = [
    "nemotron_3_ultra_pretrain_256gpu_gb200_bf16_config",
    "nemotron_3_ultra_pretrain_256gpu_gb200_bf16_ep16_config",
    "nemotron_3_ultra_pretrain_256gpu_gb200_bf16_ep32_config",
    "nemotron_3_ultra_pretrain_256gpu_gb200_bf16_hybridep_sm16_config",
    "nemotron_3_ultra_pretrain_256gpu_gb200_fp8mx_fsdp_config",
]
