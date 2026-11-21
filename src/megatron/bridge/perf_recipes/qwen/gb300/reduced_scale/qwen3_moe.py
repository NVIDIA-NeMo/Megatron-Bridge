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
"""Reduced-scale GB300 performance recipes for Qwen3-235B-A22B."""

import torch

from megatron.bridge.perf_recipes.qwen.gb300.qwen3_moe import (
    qwen3_235b_a22b_pretrain_64gpu_gb300_bf16_config,
)
from megatron.bridge.training.config import ConfigContainer


_HYBRID_EP_DOMAIN_SIZE_ENV = "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"


def _set_hybridep_domain_size(cfg: ConfigContainer, domain_size: int) -> None:
    """Set the HybridEP rank count for the reduced allocation."""
    cfg.env_vars = {
        **cfg.env_vars,
        _HYBRID_EP_DOMAIN_SIZE_ENV: domain_size,
    }


def _enable_fsdp_full_iteration_graph(cfg: ConfigContainer) -> None:
    """Enable the static full-iteration graph path used by Megatron-FSDP."""
    cfg.model.cuda_graph_impl = "full_iteration"
    cfg.model.cuda_graph_scope = []
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True
    cfg.rerun_state_machine.check_for_nan_in_loss = False
    cfg.ddp.check_for_nan_in_grad = False
    cfg.ddp.megatron_fsdp_cuda_graph_mode = True
    cfg.ddp.fsdp_all_gather_in_start_param_sync = False
    cfg.model.gradient_accumulation_fusion = False
    cfg.model.moe_pad_experts_for_cuda_graph_inference = True
    cfg.model.moe_paged_stash = True
    cfg.model.moe_use_grouped_tensor = True
    cfg.model.moe_expert_rank_capacity_factor = 1.5
    cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.2
    cfg.model.moe_paged_stash_buffer_size_factor_cpu = 1.0
    cfg.model.offload_modules = []


def qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_config() -> ConfigContainer:
    """Qwen3 235B-A22B pretrain: 32× GB300, BF16, PP=2, EP=16.

    This is a reduced-scale derivative of the canonical 64-GPU recipe. It
    retains the sequence length, global batch size, precision, kernels, and
    benchmark schedule. MBS=1 is required because MBS=2 was observed to run
    out of memory on this topology.
    """
    cfg = qwen3_235b_a22b_pretrain_64gpu_gb300_bf16_config()

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 16
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.train.micro_batch_size = 1

    _set_hybridep_domain_size(cfg, 16)
    return cfg


def qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_config() -> ConfigContainer:
    """Qwen3 235B-A22B pretrain: 16× GB300, BF16, EP=16, ZeRO-3.

    Megatron-FSDP shards optimizer state, gradients, and parameters. Full
    activation recompute and eager execution limit non-model-state memory.
    The recipe does not enable CPU offload, FP8, or reduced-precision
    optimizer states.
    """
    cfg = qwen3_235b_a22b_pretrain_64gpu_gb300_bf16_config()

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 16
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.train.micro_batch_size = 1

    cfg.dist.use_megatron_fsdp = True
    cfg.ddp.use_megatron_fsdp = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.expert_data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.keep_fp8_transpose_cache = False
    cfg.ddp.average_in_collective = False
    cfg.checkpoint.ckpt_format = "fsdp_dtensor"
    cfg.model.init_model_with_meta_device = True

    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1
    cfg.model.recompute_modules = None
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = []

    _set_hybridep_domain_size(cfg, 16)
    return cfg


def qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_config() -> ConfigContainer:
    """Qwen3 235B-A22B pretrain: 16× GB300, BF16, EP=16, ZeRO-2.

    This performance candidate retains the validated 16-GPU ZeRO-3 recipe's
    topology, recompute policy, and eager execution while keeping parameters
    replicated. Gradients and optimizer states remain sharded.
    """
    cfg = qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_config()
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads"
    cfg.ddp.expert_data_parallel_sharding_strategy = "optim_grads"
    return cfg


def qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_common_contract_config() -> ConfigContainer:
    """16-GPU GB/VR comparison contract with BF16 moments and ZeRO-1 semantics."""
    cfg = qwen3_235b_a22b_pretrain_64gpu_gb300_bf16_config()

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 16
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = []

    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16
    cfg.optimizer.use_distributed_optimizer = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.dist.use_megatron_fsdp = False
    cfg.ddp.use_megatron_fsdp = False
    cfg.model.init_model_with_meta_device = False

    _set_hybridep_domain_size(cfg, 16)
    return cfg


def qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_common_contract_tp2_config() -> ConfigContainer:
    """16-GPU common contract with TP=2 and sequence parallelism."""
    cfg = qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_common_contract_config()
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.sequence_parallel = True
    return cfg


def qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_selective_config() -> ConfigContainer:
    """16-GPU ZeRO-2 candidate with selective expert-activation recompute."""
    cfg = qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_config()
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = ["moe_act"]
    return cfg


def qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_no_recompute_config() -> ConfigContainer:
    """16-GPU ZeRO-2 eager control without activation recompute."""
    cfg = qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_config()
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = []
    return cfg


def qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_selective_graph_config() -> ConfigContainer:
    """16-GPU ZeRO-2 selective-recompute candidate with TE scoped graphs."""
    cfg = qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_selective_config()
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True
    return cfg


def qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_tp2_config() -> ConfigContainer:
    """16-GPU ZeRO-2 full-recompute control with TP=2 and sequence parallelism."""
    cfg = qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_config()
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.sequence_parallel = True
    return cfg


def qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_tp2_selective_config() -> ConfigContainer:
    """16-GPU TP=2 candidate with selective expert-activation recompute."""
    cfg = qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_tp2_config()
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = ["moe_act"]
    return cfg


def qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_tp2_selective_graph_config() -> ConfigContainer:
    """16-GPU TP=2 selective-recompute candidate with TE scoped graphs."""
    cfg = qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_tp2_selective_config()
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True
    return cfg


def qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_tp2_full_graph_config() -> ConfigContainer:
    """16-GPU TP=2 ZeRO-2 full-recompute candidate with full-iteration graphs."""
    cfg = qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_tp2_config()
    _enable_fsdp_full_iteration_graph(cfg)
    return cfg


def qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero3_full_graph_config() -> ConfigContainer:
    """16-GPU TP=1 ZeRO-3 full-recompute candidate with full-iteration graphs."""
    cfg = qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_config()
    _enable_fsdp_full_iteration_graph(cfg)
    return cfg


def qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_config() -> ConfigContainer:
    """32-GPU PP=1/EP=32 candidate at TP=1 and MBS=1."""
    cfg = qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_config()
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.expert_model_parallel_size = 32
    _set_hybridep_domain_size(cfg, 32)
    return cfg


def qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_mbs2_config() -> ConfigContainer:
    """32-GPU TP=1/PP=1/EP=32 winner candidate with MBS=2."""
    cfg = qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_config()
    cfg.train.micro_batch_size = 2
    return cfg


def qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_tp2_pp1_config() -> ConfigContainer:
    """32-GPU PP=1/EP=32 candidate with TP=2, SP, and MBS=1."""
    cfg = qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_config()
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.sequence_parallel = True
    return cfg


def qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_tp2_pp1_mbs2_config() -> ConfigContainer:
    """32-GPU TP=2/PP=1/EP=32 candidate with MBS=2."""
    cfg = qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_tp2_pp1_config()
    cfg.train.micro_batch_size = 2
    return cfg


def qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_sms16_config() -> ConfigContainer:
    """32-GPU TP=1/PP=1 winner candidate with 16 HybridEP communication SMs."""
    cfg = qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_config()
    cfg.model.moe_hybridep_num_sms = 16
    return cfg


def qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_sms24_config() -> ConfigContainer:
    """32-GPU TP=1/PP=1 winner candidate with 24 HybridEP communication SMs."""
    cfg = qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_config()
    cfg.model.moe_hybridep_num_sms = 24
    return cfg


def qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_sms40_config() -> ConfigContainer:
    """32-GPU TP=1/PP=1 winner candidate with 40 HybridEP communication SMs."""
    cfg = qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_config()
    cfg.model.moe_hybridep_num_sms = 40
    return cfg


def qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_sms48_config() -> ConfigContainer:
    """32-GPU TP=1/PP=1 winner candidate with 48 HybridEP communication SMs."""
    cfg = qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_config()
    cfg.model.moe_hybridep_num_sms = 48
    return cfg


def qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_chunk64_config() -> ConfigContainer:
    """32-GPU TP=1/PP=1 winner candidate with 64-token combine chunks."""
    cfg = qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_config()
    cfg.env_vars = {
        **cfg.env_vars,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 64,
    }
    return cfg


def qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_chunk256_config() -> ConfigContainer:
    """32-GPU TP=1/PP=1 winner candidate with 256-token combine chunks."""
    cfg = qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_config()
    cfg.env_vars = {
        **cfg.env_vars,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 256,
    }
    return cfg


__all__ = [
    "qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_config",
    "qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_common_contract_config",
    "qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_common_contract_tp2_config",
    "qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_no_recompute_config",
    "qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_config",
    "qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_selective_config",
    "qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_selective_graph_config",
    "qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_tp2_config",
    "qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_tp2_selective_config",
    "qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_tp2_selective_graph_config",
    "qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_tp2_full_graph_config",
    "qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero3_full_graph_config",
    "qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_config",
    "qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_config",
    "qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_mbs2_config",
    "qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_sms16_config",
    "qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_sms24_config",
    "qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_sms40_config",
    "qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_sms48_config",
    "qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_chunk64_config",
    "qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_chunk256_config",
    "qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_tp2_pp1_config",
    "qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_tp2_pp1_mbs2_config",
]
