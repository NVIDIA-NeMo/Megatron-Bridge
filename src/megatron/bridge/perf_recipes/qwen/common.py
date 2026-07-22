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
"""Common helpers for qwen performance recipes."""

import torch

from megatron.bridge.perf_recipes._common import (
    _benchmark_common,
    _enable_overlap_param_gather_with_optimizer_step,
    _perf_precision,
)
from megatron.bridge.recipes.qwen.h100.qwen3_moe import (
    qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config as qwen3_30b_a3b_pretrain_config,
)
from megatron.bridge.recipes.qwen.qwen3_moe import qwen3_235b_a22b_pretrain_config
from megatron.bridge.recipes.qwen.qwen3_next import qwen3_next_80b_a3b_pretrain_config
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer


def _with_global_batch_size(cfg: ConfigContainer, global_batch_size: int) -> ConfigContainer:
    cfg.train.global_batch_size = global_batch_size
    return cfg


def _apply_qwen3_moe_tuned_defaults(
    cfg: ConfigContainer,
    *,
    original_max_position_embeddings: int,
) -> None:
    """Apply defaults shared by the hardware-tuned Qwen3 MoE recipes."""
    _benchmark_common(cfg)

    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.yarn_original_max_position_embeddings = original_max_position_embeddings
    cfg.model.make_vocab_size_divisible_by = 1187
    cfg.model.moe_router_force_load_balancing = True
    cfg.model.moe_router_fusion = True
    cfg.model.moe_router_dtype = torch.float32
    cfg.model.moe_token_dispatcher_type = "flex"

    cfg.dataset.seq_length = cfg.model.seq_length
    cfg.train.manual_gc_interval = 5


def _enable_qwen_precision_aware_optimizer(cfg: ConfigContainer) -> None:
    """Enable the BF16 optimizer-state settings used by tuned Qwen3 MoE recipes."""
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16


def _enable_hybridep_full_iteration_mxfp8(cfg: ConfigContainer) -> None:
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
