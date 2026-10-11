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
"""Common helpers for deepseek performance recipes."""

import torch

from megatron.bridge.perf_recipes._common import (
    _benchmark_common,
    _enable_cutedsl_fused_grouped_mlp,
    _enable_full_iteration_cuda_graph,
    _enable_moe_a2a_overlap,
    _perf_precision,
)
from megatron.bridge.recipes.deepseek.deepseek_v3 import (
    deepseek_v3_pretrain_config,
    set_deepseek_v3_pipeline_model_parallel_layout,
)
from megatron.bridge.training.config import ConfigContainer


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
    cfg.model.recompute_modules = ["layernorm", "mla_up_proj", "moe_act"]
    cfg.model.fine_grained_activation_offloading = True
    cfg.model.offload_modules = ["core_attn", "attn_proj"]
    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    cfg.comm_overlap.overlap_grad_reduce = True

    _benchmark_common(cfg)
