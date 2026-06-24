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
"""GB200 performance recipes for Kimi K2."""

from megatron.bridge.perf_recipes.kimi.common import (
    ConfigContainer,
    _benchmark_common,
    _get_kimi_k2_pipeline_layout,
    _perf_precision,
    kimi_k2_pretrain_config,
)


def kimi_k2_pretrain_256gpu_gb200_bf16_config() -> ConfigContainer:
    """Kimi K2 pretrain: 256× GB200, BF16."""
    cfg = kimi_k2_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096
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
    return cfg


def kimi_k2_pretrain_256gpu_gb200_fp8cs_config() -> ConfigContainer:
    """Kimi K2 pretrain: 256× GB200, FP8 current-scaling."""
    cfg = kimi_k2_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096
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
    return cfg


def kimi_k2_pretrain_256gpu_gb200_fp8mx_config() -> ConfigContainer:
    """Kimi K2 pretrain: 256× GB200, MXFP8."""
    cfg = kimi_k2_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.mixed_precision.fp8_param_gather = False
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096
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
    return cfg
