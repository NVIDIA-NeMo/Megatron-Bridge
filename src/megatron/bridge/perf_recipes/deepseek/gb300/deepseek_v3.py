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
"""GB300 performance recipes for DeepSeek V3."""

from megatron.bridge.perf_recipes.deepseek.common import (
    ConfigContainer,
    _apply_deepseek_v3_64gpu_gb300_fsdp_configs,
    _benchmark_common,
    _deepseek_v3_common,
    _enable_deepseek_full_iteration_mxfp8,
    _enable_overlap_param_gather_with_optimizer_step,
    _perf_precision,
    deepseek_v3_pretrain_config,
    set_deepseek_v3_pipeline_model_parallel_layout,
)


def deepseek_v3_pretrain_256gpu_gb300_bf16_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× GB300, BF16."""
    cfg = deepseek_v3_pretrain_config()
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
    _enable_overlap_param_gather_with_optimizer_step(cfg)
    return cfg


def deepseek_v3_pretrain_256gpu_gb300_fp8cs_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× GB300, FP8 current-scaling."""
    cfg = deepseek_v3_pretrain_config()
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
    _enable_overlap_param_gather_with_optimizer_step(cfg)
    return cfg


def deepseek_v3_pretrain_256gpu_gb300_fp8mx_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× GB300, MXFP8."""
    cfg = deepseek_v3_pretrain_config()
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
    _enable_deepseek_full_iteration_mxfp8(cfg, fp8_dot_product_attention=True, fp8_output_proj=True)
    return cfg


def deepseek_v3_pretrain_256gpu_gb300_nvfp4_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× GB300, NVFP4."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("nvfp4")
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
    return cfg


def deepseek_v3_pretrain_64gpu_gb300_bf16_fsdp_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 64× GB300, BF16, Megatron FSDP."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    _apply_deepseek_v3_64gpu_gb300_fsdp_configs(cfg)
    return cfg


def deepseek_v3_pretrain_64gpu_gb300_fp8mx_fsdp_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 64× GB300, MXFP8, Megatron FSDP."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    cfg.model.fp8_output_proj = True
    _apply_deepseek_v3_64gpu_gb300_fsdp_configs(cfg)
    cfg.ddp.outer_dp_sharding_strategy = "no_shard"
    cfg.ddp.num_distributed_optimizer_instances = 1
    cfg.model.fp8_param_gather = True
    cfg.model.fp8_param = True
    cfg.model.moe_router_dtype = "bf16"
    return cfg


def deepseek_v3_pretrain_256gpu_gb300_fp8mx_large_scale_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× GB300, MXFP8, large-scale proxy (BF16_V1 layout, GBS=256)."""
    cfg = deepseek_v3_pretrain_256gpu_gb300_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    cfg.train.global_batch_size = 256
    cfg.model.fp8_output_proj = True
    cfg.comm_overlap.overlap_param_gather_with_optimizer_step = None
    cfg.optimizer.overlap_param_gather_with_optimizer_step = False
    return cfg
