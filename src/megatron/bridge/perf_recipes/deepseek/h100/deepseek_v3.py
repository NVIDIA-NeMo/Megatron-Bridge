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
"""H100 performance recipes for DeepSeek V3."""

from megatron.bridge.perf_recipes.deepseek.common import (
    ConfigContainer,
    _benchmark_common,
    _deepseek_v3_common,
    _enable_overlap_param_gather_with_optimizer_step,
    _perf_precision,
    deepseek_v3_pretrain_config,
    set_deepseek_v3_pipeline_model_parallel_layout,
)


def deepseek_v3_pretrain_1024gpu_h100_bf16_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 1024× H100, BF16."""
    cfg = deepseek_v3_pretrain_config()
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
    _enable_overlap_param_gather_with_optimizer_step(cfg)
    return cfg


def deepseek_v3_pretrain_1024gpu_h100_fp8cs_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 1024× H100, FP8 current-scaling."""
    cfg = deepseek_v3_pretrain_config()
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
    _enable_overlap_param_gather_with_optimizer_step(cfg)
    return cfg


def deepseek_v3_pretrain_1024gpu_h100_fp8sc_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 1024× H100, FP8-SC (VP=2, auto-applied default PP layout)."""
    cfg = deepseek_v3_pretrain_1024gpu_h100_fp8cs_config()
    cfg.mixed_precision.fp8_recipe = "blockwise"
    cfg.mixed_precision.fp8_param = False
    cfg.mixed_precision.fp8_param_gather = False
    cfg.mixed_precision.num_layers_at_start_in_bf16 = 0
    cfg.mixed_precision.num_layers_at_end_in_bf16 = 0
    cfg.model.virtual_pipeline_model_parallel_size = 2
    # DeepSeek-V3 has 61 layers; (61 // PP) % VP != 0 for (PP=8, VP=2), so a custom layout
    # is required. The helper's default layout map provides one for this (pp, vp) pair.
    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)
    return cfg


def deepseek_v3_pretrain_64gpu_h100_bf16_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 64× H100, BF16 (1024-GPU layout with legacy-scaled GBS)."""
    cfg = deepseek_v3_pretrain_1024gpu_h100_bf16_config()
    cfg.train.global_batch_size = 1024
    return cfg


def deepseek_v3_pretrain_64gpu_h100_fp8cs_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 64× H100, FP8 current-scaling (standard tensorwise)."""
    cfg = deepseek_v3_pretrain_config()
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
    _enable_overlap_param_gather_with_optimizer_step(cfg)
    return cfg


def deepseek_v3_pretrain_1024gpu_h100_fp8sc_large_scale_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 1024× H100, FP8-SC, large-scale proxy (GBS=1024)."""
    cfg = deepseek_v3_pretrain_1024gpu_h100_fp8sc_config()
    cfg.train.global_batch_size = 1024
    return cfg
