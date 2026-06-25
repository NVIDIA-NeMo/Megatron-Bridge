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
"""VR200 performance recipes for DeepSeek V3."""

from megatron.bridge.perf_recipes.deepseek.common import (
    ConfigContainer,
    _benchmark_common,
    _deepseek_v3_common,
    _enable_deepseek_full_iteration_mxfp8,
    _enable_overlap_param_gather_with_optimizer_step,
    _perf_precision,
    deepseek_v3_pretrain_config,
    set_deepseek_v3_pipeline_model_parallel_layout,
)
from megatron.bridge.perf_recipes.deepseek.gb200.deepseek_v3 import (
    deepseek_v3_pretrain_256gpu_gb200_bf16_config,
    deepseek_v3_pretrain_256gpu_gb200_fp8cs_config,
    deepseek_v3_pretrain_256gpu_gb200_fp8mx_config,
    deepseek_v3_pretrain_256gpu_gb200_nvfp4_config,
)


def deepseek_v3_pretrain_128gpu_vr200_bf16_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 128× VR200, BF16."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    _deepseek_v3_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = ["mla_up_proj"]

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]

    cfg.ddp.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_grad_reduce = True

    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    _benchmark_common(cfg)
    _enable_overlap_param_gather_with_optimizer_step(cfg)
    return cfg


def deepseek_v3_pretrain_128gpu_vr200_fp8cs_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 128× VR200, FP8 current-scaling."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    _deepseek_v3_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = ["mlp"]

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]

    cfg.ddp.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_grad_reduce = True

    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    _benchmark_common(cfg)
    _enable_overlap_param_gather_with_optimizer_step(cfg)
    return cfg


def deepseek_v3_pretrain_128gpu_vr200_fp8mx_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 128× VR200, MXFP8."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    _deepseek_v3_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = ["mla_up_proj"]

    cfg.ddp.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_grad_reduce = True

    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    _benchmark_common(cfg)
    _enable_deepseek_full_iteration_mxfp8(cfg)
    return cfg


def deepseek_v3_pretrain_128gpu_vr200_nvfp4_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 128× VR200, NVFP4."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("nvfp4")
    _deepseek_v3_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = ["mlp"]

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]

    cfg.ddp.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_grad_reduce = True

    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    _benchmark_common(cfg)
    return cfg


def deepseek_v3_pretrain_256gpu_vr200_bf16_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× VR200, BF16 (alias of GB200)."""
    return deepseek_v3_pretrain_256gpu_gb200_bf16_config()


def deepseek_v3_pretrain_256gpu_vr200_fp8cs_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× VR200, FP8-CS (alias of GB200)."""
    return deepseek_v3_pretrain_256gpu_gb200_fp8cs_config()


def deepseek_v3_pretrain_256gpu_vr200_fp8mx_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× VR200, FP8-MX (alias of GB200)."""
    cfg = deepseek_v3_pretrain_256gpu_gb200_fp8mx_config()
    cfg.model.fp8_output_proj = False
    return cfg


def deepseek_v3_pretrain_256gpu_vr200_nvfp4_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× VR200, NVFP4 (alias of GB200)."""
    return deepseek_v3_pretrain_256gpu_gb200_nvfp4_config()
