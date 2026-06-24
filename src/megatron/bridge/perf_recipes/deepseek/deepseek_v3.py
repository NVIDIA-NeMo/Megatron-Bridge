# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Flat performance benchmark recipes for DeepSeek V3.

Each function is self-contained: call library recipe, override fields, call
``_benchmark_common()``, return.

Naming convention::

    {model}_{size}_{task}_{num_gpus}gpu_{gpu}_{precision}_config
"""

from megatron.bridge.perf_recipes._common import (
    _benchmark_common,
    _enable_overlap_param_gather_with_optimizer_step,
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


# =============================================================================
# DeepSeek V3 pretrain — 256 GPU, GB300
# =============================================================================


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
    cfg.train.micro_batch_size = 2

    cfg.model.recompute_modules = ["mla_up_proj"]

    cfg.model.cuda_graph_scope = []
    cfg.ddp.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_grad_reduce = True

    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model, "Et*4|(t*4|)*14tmL")

    _benchmark_common(cfg)
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


# =============================================================================
# DeepSeek V3 pretrain — 256 GPU, GB200
# =============================================================================


def deepseek_v3_pretrain_256gpu_gb200_bf16_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× GB200, BF16."""
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

    cfg.model.recompute_modules = ["mla_up_proj"]

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]

    cfg.ddp.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_grad_reduce = True

    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    _benchmark_common(cfg)
    _enable_overlap_param_gather_with_optimizer_step(cfg)
    return cfg


def deepseek_v3_pretrain_256gpu_gb200_fp8cs_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× GB200, FP8 current-scaling."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    _deepseek_v3_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 4096
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


def deepseek_v3_pretrain_256gpu_gb200_fp8mx_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× GB200, MXFP8."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    _deepseek_v3_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = ["mlp"]

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]

    cfg.ddp.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_grad_reduce = True

    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# DeepSeek V3 pretrain — 256 GPU, B300
# =============================================================================


def deepseek_v3_pretrain_256gpu_b300_bf16_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× B300, BF16."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    _deepseek_v3_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = ["mla_up_proj"]

    cfg.ddp.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_grad_reduce = True

    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    _benchmark_common(cfg)
    return cfg


def deepseek_v3_pretrain_256gpu_b300_fp8cs_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× B300, FP8 current-scaling."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    _deepseek_v3_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = ["mla_up_proj"]

    cfg.ddp.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_grad_reduce = True

    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    _benchmark_common(cfg)
    return cfg


def deepseek_v3_pretrain_256gpu_b300_fp8mx_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× B300, MXFP8."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    _deepseek_v3_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = ["mla_up_proj"]

    cfg.ddp.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_grad_reduce = True

    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# DeepSeek V3 pretrain — 256 GPU, B200
# =============================================================================


def deepseek_v3_pretrain_256gpu_b200_bf16_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× B200, BF16."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    _deepseek_v3_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = ["mla_up_proj"]

    cfg.ddp.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_grad_reduce = True

    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    _benchmark_common(cfg)
    return cfg


def deepseek_v3_pretrain_256gpu_b200_fp8cs_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× B200, FP8 current-scaling."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    _deepseek_v3_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = ["mla_up_proj"]

    cfg.ddp.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_grad_reduce = True

    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    _benchmark_common(cfg)
    return cfg


def deepseek_v3_pretrain_256gpu_b200_fp8mx_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× B200, MXFP8."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    _deepseek_v3_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = ["mla_up_proj"]

    cfg.ddp.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_grad_reduce = True

    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# DeepSeek V3 pretrain — 1024 GPU, H100
# =============================================================================


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
    cfg.mixed_precision.fp8_recipe = "blockwise"
    cfg.mixed_precision.fp8_param = False
    cfg.mixed_precision.fp8_param_gather = False
    cfg.mixed_precision.num_layers_at_start_in_bf16 = 0
    cfg.mixed_precision.num_layers_at_end_in_bf16 = 0
    _deepseek_v3_common(cfg)

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.virtual_pipeline_model_parallel_size = 2
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 16384
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = ["mla_up_proj", "mlp"]

    cfg.ddp.overlap_grad_reduce = False
    cfg.comm_overlap.overlap_grad_reduce = False

    cfg.model.pipeline_model_parallel_layout = None

    _benchmark_common(cfg)
    _enable_overlap_param_gather_with_optimizer_step(cfg)
    return cfg


# =============================================================================
# DeepSeek V3 — NVFP4 aliases: same parallelism as BF16, NVFP4 precision
# =============================================================================


def deepseek_v3_pretrain_256gpu_b200_nvfp4_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× B200, NVFP4 (same layout as BF16)."""
    cfg = deepseek_v3_pretrain_256gpu_b200_bf16_config()
    cfg.mixed_precision = _perf_precision("nvfp4")
    return cfg


def deepseek_v3_pretrain_256gpu_b300_nvfp4_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× B300, NVFP4 (PP=16 matching base layout)."""
    cfg = deepseek_v3_pretrain_256gpu_b300_bf16_config()
    cfg.mixed_precision = _perf_precision("nvfp4")
    cfg.model.pipeline_model_parallel_size = 16
    cfg.model.virtual_pipeline_model_parallel_size = None
    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)
    return cfg


def deepseek_v3_pretrain_256gpu_gb200_nvfp4_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× GB200, NVFP4 (same layout as BF16, mlp recompute)."""
    cfg = deepseek_v3_pretrain_256gpu_gb200_bf16_config()
    cfg.mixed_precision = _perf_precision("nvfp4")
    cfg.model.recompute_modules = ["mlp"]
    cfg.optimizer.overlap_param_gather_with_optimizer_step = False
    cfg.comm_overlap.overlap_param_gather_with_optimizer_step = None
    return cfg


def deepseek_v3_pretrain_1024gpu_h100_fp8sc_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 1024× H100, FP8-SC (VP=2, auto-applied default PP layout)."""
    cfg = deepseek_v3_pretrain_1024gpu_h100_fp8cs_config()
    cfg.model.virtual_pipeline_model_parallel_size = 2
    # DeepSeek-V3 has 61 layers; (61 // PP) % VP != 0 for (PP=8, VP=2), so a custom layout
    # is required. The helper's default layout map provides one for this (pp, vp) pair.
    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)
    return cfg


# =============================================================================
# DeepSeek V3 pretrain — 128 GPU, VR200
# =============================================================================


def deepseek_v3_pretrain_128gpu_vr200_bf16_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 128× VR200, BF16."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    _deepseek_v3_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = ["mla_up_proj"]

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]

    cfg.ddp.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_grad_reduce = True

    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model, "Et*4|(t*4|)*14tmL")

    _benchmark_common(cfg)
    _enable_overlap_param_gather_with_optimizer_step(cfg)
    return cfg


def deepseek_v3_pretrain_128gpu_vr200_fp8cs_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 128× VR200, FP8 current-scaling."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    _deepseek_v3_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = ["mla_up_proj"]

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]

    cfg.ddp.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_grad_reduce = True

    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model, "Et*4|(t*4|)*14tmL")

    _benchmark_common(cfg)
    _enable_overlap_param_gather_with_optimizer_step(cfg)
    return cfg


def deepseek_v3_pretrain_128gpu_vr200_fp8mx_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 128× VR200, MXFP8."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    _deepseek_v3_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = ["mla_up_proj"]

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]

    cfg.ddp.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_grad_reduce = True

    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model, "Et*4|(t*4|)*14tmL")

    _benchmark_common(cfg)
    return cfg


def deepseek_v3_pretrain_128gpu_vr200_nvfp4_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 128× VR200, NVFP4."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("nvfp4")
    _deepseek_v3_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = ["mla_up_proj"]

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]

    cfg.ddp.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_grad_reduce = True

    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model, "Et*4|(t*4|)*14tmL")

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# DeepSeek V3 — VR200 aliases: identical config to GB200 counterparts
# =============================================================================


def deepseek_v3_pretrain_256gpu_vr200_bf16_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× VR200, BF16 (alias of GB200)."""
    return deepseek_v3_pretrain_256gpu_gb200_bf16_config()


def deepseek_v3_pretrain_256gpu_vr200_fp8cs_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× VR200, FP8-CS (alias of GB200)."""
    return deepseek_v3_pretrain_256gpu_gb200_fp8cs_config()


def deepseek_v3_pretrain_256gpu_vr200_fp8mx_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× VR200, FP8-MX (alias of GB200)."""
    return deepseek_v3_pretrain_256gpu_gb200_fp8mx_config()


def deepseek_v3_pretrain_256gpu_vr200_nvfp4_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× VR200, NVFP4 (alias of GB200)."""
    return deepseek_v3_pretrain_256gpu_gb200_nvfp4_config()


# =============================================================================
# DeepSeek V3 pretrain — 64 GPU aliases (same config as 1024 GPU H100)
# =============================================================================


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


# =============================================================================
# DeepSeek V3 pretrain — 64 GPU, GB300, Megatron FSDP
# =============================================================================


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


# =============================================================================
# DeepSeek V3 — Large-scale proxy variants
# =============================================================================
# Pre-refactor large_scale configs in workload_base_configs.py used distinct
# parallelism from V2 (typically BF16_V1's layout) plus a small global batch
# size to stress comm patterns without long-running training. Reproduce that
# behavior here so the `config_variant=large_scale` lookup resolves to the
# correct shape.


def deepseek_v3_pretrain_256gpu_gb300_fp8mx_large_scale_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× GB300, MXFP8, large-scale proxy (BF16_V1 layout, GBS=256)."""
    cfg = deepseek_v3_pretrain_256gpu_gb300_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    cfg.train.global_batch_size = 256
    return cfg


def deepseek_v3_pretrain_256gpu_gb200_fp8mx_large_scale_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× GB200, MXFP8, large-scale proxy (GBS=256)."""
    cfg = deepseek_v3_pretrain_256gpu_gb200_fp8mx_config()
    cfg.train.global_batch_size = 256
    return cfg


def deepseek_v3_pretrain_256gpu_b300_fp8mx_large_scale_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× B300, MXFP8, large-scale proxy (GBS=256)."""
    cfg = deepseek_v3_pretrain_256gpu_b300_fp8mx_config()
    cfg.train.global_batch_size = 256
    return cfg


def deepseek_v3_pretrain_256gpu_b200_fp8mx_large_scale_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× B200, MXFP8, large-scale proxy (GBS=256)."""
    cfg = deepseek_v3_pretrain_256gpu_b200_fp8mx_config()
    cfg.train.global_batch_size = 256
    return cfg


def deepseek_v3_pretrain_1024gpu_h100_fp8sc_large_scale_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 1024× H100, FP8-SC, large-scale proxy (GBS=1024)."""
    cfg = deepseek_v3_pretrain_1024gpu_h100_fp8sc_config()
    cfg.train.global_batch_size = 1024
    return cfg
