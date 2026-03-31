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

"""Flat performance benchmark recipes for Kimi K2.

Each function is self-contained: call library recipe, override fields, call
``_benchmark_common()``, return.

Naming convention::

    {model}_{task}_{num_gpus}gpu_{gpu}_{precision}_config

Precision short-names:
    bf16   = BF16 mixed precision
    fp8cs  = FP8 per-tensor current-scaling (first/last BF16 layers disabled)
    fp8mx  = MXFP8
    nvfp4  = NVFP4
"""

from megatron.bridge.recipes.common import _benchmark_common
from megatron.bridge.recipes.kimi.kimi_k2 import (
    _get_kimi_k2_pipeline_layout,
    kimi_k2_pretrain_config,
)
from megatron.bridge.recipes.llama.llama3_perf import _perf_precision
from megatron.bridge.training.config import ConfigContainer


# =============================================================================
# Kimi K2 pretrain — 256 GPU, GB300
# =============================================================================


def kimi_k2_pretrain_256gpu_gb300_bf16_config() -> ConfigContainer:
    """Kimi K2 pretrain: 256× GB300, BF16."""
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
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 2

    cfg.model.recompute_modules = ["mla_up_proj"]
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.pipeline_model_parallel_layout = _get_kimi_k2_pipeline_layout(4, 4)

    cfg.ddp.overlap_grad_reduce = True

    cfg.model.cuda_graph_scope = []
    _benchmark_common(cfg)
    cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True  # Must be after _benchmark_common
    return cfg


def kimi_k2_pretrain_256gpu_gb300_fp8cs_config() -> ConfigContainer:
    """Kimi K2 pretrain: 256× GB300, FP8 current-scaling."""
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
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 2

    cfg.model.recompute_modules = ["mla_up_proj"]
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.pipeline_model_parallel_layout = _get_kimi_k2_pipeline_layout(4, 4)

    cfg.ddp.overlap_grad_reduce = True

    cfg.model.cuda_graph_scope = []
    _benchmark_common(cfg)
    cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True  # Must be after _benchmark_common
    return cfg


def kimi_k2_pretrain_256gpu_gb300_fp8mx_config() -> ConfigContainer:
    """Kimi K2 pretrain: 256× GB300, MXFP8."""
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
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 2

    cfg.model.recompute_modules = ["mla_up_proj"]
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.pipeline_model_parallel_layout = _get_kimi_k2_pipeline_layout(4, 4)

    cfg.ddp.overlap_grad_reduce = True

    cfg.model.cuda_graph_scope = []
    _benchmark_common(cfg)
    cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True  # Must be after _benchmark_common
    return cfg


def kimi_k2_pretrain_256gpu_gb300_nvfp4_config() -> ConfigContainer:
    """Kimi K2 pretrain: 256× GB300, NVFP4."""
    cfg = kimi_k2_pretrain_config()
    cfg.mixed_precision = _perf_precision("nvfp4")
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
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 2

    cfg.model.recompute_modules = ["mla_up_proj"]
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.pipeline_model_parallel_layout = _get_kimi_k2_pipeline_layout(4, 4)

    cfg.ddp.overlap_grad_reduce = True

    cfg.model.cuda_graph_scope = []
    _benchmark_common(cfg)
    cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True  # Must be after _benchmark_common
    return cfg


# =============================================================================
# Kimi K2 pretrain — 256 GPU, GB200
# =============================================================================


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

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Kimi K2 pretrain — 256 GPU, B200
# =============================================================================


def kimi_k2_pretrain_256gpu_b200_bf16_config() -> ConfigContainer:
    """Kimi K2 pretrain: 256× B200, BF16."""
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
    cfg.model.pipeline_model_parallel_size = 16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 16
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = ["mla_up_proj"]
    cfg.model.pipeline_model_parallel_layout = _get_kimi_k2_pipeline_layout(16, 1)

    cfg.ddp.overlap_grad_reduce = True

    _benchmark_common(cfg)
    return cfg


def kimi_k2_pretrain_256gpu_b200_fp8cs_config() -> ConfigContainer:
    """Kimi K2 pretrain: 256× B200, FP8 current-scaling."""
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
    cfg.model.pipeline_model_parallel_size = 16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 16
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = ["mla_up_proj"]
    cfg.model.pipeline_model_parallel_layout = _get_kimi_k2_pipeline_layout(16, 1)

    cfg.ddp.overlap_grad_reduce = True

    _benchmark_common(cfg)
    return cfg


def kimi_k2_pretrain_256gpu_b200_fp8mx_config() -> ConfigContainer:
    """Kimi K2 pretrain: 256× B200, MXFP8."""
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
    cfg.model.pipeline_model_parallel_size = 16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 16
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = ["mla_up_proj"]
    cfg.model.pipeline_model_parallel_layout = _get_kimi_k2_pipeline_layout(16, 1)

    cfg.ddp.overlap_grad_reduce = True

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Kimi K2 pretrain — 1024 GPU, H100
# =============================================================================


def kimi_k2_pretrain_1024gpu_h100_bf16_config() -> ConfigContainer:
    """Kimi K2 pretrain: 1024× H100, BF16."""
    cfg = kimi_k2_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096
    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = "selective"
    cfg.dist.enable_megatron_core_experimental = True
    cfg.model.moe_router_force_load_balancing = True
    cfg.model.qk_clip = False

    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 16
    cfg.model.virtual_pipeline_model_parallel_size = 2
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 8192
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = ["mla_up_proj", "mlp"]
    cfg.model.pipeline_model_parallel_layout = "Et|(tt|)*30L"

    cfg.comm_overlap.overlap_grad_reduce = False

    _benchmark_common(cfg)
    return cfg


def kimi_k2_pretrain_1024gpu_h100_fp8cs_config() -> ConfigContainer:
    """Kimi K2 pretrain: 1024× H100, FP8 current-scaling."""
    cfg = kimi_k2_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.mixed_precision.fp8_recipe = "blockwise"
    cfg.mixed_precision.fp8_param = False
    cfg.mixed_precision.fp8_param_gather = False
    cfg.mixed_precision.num_layers_at_start_in_bf16 = 0
    cfg.mixed_precision.num_layers_at_end_in_bf16 = 0
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096
    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = "selective"
    cfg.dist.enable_megatron_core_experimental = True
    cfg.model.moe_router_force_load_balancing = True
    cfg.model.qk_clip = False

    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 16
    cfg.model.virtual_pipeline_model_parallel_size = 2
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 8192
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = ["mla_up_proj", "mlp"]
    cfg.model.pipeline_model_parallel_layout = "Et|(tt|)*30L"

    cfg.comm_overlap.overlap_grad_reduce = False

    _benchmark_common(cfg)
    return cfg
