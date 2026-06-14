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

"""Flat performance benchmark recipes for GPT-OSS 120B.

Each function is self-contained: call library recipe, override fields, call
``_benchmark_common()``, return.
"""

from megatron.bridge.perf_recipes._common import _benchmark_common, _perf_precision
from megatron.bridge.recipes.gpt_oss.gpt_oss import gpt_oss_120b_pretrain_config
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer


def _apply_full_iter_fp8mx_overrides(
    cfg: ConfigContainer, *, expert_model_parallel_size: int, clear_recompute: bool = False
) -> None:
    """Apply legacy GPT-OSS 120B FP8-MX full-iteration CUDA graph settings."""
    cfg.model.expert_model_parallel_size = expert_model_parallel_size
    cfg.model.cuda_graph_impl = "full_iteration"
    cfg.model.cuda_graph_scope = []
    cfg.model.cuda_graph_warmup_steps = 2
    cfg.model.fp8_output_proj = True
    cfg.model.high_priority_a2a_comm_stream = True
    cfg.model.moe_expert_rank_capacity_factor = 1.5
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_hybridep_num_sms = 32
    cfg.model.moe_hybridep_num_sms_preprocessing = 32
    cfg.model.moe_mlp_glu_interleave_size = 32
    cfg.model.moe_pad_experts_for_cuda_graph_inference = True
    cfg.model.moe_paged_stash = True
    cfg.model.moe_paged_stash_buffer_size_factor_cpu = 1.0
    cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.2
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.use_te_rng_tracker = True
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.model.offload_modules = []
    cfg.mixed_precision.fp8_dot_product_attention = True
    cfg.rng.te_rng_tracker = True

    if cfg.comm_overlap is None:
        cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.comm_overlap.delay_wgrad_compute = True
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = True

    if clear_recompute:
        cfg.model.recompute_granularity = "selective"
        cfg.model.recompute_modules = []


# =============================================================================
# GPT-OSS 120B pretrain (GBS=1280) — 64 GPU, GB300
# =============================================================================


def gpt_oss_120b_pretrain_64gpu_gb300_bf16_config() -> ConfigContainer:
    """GPT-OSS 120B pretrain: 64× GB300, BF16, GBS=1280."""
    cfg = gpt_oss_120b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.model.moe_router_fusion = True
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 1280
    cfg.train.micro_batch_size = 4

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# GPT-OSS 120B pretrain (GBS=1280) — 64 GPU, GB200
# =============================================================================


def gpt_oss_120b_pretrain_64gpu_gb200_bf16_config() -> ConfigContainer:
    """GPT-OSS 120B pretrain: 64× GB200, BF16, GBS=1280."""
    cfg = gpt_oss_120b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.model.moe_router_fusion = True
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 1280
    cfg.train.micro_batch_size = 4

    cfg.model.recompute_modules = ["layernorm", "moe_act"]
    cfg.model.recompute_granularity = "selective"

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.comm_overlap.delay_wgrad_compute = True
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = True

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# GPT-OSS 120B pretrain (GBS=1280) — 64 GPU, B300
# =============================================================================


def gpt_oss_120b_pretrain_64gpu_b300_bf16_config() -> ConfigContainer:
    """GPT-OSS 120B pretrain: 64× B300, BF16, GBS=1280."""
    cfg = gpt_oss_120b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.model.moe_router_fusion = True
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 1280
    cfg.train.micro_batch_size = 4

    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_hybridep_num_sms = 32

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# GPT-OSS 120B pretrain (GBS=1280) — 64 GPU, B200
# =============================================================================


def gpt_oss_120b_pretrain_64gpu_b200_bf16_config() -> ConfigContainer:
    """GPT-OSS 120B pretrain: 64× B200, BF16, GBS=1280."""
    cfg = gpt_oss_120b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.model.moe_router_fusion = True
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 1280
    cfg.train.micro_batch_size = 4

    cfg.model.recompute_modules = ["layernorm", "moe_act"]
    cfg.model.recompute_granularity = "selective"

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# GPT-OSS 120B pretrain (GBS=1280) — 64 GPU, H100
# =============================================================================


def gpt_oss_120b_pretrain_64gpu_h100_bf16_config() -> ConfigContainer:
    """GPT-OSS 120B pretrain: 64× H100, BF16, PP=4 EP=8, GBS=1280."""
    cfg = gpt_oss_120b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.model.moe_router_fusion = True
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.expert_model_parallel_size = 8
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 1280
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = ["layernorm", "moe_act"]
    cfg.model.recompute_granularity = "selective"

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# GPT-OSS 120B — FP8-MX variants: same parallelism as BF16, MXFP8 precision
# =============================================================================


def gpt_oss_120b_pretrain_64gpu_gb300_fp8mx_config() -> ConfigContainer:
    """GPT-OSS 120B pretrain: 64× GB300, FP8-MX."""
    cfg = gpt_oss_120b_pretrain_64gpu_gb300_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    _apply_full_iter_fp8mx_overrides(cfg, expert_model_parallel_size=16)
    return cfg


def gpt_oss_120b_pretrain_64gpu_gb200_fp8mx_config() -> ConfigContainer:
    """GPT-OSS 120B pretrain: 64× GB200, FP8-MX."""
    cfg = gpt_oss_120b_pretrain_64gpu_gb200_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    _apply_full_iter_fp8mx_overrides(cfg, expert_model_parallel_size=64, clear_recompute=True)
    return cfg


def gpt_oss_120b_pretrain_64gpu_b300_fp8mx_config() -> ConfigContainer:
    """GPT-OSS 120B pretrain: 64× B300, FP8-MX."""
    cfg = gpt_oss_120b_pretrain_64gpu_b300_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    return cfg


def gpt_oss_120b_pretrain_64gpu_b200_fp8mx_config() -> ConfigContainer:
    """GPT-OSS 120B pretrain: 64× B200, FP8-MX."""
    cfg = gpt_oss_120b_pretrain_64gpu_b200_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    return cfg


# =============================================================================
# GPT-OSS 120B — FP8-CS variants: same parallelism as BF16, FP8 current-scaling
# =============================================================================


def gpt_oss_120b_pretrain_64gpu_h100_fp8cs_config() -> ConfigContainer:
    """GPT-OSS 120B pretrain: 64× H100, FP8-CS."""
    cfg = gpt_oss_120b_pretrain_64gpu_h100_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    return cfg
