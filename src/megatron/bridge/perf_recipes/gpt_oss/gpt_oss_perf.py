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
    cfg.model.expert_model_parallel_size = 64
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 1280
    cfg.train.micro_batch_size = 4

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
    cfg.model.expert_model_parallel_size = 64
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
    return cfg


def gpt_oss_120b_pretrain_64gpu_gb200_fp8mx_config() -> ConfigContainer:
    """GPT-OSS 120B pretrain: 64× GB200, FP8-MX."""
    cfg = gpt_oss_120b_pretrain_64gpu_gb200_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
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


def gpt_oss_120b_pretrain_64gpu_h100_fp8mx_config() -> ConfigContainer:
    """GPT-OSS 120B pretrain: 64× H100, FP8-MX."""
    cfg = gpt_oss_120b_pretrain_64gpu_h100_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    return cfg
