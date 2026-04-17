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

"""Flat performance benchmark recipes for Wan 14B diffusion model.

Each function is self-contained: call library recipe, override fields, call
``_benchmark_common()``, return.

All configs use BF16 precision (diffusion training does not use FP8).
"""

from megatron.bridge.diffusion.recipes.wan.wan import wan_14b_pretrain_config
from megatron.bridge.recipes.common import _benchmark_common
from megatron.bridge.recipes.llama.llama3_perf import _perf_precision
from megatron.bridge.training.config import ConfigContainer


# =============================================================================
# Wan 14B pretrain — 16 GPU, GB200
# =============================================================================


def wan_14b_pretrain_16gpu_gb200_bf16_config() -> ConfigContainer:
    """Wan 14B pretrain: 16× GB200, BF16, TP=1 CP=4."""
    cfg = wan_14b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 4
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 64
    cfg.train.micro_batch_size = 1

    _benchmark_common(cfg)
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.mixed_precision.grad_reduce_in_fp32 = True
    cfg.model.moe_token_dispatcher_type = "alltoall"
    return cfg


# =============================================================================
# Wan 14B pretrain — 32 GPU, H100
# =============================================================================


def wan_14b_pretrain_32gpu_h100_bf16_config() -> ConfigContainer:
    """Wan 14B pretrain: 32× H100, BF16, TP=2 CP=4, recompute block/8."""
    cfg = wan_14b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 4
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 64
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "block"
    cfg.model.recompute_num_layers = 8

    _benchmark_common(cfg)
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.mixed_precision.grad_reduce_in_fp32 = True
    cfg.model.moe_token_dispatcher_type = "alltoall"
    return cfg
