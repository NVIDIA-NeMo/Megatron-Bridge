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
"""H100 performance recipes for Wan."""

from megatron.bridge.perf_recipes.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.perf_recipes.wan.common import (
    ConfigContainer,
    _benchmark_common,
    _perf_precision,
    wan_14b_pretrain_config,
)


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
    cfg.model.cuda_graph_scope = []
    cfg.model.moe_token_dispatcher_type = "alltoall"
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg
