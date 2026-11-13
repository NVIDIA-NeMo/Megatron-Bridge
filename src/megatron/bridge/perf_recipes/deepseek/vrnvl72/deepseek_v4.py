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
"""VRNVL72 performance recipes for DeepSeek V4."""

from megatron.bridge.perf_recipes._common import _enable_ncclep
from megatron.bridge.perf_recipes.deepseek.gb300.deepseek_v4 import (
    deepseek_v4_flash_pretrain_128gpu_gb300_fp8mx_config,
)
from megatron.bridge.perf_recipes.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.training.config import ConfigContainer


def deepseek_v4_flash_pretrain_128gpu_vrnvl72_fp8mx_config() -> ConfigContainer:
    """DeepSeek V4 Flash pretrain: 128× VRNVL72, MXFP8, NCCL EP (alias of GB300)."""
    cfg = deepseek_v4_flash_pretrain_128gpu_gb300_fp8mx_config()
    _enable_ncclep(cfg)

    # Rubin's grouped GLU kernel does not support DeepSeek V4's SwiGLU clamp parameters. This is a
    # benchmark-only override that preserves the lower-memory CutDSL fusion and full CUDA graph.
    cfg.model.activation_func_clamp_value = None
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.model.moe_use_grouped_tensor = True

    # Keep the VRNVL72 launch environment explicit instead of inheriting it from GB300.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
        "NCCL_NVLS_ENABLE": 0,
        # NCCL EP dispatcher mode and one GPU per rank.
        "NCCL_EP_HT_EM_PULL_PUSH": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
    }
    return cfg
