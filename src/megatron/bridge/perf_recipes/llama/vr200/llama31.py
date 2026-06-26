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
"""VR200 performance recipes for Llama 3.1."""

from megatron.bridge.perf_recipes.llama.common import ConfigContainer
from megatron.bridge.perf_recipes.llama.gb200.llama31 import (
    llama31_405b_pretrain_256gpu_gb200_bf16_config,
    llama31_405b_pretrain_256gpu_gb200_fp8mx_config,
    llama31_405b_pretrain_256gpu_gb200_nvfp4_config,
)


llama31_405b_pretrain_256gpu_vr200_bf16_config = llama31_405b_pretrain_256gpu_gb200_bf16_config

llama31_405b_pretrain_256gpu_vr200_fp8mx_config = llama31_405b_pretrain_256gpu_gb200_fp8mx_config


def llama31_405b_pretrain_256gpu_vr200_nvfp4_config() -> ConfigContainer:
    """Llama3.1 405B pretrain: 256x VR200, NVFP4."""
    cfg = llama31_405b_pretrain_256gpu_gb200_nvfp4_config()
    cfg.ddp.overlap_param_gather = True
    cfg.optimizer.overlap_param_gather = False
    cfg.comm_overlap.overlap_param_gather = None
    cfg.comm_overlap.align_param_gather = None
    return cfg
