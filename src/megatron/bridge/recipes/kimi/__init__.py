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

# Kimi K2 models
from .kimi_k2 import kimi_k2_pretrain_config

# Kimi K2 perf recipes
from .kimi_k2_perf import (
    kimi_k2_pretrain_256gpu_b200_bf16_config,
    kimi_k2_pretrain_256gpu_b200_fp8cs_config,
    kimi_k2_pretrain_256gpu_b200_fp8mx_config,
    kimi_k2_pretrain_256gpu_gb200_bf16_config,
    kimi_k2_pretrain_256gpu_gb200_fp8cs_config,
    kimi_k2_pretrain_256gpu_gb200_fp8mx_config,
    # 256 GPU
    kimi_k2_pretrain_256gpu_gb300_bf16_config,
    kimi_k2_pretrain_256gpu_gb300_fp8cs_config,
    kimi_k2_pretrain_256gpu_gb300_fp8mx_config,
    kimi_k2_pretrain_256gpu_gb300_nvfp4_config,
    # 1024 GPU
    kimi_k2_pretrain_1024gpu_h100_bf16_config,
    kimi_k2_pretrain_1024gpu_h100_fp8cs_config,
)


__all__ = [
    # Kimi K2
    "kimi_k2_pretrain_config",
    # Kimi K2 perf recipes — 256 GPU
    "kimi_k2_pretrain_256gpu_gb300_bf16_config",
    "kimi_k2_pretrain_256gpu_gb300_fp8cs_config",
    "kimi_k2_pretrain_256gpu_gb300_fp8mx_config",
    "kimi_k2_pretrain_256gpu_gb300_nvfp4_config",
    "kimi_k2_pretrain_256gpu_gb200_bf16_config",
    "kimi_k2_pretrain_256gpu_gb200_fp8cs_config",
    "kimi_k2_pretrain_256gpu_gb200_fp8mx_config",
    "kimi_k2_pretrain_256gpu_b200_bf16_config",
    "kimi_k2_pretrain_256gpu_b200_fp8cs_config",
    "kimi_k2_pretrain_256gpu_b200_fp8mx_config",
    # Kimi K2 perf recipes — 1024 GPU
    "kimi_k2_pretrain_1024gpu_h100_bf16_config",
    "kimi_k2_pretrain_1024gpu_h100_fp8cs_config",
]
