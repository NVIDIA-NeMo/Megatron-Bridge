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
"""VR200 performance recipes for Llama 3."""

from megatron.bridge.perf_recipes.llama.common import (
    ConfigContainer,
)
from megatron.bridge.perf_recipes.llama.gb200.llama3 import (
    llama3_8b_pretrain_8gpu_gb200_bf16_config,
    llama3_8b_pretrain_8gpu_gb200_fp8cs_config,
    llama3_8b_pretrain_8gpu_gb200_fp8mx_config,
    llama3_8b_pretrain_8gpu_gb200_nvfp4_config,
    llama3_70b_pretrain_64gpu_gb200_bf16_config,
    llama3_70b_pretrain_64gpu_gb200_fp8mx_config,
    llama3_70b_pretrain_64gpu_gb200_nvfp4_config,
)


def llama3_8b_pretrain_8gpu_vr200_bf16_config() -> ConfigContainer:
    """LLaMA 3 8B pretrain: 8× VR200, BF16 (alias of GB200)."""
    return llama3_8b_pretrain_8gpu_gb200_bf16_config()


def llama3_8b_pretrain_8gpu_vr200_fp8cs_config() -> ConfigContainer:
    """LLaMA 3 8B pretrain: 8× VR200, FP8-CS (alias of GB200)."""
    return llama3_8b_pretrain_8gpu_gb200_fp8cs_config()


def llama3_8b_pretrain_8gpu_vr200_fp8mx_config() -> ConfigContainer:
    """LLaMA 3 8B pretrain: 8× VR200, FP8-MX (alias of GB200)."""
    return llama3_8b_pretrain_8gpu_gb200_fp8mx_config()


def llama3_8b_pretrain_8gpu_vr200_nvfp4_config() -> ConfigContainer:
    """LLaMA 3 8B pretrain: 8× VR200, NVFP4 (alias of GB200)."""
    return llama3_8b_pretrain_8gpu_gb200_nvfp4_config()


def llama3_70b_pretrain_64gpu_vr200_bf16_config() -> ConfigContainer:
    """LLaMA 3 70B pretrain: 64× VR200, BF16 (alias of GB200)."""
    return llama3_70b_pretrain_64gpu_gb200_bf16_config()


def llama3_70b_pretrain_64gpu_vr200_fp8mx_config() -> ConfigContainer:
    """LLaMA 3 70B pretrain: 64× VR200, FP8-MX (alias of GB200)."""
    return llama3_70b_pretrain_64gpu_gb200_fp8mx_config()


def llama3_70b_pretrain_64gpu_vr200_nvfp4_config() -> ConfigContainer:
    """LLaMA 3 70B pretrain: 64× VR200, NVFP4 (alias of GB200)."""
    return llama3_70b_pretrain_64gpu_gb200_nvfp4_config()
