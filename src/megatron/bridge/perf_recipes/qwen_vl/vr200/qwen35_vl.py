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
"""VR200 performance recipes for Qwen3.5-VL."""

from megatron.bridge.perf_recipes.environment import perf_recipe_environment
from megatron.bridge.perf_recipes.qwen_vl.common import ConfigContainer
from megatron.bridge.perf_recipes.qwen_vl.gb300.qwen35_vl import (
    qwen35_vl_35b_a3b_pretrain_8gpu_gb300_bf16_config,
    qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8cs_config,
    qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8mx_config,
    qwen35_vl_122b_a10b_pretrain_32gpu_gb300_bf16_config,
    qwen35_vl_122b_a10b_pretrain_32gpu_gb300_fp8cs_config,
    qwen35_vl_122b_a10b_pretrain_32gpu_gb300_fp8mx_config,
    qwen35_vl_397b_a17b_pretrain_64gpu_gb300_bf16_config,
    qwen35_vl_397b_a17b_pretrain_64gpu_gb300_fp8cs_config,
    qwen35_vl_397b_a17b_pretrain_64gpu_gb300_fp8mx_config,
)


@perf_recipe_environment(model_family_name="qwen_vl")
def qwen35_vl_35b_a3b_pretrain_8gpu_vr200_bf16_config() -> ConfigContainer:
    """Qwen3.5-VL 35B-A3B pretrain: 8× VR200, BF16 (alias of GB300)."""
    return qwen35_vl_35b_a3b_pretrain_8gpu_gb300_bf16_config()


@perf_recipe_environment(model_family_name="qwen_vl")
def qwen35_vl_35b_a3b_pretrain_8gpu_vr200_fp8cs_config() -> ConfigContainer:
    """Qwen3.5-VL 35B-A3B pretrain: 8× VR200, FP8-CS (alias of GB300)."""
    return qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8cs_config()


@perf_recipe_environment(model_family_name="qwen_vl")
def qwen35_vl_35b_a3b_pretrain_8gpu_vr200_fp8mx_config() -> ConfigContainer:
    """Qwen3.5-VL 35B-A3B pretrain: 8× VR200, FP8-MX (alias of GB300)."""
    return qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8mx_config()


@perf_recipe_environment(model_family_name="qwen_vl")
def qwen35_vl_122b_a10b_pretrain_32gpu_vr200_bf16_config() -> ConfigContainer:
    """Qwen3.5-VL 122B-A10B pretrain: 32× VR200, BF16 (alias of GB300)."""
    return qwen35_vl_122b_a10b_pretrain_32gpu_gb300_bf16_config()


@perf_recipe_environment(model_family_name="qwen_vl")
def qwen35_vl_122b_a10b_pretrain_32gpu_vr200_fp8cs_config() -> ConfigContainer:
    """Qwen3.5-VL 122B-A10B pretrain: 32× VR200, FP8-CS (alias of GB300)."""
    return qwen35_vl_122b_a10b_pretrain_32gpu_gb300_fp8cs_config()


@perf_recipe_environment(model_family_name="qwen_vl")
def qwen35_vl_122b_a10b_pretrain_32gpu_vr200_fp8mx_config() -> ConfigContainer:
    """Qwen3.5-VL 122B-A10B pretrain: 32× VR200, FP8-MX (alias of GB300)."""
    return qwen35_vl_122b_a10b_pretrain_32gpu_gb300_fp8mx_config()


@perf_recipe_environment(model_family_name="qwen_vl")
def qwen35_vl_397b_a17b_pretrain_64gpu_vr200_bf16_config() -> ConfigContainer:
    """Qwen3.5-VL 397B-A17B pretrain: 64× VR200, BF16 (alias of GB300)."""
    return qwen35_vl_397b_a17b_pretrain_64gpu_gb300_bf16_config()


@perf_recipe_environment(model_family_name="qwen_vl")
def qwen35_vl_397b_a17b_pretrain_64gpu_vr200_fp8cs_config() -> ConfigContainer:
    """Qwen3.5-VL 397B-A17B pretrain: 64× VR200, FP8-CS (alias of GB300)."""
    return qwen35_vl_397b_a17b_pretrain_64gpu_gb300_fp8cs_config()


@perf_recipe_environment(model_family_name="qwen_vl")
def qwen35_vl_397b_a17b_pretrain_64gpu_vr200_fp8mx_config() -> ConfigContainer:
    """Qwen3.5-VL 397B-A17B pretrain: 64× VR200, FP8-MX (alias of GB300)."""
    return qwen35_vl_397b_a17b_pretrain_64gpu_gb300_fp8mx_config()
