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
# ruff: noqa: F401
"""Compatibility aliases for legacy recipe names."""

from __future__ import annotations

from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.glm.h100 import glm45 as _h100_module
from megatron.bridge.training.config import ConfigContainer


AutoBridge = _h100_module.AutoBridge


def glm45_355b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Compatibility alias for ``glm45_355b_peft_16gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.glm45_355b_peft_16gpu_h100_bf16_config(peft_scheme=peft_scheme)


def glm45_355b_pretrain_config() -> ConfigContainer:
    """Compatibility alias for ``glm45_355b_pretrain_128gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.glm45_355b_pretrain_128gpu_h100_bf16_config()


def glm45_355b_sft_config() -> ConfigContainer:
    """Compatibility alias for ``glm45_355b_sft_128gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.glm45_355b_sft_128gpu_h100_bf16_config()


def glm45_air_106b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Compatibility alias for ``glm45_air_106b_peft_8gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.glm45_air_106b_peft_8gpu_h100_bf16_config(peft_scheme=peft_scheme)


def glm45_air_106b_pretrain_config() -> ConfigContainer:
    """Compatibility alias for ``glm45_air_106b_pretrain_32gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.glm45_air_106b_pretrain_32gpu_h100_bf16_config()


def glm45_air_106b_sft_config() -> ConfigContainer:
    """Compatibility alias for ``glm45_air_106b_sft_32gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.glm45_air_106b_sft_32gpu_h100_bf16_config()


__all__ = [
    "glm45_355b_peft_config",
    "glm45_355b_pretrain_config",
    "glm45_355b_sft_config",
    "glm45_air_106b_peft_config",
    "glm45_air_106b_pretrain_config",
    "glm45_air_106b_sft_config",
]
