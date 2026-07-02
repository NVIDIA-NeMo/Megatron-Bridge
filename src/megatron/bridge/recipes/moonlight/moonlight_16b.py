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
from megatron.bridge.recipes.moonlight.h100 import moonlight_16b as _h100_module
from megatron.bridge.recipes.moonlight.h100.moonlight_16b import (
    _get_moonlight_pipeline_layout,
)
from megatron.bridge.training.config import ConfigContainer


AutoBridge = _h100_module.AutoBridge


def moonlight_16b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Compatibility alias for ``moonlight_16b_peft_2gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.moonlight_16b_peft_2gpu_h100_bf16_config(peft_scheme=peft_scheme)


def moonlight_16b_pretrain_config() -> ConfigContainer:
    """Compatibility alias for ``moonlight_16b_pretrain_8gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.moonlight_16b_pretrain_8gpu_h100_bf16_config()


def moonlight_16b_sft_config() -> ConfigContainer:
    """Compatibility alias for ``moonlight_16b_sft_8gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.moonlight_16b_sft_8gpu_h100_bf16_config()


__all__ = [
    "moonlight_16b_pretrain_config",
    "moonlight_16b_sft_config",
    "moonlight_16b_peft_config",
]
