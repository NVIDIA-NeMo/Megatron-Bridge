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
from megatron.bridge.recipes.llama.h100 import llama3 as _h100_module
from megatron.bridge.recipes.llama.h100.llama3 import (
    _llama3_8b_pretrain_2gpu_h100_low_precision_config,
)
from megatron.bridge.training.config import ConfigContainer


AutoBridge = _h100_module.AutoBridge

_LLAMA3_8B_LOW_PRECISION_PRETRAIN_CONFIGS = {
    "bf16_with_mxfp8_mixed": _h100_module.llama3_8b_pretrain_2gpu_h100_fp8mx_config,
    "bf16_with_fp8_current_scaling_mixed": _h100_module.llama3_8b_pretrain_2gpu_h100_fp8cs_config,
    "bf16_with_nvfp4_mixed": _h100_module.llama3_8b_pretrain_2gpu_h100_nvfp4_config,
}


def llama3_8b_low_precision_pretrain_config(mixed_precision_recipe: str) -> ConfigContainer:
    """Return the legacy low-precision Llama 3 8B pre-training config."""
    assert mixed_precision_recipe in _LLAMA3_8B_LOW_PRECISION_PRETRAIN_CONFIGS, (
        f"Invalid low precision recipe: {mixed_precision_recipe}. This recipe has not been tested yet."
    )
    _h100_module.AutoBridge = AutoBridge
    return _LLAMA3_8B_LOW_PRECISION_PRETRAIN_CONFIGS[mixed_precision_recipe]()


def llama31_405b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Compatibility alias for ``llama31_405b_peft_32gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.llama31_405b_peft_32gpu_h100_bf16_config(peft_scheme=peft_scheme)


def llama31_405b_pretrain_config() -> ConfigContainer:
    """Compatibility alias for ``llama31_405b_pretrain_256gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.llama31_405b_pretrain_256gpu_h100_bf16_config()


def llama31_405b_pretrain_deterministic_config() -> ConfigContainer:
    """Compatibility alias for ``llama31_405b_pretrain_256gpu_h100_bf16_deterministic_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.llama31_405b_pretrain_256gpu_h100_bf16_deterministic_config()


def llama31_405b_sft_config() -> ConfigContainer:
    """Compatibility alias for ``llama31_405b_sft_128gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.llama31_405b_sft_128gpu_h100_bf16_config()


def llama31_70b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Compatibility alias for ``llama31_70b_peft_8gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.llama31_70b_peft_8gpu_h100_bf16_config(peft_scheme=peft_scheme)


def llama31_70b_pretrain_config() -> ConfigContainer:
    """Compatibility alias for ``llama31_70b_pretrain_32gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.llama31_70b_pretrain_32gpu_h100_bf16_config()


def llama31_70b_sft_config() -> ConfigContainer:
    """Compatibility alias for ``llama31_70b_sft_32gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.llama31_70b_sft_32gpu_h100_bf16_config()


def llama31_8b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Compatibility alias for ``llama31_8b_peft_1gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.llama31_8b_peft_1gpu_h100_bf16_config(peft_scheme=peft_scheme)


def llama31_8b_pretrain_config() -> ConfigContainer:
    """Compatibility alias for ``llama31_8b_pretrain_2gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.llama31_8b_pretrain_2gpu_h100_bf16_config()


def llama31_8b_sft_config() -> ConfigContainer:
    """Compatibility alias for ``llama31_8b_sft_2gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.llama31_8b_sft_2gpu_h100_bf16_config()


def llama32_1b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Compatibility alias for ``llama32_1b_peft_1gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.llama32_1b_peft_1gpu_h100_bf16_config(peft_scheme=peft_scheme)


def llama32_1b_pretrain_config() -> ConfigContainer:
    """Compatibility alias for ``llama32_1b_pretrain_1gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.llama32_1b_pretrain_1gpu_h100_bf16_config()


def llama32_1b_sft_config() -> ConfigContainer:
    """Compatibility alias for ``llama32_1b_sft_1gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.llama32_1b_sft_1gpu_h100_bf16_config()


def llama32_3b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Compatibility alias for ``llama32_3b_peft_1gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.llama32_3b_peft_1gpu_h100_bf16_config(peft_scheme=peft_scheme)


def llama32_3b_pretrain_config() -> ConfigContainer:
    """Compatibility alias for ``llama32_3b_pretrain_1gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.llama32_3b_pretrain_1gpu_h100_bf16_config()


def llama32_3b_sft_config() -> ConfigContainer:
    """Compatibility alias for ``llama32_3b_sft_1gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.llama32_3b_sft_1gpu_h100_bf16_config()


def llama3_70b_16k_pretrain_config() -> ConfigContainer:
    """Compatibility alias for ``llama3_70b_pretrain_32gpu_h100_bf16_16k_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.llama3_70b_pretrain_32gpu_h100_bf16_16k_config()


def llama3_70b_64k_pretrain_config() -> ConfigContainer:
    """Compatibility alias for ``llama3_70b_pretrain_256gpu_h100_bf16_64k_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.llama3_70b_pretrain_256gpu_h100_bf16_64k_config()


def llama3_70b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Compatibility alias for ``llama3_70b_peft_8gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.llama3_70b_peft_8gpu_h100_bf16_config(peft_scheme=peft_scheme)


def llama3_70b_pretrain_config() -> ConfigContainer:
    """Compatibility alias for ``llama3_70b_pretrain_32gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.llama3_70b_pretrain_32gpu_h100_bf16_config()


def llama3_70b_pretrain_deterministic_config() -> ConfigContainer:
    """Compatibility alias for ``llama3_70b_pretrain_32gpu_h100_bf16_deterministic_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.llama3_70b_pretrain_32gpu_h100_bf16_deterministic_config()


def llama3_70b_sft_config() -> ConfigContainer:
    """Compatibility alias for ``llama3_70b_sft_32gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.llama3_70b_sft_32gpu_h100_bf16_config()


def llama3_8b_128k_pretrain_config() -> ConfigContainer:
    """Compatibility alias for ``llama3_8b_pretrain_64gpu_h100_bf16_128k_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.llama3_8b_pretrain_64gpu_h100_bf16_128k_config()


def llama3_8b_16k_pretrain_config() -> ConfigContainer:
    """Compatibility alias for ``llama3_8b_pretrain_16gpu_h100_bf16_16k_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.llama3_8b_pretrain_16gpu_h100_bf16_16k_config()


def llama3_8b_64k_pretrain_config() -> ConfigContainer:
    """Compatibility alias for ``llama3_8b_pretrain_32gpu_h100_bf16_64k_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.llama3_8b_pretrain_32gpu_h100_bf16_64k_config()


def llama3_8b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Compatibility alias for ``llama3_8b_peft_1gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.llama3_8b_peft_1gpu_h100_bf16_config(peft_scheme=peft_scheme)


def llama3_8b_pretrain_config() -> ConfigContainer:
    """Compatibility alias for ``llama3_8b_pretrain_2gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.llama3_8b_pretrain_2gpu_h100_bf16_config()


def llama3_8b_sft_config() -> ConfigContainer:
    """Compatibility alias for ``llama3_8b_sft_2gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.llama3_8b_sft_2gpu_h100_bf16_config()


__all__ = [
    "llama31_405b_peft_config",
    "llama31_405b_pretrain_config",
    "llama31_405b_pretrain_deterministic_config",
    "llama31_405b_sft_config",
    "llama31_70b_peft_config",
    "llama31_70b_pretrain_config",
    "llama31_70b_sft_config",
    "llama31_8b_peft_config",
    "llama31_8b_pretrain_config",
    "llama31_8b_sft_config",
    "llama32_1b_peft_config",
    "llama32_1b_pretrain_config",
    "llama32_1b_sft_config",
    "llama32_3b_peft_config",
    "llama32_3b_pretrain_config",
    "llama32_3b_sft_config",
    "llama3_70b_16k_pretrain_config",
    "llama3_70b_64k_pretrain_config",
    "llama3_70b_peft_config",
    "llama3_70b_pretrain_config",
    "llama3_70b_pretrain_deterministic_config",
    "llama3_70b_sft_config",
    "llama3_8b_128k_pretrain_config",
    "llama3_8b_16k_pretrain_config",
    "llama3_8b_64k_pretrain_config",
    "llama3_8b_peft_config",
    "llama3_8b_pretrain_config",
    "llama3_8b_sft_config",
    "llama3_8b_low_precision_pretrain_config",
]
