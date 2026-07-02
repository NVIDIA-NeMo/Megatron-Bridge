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
from megatron.bridge.recipes.qwen.h100 import qwen2 as _h100_module
from megatron.bridge.training.config import ConfigContainer


AutoBridge = _h100_module.AutoBridge


def qwen25_14b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Compatibility alias for ``qwen25_14b_peft_1gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.qwen25_14b_peft_1gpu_h100_bf16_config(peft_scheme=peft_scheme)


def qwen25_14b_pretrain_config() -> ConfigContainer:
    """Compatibility alias for ``qwen25_14b_pretrain_4gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.qwen25_14b_pretrain_4gpu_h100_bf16_config()


def qwen25_14b_sft_config() -> ConfigContainer:
    """Compatibility alias for ``qwen25_14b_sft_4gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.qwen25_14b_sft_4gpu_h100_bf16_config()


def qwen25_1p5b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Compatibility alias for ``qwen25_1p5b_peft_1gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.qwen25_1p5b_peft_1gpu_h100_bf16_config(peft_scheme=peft_scheme)


def qwen25_1p5b_pretrain_config() -> ConfigContainer:
    """Compatibility alias for ``qwen25_1p5b_pretrain_1gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.qwen25_1p5b_pretrain_1gpu_h100_bf16_config()


def qwen25_1p5b_sft_config() -> ConfigContainer:
    """Compatibility alias for ``qwen25_1p5b_sft_1gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.qwen25_1p5b_sft_1gpu_h100_bf16_config()


def qwen25_32b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Compatibility alias for ``qwen25_32b_peft_8gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.qwen25_32b_peft_8gpu_h100_bf16_config(peft_scheme=peft_scheme)


def qwen25_32b_pretrain_config() -> ConfigContainer:
    """Compatibility alias for ``qwen25_32b_pretrain_16gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.qwen25_32b_pretrain_16gpu_h100_bf16_config()


def qwen25_32b_sft_config() -> ConfigContainer:
    """Compatibility alias for ``qwen25_32b_sft_16gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.qwen25_32b_sft_16gpu_h100_bf16_config()


def qwen25_500m_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Compatibility alias for ``qwen25_500m_peft_1gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.qwen25_500m_peft_1gpu_h100_bf16_config(peft_scheme=peft_scheme)


def qwen25_500m_pretrain_config() -> ConfigContainer:
    """Compatibility alias for ``qwen25_500m_pretrain_1gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.qwen25_500m_pretrain_1gpu_h100_bf16_config()


def qwen25_500m_sft_config() -> ConfigContainer:
    """Compatibility alias for ``qwen25_500m_sft_1gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.qwen25_500m_sft_1gpu_h100_bf16_config()


def qwen25_72b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Compatibility alias for ``qwen25_72b_peft_8gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.qwen25_72b_peft_8gpu_h100_bf16_config(peft_scheme=peft_scheme)


def qwen25_72b_pretrain_config() -> ConfigContainer:
    """Compatibility alias for ``qwen25_72b_pretrain_32gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.qwen25_72b_pretrain_32gpu_h100_bf16_config()


def qwen25_72b_sft_config() -> ConfigContainer:
    """Compatibility alias for ``qwen25_72b_sft_32gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.qwen25_72b_sft_32gpu_h100_bf16_config()


def qwen25_7b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Compatibility alias for ``qwen25_7b_peft_1gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.qwen25_7b_peft_1gpu_h100_bf16_config(peft_scheme=peft_scheme)


def qwen25_7b_pretrain_config() -> ConfigContainer:
    """Compatibility alias for ``qwen25_7b_pretrain_2gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.qwen25_7b_pretrain_2gpu_h100_bf16_config()


def qwen25_7b_sft_config() -> ConfigContainer:
    """Compatibility alias for ``qwen25_7b_sft_2gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.qwen25_7b_sft_2gpu_h100_bf16_config()


def qwen2_1p5b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Compatibility alias for ``qwen2_1p5b_peft_1gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.qwen2_1p5b_peft_1gpu_h100_bf16_config(peft_scheme=peft_scheme)


def qwen2_1p5b_pretrain_config() -> ConfigContainer:
    """Compatibility alias for ``qwen2_1p5b_pretrain_1gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.qwen2_1p5b_pretrain_1gpu_h100_bf16_config()


def qwen2_1p5b_sft_config() -> ConfigContainer:
    """Compatibility alias for ``qwen2_1p5b_sft_1gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.qwen2_1p5b_sft_1gpu_h100_bf16_config()


def qwen2_500m_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Compatibility alias for ``qwen2_500m_peft_1gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.qwen2_500m_peft_1gpu_h100_bf16_config(peft_scheme=peft_scheme)


def qwen2_500m_pretrain_config() -> ConfigContainer:
    """Compatibility alias for ``qwen2_500m_pretrain_1gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.qwen2_500m_pretrain_1gpu_h100_bf16_config()


def qwen2_500m_sft_config() -> ConfigContainer:
    """Compatibility alias for ``qwen2_500m_sft_1gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.qwen2_500m_sft_1gpu_h100_bf16_config()


def qwen2_72b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Compatibility alias for ``qwen2_72b_peft_8gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.qwen2_72b_peft_8gpu_h100_bf16_config(peft_scheme=peft_scheme)


def qwen2_72b_pretrain_config() -> ConfigContainer:
    """Compatibility alias for ``qwen2_72b_pretrain_32gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.qwen2_72b_pretrain_32gpu_h100_bf16_config()


def qwen2_72b_sft_config() -> ConfigContainer:
    """Compatibility alias for ``qwen2_72b_sft_32gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.qwen2_72b_sft_32gpu_h100_bf16_config()


def qwen2_7b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Compatibility alias for ``qwen2_7b_peft_1gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.qwen2_7b_peft_1gpu_h100_bf16_config(peft_scheme=peft_scheme)


def qwen2_7b_pretrain_config() -> ConfigContainer:
    """Compatibility alias for ``qwen2_7b_pretrain_2gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.qwen2_7b_pretrain_2gpu_h100_bf16_config()


def qwen2_7b_sft_config() -> ConfigContainer:
    """Compatibility alias for ``qwen2_7b_sft_2gpu_h100_bf16_config``."""
    _h100_module.AutoBridge = AutoBridge
    return _h100_module.qwen2_7b_sft_2gpu_h100_bf16_config()


__all__ = [
    "qwen25_14b_peft_config",
    "qwen25_14b_pretrain_config",
    "qwen25_14b_sft_config",
    "qwen25_1p5b_peft_config",
    "qwen25_1p5b_pretrain_config",
    "qwen25_1p5b_sft_config",
    "qwen25_32b_peft_config",
    "qwen25_32b_pretrain_config",
    "qwen25_32b_sft_config",
    "qwen25_500m_peft_config",
    "qwen25_500m_pretrain_config",
    "qwen25_500m_sft_config",
    "qwen25_72b_peft_config",
    "qwen25_72b_pretrain_config",
    "qwen25_72b_sft_config",
    "qwen25_7b_peft_config",
    "qwen25_7b_pretrain_config",
    "qwen25_7b_sft_config",
    "qwen2_1p5b_peft_config",
    "qwen2_1p5b_pretrain_config",
    "qwen2_1p5b_sft_config",
    "qwen2_500m_peft_config",
    "qwen2_500m_pretrain_config",
    "qwen2_500m_sft_config",
    "qwen2_72b_peft_config",
    "qwen2_72b_pretrain_config",
    "qwen2_72b_sft_config",
    "qwen2_7b_peft_config",
    "qwen2_7b_pretrain_config",
    "qwen2_7b_sft_config",
]
