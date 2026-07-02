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

from megatron.bridge.recipes.gpt.h100 import vanilla_gpt as _h100_module
from megatron.bridge.training.config import ConfigContainer


def vanilla_gpt_pretrain_config() -> ConfigContainer:
    """Compatibility alias for ``vanilla_gpt_pretrain_1gpu_h100_bf16_config``."""
    return _h100_module.vanilla_gpt_pretrain_1gpu_h100_bf16_config()


__all__ = [
    "vanilla_gpt_pretrain_config",
]
