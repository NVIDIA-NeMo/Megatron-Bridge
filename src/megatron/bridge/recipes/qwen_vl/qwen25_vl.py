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

from megatron.bridge.recipes.qwen_vl.h100.qwen25_vl import (
    qwen25_vl_3b_peft_1gpu_h100_bf16_config as qwen25_vl_3b_peft_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen25_vl import (
    qwen25_vl_3b_sft_1gpu_h100_bf16_config as qwen25_vl_3b_sft_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen25_vl import (
    qwen25_vl_7b_peft_1gpu_h100_bf16_config as qwen25_vl_7b_peft_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen25_vl import (
    qwen25_vl_7b_sft_2gpu_h100_bf16_config as qwen25_vl_7b_sft_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen25_vl import (
    qwen25_vl_32b_peft_1gpu_h100_bf16_config as qwen25_vl_32b_peft_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen25_vl import (
    qwen25_vl_32b_sft_16gpu_h100_bf16_config as qwen25_vl_32b_sft_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen25_vl import (
    qwen25_vl_72b_peft_1gpu_h100_bf16_config as qwen25_vl_72b_peft_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen25_vl import (
    qwen25_vl_72b_sft_32gpu_h100_bf16_config as qwen25_vl_72b_sft_config,
)


__all__ = [
    "qwen25_vl_32b_peft_config",
    "qwen25_vl_32b_sft_config",
    "qwen25_vl_3b_peft_config",
    "qwen25_vl_3b_sft_config",
    "qwen25_vl_72b_peft_config",
    "qwen25_vl_72b_sft_config",
    "qwen25_vl_7b_peft_config",
    "qwen25_vl_7b_sft_config",
]
