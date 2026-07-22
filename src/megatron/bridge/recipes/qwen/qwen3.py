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

from megatron.bridge.recipes.qwen.h100.qwen3 import (
    qwen3_1p7b_peft_1gpu_h100_bf16_config as qwen3_1p7b_peft_config,
)
from megatron.bridge.recipes.qwen.h100.qwen3 import (
    qwen3_1p7b_pretrain_1gpu_h100_bf16_config as qwen3_1p7b_pretrain_config,
)
from megatron.bridge.recipes.qwen.h100.qwen3 import (
    qwen3_1p7b_sft_1gpu_h100_bf16_config as qwen3_1p7b_sft_config,
)
from megatron.bridge.recipes.qwen.h100.qwen3 import (
    qwen3_4b_peft_1gpu_h100_bf16_config as qwen3_4b_peft_config,
)
from megatron.bridge.recipes.qwen.h100.qwen3 import (
    qwen3_4b_pretrain_2gpu_h100_bf16_config as qwen3_4b_pretrain_config,
)
from megatron.bridge.recipes.qwen.h100.qwen3 import (
    qwen3_4b_sft_2gpu_h100_bf16_config as qwen3_4b_sft_config,
)
from megatron.bridge.recipes.qwen.h100.qwen3 import (
    qwen3_8b_peft_1gpu_h100_bf16_config as qwen3_8b_peft_config,
)
from megatron.bridge.recipes.qwen.h100.qwen3 import (
    qwen3_8b_pretrain_16gpu_h100_bf16_config as qwen3_8b_pretrain_config,
)
from megatron.bridge.recipes.qwen.h100.qwen3 import (
    qwen3_8b_sft_4gpu_h100_bf16_config as qwen3_8b_sft_config,
)
from megatron.bridge.recipes.qwen.h100.qwen3 import (
    qwen3_8b_sft_8gpu_h100_bf16_32k_config as qwen3_8b_sft_32k_config,
)
from megatron.bridge.recipes.qwen.h100.qwen3 import (
    qwen3_14b_peft_1gpu_h100_bf16_config as qwen3_14b_peft_config,
)
from megatron.bridge.recipes.qwen.h100.qwen3 import (
    qwen3_14b_pretrain_8gpu_h100_bf16_config as qwen3_14b_pretrain_config,
)
from megatron.bridge.recipes.qwen.h100.qwen3 import (
    qwen3_14b_sft_8gpu_h100_bf16_config as qwen3_14b_sft_config,
)
from megatron.bridge.recipes.qwen.h100.qwen3 import (
    qwen3_32b_peft_1gpu_h100_bf16_config as qwen3_32b_peft_config,
)
from megatron.bridge.recipes.qwen.h100.qwen3 import (
    qwen3_32b_pretrain_16gpu_h100_bf16_config as qwen3_32b_pretrain_config,
)
from megatron.bridge.recipes.qwen.h100.qwen3 import (
    qwen3_32b_sft_16gpu_h100_bf16_config as qwen3_32b_sft_config,
)
from megatron.bridge.recipes.qwen.h100.qwen3 import (
    qwen3_600m_peft_1gpu_h100_bf16_config as qwen3_600m_peft_config,
)
from megatron.bridge.recipes.qwen.h100.qwen3 import (
    qwen3_600m_pretrain_1gpu_h100_bf16_config as qwen3_600m_pretrain_config,
)
from megatron.bridge.recipes.qwen.h100.qwen3 import (
    qwen3_600m_sft_1gpu_h100_bf16_config as qwen3_600m_sft_config,
)
from megatron.bridge.recipes.qwen.h100.qwen3 import (
    qwen3_600m_sft_8gpu_h100_bf16_128k_config as qwen3_600m_sft_128k_config,
)
from megatron.bridge.recipes.qwen.h100.qwen3 import (
    qwen3_600m_sft_8gpu_h100_bf16_yarn_128k_config as qwen3_600m_sft_yarn_128k_config,
)


__all__ = [
    "qwen3_14b_peft_config",
    "qwen3_14b_pretrain_config",
    "qwen3_14b_sft_config",
    "qwen3_1p7b_peft_config",
    "qwen3_1p7b_pretrain_config",
    "qwen3_1p7b_sft_config",
    "qwen3_32b_peft_config",
    "qwen3_32b_pretrain_config",
    "qwen3_32b_sft_config",
    "qwen3_4b_peft_config",
    "qwen3_4b_pretrain_config",
    "qwen3_4b_sft_config",
    "qwen3_600m_peft_config",
    "qwen3_600m_pretrain_config",
    "qwen3_600m_sft_128k_config",
    "qwen3_600m_sft_config",
    "qwen3_600m_sft_yarn_128k_config",
    "qwen3_8b_peft_config",
    "qwen3_8b_pretrain_config",
    "qwen3_8b_sft_32k_config",
    "qwen3_8b_sft_config",
]
