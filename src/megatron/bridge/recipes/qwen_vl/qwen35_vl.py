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

from megatron.bridge.recipes.qwen_vl.h100.qwen35_vl import (
    qwen35_vl_2b_peft_1gpu_h100_bf16_config as qwen35_vl_2b_peft_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen35_vl import (
    qwen35_vl_2b_sft_1gpu_h100_bf16_config as qwen35_vl_2b_sft_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen35_vl import (
    qwen35_vl_4b_peft_1gpu_h100_bf16_config as qwen35_vl_4b_peft_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen35_vl import (
    qwen35_vl_4b_sft_2gpu_h100_bf16_config as qwen35_vl_4b_sft_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen35_vl import (
    qwen35_vl_9b_peft_1gpu_h100_bf16_config as qwen35_vl_9b_peft_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen35_vl import (
    qwen35_vl_9b_pretrain_4gpu_h100_bf16_mock_config as qwen35_vl_9b_pretrain_mock_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen35_vl import (
    qwen35_vl_9b_sft_4gpu_h100_bf16_config as qwen35_vl_9b_sft_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen35_vl import (
    qwen35_vl_27b_peft_2gpu_h100_bf16_config as qwen35_vl_27b_peft_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen35_vl import (
    qwen35_vl_27b_pretrain_16gpu_h100_bf16_mock_config as qwen35_vl_27b_pretrain_mock_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen35_vl import (
    qwen35_vl_27b_sft_16gpu_h100_bf16_config as qwen35_vl_27b_sft_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen35_vl import (
    qwen35_vl_35b_a3b_peft_4gpu_h100_bf16_config as qwen35_vl_35b_a3b_peft_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen35_vl import (
    qwen35_vl_35b_a3b_pretrain_8gpu_h100_bf16_mock_config as qwen35_vl_35b_a3b_pretrain_mock_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen35_vl import (
    qwen35_vl_35b_a3b_sft_2gpu_h100_bf16_fsdp_config as qwen35_vl_35b_a3b_fsdp_sft_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen35_vl import (
    qwen35_vl_35b_a3b_sft_16gpu_h100_bf16_config as qwen35_vl_35b_a3b_sft_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen35_vl import (
    qwen35_vl_122b_a10b_peft_8gpu_h100_bf16_config as qwen35_vl_122b_a10b_peft_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen35_vl import (
    qwen35_vl_122b_a10b_pretrain_128gpu_h100_bf16_mock_config as qwen35_vl_122b_a10b_pretrain_mock_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen35_vl import (
    qwen35_vl_122b_a10b_sft_48gpu_h100_bf16_config as qwen35_vl_122b_a10b_sft_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen35_vl import (
    qwen35_vl_397b_a17b_peft_32gpu_h100_bf16_config as qwen35_vl_397b_a17b_peft_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen35_vl import (
    qwen35_vl_397b_a17b_pretrain_512gpu_h100_bf16_mock_config as qwen35_vl_397b_a17b_pretrain_mock_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen35_vl import (
    qwen35_vl_397b_a17b_sft_128gpu_h100_bf16_config as qwen35_vl_397b_a17b_sft_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen35_vl import (
    qwen35_vl_800m_peft_1gpu_h100_bf16_config as qwen35_vl_800m_peft_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen35_vl import (
    qwen35_vl_800m_sft_1gpu_h100_bf16_config as qwen35_vl_800m_sft_config,
)


__all__ = [
    "qwen35_vl_122b_a10b_peft_config",
    "qwen35_vl_122b_a10b_pretrain_mock_config",
    "qwen35_vl_122b_a10b_sft_config",
    "qwen35_vl_27b_peft_config",
    "qwen35_vl_27b_pretrain_mock_config",
    "qwen35_vl_27b_sft_config",
    "qwen35_vl_2b_peft_config",
    "qwen35_vl_2b_sft_config",
    "qwen35_vl_35b_a3b_fsdp_sft_config",
    "qwen35_vl_35b_a3b_peft_config",
    "qwen35_vl_35b_a3b_pretrain_mock_config",
    "qwen35_vl_35b_a3b_sft_config",
    "qwen35_vl_397b_a17b_peft_config",
    "qwen35_vl_397b_a17b_pretrain_mock_config",
    "qwen35_vl_397b_a17b_sft_config",
    "qwen35_vl_4b_peft_config",
    "qwen35_vl_4b_sft_config",
    "qwen35_vl_800m_peft_config",
    "qwen35_vl_800m_sft_config",
    "qwen35_vl_9b_peft_config",
    "qwen35_vl_9b_pretrain_mock_config",
    "qwen35_vl_9b_sft_config",
]
