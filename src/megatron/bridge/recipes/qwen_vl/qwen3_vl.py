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

from megatron.bridge.recipes.qwen_vl.h100.qwen3_vl import (
    qwen3_vl_8b_peft_1gpu_h100_bf16_config as qwen3_vl_8b_peft_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen3_vl import (
    qwen3_vl_8b_peft_1gpu_h100_bf16_energon_config as qwen3_vl_8b_peft_energon_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen3_vl import (
    qwen3_vl_8b_pretrain_4gpu_h100_bf16_mock_config as qwen3_vl_8b_pretrain_mock_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen3_vl import (
    qwen3_vl_8b_sft_2gpu_h100_bf16_config as qwen3_vl_8b_sft_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen3_vl import (
    qwen3_vl_30b_a3b_peft_4gpu_h100_bf16_config as qwen3_vl_30b_a3b_peft_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen3_vl import (
    qwen3_vl_30b_a3b_pretrain_8gpu_h100_bf16_mock_config as qwen3_vl_30b_a3b_pretrain_mock_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen3_vl import (
    qwen3_vl_30b_a3b_sft_8gpu_h100_bf16_config as qwen3_vl_30b_a3b_sft_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen3_vl import (
    qwen3_vl_235b_a22b_peft_16gpu_h100_bf16_config as qwen3_vl_235b_a22b_peft_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen3_vl import (
    qwen3_vl_235b_a22b_pretrain_256gpu_h100_bf16_mock_config as qwen3_vl_235b_a22b_pretrain_mock_config,
)
from megatron.bridge.recipes.qwen_vl.h100.qwen3_vl import (
    qwen3_vl_235b_a22b_sft_32gpu_h100_bf16_config as qwen3_vl_235b_a22b_sft_config,
)


__all__ = [
    "qwen3_vl_235b_a22b_peft_config",
    "qwen3_vl_235b_a22b_pretrain_mock_config",
    "qwen3_vl_235b_a22b_sft_config",
    "qwen3_vl_30b_a3b_peft_config",
    "qwen3_vl_30b_a3b_pretrain_mock_config",
    "qwen3_vl_30b_a3b_sft_config",
    "qwen3_vl_8b_peft_config",
    "qwen3_vl_8b_peft_energon_config",
    "qwen3_vl_8b_pretrain_mock_config",
    "qwen3_vl_8b_sft_config",
]
