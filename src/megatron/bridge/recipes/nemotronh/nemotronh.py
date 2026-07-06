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

from megatron.bridge.recipes.nemotronh.h100.nemotronh import (
    nemotronh_4b_peft_1gpu_h100_bf16_config as nemotronh_4b_peft_config,
)
from megatron.bridge.recipes.nemotronh.h100.nemotronh import (
    nemotronh_4b_pretrain_1gpu_h100_bf16_config as nemotronh_4b_pretrain_config,
)
from megatron.bridge.recipes.nemotronh.h100.nemotronh import (
    nemotronh_4b_sft_1gpu_h100_bf16_config as nemotronh_4b_sft_config,
)
from megatron.bridge.recipes.nemotronh.h100.nemotronh import (
    nemotronh_8b_peft_1gpu_h100_bf16_config as nemotronh_8b_peft_config,
)
from megatron.bridge.recipes.nemotronh.h100.nemotronh import (
    nemotronh_8b_pretrain_2gpu_h100_bf16_config as nemotronh_8b_pretrain_config,
)
from megatron.bridge.recipes.nemotronh.h100.nemotronh import (
    nemotronh_8b_sft_2gpu_h100_bf16_config as nemotronh_8b_sft_config,
)
from megatron.bridge.recipes.nemotronh.h100.nemotronh import (
    nemotronh_47b_peft_4gpu_h100_bf16_config as nemotronh_47b_peft_config,
)
from megatron.bridge.recipes.nemotronh.h100.nemotronh import (
    nemotronh_47b_pretrain_8gpu_h100_bf16_config as nemotronh_47b_pretrain_config,
)
from megatron.bridge.recipes.nemotronh.h100.nemotronh import (
    nemotronh_47b_sft_16gpu_h100_bf16_config as nemotronh_47b_sft_config,
)
from megatron.bridge.recipes.nemotronh.h100.nemotronh import (
    nemotronh_56b_peft_4gpu_h100_bf16_config as nemotronh_56b_peft_config,
)
from megatron.bridge.recipes.nemotronh.h100.nemotronh import (
    nemotronh_56b_pretrain_8gpu_h100_bf16_config as nemotronh_56b_pretrain_config,
)
from megatron.bridge.recipes.nemotronh.h100.nemotronh import (
    nemotronh_56b_sft_8gpu_h100_bf16_config as nemotronh_56b_sft_config,
)


__all__ = [
    "nemotronh_4b_pretrain_config",
    "nemotronh_8b_pretrain_config",
    "nemotronh_47b_pretrain_config",
    "nemotronh_56b_pretrain_config",
    "nemotronh_4b_sft_config",
    "nemotronh_8b_sft_config",
    "nemotronh_47b_sft_config",
    "nemotronh_56b_sft_config",
    "nemotronh_4b_peft_config",
    "nemotronh_8b_peft_config",
    "nemotronh_47b_peft_config",
    "nemotronh_56b_peft_config",
]
