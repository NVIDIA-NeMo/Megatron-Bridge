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

from megatron.bridge.recipes.gpt_oss.h100.gpt_oss import (
    _enable_gpt_oss_blackwell_mxfp8,
    _enable_gpt_oss_hopper_fp8_current_scaling,
)
from megatron.bridge.recipes.gpt_oss.h100.gpt_oss import (
    gpt_oss_20b_peft_1gpu_h100_bf16_config as gpt_oss_20b_peft_config,
)
from megatron.bridge.recipes.gpt_oss.h100.gpt_oss import (
    gpt_oss_20b_peft_1gpu_h100_fp8cs_config as gpt_oss_20b_peft_fp8_current_scaling_config,
)
from megatron.bridge.recipes.gpt_oss.h100.gpt_oss import (
    gpt_oss_20b_peft_1gpu_h100_fp8mx_config as gpt_oss_20b_peft_mxfp8_config,
)
from megatron.bridge.recipes.gpt_oss.h100.gpt_oss import (
    gpt_oss_20b_pretrain_16gpu_h100_bf16_config as gpt_oss_20b_pretrain_config,
)
from megatron.bridge.recipes.gpt_oss.h100.gpt_oss import (
    gpt_oss_20b_pretrain_16gpu_h100_fp8cs_config as gpt_oss_20b_pretrain_fp8_current_scaling_config,
)
from megatron.bridge.recipes.gpt_oss.h100.gpt_oss import (
    gpt_oss_20b_pretrain_16gpu_h100_fp8mx_config as gpt_oss_20b_pretrain_mxfp8_config,
)
from megatron.bridge.recipes.gpt_oss.h100.gpt_oss import (
    gpt_oss_20b_sft_8gpu_h100_bf16_config as gpt_oss_20b_sft_config,
)
from megatron.bridge.recipes.gpt_oss.h100.gpt_oss import (
    gpt_oss_20b_sft_8gpu_h100_bf16_openmathinstruct2_thinking_packed_config as gpt_oss_20b_sft_openmathinstruct2_thinking_packed_config,
)
from megatron.bridge.recipes.gpt_oss.h100.gpt_oss import (
    gpt_oss_20b_sft_8gpu_h100_fp8cs_config as gpt_oss_20b_sft_fp8_current_scaling_config,
)
from megatron.bridge.recipes.gpt_oss.h100.gpt_oss import (
    gpt_oss_20b_sft_8gpu_h100_fp8mx_config as gpt_oss_20b_sft_mxfp8_config,
)
from megatron.bridge.recipes.gpt_oss.h100.gpt_oss import (
    gpt_oss_120b_peft_8gpu_h100_bf16_config as gpt_oss_120b_peft_config,
)
from megatron.bridge.recipes.gpt_oss.h100.gpt_oss import (
    gpt_oss_120b_pretrain_64gpu_h100_bf16_config as gpt_oss_120b_pretrain_config,
)
from megatron.bridge.recipes.gpt_oss.h100.gpt_oss import (
    gpt_oss_120b_sft_32gpu_h100_bf16_config as gpt_oss_120b_sft_config,
)


__all__ = [
    "gpt_oss_120b_peft_config",
    "gpt_oss_120b_pretrain_config",
    "gpt_oss_120b_sft_config",
    "gpt_oss_20b_peft_config",
    "gpt_oss_20b_peft_fp8_current_scaling_config",
    "gpt_oss_20b_peft_mxfp8_config",
    "gpt_oss_20b_pretrain_config",
    "gpt_oss_20b_pretrain_fp8_current_scaling_config",
    "gpt_oss_20b_pretrain_mxfp8_config",
    "gpt_oss_20b_sft_config",
    "gpt_oss_20b_sft_fp8_current_scaling_config",
    "gpt_oss_20b_sft_mxfp8_config",
    "gpt_oss_20b_sft_openmathinstruct2_thinking_packed_config",
]
