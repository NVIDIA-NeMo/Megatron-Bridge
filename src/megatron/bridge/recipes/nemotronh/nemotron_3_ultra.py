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

from megatron.bridge.recipes.nemotronh.h100.nemotron_3_ultra import (
    NEMOTRON_3_ULTRA_HF_MODEL_ID,
    NEMOTRON_3_ULTRA_TOKENIZER_NAME,
)
from megatron.bridge.recipes.nemotronh.h100.nemotron_3_ultra import (
    nemotron_3_ultra_peft_32gpu_h100_bf16_openmathinstruct2_packed_config as nemotron_3_ultra_peft_openmathinstruct2_packed_config,
)
from megatron.bridge.recipes.nemotronh.h100.nemotron_3_ultra import (
    nemotron_3_ultra_pretrain_24gpu_h100_bf16_config as nemotron_3_ultra_pretrain_config,
)
from megatron.bridge.recipes.nemotronh.h100.nemotron_3_ultra import (
    nemotron_3_ultra_sft_192gpu_h100_bf16_openmathinstruct2_packed_config as nemotron_3_ultra_sft_openmathinstruct2_packed_config,
)


__all__ = [
    "nemotron_3_ultra_pretrain_config",
    "nemotron_3_ultra_sft_openmathinstruct2_packed_config",
    "nemotron_3_ultra_peft_openmathinstruct2_packed_config",
    "NEMOTRON_3_ULTRA_HF_MODEL_ID",
    "NEMOTRON_3_ULTRA_TOKENIZER_NAME",
]
