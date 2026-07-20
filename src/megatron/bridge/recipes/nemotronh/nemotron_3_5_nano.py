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
"""Compatibility aliases for concise Nemotron 3.5 Nano recipe names."""

from megatron.bridge.recipes.nemotronh.h100.nemotron_3_5_nano import NEMOTRON_3_5_NANO_HF_MODEL_ID
from megatron.bridge.recipes.nemotronh.h100.nemotron_3_5_nano import (
    nemotron_3_5_nano_peft_8gpu_h100_bf16_config as nemotron_3_5_nano_peft_config,
)
from megatron.bridge.recipes.nemotronh.h100.nemotron_3_5_nano import (
    nemotron_3_5_nano_peft_8gpu_h100_bf16_openmathinstruct2_packed_config as nemotron_3_5_nano_peft_openmathinstruct2_config,
)
from megatron.bridge.recipes.nemotronh.h100.nemotron_3_5_nano import (
    nemotron_3_5_nano_pretrain_16gpu_h100_bf16_config as nemotron_3_5_nano_pretrain_config,
)
from megatron.bridge.recipes.nemotronh.h100.nemotron_3_5_nano import (
    nemotron_3_5_nano_sft_16gpu_h100_bf16_config as nemotron_3_5_nano_sft_config,
)
from megatron.bridge.recipes.nemotronh.h100.nemotron_3_5_nano import (
    nemotron_3_5_nano_sft_16gpu_h100_bf16_openmathinstruct2_packed_config as nemotron_3_5_nano_sft_openmathinstruct2_config,
)


__all__ = [
    "NEMOTRON_3_5_NANO_HF_MODEL_ID",
    "nemotron_3_5_nano_peft_config",
    "nemotron_3_5_nano_peft_openmathinstruct2_config",
    "nemotron_3_5_nano_pretrain_config",
    "nemotron_3_5_nano_sft_config",
    "nemotron_3_5_nano_sft_openmathinstruct2_config",
]
