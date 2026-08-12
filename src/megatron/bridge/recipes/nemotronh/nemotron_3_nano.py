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
"""Hardware-agnostic aliases for Nemotron 3 Nano and Nemotron 3.5 Lightning recipes."""

from __future__ import annotations

from megatron.bridge.recipes.nemotronh.h100.nemotron_3_nano import (
    nemotron_3_5_lightning_peft_config,
    nemotron_3_5_lightning_pretrain_config,
    nemotron_3_5_lightning_sft_config,
    nemotron_3_5_lightning_sft_openmathinstruct2_packed_config,
)
from megatron.bridge.recipes.nemotronh.h100.nemotron_3_nano import (
    nemotron_3_nano_peft_8gpu_h100_bf16_config as nemotron_3_nano_peft_config,
)
from megatron.bridge.recipes.nemotronh.h100.nemotron_3_nano import (
    nemotron_3_nano_pretrain_8gpu_h100_bf16_config as nemotron_3_nano_pretrain_config,
)
from megatron.bridge.recipes.nemotronh.h100.nemotron_3_nano import (
    nemotron_3_nano_sft_8gpu_h100_bf16_config as nemotron_3_nano_sft_config,
)


__all__ = [
    "nemotron_3_5_lightning_peft_config",
    "nemotron_3_5_lightning_pretrain_config",
    "nemotron_3_5_lightning_sft_config",
    "nemotron_3_5_lightning_sft_openmathinstruct2_packed_config",
    "nemotron_3_nano_pretrain_config",
    "nemotron_3_nano_sft_config",
    "nemotron_3_nano_peft_config",
]
