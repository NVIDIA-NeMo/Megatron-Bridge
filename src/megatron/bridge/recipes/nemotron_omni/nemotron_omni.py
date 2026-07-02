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

from megatron.bridge.recipes.nemotron_omni.h100.nemotron_omni import (
    _DEFAULT_HF_PATH,
)
from megatron.bridge.recipes.nemotron_omni.h100.nemotron_omni import (
    nemotron_omni_cord_v2_peft_4gpu_h100_bf16_config as nemotron_omni_cord_v2_peft_config,
)
from megatron.bridge.recipes.nemotron_omni.h100.nemotron_omni import (
    nemotron_omni_cord_v2_sft_4gpu_h100_bf16_config as nemotron_omni_cord_v2_sft_config,
)
from megatron.bridge.recipes.nemotron_omni.h100.nemotron_omni import (
    nemotron_omni_valor32k_peft_4gpu_h100_bf16_config as nemotron_omni_valor32k_peft_config,
)
from megatron.bridge.recipes.nemotron_omni.h100.nemotron_omni import (
    nemotron_omni_valor32k_sft_4gpu_h100_bf16_config as nemotron_omni_valor32k_sft_config,
)


__all__ = [
    "nemotron_omni_cord_v2_peft_config",
    "nemotron_omni_cord_v2_sft_config",
    "nemotron_omni_valor32k_peft_config",
    "nemotron_omni_valor32k_sft_config",
]
