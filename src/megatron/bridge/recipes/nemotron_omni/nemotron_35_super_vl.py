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

"""Hardware-agnostic alias for Nemotron 3.5 Super VL training."""

from megatron.bridge.recipes.nemotron_omni.h100.nemotron_35_super_vl import (
    NEMOTRON_35_SUPER_VL_HF_MODEL_ID,
)
from megatron.bridge.recipes.nemotron_omni.h100.nemotron_35_super_vl import (
    nemotron_35_super_vl_sft_64gpu_h100_bf16_config as nemotron_35_super_vl_sft_config,
)


__all__ = [
    "nemotron_35_super_vl_sft_config",
    "NEMOTRON_35_SUPER_VL_HF_MODEL_ID",
]
