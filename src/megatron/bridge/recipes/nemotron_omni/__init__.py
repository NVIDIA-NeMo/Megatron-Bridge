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

from .gb200 import (
    nemotron_35_super_vl_peft_16gpu_gb200_bf16_config,
    nemotron_35_super_vl_pretrain_64gpu_gb200_bf16_config,
    nemotron_35_super_vl_sft_64gpu_gb200_bf16_config,
)
from .nemotron_35_super_vl import (
    NEMOTRON_35_SUPER_VL_HF_MODEL_ID,
    nemotron_35_super_vl_peft_config,
    nemotron_35_super_vl_pretrain_config,
    nemotron_35_super_vl_sft_config,
)
from .nemotron_omni import (
    nemotron_omni_cord_v2_peft_config,
    nemotron_omni_cord_v2_sft_config,
    nemotron_omni_valor32k_peft_config,
    nemotron_omni_valor32k_sft_config,
)


__all__ = [
    "nemotron_35_super_vl_peft_config",
    "nemotron_35_super_vl_peft_16gpu_gb200_bf16_config",
    "nemotron_35_super_vl_pretrain_config",
    "nemotron_35_super_vl_pretrain_64gpu_gb200_bf16_config",
    "nemotron_35_super_vl_sft_config",
    "nemotron_35_super_vl_sft_64gpu_gb200_bf16_config",
    "NEMOTRON_35_SUPER_VL_HF_MODEL_ID",
    "nemotron_omni_cord_v2_sft_config",
    "nemotron_omni_cord_v2_peft_config",
    "nemotron_omni_valor32k_sft_config",
    "nemotron_omni_valor32k_peft_config",
]
