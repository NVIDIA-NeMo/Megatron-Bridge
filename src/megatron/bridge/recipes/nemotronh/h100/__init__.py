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

from megatron.bridge.recipes.nemotronh.h100.nemotron_3_nano import *  # noqa: F403
from megatron.bridge.recipes.nemotronh.h100.nemotron_3_nano_4b import *  # noqa: F403
from megatron.bridge.recipes.nemotronh.h100.nemotron_3_super import *  # noqa: F403
from megatron.bridge.recipes.nemotronh.h100.nemotron_3_ultra import *  # noqa: F403
from megatron.bridge.recipes.nemotronh.h100.nemotron_nano_v2 import *  # noqa: F403
from megatron.bridge.recipes.nemotronh.h100.nemotronh import *  # noqa: F403


__all__ = [
    "nemotron_3_nano_4b_peft_8gpu_h100_bf16_config",
    "nemotron_3_nano_4b_pretrain_8gpu_h100_bf16_config",
    "nemotron_3_nano_4b_sft_8gpu_h100_bf16_32k_config",
    "nemotron_3_nano_4b_sft_8gpu_h100_bf16_config",
    "nemotron_3_nano_mtp_peft_8gpu_h100_bf16_config",
    "nemotron_3_nano_mtp_pretrain_8gpu_h100_bf16_config",
    "nemotron_3_nano_mtp_sft_8gpu_h100_bf16_config",
    "nemotron_3_nano_peft_8gpu_h100_bf16_config",
    "nemotron_3_nano_pretrain_8gpu_h100_bf16_config",
    "nemotron_3_nano_sft_8gpu_h100_bf16_config",
    "nemotron_3_super_peft_1gpu_h100_bf16_config",
    "nemotron_3_super_pretrain_8gpu_h100_bf16_config",
    "nemotron_3_super_sft_8gpu_h100_bf16_config",
    "nemotron_3_ultra_peft_32gpu_h100_bf16_openmathinstruct2_packed_config",
    "nemotron_3_ultra_pretrain_24gpu_h100_bf16_config",
    "nemotron_3_ultra_sft_192gpu_h100_bf16_openmathinstruct2_packed_config",
    "nemotron_nano_12b_v2_peft_1gpu_h100_bf16_config",
    "nemotron_nano_12b_v2_pretrain_4gpu_h100_bf16_config",
    "nemotron_nano_12b_v2_sft_4gpu_h100_bf16_config",
    "nemotron_nano_9b_v2_peft_1gpu_h100_bf16_config",
    "nemotron_nano_9b_v2_pretrain_2gpu_h100_bf16_config",
    "nemotron_nano_9b_v2_sft_2gpu_h100_bf16_config",
    "nemotronh_47b_peft_4gpu_h100_bf16_config",
    "nemotronh_47b_pretrain_8gpu_h100_bf16_config",
    "nemotronh_47b_sft_16gpu_h100_bf16_config",
    "nemotronh_4b_peft_1gpu_h100_bf16_config",
    "nemotronh_4b_pretrain_1gpu_h100_bf16_config",
    "nemotronh_4b_sft_1gpu_h100_bf16_config",
    "nemotronh_56b_peft_4gpu_h100_bf16_config",
    "nemotronh_56b_pretrain_8gpu_h100_bf16_config",
    "nemotronh_56b_sft_8gpu_h100_bf16_config",
    "nemotronh_8b_peft_1gpu_h100_bf16_config",
    "nemotronh_8b_pretrain_2gpu_h100_bf16_config",
    "nemotronh_8b_sft_2gpu_h100_bf16_config",
]
