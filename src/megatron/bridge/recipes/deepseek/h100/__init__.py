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

from megatron.bridge.recipes.deepseek.h100.deepseek_v2 import *  # noqa: F403
from megatron.bridge.recipes.deepseek.h100.deepseek_v3 import *  # noqa: F403
from megatron.bridge.recipes.deepseek.h100.deepseek_v4 import *  # noqa: F403


__all__ = [
    "deepseek_v2_lite_pretrain_8gpu_h100_bf16_config",
    "deepseek_v2_pretrain_128gpu_h100_bf16_config",
    "deepseek_v3_pretrain_1024gpu_h100_bf16_config",
    "deepseek_v3_pretrain_256gpu_h100_bf16_32nodes_config",
    "deepseek_v4_flash_no_mtp_sft_32gpu_h100_bf16_config",
    "deepseek_v4_flash_pretrain_32gpu_h100_bf16_config",
    "deepseek_v4_flash_pretrain_32gpu_h100_bf16_muon_config",
    "deepseek_v4_flash_pretrain_32gpu_h100_fp8mx_config",
    "deepseek_v4_flash_sft_32gpu_h100_bf16_config",
]
