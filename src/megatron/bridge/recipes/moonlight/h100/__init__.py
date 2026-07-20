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

from megatron.bridge.recipes.moonlight.h100.moonlight_16b import *  # noqa: F403


__all__ = [
    "moonlight_16b_peft_2gpu_h100_bf16_config",
    "moonlight_16b_peft_4gpu_h100_bf16_config",
    "moonlight_16b_pretrain_16gpu_h100_bf16_config",
    "moonlight_16b_pretrain_8gpu_h100_bf16_config",
    "moonlight_16b_sft_8gpu_h100_bf16_8k_config",
    "moonlight_16b_sft_8gpu_h100_bf16_config",
    "moonlight_16b_sft_8gpu_h100_bf16_tp1_config",
]
