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

from megatron.bridge.recipes.gpt_oss.h100.gpt_oss import *  # noqa: F403


__all__ = [
    "gpt_oss_120b_peft_8gpu_h100_bf16_config",
    "gpt_oss_120b_pretrain_64gpu_h100_bf16_config",
    "gpt_oss_120b_sft_32gpu_h100_bf16_config",
    "gpt_oss_20b_peft_1gpu_h100_bf16_config",
    "gpt_oss_20b_peft_1gpu_h100_fp8cs_config",
    "gpt_oss_20b_peft_1gpu_h100_fp8mx_config",
    "gpt_oss_20b_pretrain_16gpu_h100_bf16_config",
    "gpt_oss_20b_pretrain_16gpu_h100_fp8cs_config",
    "gpt_oss_20b_pretrain_16gpu_h100_fp8mx_config",
    "gpt_oss_20b_sft_8gpu_h100_bf16_config",
    "gpt_oss_20b_sft_8gpu_h100_bf16_openmathinstruct2_thinking_packed_config",
    "gpt_oss_20b_sft_8gpu_h100_fp8cs_config",
    "gpt_oss_20b_sft_8gpu_h100_fp8mx_config",
]
