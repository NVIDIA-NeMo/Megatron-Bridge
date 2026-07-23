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

from megatron.bridge.recipes.llama.h100.llama2 import *  # noqa: F403
from megatron.bridge.recipes.llama.h100.llama3 import *  # noqa: F403


__all__ = [
    "llama2_7b_pretrain_2gpu_h100_bf16_config",
    "llama31_405b_peft_32gpu_h100_bf16_config",
    "llama31_405b_pretrain_256gpu_h100_bf16_config",
    "llama31_405b_pretrain_256gpu_h100_bf16_deterministic_config",
    "llama31_405b_sft_128gpu_h100_bf16_config",
    "llama31_70b_peft_8gpu_h100_bf16_config",
    "llama31_70b_pretrain_32gpu_h100_bf16_config",
    "llama31_70b_sft_32gpu_h100_bf16_config",
    "llama31_8b_peft_1gpu_h100_bf16_config",
    "llama31_8b_pretrain_2gpu_h100_bf16_config",
    "llama31_8b_sft_2gpu_h100_bf16_config",
    "llama32_1b_peft_1gpu_h100_bf16_config",
    "llama32_1b_pretrain_1gpu_h100_bf16_config",
    "llama32_1b_sft_1gpu_h100_bf16_config",
    "llama32_3b_peft_1gpu_h100_bf16_config",
    "llama32_3b_pretrain_1gpu_h100_bf16_config",
    "llama32_3b_sft_1gpu_h100_bf16_config",
    "llama3_70b_peft_8gpu_h100_bf16_config",
    "llama3_70b_pretrain_256gpu_h100_bf16_64k_config",
    "llama3_70b_pretrain_32gpu_h100_bf16_16k_config",
    "llama3_70b_pretrain_32gpu_h100_bf16_config",
    "llama3_70b_pretrain_32gpu_h100_bf16_deterministic_config",
    "llama3_70b_sft_32gpu_h100_bf16_config",
    "llama3_8b_peft_1gpu_h100_bf16_config",
    "llama3_8b_pretrain_16gpu_h100_bf16_16k_config",
    "llama3_8b_pretrain_2gpu_h100_bf16_config",
    "llama3_8b_pretrain_2gpu_h100_fp8cs_config",
    "llama3_8b_pretrain_2gpu_h100_fp8mx_config",
    "llama3_8b_pretrain_2gpu_h100_nvfp4_config",
    "llama3_8b_pretrain_32gpu_h100_bf16_64k_config",
    "llama3_8b_pretrain_64gpu_h100_bf16_128k_config",
    "llama3_8b_sft_2gpu_h100_bf16_config",
]
