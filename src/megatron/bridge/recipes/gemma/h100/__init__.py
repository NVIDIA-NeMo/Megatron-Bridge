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

from megatron.bridge.recipes.gemma.h100.gemma2 import *  # noqa: F403
from megatron.bridge.recipes.gemma.h100.gemma3 import *  # noqa: F403
from megatron.bridge.recipes.gemma.h100.gemma4 import *  # noqa: F403


__all__ = [
    "gemma2_27b_peft_4gpu_h100_bf16_config",
    "gemma2_27b_pretrain_16gpu_h100_bf16_config",
    "gemma2_27b_sft_16gpu_h100_bf16_config",
    "gemma2_2b_peft_1gpu_h100_bf16_config",
    "gemma2_2b_pretrain_2gpu_h100_bf16_config",
    "gemma2_2b_sft_1gpu_h100_bf16_config",
    "gemma2_9b_peft_1gpu_h100_bf16_config",
    "gemma2_9b_pretrain_8gpu_h100_bf16_config",
    "gemma2_9b_sft_4gpu_h100_bf16_config",
    "gemma3_1b_peft_1gpu_h100_bf16_config",
    "gemma3_1b_pretrain_1gpu_h100_bf16_config",
    "gemma3_1b_sft_1gpu_h100_bf16_config",
    "gemma4_e4b_pretrain_2gpu_h100_bf16_config",
]
