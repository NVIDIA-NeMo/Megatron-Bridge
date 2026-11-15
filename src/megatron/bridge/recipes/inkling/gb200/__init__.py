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

"""GB200 recipes for Inkling2."""

from megatron.bridge.recipes.inkling.gb200.inkling2 import (
    inkling2_small_peft_16gpu_gb200_bf16_config,
    inkling2_small_peft_32gpu_gb200_bf16_config,
    inkling2_small_peft_32gpu_gb200_bf16_multimodal_config,
    inkling2_small_pretrain_128gpu_gb200_bf16_config,
    inkling2_small_sft_96gpu_gb200_bf16_config,
    inkling2_small_sft_128gpu_gb200_bf16_config,
    inkling2_small_sft_128gpu_gb200_bf16_long_context_config,
    inkling2_small_sft_128gpu_gb200_bf16_multimodal_config,
)


__all__ = [
    "inkling2_small_peft_16gpu_gb200_bf16_config",
    "inkling2_small_peft_32gpu_gb200_bf16_config",
    "inkling2_small_peft_32gpu_gb200_bf16_multimodal_config",
    "inkling2_small_pretrain_128gpu_gb200_bf16_config",
    "inkling2_small_sft_128gpu_gb200_bf16_config",
    "inkling2_small_sft_128gpu_gb200_bf16_long_context_config",
    "inkling2_small_sft_128gpu_gb200_bf16_multimodal_config",
    "inkling2_small_sft_96gpu_gb200_bf16_config",
]
