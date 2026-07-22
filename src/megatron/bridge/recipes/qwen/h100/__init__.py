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

from megatron.bridge.recipes.qwen.h100.qwen2 import *  # noqa: F403
from megatron.bridge.recipes.qwen.h100.qwen3 import *  # noqa: F403
from megatron.bridge.recipes.qwen.h100.qwen3_moe import *  # noqa: F403
from megatron.bridge.recipes.qwen.h100.qwen3_next import *  # noqa: F403


__all__ = [
    "qwen25_14b_peft_1gpu_h100_bf16_config",
    "qwen25_14b_pretrain_4gpu_h100_bf16_config",
    "qwen25_14b_sft_4gpu_h100_bf16_config",
    "qwen25_1p5b_peft_1gpu_h100_bf16_config",
    "qwen25_1p5b_pretrain_1gpu_h100_bf16_config",
    "qwen25_1p5b_sft_1gpu_h100_bf16_config",
    "qwen25_32b_peft_8gpu_h100_bf16_config",
    "qwen25_32b_pretrain_16gpu_h100_bf16_config",
    "qwen25_32b_sft_16gpu_h100_bf16_config",
    "qwen25_500m_peft_1gpu_h100_bf16_config",
    "qwen25_500m_pretrain_1gpu_h100_bf16_config",
    "qwen25_500m_sft_1gpu_h100_bf16_config",
    "qwen25_72b_peft_8gpu_h100_bf16_config",
    "qwen25_72b_pretrain_32gpu_h100_bf16_config",
    "qwen25_72b_sft_32gpu_h100_bf16_config",
    "qwen25_7b_peft_1gpu_h100_bf16_config",
    "qwen25_7b_pretrain_2gpu_h100_bf16_config",
    "qwen25_7b_sft_2gpu_h100_bf16_config",
    "qwen2_1p5b_peft_1gpu_h100_bf16_config",
    "qwen2_1p5b_pretrain_1gpu_h100_bf16_config",
    "qwen2_1p5b_sft_1gpu_h100_bf16_config",
    "qwen2_500m_peft_1gpu_h100_bf16_config",
    "qwen2_500m_pretrain_1gpu_h100_bf16_config",
    "qwen2_500m_sft_1gpu_h100_bf16_config",
    "qwen2_72b_peft_8gpu_h100_bf16_config",
    "qwen2_72b_pretrain_32gpu_h100_bf16_config",
    "qwen2_72b_sft_32gpu_h100_bf16_config",
    "qwen2_7b_peft_1gpu_h100_bf16_config",
    "qwen2_7b_pretrain_2gpu_h100_bf16_config",
    "qwen2_7b_sft_2gpu_h100_bf16_config",
    "qwen3_14b_peft_1gpu_h100_bf16_config",
    "qwen3_14b_pretrain_8gpu_h100_bf16_config",
    "qwen3_14b_sft_8gpu_h100_bf16_config",
    "qwen3_1p7b_peft_1gpu_h100_bf16_config",
    "qwen3_1p7b_pretrain_1gpu_h100_bf16_config",
    "qwen3_1p7b_sft_1gpu_h100_bf16_config",
    "qwen3_235b_a22b_peft_16gpu_h100_bf16_config",
    "qwen3_235b_a22b_pretrain_256gpu_h100_bf16_config",  # pragma: allowlist secret
    "qwen3_235b_a22b_sft_64gpu_h100_bf16_config",
    "qwen3_30b_a3b_peft_4gpu_h100_bf16_config",
    "qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config",
    "qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config",
    "qwen3_30b_a3b_sft_16gpu_h100_bf16_config",
    "qwen3_30b_a3b_sft_8gpu_h100_bf16_config",
    "qwen3_32b_peft_1gpu_h100_bf16_config",
    "qwen3_32b_pretrain_16gpu_h100_bf16_config",
    "qwen3_32b_sft_16gpu_h100_bf16_config",
    "qwen3_4b_peft_1gpu_h100_bf16_config",
    "qwen3_4b_pretrain_2gpu_h100_bf16_config",
    "qwen3_4b_sft_2gpu_h100_bf16_config",
    "qwen3_600m_peft_1gpu_h100_bf16_config",
    "qwen3_600m_pretrain_1gpu_h100_bf16_config",
    "qwen3_600m_sft_1gpu_h100_bf16_config",
    "qwen3_600m_sft_8gpu_h100_bf16_128k_config",
    "qwen3_600m_sft_8gpu_h100_bf16_yarn_128k_config",
    "qwen3_8b_peft_1gpu_h100_bf16_config",
    "qwen3_8b_pretrain_16gpu_h100_bf16_config",
    "qwen3_8b_pretrain_4gpu_h100_bf16_config",
    "qwen3_8b_sft_4gpu_h100_bf16_config",
    "qwen3_8b_sft_8gpu_h100_bf16_32k_config",
    "qwen3_next_80b_a3b_peft_1gpu_h100_bf16_config",
    "qwen3_next_80b_a3b_pretrain_32gpu_h100_bf16_config",
    "qwen3_next_80b_a3b_sft_16gpu_h100_bf16_config",
]
