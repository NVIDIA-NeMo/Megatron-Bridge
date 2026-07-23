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

from megatron.bridge.recipes.qwen_vl.h100.qwen3_vl import *  # noqa: F403
from megatron.bridge.recipes.qwen_vl.h100.qwen25_vl import *  # noqa: F403
from megatron.bridge.recipes.qwen_vl.h100.qwen35_vl import *  # noqa: F403
from megatron.bridge.recipes.qwen_vl.h100.qwen36_vl import *  # noqa: F403


__all__ = [
    "qwen25_vl_32b_peft_1gpu_h100_bf16_config",
    "qwen25_vl_32b_sft_16gpu_h100_bf16_config",
    "qwen25_vl_3b_peft_1gpu_h100_bf16_config",
    "qwen25_vl_3b_sft_1gpu_h100_bf16_config",
    "qwen25_vl_72b_peft_1gpu_h100_bf16_config",
    "qwen25_vl_72b_sft_32gpu_h100_bf16_config",
    "qwen25_vl_7b_peft_1gpu_h100_bf16_config",
    "qwen25_vl_7b_sft_2gpu_h100_bf16_config",
    "qwen35_vl_122b_a10b_peft_8gpu_h100_bf16_config",
    "qwen35_vl_122b_a10b_pretrain_128gpu_h100_bf16_mock_config",
    "qwen35_vl_122b_a10b_sft_48gpu_h100_bf16_config",
    "qwen35_vl_27b_peft_2gpu_h100_bf16_config",
    "qwen35_vl_27b_pretrain_16gpu_h100_bf16_mock_config",  # pragma: allowlist secret
    "qwen35_vl_27b_sft_16gpu_h100_bf16_config",
    "qwen35_vl_2b_peft_1gpu_h100_bf16_config",
    "qwen35_vl_2b_sft_1gpu_h100_bf16_config",
    "qwen35_vl_35b_a3b_peft_4gpu_h100_bf16_config",
    "qwen35_vl_35b_a3b_pretrain_8gpu_h100_bf16_mock_config",
    "qwen35_vl_35b_a3b_sft_2gpu_h100_bf16_fsdp_config",
    "qwen35_vl_35b_a3b_sft_16gpu_h100_bf16_config",
    "qwen35_vl_397b_a17b_peft_32gpu_h100_bf16_config",
    "qwen35_vl_397b_a17b_pretrain_512gpu_h100_bf16_mock_config",  # pragma: allowlist secret
    "qwen35_vl_397b_a17b_sft_128gpu_h100_bf16_config",
    "qwen35_vl_4b_peft_1gpu_h100_bf16_config",
    "qwen35_vl_4b_sft_2gpu_h100_bf16_config",
    "qwen35_vl_800m_peft_1gpu_h100_bf16_config",
    "qwen35_vl_800m_sft_1gpu_h100_bf16_config",
    "qwen35_vl_9b_peft_1gpu_h100_bf16_config",
    "qwen35_vl_9b_pretrain_4gpu_h100_bf16_mock_config",  # pragma: allowlist secret
    "qwen35_vl_9b_sft_4gpu_h100_bf16_config",
    "qwen36_vl_35b_a3b_peft_4gpu_h100_bf16_config",
    "qwen36_vl_35b_a3b_sft_16gpu_h100_bf16_config",
    "qwen3_vl_235b_a22b_peft_16gpu_h100_bf16_config",
    "qwen3_vl_235b_a22b_pretrain_256gpu_h100_bf16_mock_config",  # pragma: allowlist secret
    "qwen3_vl_235b_a22b_sft_32gpu_h100_bf16_config",
    "qwen3_vl_30b_a3b_peft_4gpu_h100_bf16_config",
    "qwen3_vl_30b_a3b_pretrain_8gpu_h100_bf16_mock_config",
    "qwen3_vl_30b_a3b_sft_8gpu_h100_bf16_config",
    "qwen3_vl_8b_peft_1gpu_h100_bf16_config",
    "qwen3_vl_8b_peft_1gpu_h100_bf16_energon_config",
    "qwen3_vl_8b_pretrain_4gpu_h100_bf16_mock_config",
    "qwen3_vl_8b_sft_2gpu_h100_bf16_config",
]
