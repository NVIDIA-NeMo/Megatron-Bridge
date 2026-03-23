# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from .gpt_oss import (
    gpt_oss_20b_peft_config,
    gpt_oss_20b_peft_fp8_current_scaling_config,
    gpt_oss_20b_peft_mxfp8_config,
    gpt_oss_20b_pretrain_config,
    gpt_oss_20b_pretrain_fp8_current_scaling_config,
    gpt_oss_20b_pretrain_mxfp8_config,
    gpt_oss_20b_sft_config,
    gpt_oss_20b_sft_fp8_current_scaling_config,
    gpt_oss_20b_sft_mxfp8_config,
    gpt_oss_120b_peft_config,
    gpt_oss_120b_pretrain_config,
    gpt_oss_120b_sft_config,
)

# GPT-OSS perf recipes
from .gpt_oss_perf import (
    gpt_oss_120b_pretrain_64gpu_b200_bf16_config,
    gpt_oss_120b_pretrain_64gpu_b300_bf16_config,
    gpt_oss_120b_pretrain_64gpu_gb200_bf16_config,
    # V1
    gpt_oss_120b_pretrain_64gpu_gb300_bf16_config,
    gpt_oss_120b_pretrain_64gpu_h100_bf16_config,
    gpt_oss_120b_pretrain_v2_64gpu_b200_bf16_config,
    gpt_oss_120b_pretrain_v2_64gpu_b300_bf16_config,
    gpt_oss_120b_pretrain_v2_64gpu_gb200_bf16_config,
    # V2
    gpt_oss_120b_pretrain_v2_64gpu_gb300_bf16_config,
    gpt_oss_120b_pretrain_v2_64gpu_h100_bf16_config,
)


__all__ = [
    "gpt_oss_20b_pretrain_config",
    "gpt_oss_20b_pretrain_fp8_current_scaling_config",
    "gpt_oss_120b_pretrain_config",
    "gpt_oss_20b_sft_config",
    "gpt_oss_20b_sft_fp8_current_scaling_config",
    "gpt_oss_120b_sft_config",
    "gpt_oss_20b_peft_config",
    "gpt_oss_20b_peft_fp8_current_scaling_config",
    "gpt_oss_120b_peft_config",
    "gpt_oss_20b_pretrain_mxfp8_config",
    "gpt_oss_20b_sft_mxfp8_config",
    "gpt_oss_20b_peft_mxfp8_config",
    # GPT-OSS perf recipes — V1
    "gpt_oss_120b_pretrain_64gpu_gb300_bf16_config",
    "gpt_oss_120b_pretrain_64gpu_gb200_bf16_config",
    "gpt_oss_120b_pretrain_64gpu_b300_bf16_config",
    "gpt_oss_120b_pretrain_64gpu_b200_bf16_config",
    "gpt_oss_120b_pretrain_64gpu_h100_bf16_config",
    # GPT-OSS perf recipes — V2
    "gpt_oss_120b_pretrain_v2_64gpu_gb300_bf16_config",
    "gpt_oss_120b_pretrain_v2_64gpu_gb200_bf16_config",
    "gpt_oss_120b_pretrain_v2_64gpu_b300_bf16_config",
    "gpt_oss_120b_pretrain_v2_64gpu_b200_bf16_config",
    "gpt_oss_120b_pretrain_v2_64gpu_h100_bf16_config",
]
