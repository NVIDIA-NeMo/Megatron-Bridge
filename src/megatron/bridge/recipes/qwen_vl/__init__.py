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

# Qwen2.5-VL models
from .qwen25_vl import (
    qwen25_vl_3b_sft_config,
    qwen25_vl_7b_sft_config,
    qwen25_vl_32b_sft_config,
    qwen25_vl_72b_sft_config,
    qwen25_vl_3b_peft_config,
    qwen25_vl_7b_peft_config,
    qwen25_vl_32b_peft_config,
    qwen25_vl_72b_peft_config,
)

# Qwen3-VL models
from .qwen3_vl import (
    qwen3_vl_8b_sft_config,
    qwen3_vl_30b_a3b_sft_config,
    qwen3_vl_235b_a22b_sft_config,
    qwen3_vl_8b_peft_config,
    qwen3_vl_30b_a3b_peft_config,
    qwen3_vl_235b_a22b_peft_config,
)


__all__ = [
    # Qwen2.5-VL SFT configs
    "qwen25_vl_3b_sft_config",
    "qwen25_vl_7b_sft_config",
    "qwen25_vl_32b_sft_config",
    "qwen25_vl_72b_sft_config",
    # Qwen2.5-VL PEFT configs
    "qwen25_vl_3b_peft_config",
    "qwen25_vl_7b_peft_config",
    "qwen25_vl_32b_peft_config",
    "qwen25_vl_72b_peft_config",
    # Qwen3-VL SFT configs
    "qwen3_vl_8b_sft_config",
    "qwen3_vl_30b_a3b_sft_config",
    "qwen3_vl_235b_a22b_sft_config",
    # Qwen3-VL PEFT configs
    "qwen3_vl_8b_peft_config",
    "qwen3_vl_30b_a3b_peft_config",
    "qwen3_vl_235b_a22b_peft_config",
]
