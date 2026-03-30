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
# Qwen3-VL models
from .qwen3_vl import (
    qwen3_vl_8b_peft_config,
    qwen3_vl_8b_peft_energon_config,
    qwen3_vl_8b_sft_config,
    qwen3_vl_30b_a3b_peft_config,
    qwen3_vl_30b_a3b_sft_config,
    qwen3_vl_235b_a22b_peft_config,
    qwen3_vl_235b_a22b_sft_config,
)
from .qwen25_vl import (
    qwen25_vl_3b_peft_config,
    qwen25_vl_3b_sft_config,
    qwen25_vl_7b_peft_config,
    qwen25_vl_7b_sft_config,
    qwen25_vl_32b_peft_config,
    qwen25_vl_32b_sft_config,
    qwen25_vl_72b_peft_config,
    qwen25_vl_72b_sft_config,
)

# Qwen3.5 models
from .qwen35_llm import (
    QWEN35_LLM_SFT_SKIP_VISION_GLOBS,
    qwen35_llm_2b_sft_config,
    qwen35_llm_4b_sft_config,
    qwen35_llm_9b_sft_config,
    qwen35_llm_27b_sft_config,
    qwen35_llm_35b_a3b_sft_config,
    qwen35_llm_122b_a10b_sft_config,
    qwen35_llm_397b_a17b_sft_config,
    qwen35_llm_800m_sft_config,
)
from .qwen35_vl import (
    qwen35_vl_2b_peft_config,
    qwen35_vl_2b_sft_config,
    qwen35_vl_4b_peft_config,
    qwen35_vl_4b_sft_config,
    qwen35_vl_9b_peft_config,
    qwen35_vl_9b_sft_config,
    qwen35_vl_27b_peft_config,
    qwen35_vl_27b_sft_config,
    qwen35_vl_35b_a3b_fsdp_sft_config,
    qwen35_vl_35b_a3b_peft_config,
    qwen35_vl_35b_a3b_sft_config,
    qwen35_vl_122b_a10b_peft_config,
    qwen35_vl_122b_a10b_sft_config,
    qwen35_vl_397b_a17b_peft_config,
    qwen35_vl_397b_a17b_sft_config,
    qwen35_vl_800m_peft_config,
    qwen35_vl_800m_sft_config,
)


__all__ = [
    # Qwen3.5 LLM-only (text) SFT configs
    "QWEN35_LLM_SFT_SKIP_VISION_GLOBS",
    "qwen35_llm_800m_sft_config",
    "qwen35_llm_2b_sft_config",
    "qwen35_llm_4b_sft_config",
    "qwen35_llm_9b_sft_config",
    "qwen35_llm_27b_sft_config",
    "qwen35_llm_35b_a3b_sft_config",
    "qwen35_llm_122b_a10b_sft_config",
    "qwen35_llm_397b_a17b_sft_config",
    # Qwen3.5-VL SFT configs — dense
    "qwen35_vl_800m_sft_config",
    "qwen35_vl_2b_sft_config",
    "qwen35_vl_4b_sft_config",
    "qwen35_vl_9b_sft_config",
    "qwen35_vl_27b_sft_config",
    # Qwen3.5-VL SFT configs — MoE
    "qwen35_vl_35b_a3b_sft_config",
    "qwen35_vl_35b_a3b_fsdp_sft_config",
    "qwen35_vl_122b_a10b_sft_config",
    "qwen35_vl_397b_a17b_sft_config",
    # Qwen3.5-VL PEFT configs — dense
    "qwen35_vl_800m_peft_config",
    "qwen35_vl_2b_peft_config",
    "qwen35_vl_4b_peft_config",
    "qwen35_vl_9b_peft_config",
    "qwen35_vl_27b_peft_config",
    # Qwen3.5-VL PEFT configs — MoE
    "qwen35_vl_35b_a3b_peft_config",
    "qwen35_vl_122b_a10b_peft_config",
    "qwen35_vl_397b_a17b_peft_config",
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
    "qwen3_vl_8b_peft_energon_config",
    "qwen3_vl_30b_a3b_peft_config",
    "qwen3_vl_235b_a22b_peft_config",
]
