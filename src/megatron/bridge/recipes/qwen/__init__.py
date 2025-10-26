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

# Qwen2 models
from .qwen2 import (
    qwen2_1p5b_pretrain_config,
    qwen2_7b_pretrain_config,
    qwen2_72b_pretrain_config,
    qwen2_500m_pretrain_config,
    # Qwen2.5 models
    qwen25_1p5b_pretrain_config,
    qwen25_7b_pretrain_config,
    qwen25_14b_pretrain_config,
    qwen25_32b_pretrain_config,
    qwen25_72b_pretrain_config,
    qwen25_500m_pretrain_config,
)

# Qwen3 models
from .qwen3 import (
    qwen3_1p7b_finetune_config,
    qwen3_1p7b_pretrain_config,
    qwen3_4b_finetune_config,
    qwen3_4b_pretrain_config,
    qwen3_8b_finetune_config,
    qwen3_8b_pretrain_config,
    qwen3_14b_finetune_config,
    qwen3_14b_pretrain_config,
    qwen3_32b_finetune_config,
    qwen3_32b_pretrain_config,
    qwen3_600m_finetune_config,
    qwen3_600m_pretrain_config,
)

# Qwen3 MoE models
from .qwen3_moe import (
    qwen3_30b_a3b_pretrain_config,
    qwen3_235b_a22b_pretrain_config,
)


__all__ = [
    # Qwen2 models
    "qwen2_500m_pretrain_config",
    "qwen2_1p5b_pretrain_config",
    "qwen2_7b_pretrain_config",
    "qwen2_72b_pretrain_config",
    # Qwen2.5 models
    "qwen25_500m_pretrain_config",
    "qwen25_1p5b_pretrain_config",
    "qwen25_7b_pretrain_config",
    "qwen25_14b_pretrain_config",
    "qwen25_32b_pretrain_config",
    "qwen25_72b_pretrain_config",
    # Qwen3 models
    "qwen3_600m_pretrain_config",
    "qwen3_1p7b_pretrain_config",
    "qwen3_4b_pretrain_config",
    "qwen3_8b_pretrain_config",
    "qwen3_14b_pretrain_config",
    "qwen3_32b_pretrain_config",
    "qwen3_600m_finetune_config",
    "qwen3_1p7b_finetune_config",
    "qwen3_4b_finetune_config",
    "qwen3_8b_finetune_config",
    "qwen3_14b_finetune_config",
    "qwen3_32b_finetune_config",
    # Qwen3 MoE models
    "qwen3_30b_a3b_pretrain_config",
    "qwen3_235b_a22b_pretrain_config",
]
