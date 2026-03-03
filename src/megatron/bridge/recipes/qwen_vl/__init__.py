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

# Qwen3 models
from .qwen3_vl import (
    qwen3_vl_8b_finetune_config,
    qwen3_vl_8b_pretrain_config,
    qwen3_vl_30b_a3b_finetune_config,
    qwen3_vl_30b_a3b_pretrain_config,
    qwen3_vl_235b_a22b_finetune_config,
    qwen3_vl_235b_a22b_pretrain_config,
)


__all__ = [
    # Qwen3-VL pretrain configs
    "qwen3_vl_8b_pretrain_config",
    "qwen3_vl_30b_a3b_pretrain_config",
    "qwen3_vl_235b_a22b_pretrain_config",
    # Qwen3-VL finetune configs (with PEFT support)
    "qwen3_vl_8b_finetune_config",
    "qwen3_vl_30b_a3b_finetune_config",
    "qwen3_vl_235b_a22b_finetune_config",
]
