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
"""Compatibility aliases for legacy WAN recipe names."""

from megatron.bridge.recipes.wan.h100.wan import (
    wan_1_3b_pretrain_8gpu_h100_bf16_config as wan_1_3b_pretrain_config,
)
from megatron.bridge.recipes.wan.h100.wan import (
    wan_1_3b_sft_8gpu_h100_bf16_config as wan_1_3b_sft_config,
)
from megatron.bridge.recipes.wan.h100.wan import (
    wan_1_3b_text2image_pretrain_1gpu_h100_bf16_config as wan_1_3b_text2image_pretrain_config,
)
from megatron.bridge.recipes.wan.h100.wan import (
    wan_1_3b_text2video_pretrain_4gpu_h100_bf16_config as wan_1_3b_text2video_pretrain_config,
)
from megatron.bridge.recipes.wan.h100.wan import (
    wan_14b_pretrain_8gpu_h100_bf16_config as wan_14b_pretrain_config,
)
from megatron.bridge.recipes.wan.h100.wan import (
    wan_14b_sft_8gpu_h100_bf16_config as wan_14b_sft_config,
)


__all__ = [
    "wan_14b_pretrain_config",
    "wan_14b_sft_config",
    "wan_1_3b_pretrain_config",
    "wan_1_3b_sft_config",
    "wan_1_3b_text2image_pretrain_config",
    "wan_1_3b_text2video_pretrain_config",
]
