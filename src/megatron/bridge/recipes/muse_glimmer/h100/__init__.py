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

"""H100 recipes for Muse Glimmer."""

from megatron.bridge.recipes.muse_glimmer.h100.muse_glimmer import (
    muse_glimmer_30b_peft_8gpu_h100_bf16_config,
    muse_glimmer_30b_pretrain_32gpu_h100_bf16_multimodal_config,
    muse_glimmer_30b_sft_32gpu_h100_bf16_config,
    muse_glimmer_30b_sft_32gpu_h100_bf16_long_context_config,
)


__all__ = [
    "muse_glimmer_30b_peft_8gpu_h100_bf16_config",
    "muse_glimmer_30b_pretrain_32gpu_h100_bf16_multimodal_config",
    "muse_glimmer_30b_sft_32gpu_h100_bf16_config",
    "muse_glimmer_30b_sft_32gpu_h100_bf16_long_context_config",
]
