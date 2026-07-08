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

from megatron.bridge.recipes.stepfun.h100.step35 import *  # noqa: F403
from megatron.bridge.recipes.stepfun.h100.step37 import *  # noqa: F403


__all__ = [
    "step35_196b_a11b_pretrain_512gpu_h100_bf16_config",
    "step37_sft_4gpu_h100_bf16_flickr8k_smoke_config",
    "step37_sft_64gpu_h100_bf16_flickr8k_config",
]
