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

from megatron.bridge.recipes.glm.h100.glm5 import *  # noqa: F403
from megatron.bridge.recipes.glm.h100.glm45 import *  # noqa: F403


__all__ = [
    "glm45_355b_peft_16gpu_h100_bf16_config",
    "glm45_355b_pretrain_128gpu_h100_bf16_config",
    "glm45_355b_sft_128gpu_h100_bf16_config",
    "glm45_air_106b_peft_8gpu_h100_bf16_config",
    "glm45_air_106b_pretrain_32gpu_h100_bf16_config",
    "glm45_air_106b_sft_32gpu_h100_bf16_config",
    "glm52_peft_208gpu_h100_bf16_config",
    "glm52_pretrain_416gpu_h100_bf16_config",
    "glm52_sft_functional_416gpu_h100_bf16_config",
    "glm52_sft_long_context_608gpu_h100_bf16_config",
]
