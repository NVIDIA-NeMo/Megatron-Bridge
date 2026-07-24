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
# ruff: noqa: F401
"""Compatibility aliases for generic GLM-5.2 recipe names."""

from megatron.bridge.recipes.glm.h100.glm5 import (
    glm52_peft_208gpu_h100_bf16_config as glm52_peft_config,
)
from megatron.bridge.recipes.glm.h100.glm5 import (
    glm52_pretrain_416gpu_h100_bf16_config as glm52_pretrain_config,
)
from megatron.bridge.recipes.glm.h100.glm5 import (
    glm52_sft_functional_416gpu_h100_bf16_config as glm52_sft_config,
)
from megatron.bridge.recipes.glm.h100.glm5 import (
    glm52_sft_long_context_608gpu_h100_bf16_config as glm52_sft_long_context_config,
)


__all__ = [
    "glm52_peft_config",
    "glm52_pretrain_config",
    "glm52_sft_config",
    "glm52_sft_long_context_config",
]
