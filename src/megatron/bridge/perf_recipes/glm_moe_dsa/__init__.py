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
"""Performance benchmark recipes for GLM models."""

from megatron.bridge.perf_recipes.glm_moe_dsa.gb200.glm5 import (
    glm51_sft_192gpu_gb200_bf16_config,
    glm52_sft_192gpu_gb200_bf16_config,
)
from megatron.bridge.perf_recipes.glm_moe_dsa.h100.glm5 import (
    glm51_sft_416gpu_h100_bf16_config,
    glm52_sft_416gpu_h100_bf16_config,
)
