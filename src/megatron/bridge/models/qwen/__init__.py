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

from megatron.bridge.models.qwen.qwen2_bridge import Qwen2Bridge  # noqa: F401
from megatron.bridge.models.qwen.qwen3_bridge import Qwen3Bridge  # noqa: F401
from megatron.bridge.models.qwen.qwen3_moe_bridge import Qwen3MoEBridge  # noqa: F401
from megatron.bridge.models.qwen.qwen_provider import (
    Qwen2ModelProvider,
    Qwen2ModelProvider1P5B,
    Qwen2ModelProvider7B,
    Qwen2ModelProvider72B,
    Qwen2ModelProvider500M,
    Qwen3ModelProvider,
    Qwen3ModelProvider1P7B,
    Qwen3ModelProvider4B,
    Qwen3ModelProvider8B,
    Qwen3ModelProvider14B,
    Qwen3ModelProvider32B,
    Qwen3ModelProvider600M,
    Qwen3MoEModelProvider,
    Qwen3MoEModelProvider30B_A3B,
    Qwen3MoEModelProvider235B_A22B,
    Qwen25ModelProvider1P5B,
    Qwen25ModelProvider3B,
    Qwen25ModelProvider7B,
    Qwen25ModelProvider14B,
    Qwen25ModelProvider32B,
    Qwen25ModelProvider72B,
    Qwen25ModelProvider500M,
)


__all__ = [
    "Qwen2ModelProvider",
    "Qwen2ModelProvider500M",
    "Qwen2ModelProvider1P5B",
    "Qwen2ModelProvider7B",
    "Qwen2ModelProvider72B",
    "Qwen25ModelProvider500M",
    "Qwen25ModelProvider1P5B",
    "Qwen25ModelProvider3B",
    "Qwen25ModelProvider7B",
    "Qwen25ModelProvider14B",
    "Qwen25ModelProvider32B",
    "Qwen25ModelProvider72B",
    "Qwen3ModelProvider",
    "Qwen3ModelProvider600M",
    "Qwen3ModelProvider1P7B",
    "Qwen3ModelProvider4B",
    "Qwen3ModelProvider8B",
    "Qwen3ModelProvider14B",
    "Qwen3ModelProvider32B",
    "Qwen3MoEModelProvider",
    "Qwen3MoEModelProvider30B_A3B",
    "Qwen3MoEModelProvider235B_A22B",
]
