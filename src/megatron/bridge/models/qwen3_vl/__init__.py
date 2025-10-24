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

"""Qwen3 VL model providers and configurations."""

# Dense model providers
from .qwen3_vl_provider import ( 
    Qwen3VLModelProvider,
    Qwen3VLModelProvider600M,
    Qwen3VLModelProvider1P7B,
    Qwen3VLModelProvider4B,
    Qwen3VLModelProvider8B,
    Qwen3VLModelProvider14B,
    Qwen3VLModelProvider32B,
)

# MoE (Mixture of Experts) model providers
from .qwen3_vl_moe_provider import (
    Qwen3VLMoEModelProvider,
    Qwen3VLMoEModelProvider30B_A3B,
    Qwen3VLMoEModelProvider235B_A22B,
    Qwen3VLMoEModelProvider48B_A8B,
)

# Bridges for HuggingFace to Megatron conversion
from .qwen3_vl_bridge import Qwen3VLBridge
from .qwen3_vl_moe_bridge import Qwen3VLMoEBridge
from .model import Qwen3VLModel, Qwen3VLMoEModel

__all__ = [
    # Dense models
    "Qwen3VLModelProvider",
    "Qwen3VLModelProvider600M",
    "Qwen3VLModelProvider1P7B",
    "Qwen3VLModelProvider4B",
    "Qwen3VLModelProvider8B",
    "Qwen3VLModelProvider14B",
    "Qwen3VLModelProvider32B",
    # MoE models
    "Qwen3VLMoEModelProvider",
    "Qwen3VLMoEModelProvider30B_A3B",
    "Qwen3VLMoEModelProvider235B_A22B",
    "Qwen3VLMoEModelProvider48B_A8B",
    # Bridges
    "Qwen3VLBridge",
    "Qwen3VLMoEBridge",
    # Models
    "Qwen3VLModel",
    "Qwen3VLMoEModel",
]
