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

# Core model components
#:from megatron.bridge.models.qwen_3_vl.vision_model import Qwen3VLVisionModel  # noqa: F401
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from megatron.bridge.models.qwen_3_vl.gpt_model import Qwen3VLGPTModel  # noqa: F401
from megatron.bridge.models.qwen_3_vl.transformer_block import Qwen3VLTransformerBlock  # noqa: F401
from megatron.bridge.models.qwen_3_vl.transformer_config import Qwen3VLTransformerConfig  # noqa: F401
from megatron.bridge.models.qwen_3_vl.model import Qwen3VLModel  # noqa: F401

# Dense model providers
from megatron.bridge.models.qwen_3_vl.provider import (
    Qwen3VLModelProvider,
)

# MoE (Mixture of Experts) model providers
from megatron.bridge.models.qwen_3_vl.moe_provider import (
    Qwen3VLMoEModelProvider,
)

# Bridges for HuggingFace to Megatron conversion
from megatron.bridge.models.qwen_3_vl.bridge import Qwen3VLBridge
from megatron.bridge.models.qwen_3_vl.moe_bridge import Qwen3VLMoEBridge

__all__ = [
    # Core components
    "Qwen3VLGPTModel",
    "Qwen3VLTransformerBlock",
    "Qwen3VLTransformerConfig",
    "Qwen3VLModel",
    "Qwen3VLMoEModel",
    "Qwen3VLModelProvider",
    "Qwen3VLMoEModelProvider",
    "Qwen3VLBridge",
    "Qwen3VLMoEBridge",
]