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

"""Megatron Bridge PEFT (Parameter-Efficient Fine-Tuning) integration.

This module provides high-level APIs for integrating HuggingFace PEFT adapters
with Megatron's distributed training infrastructure.

Example:
    >>> from megatron.bridge import AutoBridge
    >>> from megatron.bridge.peft import AutoPEFTBridge, get_peft_model
    >>>
    >>> # Load base model and adapters
    >>> base_bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.1-8B")
    >>> peft_bridge = AutoPEFTBridge.from_hf_pretrained("username/llama-lora")
    >>> peft_model = peft_bridge.to_megatron_model(base_bridge)
    >>>
    >>> # Or apply PEFT directly to a provider
    >>> from megatron.bridge.peft.lora import LoRA
    >>> provider = base_bridge.to_megatron_provider()
    >>> lora = LoRA(dim=32, alpha=32)
    >>> peft_model = get_peft_model(provider, lora)
"""

# Check for required dependencies
try:
    import peft
except ImportError as e:
    raise ImportError(
        "HuggingFace PEFT library is required for PEFT integration. "
        "Please install it with: pip install megatron-bridge[peft]"
    ) from e

# Import main API components
from megatron.bridge.peft.api import get_peft_model, PEFTModel
from megatron.bridge.peft.conversion.auto_peft_bridge import AutoPEFTBridge

# Import PEFT implementations
from megatron.bridge.peft.lora.lora import LoRA
from megatron.bridge.peft.lora.canonical_lora import CanonicalLoRA
from megatron.bridge.peft.lora.dora import DoRA

__all__ = [
    # Main API
    "get_peft_model",
    "PEFTModel",
    "AutoPEFTBridge",

    # PEFT implementations
    "LoRA",
    "CanonicalLoRA",
    "DoRA",
]
