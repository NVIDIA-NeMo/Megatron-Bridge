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

from megatron.bridge.peft.api import MegatronPEFTModel, get_peft_model
from megatron.bridge.peft.lora.canonical_lora import CanonicalLoRA
from megatron.bridge.peft.lora.dora import DoRA

# Import PEFT implementations
from megatron.bridge.peft.lora.lora import LoRA


# Check for optional peft dependency (only needed for AutoPEFTBridge)
try:
    import peft

    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False

# Conditionally import AutoPEFTBridge (requires peft library)
if _PEFT_AVAILABLE:
    from megatron.bridge.peft.conversion.auto_peft_bridge import AutoPEFTBridge
else:
    # Create a lazy import that raises an error when AutoPEFTBridge is accessed
    class AutoPEFTBridge:
        """Lazy import of AutoPEFTBridge"""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "HuggingFace PEFT library is required for AutoPEFTBridge. "
                "Please install it with: pip install megatron-bridge[peft]"
            )

        @classmethod
        def from_hf_pretrained(cls, *args, **kwargs):
            raise ImportError(
                "HuggingFace PEFT library is required for AutoPEFTBridge. "
                "Please install it with: pip install megatron-bridge[peft]"
            )


__all__ = [
    # Main API
    "get_peft_model",
    "MegatronPEFTModel",
    "AutoPEFTBridge",
    # PEFT implementations
    "LoRA",
    "CanonicalLoRA",
    "DoRA",
]
