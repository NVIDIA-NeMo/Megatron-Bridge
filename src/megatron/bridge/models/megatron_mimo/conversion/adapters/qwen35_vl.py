# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""MegatronMIMO conversion adapter for Qwen3.5-VL (dense)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from megatron.bridge.models.megatron_mimo.conversion.orchestrator import MIMOComponent, register_mimo_conversion
from megatron.bridge.models.megatron_mimo.qwen35_vl_provider import Qwen35VLMegatronMIMOProvider
from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import Qwen35VLBridge


if TYPE_CHECKING:
    from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
        MegatronMIMOParallelismConfig,
    )


# Component names must match the parallelism config and modality keys.
_LANGUAGE_COMPONENT = "language"
_VISION_COMPONENT = "images"


# Target module paths on the constructed ``MimoModel``.
_LANGUAGE_TARGET_PATH = "language_model"
_VISION_TARGET_PATH = "modality_submodules.images.encoders.qwen_visual"


@register_mimo_conversion(Qwen35VLBridge)
def qwen35_vl_mimo_adapter(
    source_bridge: Qwen35VLBridge,
    hf_pretrained: Any,
    parallelism_config: "MegatronMIMOParallelismConfig",
) -> tuple[Qwen35VLMegatronMIMOProvider, list[MIMOComponent]]:
    """Build the (provider, route_table) pair for Qwen3.5-VL MIMO conversion."""
    language_provider = source_bridge.provider_bridge(hf_pretrained)
    # MIMO v1 does not support MTP; force off so the provider accepts the model.
    language_provider.mtp_num_layers = None

    provider = Qwen35VLMegatronMIMOProvider(
        language_provider=language_provider,
        megatron_mimo_parallelism_config=parallelism_config,
    )

    routes = [
        MIMOComponent(
            name=_LANGUAGE_COMPONENT,
            source_prefix="language_model.",
            target_module_path=_LANGUAGE_TARGET_PATH,
        ),
        MIMOComponent(
            name=_VISION_COMPONENT,
            source_prefix="vision_model.",
            target_module_path=_VISION_TARGET_PATH,
        ),
    ]

    return provider, routes
