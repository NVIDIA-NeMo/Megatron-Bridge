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

from dataclasses import dataclass
from typing import Any, Optional

from megatron.core.models.gpt import GPTModel as MCoreGPTModel

from megatron.bridge.models.mla_provider import MLAModelProvider


@dataclass
class KimiK25VLModelProvider(MLAModelProvider):
    """
    Model provider for Kimi K2.5 VL (Vision-Language) Models.

    Inherits language model configuration from MLAModelProvider. Core architecture
    parameters (num_layers, hidden_size, MoE config, MLA config, etc.) are
    populated by ``provider_bridge`` in the bridge class via ``hf_config_to_provider_kwargs``.

    Only VLM-specific fields (vision config, token IDs, freeze options, etc.)
    that are NOT part of the language model are defined here.

    The vision component (MoonViT3d + PatchMergerMLP) is dynamically loaded
    from the HuggingFace model repository at runtime via ``trust_remote_code``.
    """

    # VL models shouldn't scatter embeddings across sequence parallel regions because
    # the vision embeddings are going to be inserted into the language embeddings.
    scatter_embedding_sequence_parallel: bool = False

    # Vision configuration — raw HF KimiK25VisionConfig object, used to construct
    # VisionTowerConfig and ProjectorConfig for the vision tower and mm_projector.
    vision_config: Any = None

    # Path to HuggingFace model directory (required for dynamic module loading
    # of MoonViT3d, PatchMergerMLP, and other custom model classes).
    hf_model_path: Optional[str] = None

    # Token IDs (from Kimi K2.5 config.json)
    bos_token_id: int = 163584
    eos_token_id: int = 163585
    image_token_id: int = 163605  # media_placeholder_token_id in HF config
    # Fields needed by HF's _merge_input_ids_with_image_features (bound via MethodType)
    media_placeholder_token_id: int = 163605
    pad_token_id: int = 163839
    ignore_index: int = -100

    # Freeze options for fine-tuning scenarios
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    # Generation configuration
    generation_config: Any | None = None

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        """Provide a KimiK25VL model instance with vision and language components."""
        from megatron.bridge.models.kimi_vl.modeling_kimi_k25_vl import KimiK25VLModel

        model = KimiK25VLModel(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)

        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )

        return model

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        """Provide just the language model component (MoE with MLA) without vision."""
        return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
