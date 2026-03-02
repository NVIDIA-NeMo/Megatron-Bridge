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

from megatron.bridge.models.kimi.kimi_provider import KimiK2Provider


@dataclass
class KimiK25VLModelProvider(KimiK2Provider):
    """
    Model provider for Kimi K2.5 VL (Vision-Language) Models.

    Inherits language model configuration from KimiK2Provider since the
    Kimi K2.5 language backbone shares the same architecture as Kimi K2
    (MoE with MLA, 384 experts, 61 layers, etc.).

    Minor config differences (rotary_scaling_factor, layernorm_epsilon,
    init_method_std) between K2 and K2.5 are handled at runtime by
    ``get_common_configs()`` in the bridge, which reads actual values
    from the HuggingFace config.

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

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        """
        Provide a KimiK25VL model instance with vision and language components.

        Returns:
            KimiK25VLModel: Configured Kimi K2.5 VL model with vision tower,
            multimodal projector, and Kimi K2 language model.
        """
        from megatron.bridge.models.kimi_vl.modeling_kimi_k25_vl import KimiK25VLModel

        model = KimiK25VLModel(
            self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )

        # Apply freeze options if any are enabled
        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )

        return model

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        """
        Provide just the language model component (Kimi K2 MoE) without vision.

        This is called by KimiK25VLModel to construct only the language backbone,
        while the vision tower and projector are constructed separately.

        Args:
            pre_process: Whether this is the first stage in pipeline parallelism.
            post_process: Whether this is the last stage in pipeline parallelism.
            vp_stage: Virtual pipeline stage number.

        Returns:
            MCoreGPTModel instance (language model only).
        """
        return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
