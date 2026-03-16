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

from dataclasses import dataclass
from typing import Any

from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)

from megatron.bridge.models.kimi.kimi_provider import KimiK2Provider


@dataclass
class KimiK25VLModelProvider(KimiK2Provider):
    """Model provider for Kimi K2.5 VL (Vision-Language) Models.

    Inherits language model configuration from KimiK2Provider since the
    Kimi K2.5 language backbone shares the same architecture as Kimi K2
    (MoE with MLA, 384 experts, 61 layers, etc.).

    The vision component (MoonViT3d + PatchMergerMLP) is dynamically loaded
    from the HuggingFace model repository at runtime via ``trust_remote_code``.
    """

    scatter_embedding_sequence_parallel: bool = False

    vision_config: Any = None
    hf_model_path: str | None = None

    # Token IDs
    bos_token_id: int = 163584
    eos_token_id: int = 163585
    image_token_id: int = 163605
    media_placeholder_token_id: int = 163605
    pad_token_id: int = 163839
    ignore_index: int = -100

    # Freeze options
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    generation_config: Any | None = None

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        """Provide a KimiK25VL model instance with vision and language components."""
        from megatron.bridge.models.kimi_vl.modeling_kimi_k25_vl import KimiK25VLModel

        # Resolve pre_process/post_process from PP/VP stage (same logic as GPTModelProvider)
        vp_size = getattr(self, "virtual_pipeline_model_parallel_size", None) or 1
        if pre_process is None:
            pre_process = is_vp_first_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_first_stage(
                self._pg_collection.pp
            )
        if post_process is None:
            post_process = is_vp_last_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_last_stage(
                self._pg_collection.pp
            )

        model = KimiK25VLModel(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)

        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )

        return model

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        """Provide just the language model component (Kimi K2 MoE) without vision."""
        return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
