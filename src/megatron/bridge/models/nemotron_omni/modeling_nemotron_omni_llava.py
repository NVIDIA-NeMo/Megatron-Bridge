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

import warnings

from megatron.core.models.multimodal.llava_model import LLaVAModel

from megatron.bridge.models.nemotron_vl.modeling_nemotron_vl import NemotronVLModel
from megatron.bridge.models.nemotron_vl.nemotron_vl_provider import NemotronVLModelProvider


class NemotronOmniLlavaModel(NemotronVLModel):
    """Deprecated collapse/expand Omni wrapper around MCore ``LLaVAModel``.

    forward() is inherited from NemotronVLModel (which delegates to LLaVAModel),
    so sound kwargs (sound_clips, sound_length) pass through automatically when
    the selected LLaVAModel implementation supports them.

    Use :class:`~megatron.bridge.models.nemotron_omni.modeling_nemotron_omni.NemotronOmniModel`
    for the canonical processor-expanded sequence and collator-owned packing
    contract.
    """

    def __init__(
        self,
        config: NemotronVLModelProvider | None = None,
        *,
        llava_model: LLaVAModel | None = None,
        pre_process: bool | None = True,
        post_process: bool | None = True,
        vp_stage: int | None = None,
    ) -> None:
        """Construct the deprecated LLaVA compatibility model.

        Args:
            config: Provider used to construct the wrapped model.
            llava_model: Fully assembled MCore LLaVA model.
            pre_process: Whether this pipeline stage owns input processing.
            post_process: Whether this pipeline stage owns output processing.
            vp_stage: Optional virtual pipeline stage.
        """
        warnings.warn(
            "NemotronOmniLlavaModel is deprecated; use NemotronOmniModel with the canonical "
            "processor-expanded sequence contract.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(
            config=config,
            llava_model=llava_model,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )

    def freeze(
        self,
        *,
        freeze_language_model: bool = False,
        freeze_vision_model: bool = False,
        freeze_vision_projection: bool = False,
        freeze_sound_model: bool = False,
        freeze_sound_projection: bool = False,
    ) -> None:
        super().freeze(
            freeze_language_model=freeze_language_model,
            freeze_vision_model=freeze_vision_model,
            freeze_vision_projection=freeze_vision_projection,
        )
        if freeze_sound_model and self.llava_model.sound_model is not None:
            for param in self.llava_model.sound_model.parameters():
                param.requires_grad = False
        if freeze_sound_projection and self.llava_model.sound_projection is not None:
            for param in self.llava_model.sound_projection.parameters():
                param.requires_grad = False
