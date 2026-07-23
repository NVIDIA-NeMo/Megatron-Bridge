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

from megatron.bridge.models.nemotron_vl.modeling_nemotron_vl import NemotronVLModel


class NemotronOmniLlavaModel(NemotronVLModel):
    """Legacy collapse/expand Omni wrapper around MCore ``LLaVAModel``.

    forward() is inherited from NemotronVLModel (which delegates to LLaVAModel),
    so sound kwargs (sound_clips, sound_length) pass through automatically when
    the selected LLaVAModel implementation supports them.
    """

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
