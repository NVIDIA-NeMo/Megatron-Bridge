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


class NemotronOmniModel(NemotronVLModel):
    """Extends NemotronVLModel with sound modality freeze support.

    forward() is inherited from NemotronVLModel (which delegates to LLaVAModel),
    so sound kwargs (sound_clips, sound_length) pass through automatically when
    the vlm2 LLaVAModel is used.
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
            for name, param in self.llava_model.sound_model.named_parameters():
                print(f"Freezing sound_model.{name}")
                param.requires_grad = False
        if freeze_sound_projection and self.llava_model.sound_projection is not None:
            for name, param in self.llava_model.sound_projection.named_parameters():
                print(f"Freezing sound_projection.{name}")
                param.requires_grad = False
