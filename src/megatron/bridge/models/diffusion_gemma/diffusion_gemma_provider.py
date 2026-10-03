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

from megatron.bridge.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaModel
from megatron.bridge.models.gemma_vl.gemma4_vl_provider import Gemma4VLModelProvider


@dataclass
class DiffusionGemmaModelProvider(Gemma4VLModelProvider):
    """Model provider for DiffusionGemma block-diffusion checkpoints."""

    canvas_length: int = 256
    global_image_bidirectional_attention: bool = True
    freeze_self_conditioning: bool = False
    # Training-only block-diffusion objective settings.  They live on the
    # provider so recipe/config serialization keeps the stochastic process
    # alongside the model rather than hiding it in the data loader.
    diffusion_loss_variant: str = "base-sft"
    diffusion_noise_min: float = 1e-3
    diffusion_noise_max: float = 1.0 - 1e-3
    diffusion_weight_clip: float | None = None
    diffusion_self_conditioning_prob: float = 0.0
    diffusion_decoder_loss_weight: float = 1.0
    diffusion_encoder_loss_weight: float = 1.0
    moe_aux_loss_coeff: float = 0.0

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> DiffusionGemmaModel:
        if getattr(self, "audio_config", None) is not None:
            raise ValueError("DiffusionGemma does not support audio inputs")
        model = DiffusionGemmaModel(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        if any(
            (
                self.freeze_language_model,
                self.freeze_vision_model,
                self.freeze_vision_projection,
                self.freeze_self_conditioning,
            )
        ):
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
                freeze_self_conditioning=self.freeze_self_conditioning,
            )
        return model
