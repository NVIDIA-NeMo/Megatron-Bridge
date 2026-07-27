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

"""BAGEL vision and diffusion modality modules."""

from __future__ import annotations

from typing import Any

import torch
from megatron.core.models.mimo.submodules.base import ModalitySubmodules
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules


class OfficialBagelVisionEncoder(torch.nn.Module):
    """Wrap BAGEL's packed SigLIP encoder without changing its inputs."""

    def __init__(
        self,
        *,
        bagel_config: Any,
        vision_model_path: str | None,
        dtype: torch.dtype,
        recompute: bool,
    ) -> None:
        super().__init__()
        from modeling.bagel import SiglipVisionModel
        from modeling.bagel.modeling_utils import PositionEmbedding

        if vision_model_path is None:
            self.encoder = SiglipVisionModel(bagel_config.vit_config)
        else:
            self.encoder = SiglipVisionModel.from_pretrained(
                vision_model_path,
                config=bagel_config.vit_config,
            )
        self.encoder.vision_model.embeddings.convert_conv2d_to_linear(bagel_config.vit_config)
        self.encoder.to(dtype)
        self.position_embedding = PositionEmbedding(
            bagel_config.vit_max_num_patch_per_side,
            bagel_config.llm_config.hidden_size,
        ).to(dtype)
        if recompute:
            self._enable_recompute()

    def _enable_recompute(self) -> None:
        """Checkpoint each official SigLIP encoder layer."""
        from torch.utils.checkpoint import checkpoint

        for layer in self.encoder.vision_model.encoder.layers:
            original_forward = layer.forward

            def checkpointed_forward(*args: Any, _forward=original_forward, **kwargs: Any) -> Any:
                if torch.is_grad_enabled():
                    return checkpoint(_forward, *args, use_reentrant=False, **kwargs)
                return _forward(*args, **kwargs)

            layer.forward = checkpointed_forward

    def forward(
        self,
        packed_vit_tokens: torch.Tensor,
        packed_vit_position_ids: torch.Tensor,
        vit_token_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode packed patches and return BAGEL's post-connector positions."""
        dtype = next(self.encoder.parameters()).dtype
        packed_vit_tokens = packed_vit_tokens.to(dtype=dtype)
        cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0)).to(torch.int32)
        max_seqlen = int(vit_token_seqlens.max())
        use_autocast = packed_vit_tokens.is_cuda and dtype in (torch.float16, torch.bfloat16)
        with torch.amp.autocast("cuda", enabled=use_autocast, dtype=dtype):
            embeddings = self.encoder(
                packed_pixel_values=packed_vit_tokens,
                packed_flattened_position_ids=packed_vit_position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        positions = self.position_embedding(packed_vit_position_ids).to(dtype=dtype)
        return embeddings.to(dtype=dtype), positions


class BagelVisionSubmodule(VisionModalitySubmodules):
    """Project packed SigLIP embeddings and add BAGEL positions."""

    def forward(self, encoder_inputs: dict[str, Any]) -> torch.Tensor:
        """Run the one BAGEL vision encoder and connector."""
        encoder = self.encoders["vision_encoder"]
        embeddings, positions = encoder(**encoder_inputs["vision_encoder"])
        projected = self.input_projections[0](embeddings)
        if projected.shape != positions.shape:
            raise ValueError("BAGEL vision and position embedding shapes differ")
        return projected + positions


class BagelDiffusionSubmodule(ModalitySubmodules):
    """Combine noisy latents with BAGEL timestep and position embeddings."""

    def __init__(self, *args: Any, dtype: torch.dtype = torch.float32, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.dtype = dtype

    def encode(self, encoders_data_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Encode packed timesteps and latent positions."""
        return {
            "timestep": self.encoders["timestep"](encoders_data_batch["shifted_timesteps"]).to(self.dtype),
            "position": self.encoders["latent_position_ids"](encoders_data_batch["latent_position_ids"]).to(
                self.dtype
            ),
        }

    def decode(self, embeddings: torch.Tensor, data_batch: dict[str, Any]) -> torch.Tensor:
        """Reject decoding because training predicts velocity directly."""
        raise NotImplementedError("BAGEL training does not decode VAE latents")

    def forward(self, encoder_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Build the visual-generation token embeddings."""
        embeddings = self.encode(encoder_inputs)
        latents = self.input_projections[0](encoder_inputs["latents"].to(self.dtype))
        return latents + embeddings["timestep"] + embeddings["position"]

    def llm2vae(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project language hidden states into latent-patch velocity."""
        return self.output_projections[0](embeddings)
