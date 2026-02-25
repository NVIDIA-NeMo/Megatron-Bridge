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

"""
Megatron-specific adapter for FLUX models using the automodel FlowMatching pipeline.
"""

import random
from typing import Any, Dict

import torch
from megatron.core.models.common.vision_module.vision_module import VisionModule

from dfm.src.automodel.flow_matching.adapters.base import FlowMatchingContext, ModelAdapter


class MegatronFluxAdapter(ModelAdapter):
    """
    Adapter for FLUX models in Megatron training framework.

    Key differences from standard FluxAdapter:
    - Handles sequence-first tensor layout [S, B, ...] required by Megatron
    - Integrates with pipeline parallelism
    - Maps Megatron batch keys to expected format
    - Handles guidance embedding for FLUX-dev models
    """

    def __init__(self, guidance_scale: float = 3.5):
        """
        Initialize MegatronFluxAdapter.

        Args:
            guidance_scale: Guidance scale for classifier-free guidance
        """
        self.guidance_scale = guidance_scale

    def _pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Pack latents from [B, C, H, W] to Flux format [B, (H//2)*(W//2), C*4].

        Flux uses a 2x2 patch embedding, so latents are reshaped accordingly.
        """
        b, c, h, w = latents.shape
        latents = latents.view(b, c, h // 2, 2, w // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(b, (h // 2) * (w // 2), c * 4)
        return latents

    def _unpack_latents(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Unpack latents from Flux format [B, num_patches, C*4] back to [B, C, H, W].

        Args:
            latents: Packed latents of shape [B, num_patches, channels]
            height: Target latent height
            width: Target latent width
        """
        batch_size, num_patches, channels = latents.shape
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // 4, height, width)
        return latents

    def _prepare_latent_image_ids(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Prepare positional IDs for image latents.

        Returns tensor of shape [B, (H//2)*(W//2), 3] containing (batch_idx, y, x).
        """
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = torch.arange(width // 2)[None, :]

        latent_image_ids = latent_image_ids.reshape(-1, 3)
        latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1)
        return latent_image_ids.to(device=device, dtype=dtype)

    def prepare_inputs(self, context: FlowMatchingContext) -> Dict[str, Any]:
        """
        Prepare inputs for Megatron Flux model from FlowMatchingContext.

        Handles batch key mapping:
        - Megatron uses: latents, prompt_embeds, pooled_prompt_embeds, text_ids
        - Automodel expects: image_latents, text_embeddings, pooled_prompt_embeds
        """
        batch = context.batch
        device = context.device
        dtype = context.dtype

        # Get model reference if passed in batch for guidance check
        model = batch.get("_model")

        # Get latents - Megatron uses 'latents' key
        noisy_latents = context.noisy_latents
        if noisy_latents.ndim != 4:
            raise ValueError(f"MegatronFluxAdapter expects 4D latents [B, C, H, W], got {noisy_latents.ndim}D")

        batch_size, channels, height, width = noisy_latents.shape

        # Get text embeddings - Megatron uses 'prompt_embeds' (T5)
        if "prompt_embeds" in batch:
            # Megatron stores as [S, B, D], need to transpose to [B, S, D]
            text_embeddings = batch["prompt_embeds"]
            if text_embeddings.shape[1] == batch_size:  # Already [S, B, D]
                text_embeddings = text_embeddings.transpose(0, 1).to(device, dtype=dtype)
            else:
                text_embeddings = text_embeddings.to(device, dtype=dtype)
        else:
            raise ValueError("Expected 'prompt_embeds' in batch for Megatron FLUX training")

        # Get pooled embeddings (CLIP)
        if "pooled_prompt_embeds" in batch:
            pooled_projections = batch["pooled_prompt_embeds"].to(device, dtype=dtype)
        else:
            pooled_projections = torch.zeros(batch_size, 768, device=device, dtype=dtype)

        if pooled_projections.ndim == 1:
            pooled_projections = pooled_projections.unsqueeze(0)

        # Apply CFG dropout if needed
        if random.random() < context.cfg_dropout_prob:
            text_embeddings = torch.zeros_like(text_embeddings)
            pooled_projections = torch.zeros_like(pooled_projections)

        # Pack latents for Flux transformer
        packed_latents = self._pack_latents(noisy_latents)

        # Prepare positional IDs
        img_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)

        # Text positional IDs
        if "text_ids" in batch:
            txt_ids = batch["text_ids"].to(device, dtype=dtype)
        else:
            text_seq_len = text_embeddings.shape[1]
            txt_ids = torch.zeros(batch_size, text_seq_len, 3, device=device, dtype=dtype)

        # Timesteps - normalize to [0, 1] for FLUX
        timesteps = context.timesteps.to(dtype) / 1000.0

        # Guidance vector for FLUX-dev (only if model supports it)
        # Exactly match original implementation pattern
        guidance = None
        if model is not None:
            # Unwrap model wrappers (DDP, etc.)
            unwrapped = model
            while hasattr(unwrapped, "module"):
                unwrapped = unwrapped.module

            # Check if model has guidance enabled (matches original flux_step.py logic)
            if hasattr(unwrapped, "guidance_embed") and unwrapped.guidance_embed:
                guidance = torch.full((batch_size,), self.guidance_scale, device=device, dtype=torch.float32)

        # Transpose to sequence-first for Megatron: [B, ...] -> [S, B, ...]
        packed_latents = packed_latents.transpose(0, 1)
        text_embeddings = text_embeddings.transpose(0, 1)

        inputs = {
            "img": packed_latents,
            "txt": text_embeddings,
            "y": pooled_projections,
            "timesteps": timesteps,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
            # Store original shape for unpacking
            "_original_shape": (batch_size, channels, height, width),
        }

        # Only add guidance if model supports it
        if guidance is not None:
            inputs["guidance"] = guidance

        return inputs

    def forward(self, model: VisionModule, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Execute forward pass for Megatron Flux model.

        Returns unpacked prediction in [B, C, H, W] format.
        """
        original_shape = inputs.pop("_original_shape")
        batch_size, channels, height, width = original_shape

        # Megatron forward pass (guidance may be None if model doesn't support it)
        model_pred = model(
            img=inputs["img"],
            txt=inputs["txt"],
            y=inputs["y"],
            timesteps=inputs["timesteps"],
            img_ids=inputs["img_ids"],
            txt_ids=inputs["txt_ids"],
            guidance=inputs.get("guidance"),  # Use .get() in case it's None
        )

        # Handle potential tuple output and transpose back from sequence-first
        if isinstance(model_pred, tuple):
            model_pred = model_pred[0]

        # Transpose from [S, B, D] to [B, S, D]
        model_pred = model_pred.transpose(0, 1)

        # Unpack from Flux format back to [B, C, H, W]
        model_pred = self._unpack_latents(model_pred, height, width)

        return model_pred
