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

"""Official-semantics diffusion input preparation for BAGEL."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from megatron.bridge.diffusion.common.flow_matching.flow_matching_pipeline import LinearInterpolationSchedule


class BagelDiffusionScheduler:
    """Encode images and apply BAGEL's flow-matching schedule."""

    def __init__(
        self,
        *,
        bagel_repo: str,
        vae_path: str,
        latent_patch_size: int = 2,
        timestep_shift: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.bagel_repo = bagel_repo
        self.vae_path = vae_path
        self.latent_patch_size = latent_patch_size
        self.timestep_shift = timestep_shift
        self.dtype = dtype
        self.schedule = LinearInterpolationSchedule()
        self.vae: torch.nn.Module | None = None
        self.vae_params: Any = None

    def _ensure_vae(self) -> None:
        """Load BAGEL's frozen FP32 VAE on first training step."""
        if self.vae is not None:
            return
        repo = str(Path(self.bagel_repo).resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from modeling.autoencoder import load_ae

        self.vae, self.vae_params = load_ae(self.vae_path)
        self.vae.requires_grad_(False).eval().cuda()

    def shift_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Apply BAGEL's sigmoid and rational timestep shift."""
        timesteps = torch.sigmoid(timesteps)
        return self.timestep_shift * timesteps / (1 + (self.timestep_shift - 1) * timesteps)

    def add_noise(
        self,
        clean_latents: torch.Tensor,
        shifted_timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use Bridge's linear interpolation and BAGEL's velocity target."""
        noise = torch.randn_like(clean_latents)
        noisy_latents = self.schedule.forward(clean_latents, noise, shifted_timesteps)
        target = (noise - clean_latents)[shifted_timesteps > 0]
        return noisy_latents, target

    def encode_images(
        self,
        padded_images: torch.Tensor,
        latent_shapes: list[tuple[int, int]],
    ) -> torch.Tensor:
        """VAE-encode and patchify images with official BAGEL ordering."""
        self._ensure_vae()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=self.dtype):
            padded_latents = self.vae.encode(padded_images)
            packed_latents = []
            for latent, (height, width) in zip(padded_latents, latent_shapes):
                patch = self.latent_patch_size
                channels = self.vae_params.z_channels
                latent = latent[..., : height * patch, : width * patch]
                latent = latent.reshape(channels, height, patch, width, patch)
                packed_latents.append(torch.einsum("chpwq->hwpqc", latent).reshape(-1, channels * patch**2))
        return torch.cat(packed_latents)

    def prepare(self, batch: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return noisy latents, shifted timesteps, and velocity targets."""
        timesteps = batch["packed_timesteps"].cuda(non_blocking=True).to(self.dtype)
        shifted_timesteps = self.shift_timesteps(timesteps)
        clean_latents = self.encode_images(
            batch["padded_images"].cuda(non_blocking=True),
            batch["patchified_vae_latent_shapes"],
        )
        noisy_latents, target = self.add_noise(clean_latents, shifted_timesteps)
        return noisy_latents, shifted_timesteps, target
