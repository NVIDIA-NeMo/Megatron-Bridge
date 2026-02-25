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


import logging
import math
from functools import lru_cache, partial
from typing import Iterable

import torch
from megatron.bridge.training.losses import masked_next_token_loss
from megatron.bridge.training.state import GlobalState
from megatron.core import parallel_state
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.utils import get_model_config

from megatron.bridge.diffusion.models.flux.flow_matching.flux_inference_pipeline import FlowMatchEulerDiscreteScheduler


logger = logging.getLogger(__name__)


def flux_data_step(dataloader_iter, store_in_state=False):
    """Process batch data for FLUX model.

    Args:
        dataloader_iter: Iterator over the dataloader.
        store_in_state: If True, store the batch in GlobalState for callbacks.

    Returns:
        Processed batch dictionary with tensors moved to CUDA.
    """
    batch = next(dataloader_iter)
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch

    _batch = {k: v.to(device="cuda", non_blocking=True) if torch.is_tensor(v) else v for k, v in _batch.items()}

    if "loss_mask" not in _batch or _batch["loss_mask"] is None:
        _batch["loss_mask"] = torch.ones(1, device="cuda")

    # Store batch in state for callbacks (e.g., validation image generation)
    if store_in_state:
        try:
            from megatron.bridge.training.pretrain import get_current_state

            state = get_current_state()
            state._last_validation_batch = _batch
        except:
            pass  # If state access fails, silently continue

    return _batch


class FluxForwardStep:
    """Forward step for FLUX diffusion model training.

    This class handles the forward pass during training, including:
    - Timestep sampling using flow matching
    - Noise injection with latent packing
    - Model prediction
    - Loss computation

    Args:
        timestep_sampling: Method for sampling timesteps ("logit_normal", "uniform", "mode").
        logit_mean: Mean for logit-normal sampling.
        logit_std: Standard deviation for logit-normal sampling.
        mode_scale: Scale for mode sampling.
        scheduler_steps: Number of scheduler training steps.
        guidance_scale: Guidance scale for FLUX-dev models.
    """

    def __init__(
        self,
        timestep_sampling: str = "logit_normal",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        mode_scale: float = 1.29,
        scheduler_steps: int = 1000,
        guidance_scale: float = 3.5,
    ):
        self.timestep_sampling = timestep_sampling
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.mode_scale = mode_scale
        self.scheduler_steps = scheduler_steps
        self.guidance_scale = guidance_scale
        self.autocast_dtype = torch.bfloat16
        # Initialize scheduler for timestep/sigma computations
        self.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=scheduler_steps)

    def __call__(
        self, state: GlobalState, data_iterator: Iterable, model: VisionModule
    ) -> tuple[torch.Tensor, partial]:
        """Forward training step.

        Args:
            state: Global state for the run.
            data_iterator: Input data iterator.
            model: The FLUX model.

        Returns:
            Tuple containing the output tensor and the loss function.
        """
        timers = state.timers
        straggler_timer = state.straggler_timer

        config = get_model_config(model)

        timers("batch-generator", log_level=2).start()

        with straggler_timer(bdata=True):
            batch = flux_data_step(data_iterator)
            # Store batch for validation callbacks (only during evaluation)
            if not torch.is_grad_enabled():
                state._last_batch = batch
        timers("batch-generator").stop()

        check_for_nan_in_loss = state.cfg.rerun_state_machine.check_for_nan_in_loss
        check_for_spiky_loss = state.cfg.rerun_state_machine.check_for_spiky_loss

        # Run diffusion training step
        with straggler_timer:
            if parallel_state.is_pipeline_last_stage():
                output_tensor, loss, loss_mask = self._training_step(model, batch, config)
                batch["loss_mask"] = loss_mask
            else:
                output_tensor = self._training_step(model, batch, config)

        loss = output_tensor
        if "loss_mask" not in batch or batch["loss_mask"] is None:
            loss_mask = torch.ones_like(loss)
        else:
            loss_mask = batch["loss_mask"]

        loss_function = self._create_loss_function(loss_mask, check_for_nan_in_loss, check_for_spiky_loss)

        return output_tensor, loss_function

    def _training_step(
        self, model: VisionModule, batch: dict, config
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
        """Perform single training step with flow matching.

        Args:
            model: The FLUX model.
            batch: Data batch containing latents and text embeddings.
            config: Model configuration.

        Returns:
            On last pipeline stage: tuple of (output_tensor, loss, loss_mask).
            On other stages: hidden_states tensor.
        """
        # Get latents from batch - expected in [B, C, H, W] format
        if "latents" in batch:
            latents = batch["latents"]
        else:
            raise ValueError("Expected 'latents' in batch. VAE encoding should be done in data preprocessing.")

        # Prepare image latents with flow matching noise
        (
            latents,
            noise,
            packed_noisy_model_input,
            latent_image_ids,
            guidance_vec,
            timesteps,
        ) = self.prepare_image_latent(latents, model)

        # Get text embeddings (precached)
        if "prompt_embeds" in batch:
            prompt_embeds = batch["prompt_embeds"].transpose(0, 1)
            pooled_prompt_embeds = batch["pooled_prompt_embeds"]
            text_ids = batch["text_ids"]
        else:
            raise ValueError("Expected precached text embeddings in batch.")

        # Forward pass
        with torch.amp.autocast(
            "cuda", enabled=self.autocast_dtype in (torch.half, torch.bfloat16), dtype=self.autocast_dtype
        ):
            noise_pred = model(
                img=packed_noisy_model_input,
                txt=prompt_embeds,
                y=pooled_prompt_embeds,
                timesteps=timesteps / 1000,
                img_ids=latent_image_ids,
                txt_ids=text_ids,
                guidance=guidance_vec,
            )

            # Unpack predictions for loss computation
            noise_pred = self._unpack_latents(
                noise_pred.transpose(0, 1),
                latents.shape[2],
                latents.shape[3],
            ).transpose(0, 1)

            # Flow matching target: v = noise - latents (velocity formulation)
            target = noise - latents

            # MSE loss
            loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
            output_tensor = torch.mean(loss, dim=-1)

            # Create loss mask (all ones for now)
            loss_mask = torch.ones_like(output_tensor)

            return output_tensor, loss, loss_mask

        # else:
        #     hidden_states = model(
        #         img=packed_noisy_model_input,
        #         txt=prompt_embeds,
        #         y=pooled_prompt_embeds,
        #         timesteps=timesteps / 1000,
        #         img_ids=latent_image_ids,
        #         txt_ids=text_ids,
        #         guidance=guidance_vec,
        #     )
        #     return hidden_states

    def prepare_image_latent(self, latents: torch.Tensor, model: VisionModule):
        """Prepare image latents with flow matching noise.

        Args:
            latents: Input latent tensor [B, C, H, W].
            model: The FLUX model (for guidance_embed config).

        Returns:
            Tuple of (latents, noise, packed_noisy_input, latent_image_ids, guidance, timesteps).
        """
        latent_image_ids = self._prepare_latent_image_ids(
            latents.shape[0],
            latents.shape[2],
            latents.shape[3],
            latents.device,
            latents.dtype,
        )

        noise = torch.randn_like(latents, device=latents.device, dtype=latents.dtype)
        batch_size = latents.shape[0]
        u = self.compute_density_for_timestep_sampling(
            self.timestep_sampling,
            batch_size,
        )
        indices = (u * self.scheduler.num_train_timesteps).long()
        timesteps = self.scheduler.timesteps[indices].to(device=latents.device)

        sigmas = self.scheduler.sigmas.to(device=latents.device, dtype=latents.dtype)
        scheduler_timesteps = self.scheduler.timesteps.to(device=latents.device)
        step_indices = [(scheduler_timesteps == t).nonzero().item() for t in timesteps]
        timesteps = timesteps.to(dtype=latents.dtype)
        sigma = sigmas[step_indices].flatten()

        while len(sigma.shape) < latents.ndim:
            sigma = sigma.unsqueeze(-1)

        noisy_model_input = (1.0 - sigma) * latents + sigma * noise
        packed_noisy_model_input = self._pack_latents(
            noisy_model_input,
            batch_size=latents.shape[0],
            num_channels_latents=latents.shape[1],
            height=latents.shape[2],
            width=latents.shape[3],
        )

        # Guidance embedding (for FLUX-dev)
        if hasattr(model, "guidance_embed") and model.guidance_embed:
            guidance_vec = torch.full(
                (noisy_model_input.shape[0],),
                self.guidance_scale,
                device=latents.device,
                dtype=latents.dtype,
            )
        else:
            guidance_vec = None

        return (
            latents.transpose(0, 1),
            noise.transpose(0, 1),
            packed_noisy_model_input.transpose(0, 1),
            latent_image_ids,
            guidance_vec,
            timesteps,
        )

    def compute_density_for_timestep_sampling(
        self,
        weighting_scheme: str,
        batch_size: int,
        logit_mean: float = None,
        logit_std: float = None,
        mode_scale: float = None,
    ) -> torch.Tensor:
        """Compute the density for sampling the timesteps when doing SD3 training.

        Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.
        SD3 paper reference: https://arxiv.org/abs/2403.03206v1.

        Args:
            weighting_scheme: Sampling scheme ("logit_normal", "mode", or "uniform").
            batch_size: Number of samples in batch.
            logit_mean: Mean for logit-normal sampling.
            logit_std: Standard deviation for logit-normal sampling.
            mode_scale: Scale for mode sampling.

        Returns:
            Tensor of sampled u values in [0, 1].
        """
        # Use instance defaults if not provided
        logit_mean = logit_mean if logit_mean is not None else self.logit_mean
        logit_std = logit_std if logit_std is not None else self.logit_std
        mode_scale = mode_scale if mode_scale is not None else self.mode_scale

        if weighting_scheme == "logit_normal":
            # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$)
            u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
            u = torch.nn.functional.sigmoid(u)
        elif weighting_scheme == "mode":
            u = torch.rand(size=(batch_size,), device="cpu")
            u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
        else:
            u = torch.rand(size=(batch_size,), device="cpu")
        return u

    @lru_cache
    def _prepare_latent_image_ids(
        self, batch_size: int, height: int, width: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Prepare latent image IDs for positional encoding.

        Args:
            batch_size: Number of samples.
            height: Latent height.
            width: Latent width.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Tensor of shape [B, (H/2)*(W/2), 3] with position IDs.
        """
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
        latent_image_ids = latent_image_ids.reshape(
            batch_size, latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype, non_blocking=True)

    def _pack_latents(
        self, latents: torch.Tensor, batch_size: int, num_channels_latents: int, height: int, width: int
    ) -> torch.Tensor:
        """Pack latents for FLUX processing.

        Rearranges [B, C, H, W] -> [B, (H/2)*(W/2), C*4].

        Args:
            latents: Input tensor [B, C, H, W].
            batch_size: Batch size.
            num_channels_latents: Number of latent channels.
            height: Latent height.
            width: Latent width.

        Returns:
            Packed tensor [B, num_patches, C*4].
        """
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        return latents

    def _unpack_latents(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Unpack latents from FLUX format.

        Rearranges [B, num_patches, C*4] -> [B, C, H, W].

        Args:
            latents: Packed tensor [B, num_patches, C*4].
            height: Target height.
            width: Target width.

        Returns:
            Unpacked tensor [B, C, H, W].
        """
        batch_size, num_patches, channels = latents.shape

        # Adjust h and w for patching
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // 4, height, width)

        return latents

    def _create_loss_function(
        self, loss_mask: torch.Tensor, check_for_nan_in_loss: bool, check_for_spiky_loss: bool
    ) -> partial:
        """Create a partial loss function with the specified configuration.

        Args:
            loss_mask: Used to mask out some portions of the loss.
            check_for_nan_in_loss: Whether to check for NaN values in the loss.
            check_for_spiky_loss: Whether to check for spiky loss values.

        Returns:
            A partial function that can be called with output_tensor to compute the loss.
        """
        return partial(
            masked_next_token_loss,
            loss_mask,
            check_for_nan_in_loss=check_for_nan_in_loss,
            check_for_spiky_loss=check_for_spiky_loss,
        )
