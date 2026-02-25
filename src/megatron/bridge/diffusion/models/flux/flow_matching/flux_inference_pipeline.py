# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""FLUX inference pipeline for text-to-image generation."""

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from megatron.bridge.training.model_load_save import load_megatron_model as _load_megatron_model
from PIL import Image
from torch import nn
from tqdm import tqdm

from megatron.bridge.diffusion.models.flux.flux_provider import FluxProvider


@dataclass
class T5Config:
    """T5 encoder configuration."""

    version: Optional[str] = field(default_factory=lambda: "google/t5-v1_1-xxl")
    max_length: Optional[int] = field(default_factory=lambda: 512)
    load_config_only: bool = False
    device: str = "cuda"


@dataclass
class ClipConfig:
    """CLIP encoder configuration."""

    version: Optional[str] = field(default_factory=lambda: "openai/clip-vit-large-patch14")
    max_length: Optional[int] = field(default_factory=lambda: 77)
    always_return_pooled: Optional[bool] = field(default_factory=lambda: True)
    device: str = "cuda"


class FlowMatchEulerDiscreteScheduler:
    """
    Euler scheduler.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        timestep_spacing (`str`, defaults to `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        shift (`float`, defaults to 1.0):
            The shift value for the timestep schedule.
    """

    _compatibles = []
    order = 1

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting=False,
        base_shift: Optional[float] = 0.5,
        max_shift: Optional[float] = 1.15,
        base_image_seq_len: Optional[int] = 256,
        max_image_seq_len: Optional[int] = 4096,
    ):
        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / num_train_timesteps
        if not use_dynamic_shifting:
            # when use_dynamic_shifting is True, we apply the timestep shifting on the fly based on the image resolution
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps

        self._step_index = None
        self._begin_index = None

        self.sigmas = sigmas.to("cpu")  # to avoid too much CPU/GPU communication
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

        self.base_shift = base_shift
        self.max_shift = max_shift
        self.base_image_seq_len = base_image_seq_len
        self.max_image_seq_len = max_image_seq_len
        self.use_dynamic_shifting = use_dynamic_shifting
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Forward process in flow-matching

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(device=sample.device, dtype=sample.dtype)

        if sample.device.type == "mps" and torch.is_floating_point(timestep):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(sample.device, dtype=torch.float32)
            timestep = timestep.to(sample.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(sample.device)
            timestep = timestep.to(sample.device)

        # self.begin_index is None when scheduler is used for training, or pipeline does not implement set_begin_index
        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timestep]
        elif self.step_index is not None:
            # add_noise is called after first denoising step (for inpainting)
            step_indices = [self.step_index] * timestep.shape[0]
        else:
            # add noise is called before first denoising step to create initial latent(img2img)
            step_indices = [self.begin_index] * timestep.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(sample.shape):
            sigma = sigma.unsqueeze(-1)

        sample = sigma * noise + (1.0 - sigma) * sample

        return sample

    def _sigma_to_t(self, sigma):
        return sigma * self.num_train_timesteps

    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """

        if self.use_dynamic_shifting and mu is None:
            raise ValueError(" you have a pass a value for `mu` when `use_dynamic_shifting` is set to be `True`")

        if sigmas is None:
            self.num_inference_steps = num_inference_steps
            timesteps = np.linspace(
                self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
            )

            sigmas = timesteps / self.num_train_timesteps

        if self.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        timesteps = sigmas * self.num_train_timesteps

        self.timesteps = timesteps.to(device=device)
        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0):
                Scaling factor for noise added to the sample.
            generator (`torch.Generator`, *optional*):
                A random number generator.

        Returns:
            A tuple is returned where the first element is the sample tensor.
        """

        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
        prev_sample = sample + (sigma_next - sigma) * model_output

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        return (prev_sample,)

    def __len__(self):
        return self.num_train_timesteps


class FluxInferencePipeline(nn.Module):
    """
    FLUX inference pipeline for text-to-image generation.

    This pipeline orchestrates the full inference process including:
    - Text encoding with T5 and CLIP
    - Latent preparation and denoising
    - VAE decoding to images

    Args:
        params: FluxModelParams configuration.
        flux: Optional pre-initialized Flux model.
        scheduler_steps: Number of scheduler steps.

    Example:
        >>> params = FluxModelParams()
        >>> pipeline = FluxInferencePipeline(params)
        >>> pipeline.load_from_pretrained("path/to/flux_ckpt")
        >>> images = pipeline(
        ...     prompt=["A cat holding a sign that says hello world"],
        ...     height=1024,
        ...     width=1024,
        ...     num_inference_steps=20,
        ... )
    """

    def __init__(
        self,
        flux_checkpoint_dir: Optional[str] = None,
        t5_checkpoint_dir: Optional[str] = None,
        clip_checkpoint_dir: Optional[str] = None,
        vae_checkpoint_dir: Optional[str] = None,
        scheduler_steps: int = 1000,
    ):
        super().__init__()

        # Initialize transformer
        self.transformer = self.setup_model_from_checkpoint(flux_checkpoint_dir)
        self.device = "cuda:0"

        # VAE scale factor based on channel multipliers
        # if params and params.vae_config:
        #     self.vae_scale_factor = 2 ** len(params.vae_config.ch_mult)
        # else:
        self.vae_scale_factor = 16  # Default for FLUX

        # Initialize scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=scheduler_steps)

        # Placeholders for encoders (to be loaded separately)
        self.load_text_encoders(t5_checkpoint_dir, clip_checkpoint_dir)
        self.load_vae(vae_checkpoint_dir)

    def setup_model_from_checkpoint(self, checkpoint_dir):
        provider = FluxProvider()
        # provider.tensor_model_parallel_size = self.tensor_parallel_size
        # provider.pipeline_model_parallel_size = self.pipeline_parallel_size
        # provider.context_parallel_size = self.context_parallel_size
        # provider.sequence_parallel = self.sequence_parallel
        # provider.pipeline_dtype = self.pipeline_dtype
        # Once all overrides are set, finalize the model provider to ensure the post initialization logic is run
        provider.finalize()
        provider.initialize_model_parallel(seed=0)

        ## Read from megatron checkpoint
        model = _load_megatron_model(
            checkpoint_dir,
            # mp_overrides={
            #     "tensor_model_parallel_size": self.tensor_parallel_size,
            #     "pipeline_model_parallel_size": self.pipeline_parallel_size,
            #     "context_parallel_size": self.context_parallel_size,
            #     "sequence_parallel": self.sequence_parallel,
            #     "pipeline_dtype": self.pipeline_dtype,
            # },
        )
        if isinstance(model, list):
            model = model[0]
        if hasattr(model, "module"):
            model = model.module

        return model

    def load_text_encoders(self, t5_version: str = None, clip_version: str = None):
        """
        Load T5 and CLIP text encoders.

        Args:
            t5_version: HuggingFace model ID or path for T5.
            clip_version: HuggingFace model ID or path for CLIP.
        """
        try:
            from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer

            # Load T5
            t5_version = t5_version or "google/t5-v1_1-xxl"
            print(f"Loading T5 encoder from {t5_version}...")
            self.t5_tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
            self.t5_encoder = T5EncoderModel.from_pretrained(t5_version).to(self.device).eval()

            # Load CLIP
            clip_version = clip_version or "openai/clip-vit-large-patch14"
            print(f"Loading CLIP encoder from {clip_version}...")
            self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            self.clip_encoder = CLIPTextModel.from_pretrained(clip_version).to(self.device).eval()

            print("Text encoders loaded successfully")
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")

    def load_vae(self, vae_path: str):
        """
        Load VAE from checkpoint.

        Args:
            vae_path: Path to VAE checkpoint (ae.safetensors).
        """
        try:
            from diffusers import AutoencoderKL

            self.vae = AutoencoderKL.from_pretrained(vae_path).to(self.device).eval()
        except ImportError:
            raise ImportError("Please install diffusers: pip install diffusers")

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        max_sequence_length: int = 512,
        num_images_per_prompt: int = 1,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Encode text prompts using T5 and CLIP.

        Returns:
            Tuple of (prompt_embeds, pooled_prompt_embeds, text_ids).
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        # T5 encoding
        t5_inputs = self.t5_tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            prompt_embeds = self.t5_encoder(input_ids=t5_inputs.input_ids).last_hidden_state

        # CLIP encoding
        clip_inputs = self.clip_tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            clip_output = self.clip_encoder(input_ids=clip_inputs.input_ids)
            pooled_prompt_embeds = clip_output.pooler_output

        # Repeat for multiple images per prompt
        seq_len = prompt_embeds.shape[1]
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1).to(dtype=dtype)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1).to(dtype=dtype)

        # Create text IDs
        text_ids = torch.zeros(batch_size * num_images_per_prompt, seq_len, 3, device=device, dtype=dtype)

        return prompt_embeds.transpose(0, 1), pooled_prompt_embeds, text_ids

    @staticmethod
    def _prepare_latent_image_ids(batch_size: int, height: int, width: int, device: torch.device, dtype: torch.dtype):
        """Prepare latent image IDs for position encoding."""
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

        latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
        latent_image_ids = latent_image_ids.reshape(batch_size, (height // 2) * (width // 2), 3)

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        """Pack latents for FLUX processing."""
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        """Unpack latents for VAE decoding."""
        batch_size, num_patches, channels = latents.shape
        height = height // vae_scale_factor
        width = width // vae_scale_factor

        latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // 4, height * 2, width * 2)
        return latents

    @staticmethod
    def _calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.16):
        """Calculate timestep shift based on sequence length."""
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        return image_seq_len * m + b

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator=None):
        """Prepare random latents for generation."""
        height = 2 * int(height) // self.vae_scale_factor
        width = 2 * int(width) // self.vae_scale_factor

        shape = (batch_size, num_channels_latents, height, width)
        latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)

        return latents.transpose(0, 1), latent_image_ids

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 10,
        guidance_scale: float = 3.5,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        max_sequence_length: int = 512,
        output_type: str = "pil",
        output_path: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Generate images from text prompts.

        Args:
            prompt: Text prompt(s) for image generation.
            height: Output image height.
            width: Output image width.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            num_images_per_prompt: Number of images per prompt.
            generator: Random number generator for reproducibility.
            max_sequence_length: Maximum sequence length for text encoding.
            output_type: "pil" for PIL images, "latent" for latent tensors.
            output_path: Path to save generated images.
            dtype: Data type for inference.

        Returns:
            List of PIL images or latent tensors.
        """
        device = self.device

        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        # Encode prompts
        if self.t5_encoder is None or self.clip_encoder is None:
            raise RuntimeError("Text encoders not loaded. Call load_text_encoders() first.")

        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt,
            max_sequence_length=max_sequence_length,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=dtype,
        )

        # Prepare latents
        num_channels_latents = self.transformer.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
        )

        # Setup timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[0]
        mu = self._calculate_shift(
            image_seq_len,
            self.scheduler.base_image_seq_len,
            self.scheduler.max_image_seq_len,
            self.scheduler.base_shift,
            self.scheduler.max_shift,
        )
        self.scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
        timesteps = self.scheduler.timesteps

        # Denoising loop
        for t in tqdm(timesteps, desc="Denoising"):
            timestep = t.expand(latents.shape[1]).to(device=device, dtype=dtype)

            if self.transformer.guidance_embed:
                guidance = torch.full((latents.shape[1],), guidance_scale, device=device, dtype=dtype)
            else:
                guidance = None

            with torch.autocast(device_type="cuda", dtype=dtype):
                pred = self.transformer(
                    img=latents,
                    txt=prompt_embeds,
                    y=pooled_prompt_embeds,
                    timesteps=timestep / 1000,
                    img_ids=latent_image_ids,
                    txt_ids=text_ids,
                    guidance=guidance,
                )
                latents = self.scheduler.step(pred, t, latents)[0]

        if output_type == "latent":
            return latents.transpose(0, 1)

        # Decode latents to images
        if self.vae is None:
            raise RuntimeError("VAE not loaded. Call load_vae() first.")

        latents = self._unpack_latents(latents.transpose(0, 1), height, width, self.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        with torch.autocast(device_type="cuda", dtype=dtype):
            images = self.vae.decode(latents, return_dict=False)[0]

        # Post-process
        images = FluxInferencePipeline.denormalize(images)
        images = FluxInferencePipeline.torch_to_numpy(images)
        images = FluxInferencePipeline.numpy_to_pil(images)

        # Save if requested
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            assert len(images) == int(len(prompt) * num_images_per_prompt)
            prompt = [p[:40] + f"_{idx}" for p in prompt for idx in range(num_images_per_prompt)]
            for file_name, image in zip(prompt, images):
                image.save(os.path.join(output_path, f"{file_name}.png"))

        return images

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    @staticmethod
    def torch_to_numpy(images):
        """
        Convert a torch image or a batch of images to a numpy image.
        """
        numpy_images = images.float().cpu().permute(0, 2, 3, 1).numpy()
        return numpy_images

    @staticmethod
    def denormalize(image):
        # pylint: disable=C0116
        return (image / 2 + 0.5).clamp(0, 1)
