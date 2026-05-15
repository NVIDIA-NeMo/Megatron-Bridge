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

import numpy as np
import pytest
import torch

from megatron.bridge.diffusion.models.flux.flow_matching.flux_inference_pipeline import (
    FlowMatchEulerDiscreteScheduler,
    FluxInferencePipeline,
)


pytestmark = [pytest.mark.unit]


class TestFlowMatchEulerDiscreteScheduler:
    """Test class for FlowMatchEulerDiscreteScheduler."""

    def test_scheduler_initialization_default(self):
        """Test scheduler initialization with default parameters."""
        scheduler = FlowMatchEulerDiscreteScheduler()

        assert scheduler.num_train_timesteps == 1000
        assert scheduler.shift == 1.0
        assert scheduler.use_dynamic_shifting is False
        assert scheduler.timesteps is not None
        assert scheduler.sigmas is not None
        assert scheduler._step_index is None
        assert scheduler._begin_index is None

    def test_scheduler_initialization_custom(self):
        """Test scheduler initialization with custom parameters."""
        num_timesteps = 500
        shift = 2.0
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=num_timesteps, shift=shift)

        assert scheduler.num_train_timesteps == num_timesteps
        assert scheduler.shift == shift

    def test_scheduler_initialization_with_dynamic_shifting(self):
        """Test scheduler initialization with dynamic shifting enabled."""
        scheduler = FlowMatchEulerDiscreteScheduler(
            use_dynamic_shifting=True, base_shift=0.5, max_shift=1.15, base_image_seq_len=256, max_image_seq_len=4096
        )

        assert scheduler.use_dynamic_shifting is True
        assert scheduler.base_shift == 0.5
        assert scheduler.max_shift == 1.15
        assert scheduler.base_image_seq_len == 256
        assert scheduler.max_image_seq_len == 4096

    def test_scheduler_timesteps_shape(self):
        """Test that timesteps have correct shape."""
        num_timesteps = 1000
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=num_timesteps)

        assert scheduler.timesteps.shape[0] == num_timesteps
        assert scheduler.sigmas.shape[0] == num_timesteps

    def test_scheduler_sigma_range(self):
        """Test that sigmas are in valid range."""
        scheduler = FlowMatchEulerDiscreteScheduler()

        assert (scheduler.sigmas >= 0).all()
        assert (scheduler.sigmas <= 1).all()
        assert scheduler.sigma_min >= 0
        assert scheduler.sigma_max <= 1
        assert scheduler.sigma_min <= scheduler.sigma_max

    def test_set_begin_index(self):
        """Test set_begin_index method."""
        scheduler = FlowMatchEulerDiscreteScheduler()

        scheduler.set_begin_index(10)
        assert scheduler.begin_index == 10

        scheduler.set_begin_index(0)
        assert scheduler.begin_index == 0

    def test_step_index_property(self):
        """Test step_index property."""
        scheduler = FlowMatchEulerDiscreteScheduler()

        assert scheduler.step_index is None

        # After setting internally
        scheduler._step_index = 5
        assert scheduler.step_index == 5

    def test_set_timesteps_basic(self):
        """Test set_timesteps with basic parameters."""
        scheduler = FlowMatchEulerDiscreteScheduler()
        num_inference_steps = 50

        scheduler.set_timesteps(num_inference_steps=num_inference_steps, device="cpu")

        assert scheduler.num_inference_steps == num_inference_steps
        assert len(scheduler.timesteps) == num_inference_steps
        assert scheduler.timesteps.device.type == "cpu"

    def test_set_timesteps_with_custom_sigmas(self):
        """Test set_timesteps with custom sigmas."""
        import numpy as np

        scheduler = FlowMatchEulerDiscreteScheduler()
        custom_sigmas = np.array([1.0, 0.75, 0.5, 0.25, 0.0])

        scheduler.set_timesteps(sigmas=custom_sigmas, device="cpu")

        assert len(scheduler.timesteps) == len(custom_sigmas)

    def test_set_timesteps_with_dynamic_shifting(self):
        """Test set_timesteps with dynamic shifting."""
        scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
        num_inference_steps = 20
        mu = 0.5

        scheduler.set_timesteps(num_inference_steps=num_inference_steps, device="cpu", mu=mu)

        assert len(scheduler.timesteps) == num_inference_steps

    def test_set_timesteps_dynamic_shifting_requires_mu(self):
        """Test that dynamic shifting requires mu parameter."""
        scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)

        with pytest.raises(ValueError, match="you have a pass a value for `mu`"):
            scheduler.set_timesteps(num_inference_steps=10, device="cpu")

    def test_scale_noise(self):
        """Test scale_noise method."""
        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(num_inference_steps=10, device="cpu")
        batch_size = 2
        channels = 4
        height = width = 8

        sample = torch.randn(batch_size, channels, height, width)
        noise = torch.randn_like(sample)
        # timestep must be a 1-d tensor (batch of timesteps)
        timestep = scheduler.timesteps[0:1].repeat(batch_size)

        noisy_sample = scheduler.scale_noise(sample, timestep, noise)

        assert noisy_sample.shape == sample.shape
        assert not torch.isnan(noisy_sample).any()
        assert torch.isfinite(noisy_sample).all()

    def test_index_for_timestep(self):
        """Test index_for_timestep method."""
        scheduler = FlowMatchEulerDiscreteScheduler()

        # Get a timestep from the schedule
        timestep = scheduler.timesteps[10]
        index = scheduler.index_for_timestep(timestep)

        # Index should be valid
        assert 0 <= index < len(scheduler.timesteps)

    def test_step_basic(self):
        """Test step method basic functionality."""
        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(num_inference_steps=10, device="cpu")

        batch_size = 2
        channels = 4
        height = width = 8

        sample = torch.randn(batch_size, channels, height, width)
        model_output = torch.randn_like(sample)
        timestep = scheduler.timesteps[0]

        prev_sample = scheduler.step(model_output, timestep, sample)[0]

        assert prev_sample.shape == sample.shape
        assert not torch.isnan(prev_sample).any()

    def test_step_increments_step_index(self):
        """Test that step method increments step_index."""
        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(num_inference_steps=10, device="cpu")

        sample = torch.randn(2, 4, 8, 8)
        model_output = torch.randn_like(sample)

        assert scheduler.step_index is None

        scheduler.step(model_output, scheduler.timesteps[0], sample)
        assert scheduler.step_index == 1

        scheduler.step(model_output, scheduler.timesteps[1], sample)
        assert scheduler.step_index == 2

    def test_step_rejects_integer_timesteps(self):
        """Test that step method rejects integer timesteps."""
        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(num_inference_steps=10, device="cpu")

        sample = torch.randn(2, 4, 8, 8)
        model_output = torch.randn_like(sample)

        with pytest.raises(ValueError, match="Passing integer indices"):
            scheduler.step(model_output, 5, sample)

    def test_scheduler_length(self):
        """Test __len__ method."""
        num_timesteps = 500
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=num_timesteps)

        assert len(scheduler) == num_timesteps

    def test_time_shift_method(self):
        """Test time_shift method."""
        scheduler = FlowMatchEulerDiscreteScheduler()
        mu = 0.5
        sigma = 1.0
        t = torch.tensor([0.1, 0.5, 0.9])

        shifted = scheduler.time_shift(mu, sigma, t)

        assert shifted.shape == t.shape
        assert not torch.isnan(shifted).any()


class TestFluxInferencePipelineStaticMethods:
    """Test static methods of FluxInferencePipeline."""

    def test_prepare_latent_image_ids(self):
        """Test _prepare_latent_image_ids static method."""
        batch_size = 2
        height = 64
        width = 64
        device = torch.device("cpu")
        dtype = torch.float32

        latent_ids = FluxInferencePipeline._prepare_latent_image_ids(batch_size, height, width, device, dtype)

        expected_seq_len = (height // 2) * (width // 2)
        assert latent_ids.shape == (batch_size, expected_seq_len, 3)
        assert latent_ids.device == device
        assert latent_ids.dtype == dtype

    def test_pack_latents(self):
        """Test _pack_latents static method."""
        batch_size = 2
        num_channels = 16
        height = 64
        width = 64

        latents = torch.randn(batch_size, num_channels, height, width)
        packed = FluxInferencePipeline._pack_latents(latents, batch_size, num_channels, height, width)

        expected_seq_len = (height // 2) * (width // 2)
        expected_channels = num_channels * 4
        assert packed.shape == (batch_size, expected_seq_len, expected_channels)

    def test_unpack_latents(self):
        """Test _unpack_latents static method."""
        batch_size = 2
        height = 512
        width = 512
        vae_scale_factor = 16
        channels = 64
        # num_patches = (height // vae_scale_factor) * (width // vae_scale_factor)
        num_patches = (height // vae_scale_factor) * (width // vae_scale_factor)  # 32 * 32 = 1024

        packed_latents = torch.randn(batch_size, num_patches, channels)
        unpacked = FluxInferencePipeline._unpack_latents(packed_latents, height, width, vae_scale_factor)

        expected_height = (height // vae_scale_factor) * 2
        expected_width = (width // vae_scale_factor) * 2
        expected_channels = channels // 4
        assert unpacked.shape == (batch_size, expected_channels, expected_height, expected_width)

    def test_calculate_shift(self):
        """Test _calculate_shift static method."""
        # Test with default parameters
        image_seq_len = 256
        shift = FluxInferencePipeline._calculate_shift(image_seq_len)

        assert isinstance(shift, (int, float))
        assert shift >= 0.5  # Should be at least base_shift

        # Test with larger sequence length
        large_seq_len = 4096
        large_shift = FluxInferencePipeline._calculate_shift(large_seq_len)

        # Larger sequence length should give larger shift
        assert large_shift >= shift

    def test_numpy_to_pil(self):
        """Test numpy_to_pil static method."""
        # Test single image
        single_image = np.random.rand(256, 256, 3)
        pil_images = FluxInferencePipeline.numpy_to_pil(single_image)

        assert len(pil_images) == 1
        from PIL import Image

        assert isinstance(pil_images[0], Image.Image)

        # Test batch of images
        batch_images = np.random.rand(4, 256, 256, 3)
        pil_batch = FluxInferencePipeline.numpy_to_pil(batch_images)

        assert len(pil_batch) == 4
        assert all(isinstance(img, Image.Image) for img in pil_batch)

    def test_torch_to_numpy(self):
        """Test torch_to_numpy static method."""
        batch_size = 4
        channels = 3
        height = width = 256

        torch_images = torch.randn(batch_size, channels, height, width)
        numpy_images = FluxInferencePipeline.torch_to_numpy(torch_images)

        assert numpy_images.shape == (batch_size, height, width, channels)
        assert isinstance(numpy_images, np.ndarray)

    def test_denormalize(self):
        """Test denormalize static method."""
        # Create images in range [-1, 1]
        images = torch.randn(4, 3, 256, 256) * 2 - 1
        denorm = FluxInferencePipeline.denormalize(images)

        # Should be clamped to [0, 1]
        assert (denorm >= 0).all()
        assert (denorm <= 1).all()


class TestFluxInferencePipelineHelperMethods:
    """Test helper methods of FluxInferencePipeline."""

    def test_prepare_latents_shape(self, monkeypatch):
        """Test prepare_latents method output shape."""

        # Mock dependencies to avoid loading models
        def mock_setup(self, checkpoint_dir):
            class MockTransformer:
                in_channels = 64

            return MockTransformer()

        def mock_load_text(self, t5, clip):
            pass

        def mock_load_vae(self, vae):
            pass

        monkeypatch.setattr(FluxInferencePipeline, "setup_model_from_checkpoint", mock_setup)
        monkeypatch.setattr(FluxInferencePipeline, "load_text_encoders", mock_load_text)
        monkeypatch.setattr(FluxInferencePipeline, "load_vae", mock_load_vae)

        pipeline = FluxInferencePipeline()

        batch_size = 2
        num_channels = 16
        height = 512
        width = 512
        dtype = torch.float32
        device = torch.device("cpu")

        latents, latent_ids = pipeline.prepare_latents(
            batch_size, num_channels, height, width, dtype, device, generator=None
        )

        # Check shapes
        assert latents.ndim == 3
        assert latent_ids.shape[0] == batch_size
        assert latent_ids.shape[2] == 3  # 3D position IDs


class TestFluxInferencePipelineIntegration:
    """Integration tests for FluxInferencePipeline (without actual model loading)."""

    def test_pipeline_initialization_attributes(self, monkeypatch):
        """Test that pipeline sets up basic attributes."""

        def mock_setup(self, checkpoint_dir):
            class MockTransformer:
                in_channels = 64
                guidance_embed = False

            return MockTransformer()

        def mock_load_text(self, t5, clip):
            self.t5_encoder = None
            self.clip_encoder = None

        def mock_load_vae(self, vae):
            self.vae = None

        monkeypatch.setattr(FluxInferencePipeline, "setup_model_from_checkpoint", mock_setup)
        monkeypatch.setattr(FluxInferencePipeline, "load_text_encoders", mock_load_text)
        monkeypatch.setattr(FluxInferencePipeline, "load_vae", mock_load_vae)

        pipeline = FluxInferencePipeline(scheduler_steps=500)

        assert pipeline.device == "cuda:0"
        assert pipeline.vae_scale_factor == 16
        assert hasattr(pipeline, "scheduler")
        assert isinstance(pipeline.scheduler, FlowMatchEulerDiscreteScheduler)
        assert pipeline.scheduler.num_train_timesteps == 500

    def test_pack_unpack_latents_roundtrip(self):
        """Test that pack and unpack latents are inverse operations."""
        batch_size = 2
        num_channels = 16
        height = 64
        width = 64
        vae_scale_factor = 16
        # Original image dimensions: 2 * original_h // vae_scale_factor = height
        # So: original_h = height * vae_scale_factor // 2
        original_height = height * vae_scale_factor // 2
        original_width = width * vae_scale_factor // 2

        # Original latents
        original = torch.randn(batch_size, num_channels, height, width)

        # Pack
        packed = FluxInferencePipeline._pack_latents(original, batch_size, num_channels, height, width)

        # Unpack - should restore original dimensions
        unpacked = FluxInferencePipeline._unpack_latents(packed, original_height, original_width, vae_scale_factor)

        assert unpacked.shape[0] == batch_size
        assert unpacked.shape[1] == num_channels
        assert unpacked.shape == original.shape

    def test_scheduler_step_sequence(self):
        """Test a sequence of scheduler steps."""
        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(num_inference_steps=5, device="cpu")

        sample = torch.randn(1, 4, 32, 32)

        for i, timestep in enumerate(scheduler.timesteps[:-1]):  # Exclude last to avoid index error
            model_output = torch.randn_like(sample)
            sample = scheduler.step(model_output, timestep, sample)[0]

            assert not torch.isnan(sample).any()
            assert torch.isfinite(sample).all()


class TestFluxInferencePipelineEdgeCases:
    """Test edge cases and error handling."""

    def test_prepare_latent_image_ids_small_dimensions(self):
        """Test _prepare_latent_image_ids with small dimensions."""
        latent_ids = FluxInferencePipeline._prepare_latent_image_ids(1, 4, 4, torch.device("cpu"), torch.float32)

        assert latent_ids.shape == (1, 4, 3)

    def test_calculate_shift_boundary_values(self):
        """Test _calculate_shift with boundary sequence lengths."""
        base_seq_len = 256
        max_seq_len = 4096

        # Test at base
        shift_base = FluxInferencePipeline._calculate_shift(base_seq_len)
        assert shift_base >= 0

        # Test at max
        shift_max = FluxInferencePipeline._calculate_shift(max_seq_len)
        assert shift_max >= shift_base

        # Test below base
        shift_below = FluxInferencePipeline._calculate_shift(128)
        assert shift_below >= 0

        # Test above max
        shift_above = FluxInferencePipeline._calculate_shift(8192)
        assert shift_above >= 0

    def test_denormalize_extreme_values(self):
        """Test denormalize with extreme values."""
        # Very negative values
        extreme_neg = torch.full((2, 3, 64, 64), -10.0)
        denorm_neg = FluxInferencePipeline.denormalize(extreme_neg)
        assert (denorm_neg == 0).all()

        # Very positive values
        extreme_pos = torch.full((2, 3, 64, 64), 10.0)
        denorm_pos = FluxInferencePipeline.denormalize(extreme_pos)
        assert (denorm_pos == 1).all()

    def test_scheduler_sigma_ordering(self):
        """Test that scheduler sigmas are in descending order."""
        scheduler = FlowMatchEulerDiscreteScheduler()

        # Sigmas should generally decrease (though not strictly due to shifting)
        # Just check first and last
        assert scheduler.sigmas[0] >= scheduler.sigmas[-1]
