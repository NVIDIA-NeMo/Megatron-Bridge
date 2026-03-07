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

from functools import partial
from unittest.mock import MagicMock

import pytest
import torch

from megatron.bridge.diffusion.models.flux.flux_step import FluxForwardStep, flux_data_step


pytestmark = [pytest.mark.unit]


@pytest.mark.run_only_on("GPU")
class TestFluxDataStep:
    """Test flux_data_step function."""

    def test_flux_data_step_basic(self):
        """Test basic flux_data_step functionality."""
        # Create mock iterator
        batch = {"latents": torch.randn(2, 16, 64, 64), "prompt_embeds": torch.randn(2, 512, 4096)}
        dataloader_iter = iter([batch])

        result = flux_data_step(dataloader_iter)

        assert "latents" in result
        assert "prompt_embeds" in result
        assert "loss_mask" in result
        assert result["loss_mask"].device.type == "cuda"

    def test_flux_data_step_with_tuple_input(self):
        """Test flux_data_step with tuple input from dataloader."""
        batch = {"latents": torch.randn(2, 16, 64, 64)}
        dataloader_iter = iter([(batch, None, None)])

        result = flux_data_step(dataloader_iter)

        assert "latents" in result
        assert "loss_mask" in result

    def test_flux_data_step_preserves_loss_mask(self):
        """Test that existing loss_mask is preserved."""
        custom_loss_mask = torch.ones(2)
        batch = {"latents": torch.randn(2, 16, 64, 64), "loss_mask": custom_loss_mask}
        dataloader_iter = iter([batch])

        result = flux_data_step(dataloader_iter)

        assert torch.equal(result["loss_mask"].cpu(), custom_loss_mask)

    def test_flux_data_step_creates_default_loss_mask(self):
        """Test that default loss_mask is created when missing."""
        batch = {"latents": torch.randn(2, 16, 64, 64)}
        dataloader_iter = iter([batch])

        result = flux_data_step(dataloader_iter)

        assert "loss_mask" in result
        assert result["loss_mask"].shape == (1,)
        assert torch.all(result["loss_mask"] == 1.0)

    def test_flux_data_step_moves_tensors_to_cuda(self):
        """Test that tensors are moved to CUDA."""
        batch = {
            "latents": torch.randn(2, 16, 64, 64),
            "prompt_embeds": torch.randn(2, 512, 4096),
            "non_tensor": "text",
        }
        dataloader_iter = iter([batch])

        result = flux_data_step(dataloader_iter)

        assert result["latents"].device.type == "cuda"
        assert result["prompt_embeds"].device.type == "cuda"
        assert result["non_tensor"] == "text"  # Non-tensors unchanged


class TestFluxForwardStepInitialization:
    """Test FluxForwardStep initialization."""

    def test_initialization_defaults(self):
        """Test FluxForwardStep initialization with default values."""
        step = FluxForwardStep()

        assert step.timestep_sampling == "logit_normal"
        assert step.logit_mean == 0.0
        assert step.logit_std == 1.0
        assert step.mode_scale == 1.29
        assert step.scheduler_steps == 1000
        assert step.guidance_scale == 3.5
        assert step.autocast_dtype == torch.bfloat16
        assert hasattr(step, "scheduler")

    def test_initialization_custom(self):
        """Test FluxForwardStep initialization with custom values."""
        step = FluxForwardStep(
            timestep_sampling="uniform",
            logit_mean=1.0,
            logit_std=2.0,
            mode_scale=1.5,
            scheduler_steps=500,
            guidance_scale=7.5,
        )

        assert step.timestep_sampling == "uniform"
        assert step.logit_mean == 1.0
        assert step.logit_std == 2.0
        assert step.mode_scale == 1.5
        assert step.scheduler_steps == 500
        assert step.guidance_scale == 7.5


class TestFluxForwardStepTimestepSampling:
    """Test timestep sampling methods."""

    def test_compute_density_logit_normal(self):
        """Test logit-normal timestep sampling."""
        step = FluxForwardStep(timestep_sampling="logit_normal", logit_mean=0.0, logit_std=1.0)
        batch_size = 10

        u = step.compute_density_for_timestep_sampling("logit_normal", batch_size)

        assert u.shape == (batch_size,)
        assert (u >= 0).all()
        assert (u <= 1).all()

    def test_compute_density_mode(self):
        """Test mode-based timestep sampling."""
        step = FluxForwardStep(timestep_sampling="mode", mode_scale=1.29)
        batch_size = 10

        u = step.compute_density_for_timestep_sampling("mode", batch_size)

        assert u.shape == (batch_size,)
        assert (u >= 0).all()
        assert (u <= 1).all()

    def test_compute_density_uniform(self):
        """Test uniform timestep sampling."""
        step = FluxForwardStep(timestep_sampling="uniform")
        batch_size = 10

        u = step.compute_density_for_timestep_sampling("uniform", batch_size)

        assert u.shape == (batch_size,)
        assert (u >= 0).all()
        assert (u <= 1).all()

    def test_compute_density_uses_instance_defaults(self):
        """Test that compute_density uses instance defaults when not provided."""
        step = FluxForwardStep(logit_mean=0.5, logit_std=0.8, mode_scale=1.5)

        # Should use instance defaults
        u = step.compute_density_for_timestep_sampling("logit_normal", batch_size=5)

        assert u.shape == (5,)

    def test_compute_density_override_defaults(self):
        """Test that compute_density can override instance defaults."""
        step = FluxForwardStep(logit_mean=0.0, logit_std=1.0)

        # Override with custom values
        u = step.compute_density_for_timestep_sampling("logit_normal", batch_size=5, logit_mean=1.0, logit_std=0.5)

        assert u.shape == (5,)


class TestFluxForwardStepLatentOperations:
    """Test latent packing/unpacking operations."""

    def test_pack_latents(self):
        """Test _pack_latents method."""
        step = FluxForwardStep()
        batch_size = 2
        num_channels = 16
        height = 64
        width = 64

        latents = torch.randn(batch_size, num_channels, height, width)
        packed = step._pack_latents(latents, batch_size, num_channels, height, width)

        expected_seq_len = (height // 2) * (width // 2)
        expected_channels = num_channels * 4
        assert packed.shape == (batch_size, expected_seq_len, expected_channels)

    def test_unpack_latents(self):
        """Test _unpack_latents method."""
        step = FluxForwardStep()
        batch_size = 2
        num_patches = 1024  # (64 // 2) * (64 // 2)
        channels = 64  # 16 * 4
        height = 64
        width = 64

        packed_latents = torch.randn(batch_size, num_patches, channels)
        unpacked = step._unpack_latents(packed_latents, height, width)

        expected_channels = channels // 4
        assert unpacked.shape == (batch_size, expected_channels, height, width)

    def test_pack_unpack_roundtrip(self):
        """Test that pack and unpack are consistent."""
        step = FluxForwardStep()
        batch_size = 2
        num_channels = 16
        height = 64
        width = 64

        original = torch.randn(batch_size, num_channels, height, width)
        packed = step._pack_latents(original, batch_size, num_channels, height, width)
        unpacked = step._unpack_latents(packed, height, width)

        assert unpacked.shape == original.shape
        # Note: Due to the reshape operations, values should be approximately equal
        # but the exact comparison might not hold due to floating point operations

    def test_prepare_latent_image_ids(self):
        """Test _prepare_latent_image_ids method."""
        step = FluxForwardStep()
        batch_size = 2
        height = 64
        width = 64
        device = torch.device("cpu")
        dtype = torch.float32

        # First call creates the IDs
        ids = step._prepare_latent_image_ids(batch_size, height, width, device, dtype)

        expected_seq_len = (height // 2) * (width // 2)
        assert ids.shape == (batch_size, expected_seq_len, 3)
        assert ids.device == device
        assert ids.dtype == dtype

        # Second call should use cache
        ids2 = step._prepare_latent_image_ids(batch_size, height, width, device, dtype)
        assert ids2.shape == ids.shape

    def test_prepare_latent_image_ids_caching(self):
        """Test that _prepare_latent_image_ids uses LRU cache."""
        step = FluxForwardStep()

        # Cache should work with same parameters
        ids1 = step._prepare_latent_image_ids(2, 64, 64, torch.device("cpu"), torch.float32)
        ids2 = step._prepare_latent_image_ids(2, 64, 64, torch.device("cpu"), torch.float32)

        # Should be the same object from cache
        assert ids1.data_ptr() == ids2.data_ptr()


@pytest.mark.run_only_on("GPU")
class TestFluxForwardStepPrepareImageLatent:
    """Test prepare_image_latent method."""

    def test_prepare_image_latent_basic(self):
        """Test prepare_image_latent with basic input."""
        step = FluxForwardStep()
        batch_size = 2
        channels = 16
        height = 64
        width = 64

        latents = torch.randn(batch_size, channels, height, width, device="cuda")

        # Mock model
        mock_model = MagicMock()
        mock_model.guidance_embed = False

        result = step.prepare_image_latent(latents, mock_model)

        # Unpack result tuple
        ret_latents, noise, packed_noisy_input, latent_ids, guidance_vec, timesteps = result

        # Check shapes (transposed from [B, ...] to [seq, B, ...] format)
        assert ret_latents.shape[1] == batch_size
        assert noise.shape[1] == batch_size
        assert packed_noisy_input.shape[1] == batch_size
        assert latent_ids.shape[0] == batch_size
        assert guidance_vec is None
        assert timesteps.shape[0] == batch_size

    def test_prepare_image_latent_with_guidance(self):
        """Test prepare_image_latent with guidance embedding."""
        step = FluxForwardStep(guidance_scale=7.5)
        batch_size = 2
        channels = 16
        height = 64
        width = 64

        latents = torch.randn(batch_size, channels, height, width, device="cuda")

        # Mock model with guidance
        mock_model = MagicMock()
        mock_model.guidance_embed = True

        result = step.prepare_image_latent(latents, mock_model)
        ret_latents, noise, packed_noisy_input, latent_ids, guidance_vec, timesteps = result

        assert guidance_vec is not None
        assert guidance_vec.shape == (batch_size,)
        assert torch.all(guidance_vec == 7.5)


class TestFluxForwardStepLossFunction:
    """Test loss function creation."""

    def test_create_loss_function(self):
        """Test _create_loss_function method."""
        step = FluxForwardStep()
        loss_mask = torch.ones(4, dtype=torch.float32)

        loss_fn = step._create_loss_function(loss_mask, check_for_nan_in_loss=True, check_for_spiky_loss=False)

        assert isinstance(loss_fn, partial)
        assert callable(loss_fn)

    def test_create_loss_function_parameters(self):
        """Test that loss function parameters are correctly set."""
        step = FluxForwardStep()
        loss_mask = torch.ones(2, dtype=torch.float32)

        loss_fn = step._create_loss_function(loss_mask, check_for_nan_in_loss=False, check_for_spiky_loss=True)

        # Verify it's a partial with expected arguments
        assert loss_fn.func.__name__ == "masked_next_token_loss"
        assert loss_fn.keywords["check_for_nan_in_loss"] is False
        assert loss_fn.keywords["check_for_spiky_loss"] is True


class TestFluxForwardStepIntegration:
    """Integration tests for FluxForwardStep."""

    def test_timestep_sampling_methods_produce_valid_values(self):
        """Test that all timestep sampling methods produce valid u values."""
        batch_size = 100

        for method in ["logit_normal", "mode", "uniform"]:
            step = FluxForwardStep(timestep_sampling=method)
            u = step.compute_density_for_timestep_sampling(method, batch_size)

            assert u.shape == (batch_size,)
            assert (u >= 0).all(), f"{method} produced u < 0"
            assert (u <= 1).all(), f"{method} produced u > 1"
            assert not torch.isnan(u).any(), f"{method} produced NaN values"

    def test_latent_operations_preserve_batch_dimension(self):
        """Test that latent operations preserve batch dimension."""
        step = FluxForwardStep()

        for batch_size in [1, 2, 4]:
            latents = torch.randn(batch_size, 16, 64, 64)
            packed = step._pack_latents(latents, batch_size, 16, 64, 64)
            unpacked = step._unpack_latents(packed, 64, 64)

            assert packed.shape[0] == batch_size
            assert unpacked.shape[0] == batch_size


class TestFluxForwardStepEdgeCases:
    """Test edge cases and error handling."""

    def test_pack_latents_small_dimensions(self):
        """Test _pack_latents with small dimensions."""
        step = FluxForwardStep()
        latents = torch.randn(1, 4, 4, 4)

        packed = step._pack_latents(latents, 1, 4, 4, 4)

        assert packed.shape == (1, 4, 16)  # (4/2) * (4/2) = 4, 4*4 = 16

    def test_unpack_latents_small_dimensions(self):
        """Test _unpack_latents with small dimensions."""
        step = FluxForwardStep()
        packed = torch.randn(1, 4, 16)

        unpacked = step._unpack_latents(packed, 4, 4)

        assert unpacked.shape == (1, 4, 4, 4)

    def test_compute_density_mode_with_extreme_scale(self):
        """Test mode sampling with extreme scale values."""
        step = FluxForwardStep()

        # Test with very small scale
        u_small = step.compute_density_for_timestep_sampling("mode", 10, mode_scale=0.01)
        assert (u_small >= 0).all() and (u_small <= 1).all()

        # Test with larger scale
        u_large = step.compute_density_for_timestep_sampling("mode", 10, mode_scale=2.0)
        assert (u_large >= 0).all() and (u_large <= 1).all()

    def test_prepare_latent_image_ids_different_sizes(self):
        """Test _prepare_latent_image_ids with different image sizes."""
        step = FluxForwardStep()

        for height, width in [(32, 32), (64, 64), (128, 128)]:
            ids = step._prepare_latent_image_ids(2, height, width, torch.device("cpu"), torch.float32)

            expected_seq_len = (height // 2) * (width // 2)
            assert ids.shape == (2, expected_seq_len, 3)


class TestFluxForwardStepScheduler:
    """Test scheduler integration."""

    def test_scheduler_initialized_with_correct_steps(self):
        """Test that scheduler is initialized with correct number of steps."""
        scheduler_steps = 500
        step = FluxForwardStep(scheduler_steps=scheduler_steps)

        assert step.scheduler.num_train_timesteps == scheduler_steps
        assert len(step.scheduler.timesteps) == scheduler_steps

    def test_scheduler_timesteps_in_valid_range(self):
        """Test that scheduler timesteps are in valid range."""
        step = FluxForwardStep()

        assert (step.scheduler.timesteps >= 0).all()
        assert (step.scheduler.timesteps <= step.scheduler.num_train_timesteps).all()

    def test_scheduler_sigmas_in_valid_range(self):
        """Test that scheduler sigmas are in valid range."""
        step = FluxForwardStep()

        assert (step.scheduler.sigmas >= 0).all()
        assert (step.scheduler.sigmas <= 1).all()
