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

        assert step.autocast_dtype == torch.bfloat16
        assert hasattr(step, "pipeline")
        # Pipeline holds timestep/config; check pipeline was created with defaults
        assert step.pipeline.timestep_sampling == "logit_normal"
        assert step.pipeline.flow_shift == 1.0
        assert step.pipeline.logit_mean == 0.0
        assert step.pipeline.logit_std == 1.0
        assert step.pipeline.num_train_timesteps == 1000
        assert step.pipeline.model_adapter.guidance_scale == 3.5

    def test_initialization_custom(self):
        """Test FluxForwardStep initialization with custom values."""
        step = FluxForwardStep(
            timestep_sampling="uniform",
            logit_mean=1.0,
            logit_std=2.0,
            flow_shift=1.5,
            scheduler_steps=500,
            guidance_scale=7.5,
        )

        assert step.pipeline.timestep_sampling == "uniform"
        assert step.pipeline.logit_mean == 1.0
        assert step.pipeline.logit_std == 2.0
        assert step.pipeline.flow_shift == 1.5
        assert step.pipeline.num_train_timesteps == 500
        assert step.pipeline.model_adapter.guidance_scale == 7.5

    def test_initialization_use_loss_weighting(self):
        """Test FluxForwardStep with use_loss_weighting=True."""
        step = FluxForwardStep(use_loss_weighting=True)
        assert step.pipeline.use_loss_weighting is True


class TestFluxForwardStepPrepareBatch:
    """Test _prepare_batch_for_pipeline."""

    def test_prepare_batch_maps_keys(self):
        """Test that Megatron keys are mapped to pipeline keys."""
        step = FluxForwardStep()
        batch = {
            "latents": torch.randn(2, 16, 64, 64),
            "prompt_embeds": torch.randn(2, 77, 4096),
            "pooled_prompt_embeds": torch.randn(2, 768),
            "text_ids": torch.zeros(2, 77, 3),
        }

        pipeline_batch = step._prepare_batch_for_pipeline(batch)

        assert "image_latents" in pipeline_batch
        assert pipeline_batch["image_latents"] is batch["latents"]
        assert pipeline_batch["prompt_embeds"] is batch["prompt_embeds"]
        assert pipeline_batch["pooled_prompt_embeds"] is batch["pooled_prompt_embeds"]
        assert pipeline_batch["text_ids"] is batch["text_ids"]
        assert pipeline_batch["data_type"] == "image"
        assert "latents" not in pipeline_batch

    def test_prepare_batch_extra_keys_copied(self):
        """Test that extra batch keys are copied (except latents)."""
        step = FluxForwardStep()
        batch = {
            "latents": torch.randn(1, 16, 32, 32),
            "custom_key": "value",
        }

        pipeline_batch = step._prepare_batch_for_pipeline(batch)

        assert pipeline_batch["custom_key"] == "value"
        assert pipeline_batch["image_latents"] is batch["latents"]

    def test_prepare_batch_optional_keys(self):
        """Test prepare_batch when optional keys are missing."""
        step = FluxForwardStep()
        batch = {"latents": torch.randn(1, 16, 32, 32)}

        pipeline_batch = step._prepare_batch_for_pipeline(batch)

        assert pipeline_batch["image_latents"] is batch["latents"]
        assert pipeline_batch.get("prompt_embeds") is None
        assert pipeline_batch.get("pooled_prompt_embeds") is None
        assert pipeline_batch.get("text_ids") is None


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

        assert loss_fn.func.__name__ == "masked_next_token_loss"
        assert loss_fn.keywords["check_for_nan_in_loss"] is False
        assert loss_fn.keywords["check_for_spiky_loss"] is True


class TestFluxForwardStepPipelineConfig:
    """Test pipeline configuration exposed by FluxForwardStep."""

    def test_pipeline_num_train_timesteps(self):
        """Test that pipeline has correct num_train_timesteps."""
        step = FluxForwardStep(scheduler_steps=500)
        assert step.pipeline.num_train_timesteps == 500

    def test_pipeline_timestep_sampling_options(self):
        """Test pipeline accepts different timestep_sampling values."""
        for method in ("logit_normal", "uniform", "mode"):
            step = FluxForwardStep(timestep_sampling=method)
            assert step.pipeline.timestep_sampling == method
