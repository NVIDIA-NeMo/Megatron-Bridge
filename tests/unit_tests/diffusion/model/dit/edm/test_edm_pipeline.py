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

import pytest
import torch

from megatron.bridge.diffusion.common.utils.batch_ops import batch_mul
from megatron.bridge.diffusion.models.dit.edm.edm_pipeline import EDMPipeline


class _DummyModel:
    """Dummy model for testing that mimics the DiT network interface."""

    def __call__(self, x, timesteps, **condition):
        # Return zeros matching input shape
        return torch.zeros_like(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestEDMPipeline:
    """Test class for EDMPipeline with shared setup."""

    def setup_method(self, method, monkeypatch=None):
        """Set up test fixtures before each test method."""
        # Stub parallel_state functions to avoid requiring initialization
        from megatron.core import parallel_state

        if monkeypatch:
            monkeypatch.setattr(
                parallel_state, "get_data_parallel_rank", lambda with_context_parallel=False: 0, raising=False
            )
            monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: 1, raising=False)
            monkeypatch.setattr(parallel_state, "is_pipeline_last_stage", lambda: True, raising=False)
            monkeypatch.setattr(parallel_state, "get_context_parallel_group", lambda: None, raising=False)

        # Create pipeline with common parameters
        self.sigma_data = 0.5
        self.pipeline = EDMPipeline(
            vae=None,
            p_mean=0.0,
            p_std=1.0,
            sigma_max=80.0,
            sigma_min=0.0002,
            sigma_data=self.sigma_data,
            seed=1234,
        )

        # Create and assign dummy model
        self.model = _DummyModel()
        self.pipeline.net = self.model

        # Create common test data shapes
        self.batch_size = 2
        self.channels = 4
        self.height = self.width = 8

        # Create common test tensors
        self.x0 = torch.randn(self.batch_size, self.channels, self.height, self.width).to(
            **self.pipeline.tensor_kwargs
        )
        self.sigma = torch.ones(self.batch_size).to(**self.pipeline.tensor_kwargs) * 1.0
        self.condition = {"crossattn_emb": torch.randn(self.batch_size, 10, 512).to(**self.pipeline.tensor_kwargs)}
        self.epsilon = torch.randn(self.batch_size, self.channels, self.height, self.width).to(
            **self.pipeline.tensor_kwargs
        )

    def test_denoise(self, monkeypatch):
        """Test the denoise method produces correct output shapes and values."""
        # Initialize with monkeypatch
        self.setup_method(None, monkeypatch)

        # Create test inputs (xt on CPU for conversion test)
        xt = torch.randn(self.batch_size, self.channels, self.height, self.width)
        sigma = torch.ones(self.batch_size) * 1.0

        # Test Case 1: is_pipeline_last_stage = True
        # Call denoise
        x0_pred, eps_pred = self.pipeline.denoise(xt, sigma, self.condition)

        # Verify outputs have correct shapes
        assert x0_pred.shape == xt.shape, f"Expected x0_pred shape {xt.shape}, got {x0_pred.shape}"
        assert eps_pred.shape == xt.shape, f"Expected eps_pred shape {xt.shape}, got {eps_pred.shape}"

        # Verify outputs are on CUDA with correct dtype
        assert x0_pred.device.type == "cuda"
        assert x0_pred.dtype == torch.bfloat16
        assert eps_pred.device.type == "cuda"
        assert eps_pred.dtype == torch.bfloat16

        # Verify the outputs follow the expected formulas
        # Convert inputs to expected dtype/device for comparison
        xt_converted = xt.to(**self.pipeline.tensor_kwargs)
        sigma_converted = sigma.to(**self.pipeline.tensor_kwargs)

        # Get scaling factors
        c_skip, c_out, c_in, c_noise = self.pipeline.scaling(sigma=sigma_converted)

        # Since model returns zeros, net_output = 0
        # Expected: x0_pred = c_skip * xt + c_out * 0 = c_skip * xt
        expected_x0_pred = batch_mul(c_skip, xt_converted)
        assert torch.allclose(x0_pred, expected_x0_pred, rtol=1e-3, atol=1e-5), "x0_pred doesn't match expected value"

        # Expected: eps_pred = (xt - x0_pred) / sigma
        expected_eps_pred = batch_mul(xt_converted - x0_pred, 1.0 / sigma_converted)
        assert torch.allclose(eps_pred, expected_eps_pred, rtol=1e-3, atol=1e-5), (
            "eps_pred doesn't match expected value"
        )

        # Test Case 2: is_pipeline_last_stage = False
        # Mock is_pipeline_last_stage to return False
        from megatron.core import parallel_state

        monkeypatch.setattr(parallel_state, "is_pipeline_last_stage", lambda: False)

        # Call denoise again
        net_output = self.pipeline.denoise(xt, sigma, self.condition)

        # Verify output is a single tensor (not a tuple)
        assert isinstance(net_output, torch.Tensor), "Expected net_output to be a single tensor when not last stage"
        assert not isinstance(net_output, tuple), "Expected net_output to not be a tuple when not last stage"

        # Verify output has correct shape (same as model output)
        assert net_output.shape == xt.shape, f"Expected net_output shape {xt.shape}, got {net_output.shape}"

        # Verify output is on CUDA with correct dtype
        assert net_output.device.type == "cuda"
        assert net_output.dtype == torch.bfloat16

        # Since model returns zeros, net_output should be zeros
        assert torch.allclose(net_output, torch.zeros_like(xt_converted), rtol=1e-3, atol=1e-5), (
            "net_output doesn't match expected value (zeros from dummy model)"
        )

    def test_compute_loss_with_epsilon_and_sigma(self, monkeypatch):
        """Test the compute_loss_with_epsilon_and_sigma method produces correct output shapes and values."""
        # Initialize with monkeypatch
        self.setup_method(None, monkeypatch)

        # Call compute_loss_with_epsilon_and_sigma
        output_batch, pred_mse, edm_loss = self.pipeline.compute_loss_with_epsilon_and_sigma(
            self.x0, self.condition, self.epsilon, self.sigma
        )

        # Verify output_batch contains expected keys
        assert "x0" in output_batch
        assert "xt" in output_batch
        assert "sigma" in output_batch
        assert "weights_per_sigma" in output_batch
        assert "condition" in output_batch
        assert "model_pred" in output_batch
        assert "mse_loss" in output_batch
        assert "edm_loss" in output_batch

        # Verify shapes
        assert output_batch["x0"].shape == self.x0.shape
        assert output_batch["xt"].shape == self.x0.shape
        assert output_batch["sigma"].shape == self.sigma.shape
        assert output_batch["weights_per_sigma"].shape == self.sigma.shape
        assert pred_mse.shape == self.x0.shape
        assert edm_loss.shape == self.x0.shape

        # Verify the loss computation follows expected formulas
        # 1. Compute expected xt from marginal probability
        mean, std = self.pipeline.sde.marginal_prob(self.x0, self.sigma)
        expected_xt = mean + batch_mul(std, self.epsilon)
        assert torch.allclose(output_batch["xt"], expected_xt, rtol=1e-3, atol=1e-5), "xt doesn't match expected value"

        # 2. Verify loss weights
        expected_weights = (self.sigma**2 + self.sigma_data**2) / (self.sigma * self.sigma_data) ** 2
        assert torch.allclose(output_batch["weights_per_sigma"], expected_weights, rtol=1e-3, atol=1e-5), (
            "weights_per_sigma doesn't match expected value"
        )

        # 3. Verify edm_loss = weights * (x0 - x0_pred)^2
        x0_pred = output_batch["model_pred"]["x0_pred"]
        expected_pred_mse = (self.x0 - x0_pred) ** 2
        assert torch.allclose(pred_mse, expected_pred_mse, rtol=1e-3, atol=1e-5), (
            "pred_mse doesn't match expected value"
        )

        expected_edm_loss = batch_mul(expected_pred_mse, expected_weights)
        assert torch.allclose(edm_loss, expected_edm_loss, rtol=1e-3, atol=1e-5), (
            "edm_loss doesn't match expected value"
        )

        # 4. Verify scalar losses are proper means
        assert torch.isclose(output_batch["mse_loss"], pred_mse.mean(), rtol=1e-3, atol=1e-5)
        assert torch.isclose(output_batch["edm_loss"], edm_loss.mean(), rtol=1e-3, atol=1e-5)

    def test_training_step(self, monkeypatch):
        """Test the training_step method with mocked compute_loss_with_epsilon_and_sigma."""
        from unittest.mock import patch

        # Initialize with monkeypatch
        self.setup_method(None, monkeypatch)

        # Create test data batch
        data_batch = {
            "video": self.x0,
            "context_embeddings": torch.randn(self.batch_size, 10, 512).to(**self.pipeline.tensor_kwargs),
        }
        iteration = 0

        # Test Case 1: is_pipeline_last_stage = True
        # Mock compute_loss_with_epsilon_and_sigma to return expected values
        mock_output_batch = {
            "x0": self.x0,
            "xt": torch.randn_like(self.x0),
            "sigma": self.sigma,
            "weights_per_sigma": torch.ones_like(self.sigma),
            "condition": self.condition,
            "model_pred": {"x0_pred": torch.randn_like(self.x0), "eps_pred": torch.randn_like(self.x0)},
            "mse_loss": torch.tensor(0.5, **self.pipeline.tensor_kwargs),
            "edm_loss": torch.tensor(0.3, **self.pipeline.tensor_kwargs),
        }
        mock_pred_mse = torch.randn_like(self.x0)
        mock_edm_loss = torch.randn_like(self.x0)

        with patch.object(
            self.pipeline,
            "compute_loss_with_epsilon_and_sigma",
            return_value=(mock_output_batch, mock_pred_mse, mock_edm_loss),
        ) as mock_compute_loss:
            # Call training_step
            result = self.pipeline.training_step(self.model, data_batch, iteration)

            # Verify compute_loss_with_epsilon_and_sigma was called once
            assert mock_compute_loss.call_count == 1

            # Verify return values are correct (output_batch, edm_loss)
            assert len(result) == 2
            output_batch, edm_loss = result
            assert output_batch == mock_output_batch
            assert torch.equal(edm_loss, mock_edm_loss)

        # Test Case 2: is_pipeline_last_stage = False
        # Mock is_pipeline_last_stage to return False
        from megatron.core import parallel_state

        monkeypatch.setattr(parallel_state, "is_pipeline_last_stage", lambda: False)

        # Mock compute_loss_with_epsilon_and_sigma to return net_output only
        mock_net_output = torch.randn_like(self.x0)

        with patch.object(
            self.pipeline, "compute_loss_with_epsilon_and_sigma", return_value=mock_net_output
        ) as mock_compute_loss:
            # Call training_step
            result = self.pipeline.training_step(self.model, data_batch, iteration)

            # Verify compute_loss_with_epsilon_and_sigma was called once
            assert mock_compute_loss.call_count == 1

            # Verify return value is just net_output (not a tuple)
            assert torch.equal(result, mock_net_output)

    def test_get_data_and_condition(self, monkeypatch):
        """Test the get_data_and_condition method with different dropout rates."""
        # Initialize with monkeypatch
        self.setup_method(None, monkeypatch)

        # Create test data batch
        video_data = torch.randn(self.batch_size, self.channels, self.height, self.width).to(
            **self.pipeline.tensor_kwargs
        )
        context_embeddings = torch.randn(self.batch_size, 10, 512).to(**self.pipeline.tensor_kwargs)

        data_batch = {"video": video_data.clone(), "context_embeddings": context_embeddings.clone()}

        # Test Case 1: With default dropout_rate (0.2)
        latent_state, condition = self.pipeline.get_data_and_condition(data_batch.copy(), dropout_rate=0.2)

        # Verify raw_state is video * sigma_data
        expected_raw_state = video_data * self.sigma_data
        assert torch.allclose(latent_state, expected_raw_state, rtol=1e-3, atol=1e-5), (
            "raw_state doesn't match expected value (video * sigma_data)"
        )

        # Verify condition contains crossattn_emb
        assert "crossattn_emb" in condition, "condition should contain 'crossattn_emb' key"
        assert condition["crossattn_emb"].shape == context_embeddings.shape, (
            f"Expected crossattn_emb shape {context_embeddings.shape}, got {condition['crossattn_emb'].shape}"
        )

        # Verify crossattn_emb is on CUDA with correct dtype
        assert condition["crossattn_emb"].device.type == "cuda"
        assert condition["crossattn_emb"].dtype == torch.bfloat16

        # Test Case 2: With dropout_rate=0.0 (no dropout, should keep all values)
        data_batch_no_dropout = {"video": video_data.clone(), "context_embeddings": context_embeddings.clone()}
        latent_state_no_dropout, condition_no_dropout = self.pipeline.get_data_and_condition(
            data_batch_no_dropout, dropout_rate=0.0
        )

        # With dropout_rate=0.0, crossattn_emb should equal context_embeddings
        assert torch.allclose(condition_no_dropout["crossattn_emb"], context_embeddings, rtol=1e-3, atol=1e-5), (
            "With dropout_rate=0.0, crossattn_emb should equal original context_embeddings"
        )

        # Test Case 3: With dropout_rate=1.0 (complete dropout, should zero out all values)
        data_batch_full_dropout = {"video": video_data.clone(), "context_embeddings": context_embeddings.clone()}
        latent_state_full_dropout, condition_full_dropout = self.pipeline.get_data_and_condition(
            data_batch_full_dropout, dropout_rate=1.0
        )

        # With dropout_rate=1.0, crossattn_emb should be all zeros
        expected_zeros = torch.zeros_like(context_embeddings)
        assert torch.allclose(condition_full_dropout["crossattn_emb"], expected_zeros, rtol=1e-3, atol=1e-5), (
            "With dropout_rate=1.0, crossattn_emb should be all zeros"
        )

        # test latent_state_full_dropout and latent_state_no_dropout are equal to each other
        assert torch.allclose(latent_state_full_dropout, latent_state_no_dropout, rtol=1e-3, atol=1e-5), (
            "latent_state_full_dropout and latent_state_no_dropout should be equal to each other"
        )
        assert torch.allclose(latent_state_no_dropout, video_data * self.sigma_data, rtol=1e-3, atol=1e-5), (
            "latent_state with dropout=0 should equal video data * sigma_data"
        )

    def test_get_x0_fn_from_batch(self, monkeypatch):
        """Test the get_x0_fn_from_batch method returns a callable with correct guidance behavior."""
        from unittest.mock import patch

        # Initialize with monkeypatch
        self.setup_method(None, monkeypatch)

        # Create test data batch
        video_data = torch.randn(self.batch_size, self.channels, self.height, self.width).to(
            **self.pipeline.tensor_kwargs
        )
        context_embeddings = torch.randn(self.batch_size, 10, 512).to(**self.pipeline.tensor_kwargs)

        data_batch = {"video": video_data, "context_embeddings": context_embeddings}

        # Create mock condition and uncondition
        mock_condition = {"crossattn_emb": torch.randn(self.batch_size, 10, 512).to(**self.pipeline.tensor_kwargs)}
        mock_uncondition = {"crossattn_emb": torch.randn(self.batch_size, 10, 512).to(**self.pipeline.tensor_kwargs)}

        # Mock get_condition_uncondition to return our mock conditions
        with patch.object(self.pipeline, "get_condition_uncondition", return_value=(mock_condition, mock_uncondition)):
            # Test Case 1: Default guidance (1.5)
            guidance = 1.5
            x0_fn = self.pipeline.get_x0_fn_from_batch(data_batch, guidance=guidance)

            # Verify x0_fn is callable
            assert callable(x0_fn), "get_x0_fn_from_batch should return a callable"

            # Create test inputs for the returned function
            noise_x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(
                **self.pipeline.tensor_kwargs
            )
            sigma = torch.ones(self.batch_size).to(**self.pipeline.tensor_kwargs) * 1.0

            # Create mock outputs for denoise calls
            mock_cond_x0 = torch.randn_like(noise_x)
            mock_uncond_x0 = torch.randn_like(noise_x)
            mock_eps = torch.randn_like(noise_x)  # dummy eps_pred (not used in x0_fn)

            # Mock denoise to return different values for condition vs uncondition
            call_count = [0]

            def mock_denoise(xt, sig, cond):
                call_count[0] += 1
                if call_count[0] == 1:  # First call (with condition)
                    return mock_cond_x0, mock_eps
                else:  # Second call (with uncondition)
                    return mock_uncond_x0, mock_eps

            with patch.object(self.pipeline, "denoise", side_effect=mock_denoise):
                # Call the returned x0_fn
                result = x0_fn(noise_x, sigma)

                # Verify denoise was called twice
                assert call_count[0] == 2, "mock_denoise should be called twice (condition and uncondition)"

                # Verify the result follows the guidance formula: cond_x0 + guidance * (cond_x0 - uncond_x0)
                expected_result = mock_cond_x0 + guidance * (mock_cond_x0 - mock_uncond_x0)
                assert torch.allclose(result, expected_result, rtol=1e-3, atol=1e-5), (
                    "x0_fn output doesn't match expected guidance formula"
                )

                # Verify output shape and dtype
                assert result.shape == noise_x.shape, f"Expected result shape {noise_x.shape}, got {result.shape}"
                assert result.device.type == "cuda"
                assert result.dtype == torch.bfloat16
