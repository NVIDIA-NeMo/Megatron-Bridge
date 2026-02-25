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

from unittest.mock import MagicMock, patch

import pytest
import torch
from dfm.src.automodel.flow_matching.adapters.base import FlowMatchingContext

from megatron.bridge.diffusion.models.wan.flow_matching.flow_matching_pipeline_wan import (
    WanAdapter,
    WanFlowMatchingPipeline,
)


class TestWanAdapter:
    @pytest.fixture
    def adapter(self):
        return WanAdapter()

    @pytest.fixture
    def context(self):
        # Create a mock context with necessary attributes
        ctx = MagicMock(spec=FlowMatchingContext)

        # Setup inputs
        batch_size = 2
        seq_len = 8
        hidden_dim = 16

        # Input latents are typically (B, S, H) before adapter
        ctx.noisy_latents = torch.randn(batch_size, seq_len, hidden_dim)
        ctx.video_latents = torch.randn(batch_size, seq_len, hidden_dim)
        ctx.timesteps = torch.tensor([0.5, 0.5])

        ctx.batch = {
            "grid_sizes": [(4, 4, 4)] * batch_size,
            "loss_mask": torch.ones(batch_size, seq_len),
            "context_embeddings": torch.randn(batch_size, seq_len, hidden_dim),  # B, S, H
            "packed_seq_params": {
                "self_attention": MagicMock(cu_seqlens_q_padded=None),
                "cross_attention": MagicMock(cu_seqlens_kv_padded=None),
            },
        }
        return ctx

    def test_prepare_inputs_no_cp(self, adapter, context):
        with patch(
            "megatron.bridge.diffusion.models.wan.flow_matching.flow_matching_pipeline_wan.parallel_state"
        ) as mock_ps:
            mock_ps.get_context_parallel_world_size.return_value = 1

            inputs = adapter.prepare_inputs(context)

            # Check keys
            assert "noisy_latents" in inputs
            assert "grid_sizes" in inputs
            assert "timesteps" in inputs
            assert "context_embeddings" in inputs
            assert "packed_seq_params" in inputs

            # Check shapes and types
            # noisy_latents should be transposed to (S, B, H) from (B, S, H) and cast to bf16
            # Input was (2, 8, 16), so expected is (8, 2, 16)
            assert inputs["noisy_latents"].shape == (8, 2, 16)
            assert inputs["noisy_latents"].dtype == torch.bfloat16

            # context_embeddings should be (B, S, H) (2, 8, 16) and cast to bf16
            assert inputs["context_embeddings"].shape == (2, 8, 16)
            assert inputs["context_embeddings"].dtype == torch.bfloat16

            # Timesteps should be bf16
            assert inputs["timesteps"].dtype == torch.bfloat16

    def test_prepare_inputs_with_cp(self, adapter, context):
        with (
            patch(
                "megatron.bridge.diffusion.models.wan.flow_matching.flow_matching_pipeline_wan.parallel_state"
            ) as mock_ps,
            patch(
                "megatron.bridge.diffusion.models.wan.flow_matching.flow_matching_pipeline_wan.thd_split_inputs_cp"
            ) as mock_split,
        ):
            mock_ps.get_context_parallel_world_size.return_value = 2
            mock_ps.get_context_parallel_group.return_value = "fake_group"

            # Mock split to return a dummy value so we know it was processed
            # We return the input as is, but we check the call arguments
            mock_split.side_effect = lambda x, *args: x

            inputs = adapter.prepare_inputs(context)

            # Verify thd_split_inputs_cp was called for noisy_latents and context_embeddings
            assert mock_split.call_count == 2

            # Verify args for the calls
            # We can't easily distinguish order without checking args, but we know both should be called.

            # Check types are correct (bf16)
            assert inputs["noisy_latents"].dtype == torch.bfloat16
            assert inputs["context_embeddings"].dtype == torch.bfloat16

    def test_forward(self, adapter):
        model = MagicMock()
        model.return_value = torch.randn(8, 2, 16)  # S, B, H

        inputs = {
            "noisy_latents": torch.randn(8, 2, 16),
            "grid_sizes": [],
            "timesteps": torch.tensor([1.0]),
            "context_embeddings": torch.randn(2, 8, 16),
            "packed_seq_params": {},
        }

        # Mock post_process_prediction inherited from ModelAdapter
        adapter.post_process_prediction = MagicMock(side_effect=lambda x: x)

        out = adapter.forward(model, inputs)

        model.assert_called_once_with(
            x=inputs["noisy_latents"],
            grid_sizes=inputs["grid_sizes"],
            t=inputs["timesteps"],
            context=inputs["context_embeddings"],
            packed_seq_params=inputs["packed_seq_params"],
        )
        assert out is not None


class TestWanFlowMatchingPipeline:
    @pytest.fixture
    def pipeline(self):
        # Use object.__new__ to avoid __init__ if it's heavy
        pip = object.__new__(WanFlowMatchingPipeline)
        return pip

    def test_determine_task_type(self, pipeline):
        assert pipeline.determine_task_type("any") == "t2v"

    def test_compute_loss_no_cp(self, pipeline):
        model_pred = torch.randn(8, 2, 16)  # S, B, H
        target = torch.randn(2, 8, 16)  # B, S, H
        sigma = torch.randn(2)

        batch = {
            "loss_mask": torch.ones(2, 8),
            "packed_seq_params": {"self_attention": MagicMock(cu_seqlens_q_padded=None)},
        }

        with (
            patch(
                "dfm.src.automodel.flow_matching.flow_matching_pipeline.FlowMatchingPipeline.compute_loss"
            ) as mock_super_loss,
            patch(
                "megatron.bridge.diffusion.models.wan.flow_matching.flow_matching_pipeline_wan.parallel_state"
            ) as mock_ps,
        ):
            mock_ps.get_context_parallel_world_size.return_value = 1
            mock_super_loss.return_value = (1, 2, 3, 4, 5, batch["loss_mask"])

            pipeline.compute_loss(model_pred, target, sigma, batch)

            # target should be transposed to (S, B, H) before passing to super
            # Input target was (2, 8, 16), so expected is (8, 2, 16)
            args, _ = mock_super_loss.call_args
            passed_target = args[1]
            assert passed_target.shape == (8, 2, 16)

    def test_compute_loss_with_cp(self, pipeline):
        model_pred = torch.randn(8, 2, 16)
        target = torch.randn(2, 8, 16)
        sigma = torch.randn(2)

        batch = {
            "loss_mask": torch.ones(2, 8),
            "packed_seq_params": {"self_attention": MagicMock(cu_seqlens_q_padded="dummy_seq_len")},
        }

        with (
            patch(
                "dfm.src.automodel.flow_matching.flow_matching_pipeline.FlowMatchingPipeline.compute_loss"
            ) as mock_super_loss,
            patch(
                "megatron.bridge.diffusion.models.wan.flow_matching.flow_matching_pipeline_wan.parallel_state"
            ) as mock_ps,
            patch(
                "megatron.bridge.diffusion.models.wan.flow_matching.flow_matching_pipeline_wan.thd_split_inputs_cp"
            ) as mock_split,
        ):
            mock_ps.get_context_parallel_world_size.return_value = 2
            mock_ps.get_context_parallel_group.return_value = "fake_group"
            mock_super_loss.return_value = (1, 2, 3, 4, 5, batch["loss_mask"])

            mock_split.side_effect = lambda x, *args: x  # Identity for simplicity

            pipeline.compute_loss(model_pred, target, sigma, batch)

            # Check thd_split_inputs_cp calls
            # Should be called for target and split_loss_mask
            assert mock_split.call_count == 2
