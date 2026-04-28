# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for MegatronMIMO forward step functions."""

from unittest.mock import MagicMock, patch

import pytest
import torch


class TestLossFunc:
    """Test cases for loss_func()."""

    def test_loss_computation(self):
        """Test loss is computed correctly with mask."""
        from megatron.bridge.training.megatron_mimo_step import loss_func

        # Create test data
        output_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        loss_mask = torch.tensor([1.0, 1.0, 0.0, 1.0])  # Mask out 3rd element

        total_loss, num_tokens, metrics = loss_func(loss_mask, output_tensor)

        # Expected: (1.0*1 + 2.0*1 + 3.0*0 + 4.0*1) = 7.0
        assert total_loss.item() == 7.0
        # Expected tokens: 3 (sum of mask)
        assert num_tokens.item() == 3
        # Check metrics dict structure
        assert "lm loss" in metrics

    def test_loss_with_all_ones_mask(self):
        """Test loss with all-ones mask."""
        from megatron.bridge.training.megatron_mimo_step import loss_func

        output_tensor = torch.tensor([1.0, 2.0, 3.0])
        loss_mask = torch.ones(3)

        total_loss, num_tokens, metrics = loss_func(loss_mask, output_tensor)

        assert total_loss.item() == 6.0
        assert num_tokens.item() == 3

    def test_loss_with_all_zeros_mask(self):
        """Test loss with all-zeros mask."""
        from megatron.bridge.training.megatron_mimo_step import loss_func

        output_tensor = torch.tensor([1.0, 2.0, 3.0])
        loss_mask = torch.zeros(3)

        total_loss, num_tokens, metrics = loss_func(loss_mask, output_tensor)

        assert total_loss.item() == 0.0
        assert num_tokens.item() == 0


class TestGetBatch:
    """Test cases for get_batch()."""

    def test_returns_none_for_none_iterator(self):
        """Test returns None when iterator is None."""
        from megatron.bridge.training.megatron_mimo_step import get_batch

        result = get_batch(None)
        assert result is None

    def test_returns_none_on_stop_iteration(self):
        """Test returns None when iterator is exhausted."""
        from megatron.bridge.training.megatron_mimo_step import get_batch

        empty_iter = iter([])
        result = get_batch(empty_iter)
        assert result is None

    def test_returns_batch_from_iterator(self):
        """Test returns batch from iterator."""
        from megatron.bridge.training.megatron_mimo_step import get_batch

        batch = {"input_ids": torch.tensor([1, 2, 3])}
        data_iter = iter([batch])

        result = get_batch(data_iter)

        assert result is not None
        assert "input_ids" in result


class TestForwardStep:
    """Test cases for forward_step()."""

    def test_colocated_data_alignment_check_accepts_fan_in_layout(self, monkeypatch):
        """The optional L3 alignment check should accept encoder-DP fan-in."""
        from megatron.bridge.training import megatron_mimo_step

        class FakePG:
            def __init__(self, rank, size):
                self._rank = rank
                self._size = size

            def rank(self):
                return self._rank

            def size(self):
                return self._size

        class FakeGrid:
            rank_offset = 0
            size = 8

            def __init__(self, dp_rank, dp_size):
                self._dp = FakePG(dp_rank, dp_size)

            def get_pg(self, dims):
                assert dims == ["dp"]
                return self._dp

        monkeypatch.setenv("MIMO_CHECK_DATA_ALIGNMENT", "true")
        monkeypatch.setenv("MIMO_CHECK_DATA_ALIGNMENT_STEPS", "1")
        monkeypatch.setattr(megatron_mimo_step.dist, "is_available", lambda: True)
        monkeypatch.setattr(megatron_mimo_step.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(megatron_mimo_step.dist, "get_rank", lambda: 0)
        megatron_mimo_step._DATA_ALIGNMENT_CHECK_COUNT = 0

        original_batch = {
            "input_ids": torch.zeros(4, 8),
            "modality_inputs": {"images": {"clip": {"x": torch.zeros(4, 3, 336, 336)}}},
        }
        sliced_batch = {
            "input_ids": torch.zeros(2, 8),
            "modality_inputs": {"images": {"clip": {"x": torch.zeros(1, 3, 336, 336)}}},
        }
        grids = {
            "language": FakeGrid(dp_rank=0, dp_size=2),
            "images": FakeGrid(dp_rank=0, dp_size=4),
        }

        megatron_mimo_step._maybe_check_colocated_data_alignment(original_batch, sliced_batch, grids)

        assert megatron_mimo_step._DATA_ALIGNMENT_CHECK_COUNT == 1

    @patch("megatron.bridge.training.megatron_mimo_step.unwrap_megatron_mimo_model")
    def test_forward_step_last_stage(self, mock_unwrap):
        """Test forward step at last pipeline stage returns loss func."""
        from megatron.bridge.training.megatron_mimo_step import forward_step

        # Create mock state
        mock_state = MagicMock()

        # Create mock model with role=None (indicates last stage)
        mock_model = MagicMock()
        mock_model.role = None  # role=None means is_last_stage=True
        mock_output = torch.tensor([1.0, 2.0])
        mock_loss_mask = torch.ones(2)
        mock_model.return_value = (mock_output, mock_loss_mask)

        # unwrap_megatron_mimo_model returns the mock model itself
        mock_unwrap.return_value = mock_model

        # Create mock iterator
        batch = {"input_ids": torch.tensor([1, 2])}
        data_iter = iter([batch])

        output, loss_fn = forward_step(mock_state, data_iter, mock_model)

        # At last stage, should return loss function
        assert loss_fn is not None
        assert callable(loss_fn)

    @patch("megatron.bridge.training.megatron_mimo_step.unwrap_megatron_mimo_model")
    def test_forward_step_intermediate_stage(self, mock_unwrap):
        """Test forward step at intermediate stage returns None for loss func."""
        from megatron.bridge.training.megatron_mimo_step import forward_step

        mock_state = MagicMock()
        mock_model = MagicMock()
        # Configure role to indicate intermediate stage (not last stage)
        mock_role = MagicMock()
        mock_role.has_language_module = True
        mock_role.has_modality_modules = False
        mock_role.is_last_stage.return_value = False
        mock_role.is_first_stage.return_value = True
        mock_model.role = mock_role
        mock_model.return_value = (torch.tensor([1.0]), None)

        mock_unwrap.return_value = mock_model

        batch = {"input_ids": torch.tensor([1, 2])}
        data_iter = iter([batch])

        output, loss_fn = forward_step(mock_state, data_iter, mock_model)

        # Intermediate stage should return None for loss_fn
        assert loss_fn is None

    @patch("megatron.bridge.training.megatron_mimo_step.unwrap_megatron_mimo_model")
    def test_forward_step_rejects_dict_at_last_stage(self, mock_unwrap):
        """Test forward step raises error if dict returned at last stage."""
        from megatron.bridge.training.megatron_mimo_step import forward_step

        mock_state = MagicMock()
        mock_model = MagicMock()
        mock_model.role = None  # role=None means is_last_stage=True
        # Return dict (incorrect for last stage)
        mock_model.return_value = ({"encoder": torch.tensor([1.0])}, None)

        mock_unwrap.return_value = mock_model

        batch = {"input_ids": torch.tensor([1, 2])}
        data_iter = iter([batch])

        with pytest.raises(ValueError, match="Last pipeline stage must return scalar loss"):
            forward_step(mock_state, data_iter, mock_model)

    def test_forward_step_uses_global_state_signature(self):
        """Test forward step uses 3-arg signature with GlobalState."""
        import inspect

        from megatron.bridge.training.megatron_mimo_step import forward_step

        sig = inspect.signature(forward_step)
        params = list(sig.parameters.keys())

        # Should have state as first parameter
        assert params[0] == "state"
        assert len(params) == 3

    @patch("megatron.bridge.training.megatron_mimo_step.unwrap_megatron_mimo_model")
    @patch("megatron.bridge.training.megatron_mimo_step.slice_batch_for_megatron_mimo_modules")
    def test_forward_step_invokes_module_aware_slicing_with_grids(self, mock_slice, mock_unwrap):
        """Composition test: forward_step routes the global micro-batch through
        ``slice_batch_for_megatron_mimo_modules`` with the model's grid map.

        This is the load-bearing wire-up for asymmetric DP: a regression where
        forward_step still calls the legacy single-DP helper would silently
        collapse modality_inputs and language keys onto the same DP, breaking
        ``align_embeddings_by_token_positions`` on the first forward.
        """
        from megatron.bridge.training.megatron_mimo_step import forward_step

        mock_state = MagicMock()
        mock_model = MagicMock()
        mock_model.role = None  # last stage → returns loss tuple

        # Two grids same offset/size — colocated layout.
        fake_grids = {
            "vision": MagicMock(rank_offset=0, size=2),
            "language": MagicMock(rank_offset=0, size=2),
        }
        mock_model.mimo_config.module_to_grid_map = fake_grids

        # The helper returns the same dict so the rest of forward_step proceeds.
        original_batch = {"input_ids": torch.tensor([[1, 2], [3, 4]])}
        sliced_batch = {"input_ids": torch.tensor([[1, 2]])}
        mock_slice.return_value = sliced_batch

        mock_model.return_value = (torch.tensor([1.0]), torch.ones(1))
        mock_unwrap.return_value = mock_model

        data_iter = iter([original_batch])
        forward_step(mock_state, data_iter, mock_model)

        # Helper called once with the global batch and the model's grids.
        # ``get_batch`` moves tensors to cuda (creating a new dict), so we
        # check structural equivalence and grid identity, not dict identity.
        mock_slice.assert_called_once()
        call_args = mock_slice.call_args
        passed_batch = call_args.args[0] if call_args.args else call_args.kwargs.get("batch")
        passed_grids = call_args.kwargs["grids"]
        assert set(passed_batch.keys()) == set(original_batch.keys())
        torch.testing.assert_close(passed_batch["input_ids"].cpu(), original_batch["input_ids"])
        assert passed_grids is fake_grids
