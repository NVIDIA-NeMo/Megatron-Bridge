# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for MegatronMIMO forward step functions."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch


class FakeColocatedPPPG:
    def __init__(self, *, rank: int = 0, size: int = 1) -> None:
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


class FakeColocatedPPGrid:
    def __init__(self, *, dp_rank: int = 0, dp_size: int = 1) -> None:
        self._dp = FakeColocatedPPPG(rank=dp_rank, size=dp_size)

    def get_pg(self, dims):
        assert dims in (["dp"], "dp")
        return self._dp


class FakeColocatedPPRole:
    def __init__(self, *, is_first: bool = True, is_last: bool = True) -> None:
        self._is_first = is_first
        self._is_last = is_last

    def is_first_stage(self, module_name):
        assert module_name == "language"
        return self._is_first

    def is_last_stage(self, module_name):
        assert module_name == "language"
        return self._is_last


class FakeColocatedPPMimoModel:
    def __init__(self, finalize_model_grads_func=None) -> None:
        self.role = FakeColocatedPPRole()
        self.special_token_ids = {"images": 99}
        self.config = SimpleNamespace(finalize_model_grads_func=finalize_model_grads_func)
        self.encoder_weight = torch.tensor(2.0, requires_grad=True)
        self.forward_calls = []

    def encode_and_communicate(self, modality_inputs):
        x = modality_inputs["images"]["x"].float().reshape(-1, 1)
        return {"images": x * self.encoder_weight}

    def __call__(self, **kwargs):
        self.forward_calls.append(kwargs)
        encoder_embeddings = kwargs["encoder_embeddings"]
        output = encoder_embeddings["images"].reshape(-1)
        return output, torch.ones_like(output)


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


class TestColocatedPPScheduleAdapter:
    """Tests for the colocated language-PP three-phase adapter."""

    def test_language_micro_batch_size_uses_language_dp(self):
        from megatron.bridge.training import megatron_mimo_step

        assert (
            megatron_mimo_step._language_micro_batch_size(
                micro_batch_size=8,
                language_grid=FakeColocatedPPGrid(dp_size=4),
            )
            == 2
        )

    def test_language_micro_batch_size_rejects_uneven_dp(self):
        from megatron.bridge.training import megatron_mimo_step

        with pytest.raises(ValueError, match="must be divisible"):
            megatron_mimo_step._language_micro_batch_size(
                micro_batch_size=7,
                language_grid=FakeColocatedPPGrid(dp_size=4),
            )

    def test_load_and_slice_microbatches_slices_inside_adapter_contract(self, monkeypatch):
        from megatron.bridge.training import megatron_mimo_step

        grids = {"language": object(), "images": object()}
        full_batches = [
            {"input_ids": torch.tensor([[1], [2]]), "modality_inputs": {"images": {"x": torch.tensor([[10], [11]])}}},
            {"input_ids": torch.tensor([[3], [4]]), "modality_inputs": {"images": {"x": torch.tensor([[12], [13]])}}},
        ]
        sliced_batches = [
            {"input_ids": torch.tensor([[1]]), "modality_inputs": {"images": {"x": torch.tensor([[10]])}}},
            {"input_ids": torch.tensor([[3]]), "modality_inputs": {"images": {"x": torch.tensor([[12]])}}},
        ]
        slice_calls = []

        def fake_slice(batch, *, grids):
            slice_calls.append((batch, grids))
            return sliced_batches[len(slice_calls) - 1]

        monkeypatch.setattr(megatron_mimo_step, "get_batch", lambda iterator: next(iterator))
        monkeypatch.setattr(megatron_mimo_step, "slice_batch_for_megatron_mimo_modules", fake_slice)

        original, sliced = megatron_mimo_step._load_and_slice_microbatches(
            data_iterator=iter(full_batches),
            grids=grids,
            num_microbatches=2,
        )

        assert original[0] is full_batches[0]
        assert original[1] is full_batches[1]
        assert sliced[0] is sliced_batches[0]
        assert sliced[1] is sliced_batches[1]
        assert slice_calls == [(full_batches[0], grids), (full_batches[1], grids)]

    def test_build_pp_encoder_input_recurses_and_preserves_metadata(self):
        from megatron.bridge.training import megatron_mimo_step

        original_batches = [
            {
                "modality_inputs": {
                    "images": {"clip": {"x": torch.tensor([[1], [2]]), "mode": "rgb"}},
                },
            },
            {
                "modality_inputs": {
                    "images": {"clip": {"x": torch.tensor([[3]]), "mode": "rgb"}},
                },
            },
        ]

        result = megatron_mimo_step._build_pp_encoder_input(
            original_batches=original_batches,
            encoder_module_name="images",
            encoder_grid=FakeColocatedPPGrid(),
            language_grid=FakeColocatedPPGrid(),
        )

        torch.testing.assert_close(result["clip"]["x"], torch.tensor([[1], [2], [3]]))
        assert result["clip"]["mode"] == "rgb"

    def test_build_pp_encoder_input_rejects_metadata_mismatch(self):
        from megatron.bridge.training import megatron_mimo_step

        original_batches = [
            {"modality_inputs": {"images": {"clip": {"x": torch.tensor([[1]]), "mode": "rgb"}}}},
            {"modality_inputs": {"images": {"clip": {"x": torch.tensor([[2]]), "mode": "bgr"}}}},
        ]

        with pytest.raises(ValueError, match="metadata differs"):
            megatron_mimo_step._build_pp_encoder_input(
                original_batches=original_batches,
                encoder_module_name="images",
                encoder_grid=FakeColocatedPPGrid(),
                language_grid=FakeColocatedPPGrid(),
            )

    def test_build_pp_encoder_input_equal_dp_takes_partition_per_microbatch(self):
        """Equal DP: each rank takes its DP partition slice from each microbatch and concatenates."""
        from megatron.bridge.training import megatron_mimo_step

        # Two microbatches of 4 samples each. With dp=2, rank 0 should see
        # the first 2 samples from each microbatch.
        original_batches = [
            {"modality_inputs": {"images": torch.tensor([[0.0], [1.0], [2.0], [3.0]])}},
            {"modality_inputs": {"images": torch.tensor([[4.0], [5.0], [6.0], [7.0]])}},
        ]

        result_rank0 = megatron_mimo_step._build_pp_encoder_input(
            original_batches=original_batches,
            encoder_module_name="images",
            encoder_grid=FakeColocatedPPGrid(dp_rank=0, dp_size=2),
            language_grid=FakeColocatedPPGrid(dp_rank=0, dp_size=2),
        )
        torch.testing.assert_close(result_rank0, torch.tensor([[0.0], [1.0], [4.0], [5.0]]))

        result_rank1 = megatron_mimo_step._build_pp_encoder_input(
            original_batches=original_batches,
            encoder_module_name="images",
            encoder_grid=FakeColocatedPPGrid(dp_rank=1, dp_size=2),
            language_grid=FakeColocatedPPGrid(dp_rank=1, dp_size=2),
        )
        torch.testing.assert_close(result_rank1, torch.tensor([[2.0], [3.0], [6.0], [7.0]]))

    def test_build_pp_encoder_input_fan_in_gather_recovers_microbatch_major_order(self):
        """Across all encoder-DP ranks, per-rank slices reassembled in rank
        order must equal the microbatch-major full language-DP partition.

        This is the property that ``_split_encoder_output`` downstream relies
        on: chunking the gathered tensor per microbatch must hand back each
        microbatch's encoder embeddings in microbatch order.
        """
        from megatron.bridge.training import megatron_mimo_step

        # 2 microbatches × 8 samples each. enc_dp=4, lm_dp=2 (scale=2).
        # Language partition 0 picks samples 0..3 of each microbatch.
        original_batches = [
            {"modality_inputs": {"images": torch.arange(8).float().reshape(8, 1)}},
            {"modality_inputs": {"images": (torch.arange(8) + 8).float().reshape(8, 1)}},
        ]
        language_grid = FakeColocatedPPGrid(dp_rank=0, dp_size=2)

        # Encoder-DP ranks 0 and 1 feed language partition 0 (rank // scale).
        per_rank_pieces = []
        for enc_dp_rank in (0, 1):
            piece = megatron_mimo_step._build_pp_encoder_input(
                original_batches=original_batches,
                encoder_module_name="images",
                encoder_grid=FakeColocatedPPGrid(dp_rank=enc_dp_rank, dp_size=4),
                language_grid=language_grid,
            )
            per_rank_pieces.append(piece)

        # Simulate the colocated bridge's all-gather along dim 0 in rank order.
        gathered = torch.cat(per_rank_pieces, dim=0)

        # Microbatch-major order for language partition 0:
        # [mb0_samples_0..3, mb1_samples_0..3] = [0,1,2,3, 8,9,10,11].
        expected = torch.tensor([[0.0], [1.0], [2.0], [3.0], [8.0], [9.0], [10.0], [11.0]])
        torch.testing.assert_close(gathered, expected)

    def test_build_pp_encoder_input_fan_out_slices_per_microbatch_by_encoder_dp(self):
        """Fan-out (encoder DP < language DP): each encoder rank holds its own
        encoder-DP slice of every microbatch; the bridge narrows in forward.
        """
        from megatron.bridge.training import megatron_mimo_step

        original_batches = [
            {"modality_inputs": {"images": torch.tensor([[0.0], [1.0], [2.0], [3.0]])}},
            {"modality_inputs": {"images": torch.tensor([[4.0], [5.0], [6.0], [7.0]])}},
        ]

        result_rank0 = megatron_mimo_step._build_pp_encoder_input(
            original_batches=original_batches,
            encoder_module_name="images",
            encoder_grid=FakeColocatedPPGrid(dp_rank=0, dp_size=2),
            language_grid=FakeColocatedPPGrid(dp_rank=0, dp_size=4),
        )
        torch.testing.assert_close(result_rank0, torch.tensor([[0.0], [1.0], [4.0], [5.0]]))

        result_rank1 = megatron_mimo_step._build_pp_encoder_input(
            original_batches=original_batches,
            encoder_module_name="images",
            encoder_grid=FakeColocatedPPGrid(dp_rank=1, dp_size=2),
            language_grid=FakeColocatedPPGrid(dp_rank=0, dp_size=4),
        )
        torch.testing.assert_close(result_rank1, torch.tensor([[2.0], [3.0], [6.0], [7.0]]))

    def test_build_pp_encoder_input_rejects_uneven_encoder_to_language_dp(self):
        from megatron.bridge.training import megatron_mimo_step

        original_batches = [
            {"modality_inputs": {"images": torch.zeros(6, 1)}},
        ]
        with pytest.raises(ValueError, match="must be divisible"):
            megatron_mimo_step._build_pp_encoder_input(
                original_batches=original_batches,
                encoder_module_name="images",
                encoder_grid=FakeColocatedPPGrid(dp_rank=0, dp_size=3),
                language_grid=FakeColocatedPPGrid(dp_rank=0, dp_size=2),
            )

    def test_pp_concat_alignment_check_accepts_aggregated_microbatches(self, monkeypatch):
        from megatron.bridge.training import megatron_mimo_step

        monkeypatch.setenv("MIMO_CHECK_DATA_ALIGNMENT", "true")
        monkeypatch.setenv("MIMO_CHECK_DATA_ALIGNMENT_STEPS", "1")
        megatron_mimo_step._DATA_ALIGNMENT_CHECK_COUNT = 0
        sliced_batches = [
            {
                "input_ids": torch.zeros(2, 4),
                "modality_inputs": {"images": {"x": torch.zeros(1, 3)}},
            },
            {
                "input_ids": torch.zeros(3, 4),
                "modality_inputs": {"images": {"x": torch.zeros(2, 3)}},
            },
        ]
        concatenated_input = {"x": torch.zeros(3, 3)}

        megatron_mimo_step._maybe_check_colocated_pp_concat_alignment(
            sliced_batches=sliced_batches,
            concatenated_input=concatenated_input,
            encoder_module_name="images",
        )

        assert megatron_mimo_step._DATA_ALIGNMENT_CHECK_COUNT == 1

    def test_pp_concat_alignment_check_rejects_bad_encoder_concat_size(self, monkeypatch):
        from megatron.bridge.training import megatron_mimo_step

        monkeypatch.setenv("MIMO_CHECK_DATA_ALIGNMENT", "true")
        monkeypatch.setenv("MIMO_CHECK_DATA_ALIGNMENT_STEPS", "1")
        megatron_mimo_step._DATA_ALIGNMENT_CHECK_COUNT = 0
        sliced_batches = [
            {
                "input_ids": torch.zeros(2, 4),
                "modality_inputs": {"images": {"x": torch.zeros(1, 3)}},
            },
            {
                "input_ids": torch.zeros(3, 4),
                "modality_inputs": {"images": {"x": torch.zeros(2, 3)}},
            },
        ]
        concatenated_input = {"x": torch.zeros(4, 3)}

        with pytest.raises(RuntimeError, match="concatenated encoder batch"):
            megatron_mimo_step._maybe_check_colocated_pp_concat_alignment(
                sliced_batches=sliced_batches,
                concatenated_input=concatenated_input,
                encoder_module_name="images",
            )

    def test_split_flattened_encoder_output_uses_placeholder_counts(self):
        from megatron.bridge.training import megatron_mimo_step

        output = torch.arange(10, dtype=torch.float32).reshape(5, 2)

        chunks = megatron_mimo_step._split_encoder_output(
            output,
            token_counts=[2, 0, 3],
            language_batch_sizes=[1, 1, 1],
            encoder_module_name="images",
        )

        torch.testing.assert_close(chunks[0], output[:2])
        assert chunks[1].shape == (0, 2)
        torch.testing.assert_close(chunks[2], output[2:])

    def test_split_sbh_encoder_output_uses_language_batch_sizes(self):
        from megatron.bridge.training import megatron_mimo_step

        output = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)

        chunks = megatron_mimo_step._split_encoder_output(
            output,
            token_counts=[1, 1],
            language_batch_sizes=[1, 2],
            encoder_module_name="images",
        )

        torch.testing.assert_close(chunks[0], output[:, :1, :])
        torch.testing.assert_close(chunks[1], output[:, 1:, :])

    def test_build_cached_language_microbatches_preserves_uneven_token_boundaries(self):
        from megatron.bridge.training import megatron_mimo_step

        sliced_batches = [
            {
                "input_ids": torch.tensor([[99, 1, 99]]),
                "position_ids": torch.tensor([[0, 1, 2]]),
                "labels": torch.tensor([[2, 3, 4]]),
                "loss_mask": torch.tensor([[1.0, 1.0, 0.0]]),
            },
            {
                "input_ids": torch.tensor([[1, 99, 2]]),
                "position_ids": torch.tensor([[0, 1, 2]]),
                "labels": torch.tensor([[3, 4, 5]]),
                "loss_mask": torch.tensor([[1.0, 0.0, 1.0]]),
            },
        ]
        detached_outputs = {"images": torch.tensor([[10.0], [11.0], [12.0]])}

        cached = megatron_mimo_step._build_cached_language_microbatches(
            detached_encoder_outputs=detached_outputs,
            sliced_batches=sliced_batches,
            encoder_module_name="images",
            special_token_ids={"images": 99},
        )

        torch.testing.assert_close(cached[0]["encoder_embeddings"]["images"], torch.tensor([[10.0], [11.0]]))
        torch.testing.assert_close(cached[1]["encoder_embeddings"]["images"], torch.tensor([[12.0]]))

    def test_deferred_finalize_captures_and_restores_on_exception(self):
        from megatron.bridge.training import megatron_mimo_step

        original_calls = []

        def original_finalize(*args, **kwargs):
            original_calls.append((args, kwargs))

        config = SimpleNamespace(finalize_model_grads_func=original_finalize)
        num_tokens = torch.tensor(7)

        with pytest.raises(RuntimeError, match="phase 2 failed"):
            with megatron_mimo_step._deferred_finalize(config) as (original_finalize_from_context, capture):
                assert original_finalize_from_context is original_finalize
                config.finalize_model_grads_func([object()], num_tokens, force_all_reduce=True)
                raise RuntimeError("phase 2 failed")

        assert config.finalize_model_grads_func is original_finalize
        assert capture.called
        assert capture.num_tokens is num_tokens
        assert capture.force_all_reduce is True
        assert original_calls == []

    def test_inner_forward_step_passes_stage_specific_kwargs(self, monkeypatch):
        from megatron.bridge.training import megatron_mimo_step

        first_stage_model = SimpleNamespace(role=FakeColocatedPPRole(is_first=True, is_last=False))
        calls = []

        class CallableModel:
            def __call__(self, **kwargs):
                calls.append(kwargs)
                return torch.tensor([1.0, 2.0])

        monkeypatch.setattr(megatron_mimo_step, "unwrap_megatron_mimo_model", lambda _model: first_stage_model)
        forward_step = megatron_mimo_step._make_inner_language_forward_step()
        cached = {
            "input_ids": torch.tensor([[99, 1]]),
            "position_ids": torch.tensor([[0, 1]]),
            "labels": torch.tensor([[1, 2]]),
            "loss_mask": torch.ones(1, 2),
            "attention_mask": torch.ones(1, 1, 2, 2),
            "encoder_embeddings": {"images": torch.tensor([[0.5]])},
        }

        output, loss_fn = forward_step(iter([cached]), CallableModel())

        torch.testing.assert_close(output, torch.tensor([1.0, 2.0]))
        assert loss_fn is None
        assert calls[0]["input_ids"] is cached["input_ids"]
        assert calls[0]["encoder_embeddings"] is cached["encoder_embeddings"]
        assert calls[0]["labels"] is None
        assert calls[0]["loss_mask"] is None

    def test_inner_forward_step_last_stage_uses_cached_loss_fields(self, monkeypatch):
        from megatron.bridge.training import megatron_mimo_step

        last_stage_model = SimpleNamespace(role=FakeColocatedPPRole(is_first=False, is_last=True))
        calls = []
        output_tensor = torch.tensor([3.0, 4.0])

        class CallableModel:
            def __call__(self, **kwargs):
                calls.append(kwargs)
                return output_tensor

        monkeypatch.setattr(megatron_mimo_step, "unwrap_megatron_mimo_model", lambda _model: last_stage_model)
        monkeypatch.setattr(
            megatron_mimo_step,
            "get_batch",
            lambda _iterator: (_ for _ in ()).throw(AssertionError("inner forward step must not call get_batch")),
        )
        forward_step = megatron_mimo_step._make_inner_language_forward_step()
        cached = {
            "input_ids": torch.tensor([[99, 1]]),
            "position_ids": torch.tensor([[0, 1]]),
            "labels": torch.tensor([[1, 2]]),
            "loss_mask": torch.tensor([1.0, 0.0]),
            "attention_mask": torch.ones(1, 1, 2, 2),
            "encoder_embeddings": {"images": torch.tensor([[0.5]])},
        }

        output, loss_fn = forward_step(iter([cached]), CallableModel())
        loss, num_tokens, metrics = loss_fn(output)

        torch.testing.assert_close(output, output_tensor)
        assert loss.item() == pytest.approx(3.0)
        assert num_tokens.item() == 1
        assert "lm loss" in metrics
        assert calls[0]["input_ids"] is None
        assert calls[0]["position_ids"] is None
        assert calls[0]["encoder_embeddings"] is None
        assert calls[0]["labels"] is cached["labels"]
        assert calls[0]["loss_mask"] is cached["loss_mask"]
        assert calls[0]["attention_mask"] is cached["attention_mask"]

    def test_inner_forward_step_middle_stage_uses_only_attention_metadata(self, monkeypatch):
        from megatron.bridge.training import megatron_mimo_step

        middle_stage_model = SimpleNamespace(role=FakeColocatedPPRole(is_first=False, is_last=False))
        calls = []
        output_tensor = torch.tensor([5.0, 6.0])

        class CallableModel:
            def __call__(self, **kwargs):
                calls.append(kwargs)
                return output_tensor

        monkeypatch.setattr(megatron_mimo_step, "unwrap_megatron_mimo_model", lambda _model: middle_stage_model)
        forward_step = megatron_mimo_step._make_inner_language_forward_step()
        cached = {
            "input_ids": torch.tensor([[99, 1]]),
            "position_ids": torch.tensor([[0, 1]]),
            "labels": torch.tensor([[1, 2]]),
            "loss_mask": torch.ones(1, 2),
            "attention_mask": torch.ones(1, 1, 2, 2),
            "encoder_embeddings": {"images": torch.tensor([[0.5]])},
        }

        output, loss_fn = forward_step(iter([cached]), CallableModel())

        torch.testing.assert_close(output, output_tensor)
        assert loss_fn is None
        assert calls[0]["input_ids"] is None
        assert calls[0]["position_ids"] is None
        assert calls[0]["labels"] is None
        assert calls[0]["loss_mask"] is None
        assert calls[0]["encoder_embeddings"] is None
        assert calls[0]["attention_mask"] is cached["attention_mask"]

    def test_adapter_passes_language_pipeline_schedule_kwargs(self, monkeypatch):
        from megatron.bridge.training import megatron_mimo_step

        model = FakeColocatedPPMimoModel()
        language_pg = SimpleNamespace(pp=FakeColocatedPPPG())
        language_grid = FakeColocatedPPGrid(dp_size=2)
        infra = SimpleNamespace(
            module_to_grid_map={"language": language_grid, "images": FakeColocatedPPGrid()},
            pg_collections={"language": language_pg, "images": SimpleNamespace()},
        )
        p2p_communicator = object()
        batches = [
            {
                "input_ids": torch.tensor([[99, 1], [2, 3]]),
                "position_ids": torch.tensor([[0, 1], [0, 1]]),
                "labels": torch.tensor([[1, 2], [3, 4]]),
                "loss_mask": torch.ones(2, 2),
                "modality_inputs": {"images": {"x": torch.tensor([[3.0]])}},
            },
        ]
        schedule_calls = []

        def fake_schedule(**kwargs):
            schedule_calls.append(kwargs)
            output, batch_loss_func = kwargs["forward_step_func"](kwargs["data_iterator"], kwargs["model"][0])
            loss, num_tokens, metrics = batch_loss_func(output)
            assert loss.requires_grad is False
            assert num_tokens.item() == 1
            return [metrics]

        monkeypatch.setattr(megatron_mimo_step, "unwrap_megatron_mimo_model", lambda _model: model)
        monkeypatch.setattr(megatron_mimo_step, "get_batch", lambda iterator: next(iterator))
        monkeypatch.setattr(
            megatron_mimo_step, "_maybe_check_colocated_data_alignment", lambda *_args, **_kwargs: None
        )
        monkeypatch.setattr(megatron_mimo_step, "forward_backward_pipelining_without_interleaving", fake_schedule)

        losses = megatron_mimo_step.forward_backward_colocated_mimo_with_pp(
            model=model,
            data_iterator=iter(batches),
            infra=infra,
            encoder_module_name="images",
            num_microbatches=1,
            seq_length=7,
            micro_batch_size=4,
            forward_only=True,
            p2p_communicator=p2p_communicator,
        )

        assert len(losses) == 1
        call_kwargs = schedule_calls[0]
        assert call_kwargs["model"] == [model]
        assert call_kwargs["num_microbatches"] == 1
        assert call_kwargs["seq_length"] == 7
        assert call_kwargs["decoder_seq_length"] == 7
        assert call_kwargs["micro_batch_size"] == 2
        assert call_kwargs["forward_only"] is True
        assert call_kwargs["p2p_communicator"] is p2p_communicator
        assert call_kwargs["pg_collection"] is language_pg

    @pytest.mark.parametrize("force_all_reduce", [False, True])
    def test_adapter_training_defers_finalize_until_after_encoder_backward(self, monkeypatch, force_all_reduce):
        from megatron.bridge.training import megatron_mimo_step

        finalize_calls = []
        bound_module_to_grid_tuple = [("images", object()), ("language", object())]

        def multimodule_finalize(model_list, num_tokens, *, pg_collection, force_all_reduce, module_to_grid_tuple):
            assert model.encoder_weight.grad is not None
            assert module_to_grid_tuple is bound_module_to_grid_tuple
            finalize_calls.append(
                {
                    "model_list": model_list,
                    "num_tokens": num_tokens,
                    "pg_collection": pg_collection,
                    "force_all_reduce": force_all_reduce,
                }
            )

        original_finalize = partial(multimodule_finalize, module_to_grid_tuple=bound_module_to_grid_tuple)
        model = FakeColocatedPPMimoModel(finalize_model_grads_func=original_finalize)
        language_pg = SimpleNamespace(pp=FakeColocatedPPPG())
        infra = SimpleNamespace(
            module_to_grid_map={"language": FakeColocatedPPGrid(), "images": FakeColocatedPPGrid()},
            pg_collections={"language": language_pg, "images": SimpleNamespace()},
        )
        batches = [
            {
                "input_ids": torch.tensor([[99, 1]]),
                "position_ids": torch.tensor([[0, 1]]),
                "labels": torch.tensor([[1, 2]]),
                "loss_mask": torch.tensor([[1.0, 1.0]]),
                "modality_inputs": {"images": {"x": torch.tensor([[3.0]])}},
            },
            {
                "input_ids": torch.tensor([[99, 2]]),
                "position_ids": torch.tensor([[0, 1]]),
                "labels": torch.tensor([[2, 3]]),
                "loss_mask": torch.tensor([[1.0, 1.0]]),
                "modality_inputs": {"images": {"x": torch.tensor([[4.0]])}},
            },
        ]

        def fake_schedule(**kwargs):
            losses = []
            total_num_tokens = torch.tensor(0, dtype=torch.int32)
            for _ in range(kwargs["num_microbatches"]):
                output, batch_loss_func = kwargs["forward_step_func"](kwargs["data_iterator"], kwargs["model"][0])
                loss, num_tokens, metrics = batch_loss_func(output)
                total_num_tokens += num_tokens
                loss.backward()
                losses.append(metrics)
            kwargs["model"][0].config.finalize_model_grads_func(
                kwargs["model"],
                total_num_tokens,
                pg_collection=kwargs["pg_collection"],
                force_all_reduce=force_all_reduce,
            )
            return losses

        monkeypatch.setattr(megatron_mimo_step, "unwrap_megatron_mimo_model", lambda _model: model)
        monkeypatch.setattr(megatron_mimo_step, "get_batch", lambda iterator: next(iterator))
        monkeypatch.setattr(
            megatron_mimo_step, "_maybe_check_colocated_data_alignment", lambda *_args, **_kwargs: None
        )
        monkeypatch.setattr(megatron_mimo_step, "forward_backward_pipelining_without_interleaving", fake_schedule)

        losses = megatron_mimo_step.forward_backward_colocated_mimo_with_pp(
            model=model,
            data_iterator=iter(batches),
            infra=infra,
            encoder_module_name="images",
            num_microbatches=2,
            seq_length=2,
            micro_batch_size=1,
            forward_only=False,
            p2p_communicator=object(),
        )

        assert len(losses) == 2
        assert model.encoder_weight.grad is not None
        assert len(finalize_calls) == 1
        assert finalize_calls[0]["model_list"] == [model]
        assert finalize_calls[0]["pg_collection"] is language_pg
        assert finalize_calls[0]["force_all_reduce"] is force_all_reduce
        assert finalize_calls[0]["num_tokens"].item() == 2
        assert model.config.finalize_model_grads_func is original_finalize

    def test_adapter_restores_finalizer_when_phase2_schedule_raises(self, monkeypatch):
        from megatron.bridge.training import megatron_mimo_step

        original_calls = []

        def original_finalize(*args, **kwargs):
            original_calls.append((args, kwargs))

        model = FakeColocatedPPMimoModel(finalize_model_grads_func=original_finalize)
        language_pg = SimpleNamespace(pp=FakeColocatedPPPG())
        infra = SimpleNamespace(
            module_to_grid_map={"language": FakeColocatedPPGrid(), "images": FakeColocatedPPGrid()},
            pg_collections={"language": language_pg, "images": SimpleNamespace()},
        )
        batches = [
            {
                "input_ids": torch.tensor([[99, 1]]),
                "position_ids": torch.tensor([[0, 1]]),
                "labels": torch.tensor([[1, 2]]),
                "loss_mask": torch.tensor([[1.0, 1.0]]),
                "modality_inputs": {"images": {"x": torch.tensor([[3.0]])}},
            },
        ]

        def fake_schedule(**kwargs):
            assert model.config.finalize_model_grads_func is not original_finalize
            kwargs["model"][0].config.finalize_model_grads_func(
                kwargs["model"],
                torch.tensor(3),
                pg_collection=kwargs["pg_collection"],
                force_all_reduce=False,
            )
            raise RuntimeError("phase 2 failed")

        monkeypatch.setattr(megatron_mimo_step, "unwrap_megatron_mimo_model", lambda _model: model)
        monkeypatch.setattr(megatron_mimo_step, "get_batch", lambda iterator: next(iterator))
        monkeypatch.setattr(
            megatron_mimo_step, "_maybe_check_colocated_data_alignment", lambda *_args, **_kwargs: None
        )
        monkeypatch.setattr(megatron_mimo_step, "forward_backward_pipelining_without_interleaving", fake_schedule)

        with pytest.raises(RuntimeError, match="phase 2 failed"):
            megatron_mimo_step.forward_backward_colocated_mimo_with_pp(
                model=model,
                data_iterator=iter(batches),
                infra=infra,
                encoder_module_name="images",
                num_microbatches=1,
                seq_length=2,
                micro_batch_size=1,
                forward_only=False,
                p2p_communicator=object(),
            )

        assert model.config.finalize_model_grads_func is original_finalize
        assert original_calls == []
        assert model.encoder_weight.grad is None

    def test_backward_encoder_outputs_uses_restored_grad_for_encoder_backward(self, monkeypatch):
        from megatron.bridge.training import megatron_mimo_step

        pp_group = object()
        encoder_weight = torch.tensor(2.0, requires_grad=True)
        encoder_output = torch.tensor([2.0, 3.0]) * encoder_weight
        detached_output = encoder_output.detach().requires_grad_(True)
        broadcast_calls = []

        def fake_broadcast_encoder_grads(*, detached_encoder_outputs, pp_group):
            broadcast_calls.append(pp_group)
            detached_encoder_outputs["images"].grad = torch.tensor([5.0, 7.0])

        monkeypatch.setattr(megatron_mimo_step, "_broadcast_encoder_grads", fake_broadcast_encoder_grads)

        megatron_mimo_step._backward_encoder_outputs(
            detached_encoder_outputs={"images": detached_output},
            encoder_outputs={"images": encoder_output},
            pp_group=pp_group,
        )

        assert broadcast_calls == [pp_group]
        torch.testing.assert_close(encoder_weight.grad, torch.tensor(31.0))

    def test_broadcast_encoder_grads_first_stage_broadcasts_existing_grad(self, monkeypatch):
        from megatron.bridge.training import megatron_mimo_step

        pp_group = object()
        detached_output = torch.tensor([1.0, 2.0], requires_grad=True)
        detached_output.grad = torch.tensor([4.0, 5.0])
        broadcast_calls = []

        def fake_broadcast(tensor, *, src, group):
            broadcast_calls.append({"tensor": tensor, "src": src, "group": group})

        monkeypatch.setattr(megatron_mimo_step, "_process_group_size", lambda group: 2)
        monkeypatch.setattr(megatron_mimo_step, "_process_group_rank", lambda group: 0)
        monkeypatch.setattr(megatron_mimo_step.dist, "get_global_rank", lambda group, rank: 13)
        monkeypatch.setattr(megatron_mimo_step.dist, "broadcast", fake_broadcast)

        megatron_mimo_step._broadcast_encoder_grads(
            detached_encoder_outputs={"images": detached_output},
            pp_group=pp_group,
        )

        assert len(broadcast_calls) == 1
        assert broadcast_calls[0]["tensor"] is detached_output.grad
        assert broadcast_calls[0]["src"] == 13
        assert broadcast_calls[0]["group"] is pp_group
        torch.testing.assert_close(detached_output.grad, torch.tensor([4.0, 5.0]))

    def test_broadcast_encoder_grads_non_first_stage_receives_stage_zero_grad(self, monkeypatch):
        from megatron.bridge.training import megatron_mimo_step

        pp_group = object()
        detached_output = torch.tensor([1.0, 2.0], requires_grad=True)
        received_grad = torch.tensor([4.0, 5.0])
        broadcast_calls = []

        def fake_broadcast(tensor, *, src, group):
            broadcast_calls.append({"tensor": tensor, "src": src, "group": group})
            tensor.copy_(received_grad)

        monkeypatch.setattr(megatron_mimo_step, "_process_group_size", lambda group: 2)
        monkeypatch.setattr(megatron_mimo_step, "_process_group_rank", lambda group: 1)
        monkeypatch.setattr(megatron_mimo_step.dist, "get_global_rank", lambda group, rank: 13)
        monkeypatch.setattr(megatron_mimo_step.dist, "broadcast", fake_broadcast)

        megatron_mimo_step._broadcast_encoder_grads(
            detached_encoder_outputs={"images": detached_output},
            pp_group=pp_group,
        )

        assert len(broadcast_calls) == 1
        assert broadcast_calls[0]["src"] == 13
        assert broadcast_calls[0]["group"] is pp_group
        torch.testing.assert_close(detached_output.grad, received_grad)

    def test_broadcast_encoder_grads_first_stage_requires_grad(self, monkeypatch):
        from megatron.bridge.training import megatron_mimo_step

        monkeypatch.setattr(megatron_mimo_step, "_process_group_size", lambda group: 2)
        monkeypatch.setattr(megatron_mimo_step, "_process_group_rank", lambda group: 0)
        monkeypatch.setattr(megatron_mimo_step.dist, "get_global_rank", lambda group, rank: 13)

        with pytest.raises(RuntimeError, match="No encoder gradient available"):
            megatron_mimo_step._broadcast_encoder_grads(
                detached_encoder_outputs={"images": torch.tensor([1.0, 2.0], requires_grad=True)},
                pp_group=object(),
            )

    def test_adapter_forward_only_skips_encoder_backward_and_finalize(self, monkeypatch):
        from megatron.bridge.training import megatron_mimo_step

        def original_finalize(*_args, **_kwargs):
            raise AssertionError("forward_only must not replay finalize")

        model = FakeColocatedPPMimoModel(finalize_model_grads_func=original_finalize)
        language_pg = SimpleNamespace(pp=FakeColocatedPPPG())
        infra = SimpleNamespace(
            module_to_grid_map={"language": FakeColocatedPPGrid(), "images": FakeColocatedPPGrid()},
            pg_collections={"language": language_pg, "images": SimpleNamespace()},
        )
        batches = [
            {
                "input_ids": torch.tensor([[99, 1]]),
                "position_ids": torch.tensor([[0, 1]]),
                "labels": torch.tensor([[1, 2]]),
                "loss_mask": torch.tensor([[1.0, 1.0]]),
                "modality_inputs": {"images": {"x": torch.tensor([[3.0]])}},
            }
        ]

        def fake_schedule(**kwargs):
            output, batch_loss_func = kwargs["forward_step_func"](kwargs["data_iterator"], kwargs["model"][0])
            loss, _num_tokens, metrics = batch_loss_func(output)
            assert loss.requires_grad is False
            return [metrics]

        monkeypatch.setattr(megatron_mimo_step, "unwrap_megatron_mimo_model", lambda _model: model)
        monkeypatch.setattr(megatron_mimo_step, "get_batch", lambda iterator: next(iterator))
        monkeypatch.setattr(
            megatron_mimo_step, "_maybe_check_colocated_data_alignment", lambda *_args, **_kwargs: None
        )
        monkeypatch.setattr(megatron_mimo_step, "forward_backward_pipelining_without_interleaving", fake_schedule)

        losses = megatron_mimo_step.forward_backward_colocated_mimo_with_pp(
            model=model,
            data_iterator=iter(batches),
            infra=infra,
            encoder_module_name="images",
            num_microbatches=1,
            seq_length=2,
            micro_batch_size=1,
            forward_only=True,
            p2p_communicator=object(),
        )

        assert len(losses) == 1
        assert model.encoder_weight.grad is None
