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
    """Tests for the colocated language-PP adapter."""

    def test_language_micro_batch_size_rejects_uneven_dp(self):
        from megatron.bridge.training import megatron_mimo_step

        with pytest.raises(ValueError, match="must be divisible"):
            megatron_mimo_step._language_micro_batch_size(
                micro_batch_size=7,
                language_grid=FakeColocatedPPGrid(dp_size=4),
            )

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

    def test_adapter_restores_finalizer_when_language_schedule_raises(self, monkeypatch):
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
            raise RuntimeError("language schedule failed")

        monkeypatch.setattr(megatron_mimo_step, "unwrap_megatron_mimo_model", lambda _model: model)
        monkeypatch.setattr(megatron_mimo_step, "get_batch", lambda iterator: next(iterator))
        monkeypatch.setattr(
            megatron_mimo_step, "_maybe_check_colocated_data_alignment", lambda *_args, **_kwargs: None
        )
        monkeypatch.setattr(megatron_mimo_step, "forward_backward_pipelining_without_interleaving", fake_schedule)

        with pytest.raises(RuntimeError, match="language schedule failed"):
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
