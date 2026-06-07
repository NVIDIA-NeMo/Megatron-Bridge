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
from unittest.mock import MagicMock, Mock, patch

import modelopt.torch.distill as mtd
import torch
from megatron.core.packed_seq_params import PackedSeqParams

from megatron.bridge.training.gpt_step import (
    _create_loss_function_modelopt,
    _forward_step_common,
    get_batch,
    get_packed_seq_params,
)
from megatron.bridge.training.losses import (
    create_masked_next_token_loss_function as _create_loss_function,
)


class _Iterator:
    def __init__(self, batch):
        self.batch = batch
        self._done = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._done:
            raise StopIteration
        self._done = True
        return self.batch


class _MockProcessGroup:
    def rank(self):
        return 0

    def size(self):
        return 1


class _MockPGCollection:
    def __init__(self, cp_size=1):
        self.pp = _MockProcessGroup()
        self._cp_size = cp_size

    @property
    def cp(self):
        pg = _MockProcessGroup()
        pg.size = lambda: self._cp_size
        return pg


class _NoCudaTensor(torch.Tensor):
    def cuda(self, non_blocking=False):  # type: ignore[override]
        return self


def _as_nocuda(tensor):
    return tensor.as_subclass(_NoCudaTensor)


def _make_cfg(*, packed_sequence_specs=None, skip_getting_attention_mask_from_dataset=True):
    cfg = type("Cfg", (), {})()
    cfg.dataset = type(
        "D",
        (),
        {
            "packed_sequence_specs": packed_sequence_specs,
            "skip_getting_attention_mask_from_dataset": skip_getting_attention_mask_from_dataset,
        },
    )()
    return cfg


def _set_middle_pp_stage(monkeypatch):
    monkeypatch.setattr("megatron.bridge.training.gpt_step.is_pp_first_stage", lambda pg: False)
    monkeypatch.setattr("megatron.bridge.training.gpt_step.is_pp_last_stage", lambda pg: False)


class _NoopTimer:
    def __call__(self, *args, **kwargs):
        return self

    def start(self):
        return None

    def stop(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class TestGetBatch:
    """Tests for the get_batch helper."""

    def test_middle_pp_stage_preserves_full_packed_batch(self, monkeypatch):
        """Middle PP stages load full tensors when packed metadata is active."""
        _set_middle_pp_stage(monkeypatch)
        monkeypatch.setattr(
            "megatron.bridge.training.gpt_step.get_batch_on_this_cp_rank",
            lambda batch, is_hybrid_cp=False, cp_group=None, hybrid_cp_group_func=None: batch,
        )

        tokens = _as_nocuda(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]))
        labels = _as_nocuda(torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]]))
        loss_mask = _as_nocuda(torch.ones(1, 8))
        attention_mask = _as_nocuda(torch.ones(1, 1, 8, 8, dtype=torch.bool))
        position_ids = _as_nocuda(torch.arange(8).unsqueeze(0))
        cu_seqlens = _as_nocuda(torch.tensor([[0, 3, 8, -1]], dtype=torch.int32))
        cu_seqlens_unpadded = _as_nocuda(torch.tensor([[0, 2, 7, -1]], dtype=torch.int32))
        cu_seqlens_argmin = torch.tensor(3)
        cu_seqlens_unpadded_argmin = torch.tensor(3)
        max_seqlen = torch.tensor([[5]], dtype=torch.int32)
        batch = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "cu_seqlens": cu_seqlens,
            "cu_seqlens_argmin": cu_seqlens_argmin,
            "max_seqlen": max_seqlen,
            "cu_seqlens_unpadded": cu_seqlens_unpadded,
            "cu_seqlens_unpadded_argmin": cu_seqlens_unpadded_argmin,
        }

        (
            tokens,
            labels,
            loss_mask,
            attention_mask,
            position_ids,
            out_cu_seqlens,
            out_cu_seqlens_argmin,
            out_max_seqlen,
            out_cu_seqlens_unpadded,
            out_cu_seqlens_unpadded_argmin,
        ) = get_batch(
            _Iterator(batch),
            _make_cfg(packed_sequence_specs=object()),
            use_mtp=False,
            pg_collection=_MockPGCollection(),
        )

        assert torch.equal(tokens, batch["tokens"])
        assert torch.equal(labels, batch["labels"])
        assert torch.equal(loss_mask, batch["loss_mask"])
        assert torch.equal(attention_mask, batch["attention_mask"])
        assert torch.equal(position_ids, batch["position_ids"])
        assert torch.equal(out_cu_seqlens, cu_seqlens)
        assert torch.equal(out_cu_seqlens_argmin, cu_seqlens_argmin)
        assert torch.equal(out_max_seqlen, max_seqlen)
        assert torch.equal(out_cu_seqlens_unpadded, cu_seqlens_unpadded)
        assert torch.equal(out_cu_seqlens_unpadded_argmin, cu_seqlens_unpadded_argmin)

    def test_middle_pp_stage_keeps_non_packed_fast_path(self, monkeypatch):
        """Middle PP stages without attention metadata keep the all-None fast path."""
        _set_middle_pp_stage(monkeypatch)
        data_iterator = MagicMock()

        result = get_batch(
            data_iterator,
            _make_cfg(packed_sequence_specs=None),
            use_mtp=False,
            pg_collection=_MockPGCollection(),
        )

        assert result == (None, None, None, None, None, None, None, None, None, None)
        data_iterator.__next__.assert_not_called()

    def test_forward_common_passes_packed_seq_params_on_middle_pp_stage(self, monkeypatch):
        """Forward path must pass packed metadata on middle PP stages."""
        sentinel_packed_seq_params = object()
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        labels = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
        loss_mask = torch.ones(1, 8)
        position_ids = torch.arange(8).unsqueeze(0)
        cu_seqlens = torch.tensor([[0, 3, 8, -1]], dtype=torch.int32)
        cu_seqlens_argmin = torch.tensor(3)
        max_seqlen = torch.tensor([[5]], dtype=torch.int32)
        model = Mock(return_value=torch.tensor(1.0))
        state = Mock()
        state.cfg = _make_cfg(packed_sequence_specs=object())
        state.timers = _NoopTimer()
        state.straggler_timer = _NoopTimer()
        config = type(
            "Config",
            (),
            {
                "is_hybrid_model": False,
                "mtp_num_layers": 0,
                "overlap_moe_expert_parallel_comm": False,
            },
        )()

        monkeypatch.setattr("megatron.bridge.training.gpt_step.get_model_config", lambda model: config)
        monkeypatch.setattr("megatron.bridge.training.gpt_step.get_pg_collection", lambda model: _MockPGCollection())
        monkeypatch.setattr(
            "megatron.bridge.training.gpt_step.get_batch",
            lambda data_iterator, cfg, use_mtp, pg_collection: (
                tokens,
                labels,
                loss_mask,
                None,
                position_ids,
                cu_seqlens,
                cu_seqlens_argmin,
                max_seqlen,
                None,
                None,
            ),
        )
        monkeypatch.setattr(
            "megatron.bridge.training.gpt_step.get_packed_seq_params",
            Mock(return_value=sentinel_packed_seq_params),
        )

        output, returned_loss_mask = _forward_step_common(state, _Iterator({}), model)

        assert torch.equal(output, torch.tensor(1.0))
        assert torch.equal(returned_loss_mask, loss_mask)
        model.assert_called_once_with(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=None,
            labels=labels,
            packed_seq_params=sentinel_packed_seq_params,
        )


class TestGetPackedSeqParams:
    """Tests for the get_packed_seq_params function."""

    def test_basic_packed_seq_params_with_max_seqlen(self):
        """Test basic functionality with cu_seqlens and max_seqlen."""
        # Create test batch with packed sequence data
        batch = {
            "cu_seqlens": torch.tensor([[0, 5, 12, 20, -1, -1]], dtype=torch.int32),  # batch size 1
            "max_seqlen": torch.tensor([[15]], dtype=torch.int32),  # batch size 1
        }

        result = get_packed_seq_params(batch)

        # Verify the result is a PackedSeqParams object
        assert isinstance(result, PackedSeqParams)

        # Verify cu_seqlens was squeezed and padding removed (stops at first -1)
        expected_cu_seqlens = torch.tensor([0, 5, 12, 20], dtype=torch.int32)
        assert torch.equal(result.cu_seqlens_q, expected_cu_seqlens)
        assert torch.equal(result.cu_seqlens_kv, expected_cu_seqlens)

        # Verify max_seqlen was squeezed
        expected_max_seqlen = torch.tensor(15, dtype=torch.int32)
        assert torch.equal(result.max_seqlen_q, expected_max_seqlen)
        assert torch.equal(result.max_seqlen_kv, expected_max_seqlen)

        # Verify qkv_format is correct
        assert result.qkv_format == "thd"

    def test_packed_seq_params_without_max_seqlen(self):
        """Test functionality when max_seqlen is not provided."""
        batch = {
            "cu_seqlens": torch.tensor([[0, 3, 8, 15, -1]], dtype=torch.int32),
        }

        result = get_packed_seq_params(batch)

        # Verify the result is a PackedSeqParams object
        assert isinstance(result, PackedSeqParams)

        # Verify cu_seqlens was processed correctly
        expected_cu_seqlens = torch.tensor([0, 3, 8, 15], dtype=torch.int32)
        assert torch.equal(result.cu_seqlens_q, expected_cu_seqlens)
        assert torch.equal(result.cu_seqlens_kv, expected_cu_seqlens)

        # Verify max_seqlen is None when not provided
        assert result.max_seqlen_q is None
        assert result.max_seqlen_kv is None

        # Verify qkv_format is correct
        assert result.qkv_format == "thd"

    def test_packed_seq_params_with_cu_seqlens_argmin(self):
        """Test functionality when cu_seqlens_argmin is provided for performance."""
        batch = {
            "cu_seqlens": torch.tensor([[0, 4, 9, 16, 22, -1, -1, -1]], dtype=torch.int32),
            "cu_seqlens_argmin": torch.tensor(5),  # Index where -1 starts
            "max_seqlen": torch.tensor([[18]], dtype=torch.int32),
        }

        result = get_packed_seq_params(batch)

        # Verify the result is a PackedSeqParams object
        assert isinstance(result, PackedSeqParams)

        # Verify cu_seqlens was truncated using cu_seqlens_argmin
        expected_cu_seqlens = torch.tensor([0, 4, 9, 16, 22], dtype=torch.int32)
        assert torch.equal(result.cu_seqlens_q, expected_cu_seqlens)
        assert torch.equal(result.cu_seqlens_kv, expected_cu_seqlens)

        # Verify max_seqlen was processed correctly
        expected_max_seqlen = torch.tensor(18, dtype=torch.int32)
        assert torch.equal(result.max_seqlen_q, expected_max_seqlen)
        assert torch.equal(result.max_seqlen_kv, expected_max_seqlen)

    def test_packed_seq_params_with_cu_seqlens_argmin_zero(self):
        """Test edge case when cu_seqlens_argmin is 0."""
        batch = {
            "cu_seqlens": torch.tensor([[-1, -1, -1]], dtype=torch.int32),
            "cu_seqlens_argmin": torch.tensor(0),  # All are padding
        }

        result = get_packed_seq_params(batch)

        # Verify empty cu_seqlens when argmin is 0
        expected_cu_seqlens = torch.empty(0, dtype=torch.int32)
        assert torch.equal(result.cu_seqlens_q, expected_cu_seqlens)
        assert torch.equal(result.cu_seqlens_kv, expected_cu_seqlens)

    def test_packed_seq_params_batch_dimension_removal(self):
        """Test that batch dimensions are properly squeezed."""
        # Test with different batch size dimensions
        batch = {
            "cu_seqlens": torch.tensor([[[0, 6, 12, -1]]], dtype=torch.int32),  # Shape [1, 1, 4]
            "max_seqlen": torch.tensor([[[20]]], dtype=torch.int32),  # Shape [1, 1, 1]
        }

        result = get_packed_seq_params(batch)

        # Verify dimensions were squeezed properly
        expected_cu_seqlens = torch.tensor([0, 6, 12], dtype=torch.int32)
        assert torch.equal(result.cu_seqlens_q, expected_cu_seqlens)

        expected_max_seqlen = torch.tensor(20, dtype=torch.int32)
        assert torch.equal(result.max_seqlen_q, expected_max_seqlen)

    def test_packed_seq_params_with_different_dtypes(self):
        """Test functionality with different tensor dtypes."""
        batch = {
            "cu_seqlens": torch.tensor([[0, 10, 20, -1]], dtype=torch.int64),  # int64 instead of int32
            "max_seqlen": torch.tensor([[25]], dtype=torch.int64),
        }

        result = get_packed_seq_params(batch)

        # Function should handle different dtypes
        expected_cu_seqlens = torch.tensor([0, 10, 20], dtype=torch.int64)
        assert torch.equal(result.cu_seqlens_q, expected_cu_seqlens)

        expected_max_seqlen = torch.tensor(25, dtype=torch.int64)
        assert torch.equal(result.max_seqlen_q, expected_max_seqlen)

    def test_packed_seq_params_all_fields_match(self):
        """Test that cu_seqlens_q/kv and max_seqlen_q/kv are identical."""
        batch = {
            "cu_seqlens": torch.tensor([[0, 5, 11, 18, -1]], dtype=torch.int32),
            "max_seqlen": torch.tensor([[12]], dtype=torch.int32),
        }

        result = get_packed_seq_params(batch)

        # Verify that q and kv parameters are identical (as expected for this function)
        assert torch.equal(result.cu_seqlens_q, result.cu_seqlens_kv)
        assert torch.equal(result.max_seqlen_q, result.max_seqlen_kv)

    def test_packed_seq_params_with_cu_seqlens_unpadded(self):
        """Test functionality with cu_seqlens_unpadded for THD CP support."""
        # Padded cu_seqlens (includes padding for CP divisibility)
        cu_seqlens_padded = torch.tensor([[0, 8, 16, -1, -1]], dtype=torch.int32)
        # Unpadded cu_seqlens (actual sequence boundaries)
        cu_seqlens_unpadded = torch.tensor([[0, 6, 14, -1, -1]], dtype=torch.int32)

        batch = {
            "cu_seqlens": cu_seqlens_padded,
            "cu_seqlens_unpadded": cu_seqlens_unpadded,
            "max_seqlen": torch.tensor([[10]], dtype=torch.int32),
        }

        result = get_packed_seq_params(batch)

        # cu_seqlens_q and cu_seqlens_kv should use unpadded values
        expected_unpadded = torch.tensor([0, 6, 14], dtype=torch.int32)
        assert torch.equal(result.cu_seqlens_q, expected_unpadded)
        assert torch.equal(result.cu_seqlens_kv, expected_unpadded)

        # cu_seqlens_q_padded and cu_seqlens_kv_padded should use padded values
        expected_padded = torch.tensor([0, 8, 16], dtype=torch.int32)
        assert torch.equal(result.cu_seqlens_q_padded, expected_padded)
        assert torch.equal(result.cu_seqlens_kv_padded, expected_padded)

    def test_packed_seq_params_cu_seqlens_unpadded_with_argmin(self):
        """Test cu_seqlens_unpadded processing with argmin hint."""
        batch = {
            "cu_seqlens": torch.tensor([[0, 4, 8, 12, -1, -1]], dtype=torch.int32),
            "cu_seqlens_argmin": torch.tensor(4),  # Index where -1 starts
            "cu_seqlens_unpadded": torch.tensor([[0, 3, 7, 10, -1, -1]], dtype=torch.int32),
            "cu_seqlens_unpadded_argmin": torch.tensor(4),  # Index where -1 starts
        }

        result = get_packed_seq_params(batch)

        # Verify unpadded values are used for q/kv
        expected_unpadded = torch.tensor([0, 3, 7, 10], dtype=torch.int32)
        assert torch.equal(result.cu_seqlens_q, expected_unpadded)
        assert torch.equal(result.cu_seqlens_kv, expected_unpadded)

        # Verify padded values are set for _padded fields
        expected_padded = torch.tensor([0, 4, 8, 12], dtype=torch.int32)
        assert torch.equal(result.cu_seqlens_q_padded, expected_padded)
        assert torch.equal(result.cu_seqlens_kv_padded, expected_padded)

    def test_packed_seq_params_without_unpadded_fallback(self):
        """Test fallback to cu_seqlens when cu_seqlens_unpadded is not provided."""
        batch = {
            "cu_seqlens": torch.tensor([[0, 5, 10, 15, -1]], dtype=torch.int32),
            "max_seqlen": torch.tensor([[8]], dtype=torch.int32),
        }

        result = get_packed_seq_params(batch)

        expected_cu_seqlens = torch.tensor([0, 5, 10, 15], dtype=torch.int32)

        # Without unpadded, q/kv should use padded values
        assert torch.equal(result.cu_seqlens_q, expected_cu_seqlens)
        assert torch.equal(result.cu_seqlens_kv, expected_cu_seqlens)

        # Padded fields should be None when cu_seqlens_unpadded is not provided
        # (to avoid slower TE kernel paths)
        assert result.cu_seqlens_q_padded is None
        assert result.cu_seqlens_kv_padded is None

    def test_packed_seq_params_qkv_format_is_thd(self):
        """Test that qkv_format is always set to 'thd'."""
        batch = {
            "cu_seqlens": torch.tensor([[0, 10, -1]], dtype=torch.int32),
        }

        result = get_packed_seq_params(batch)

        assert result.qkv_format == "thd"


class TestCreateLossFunction:
    """Tests for the _create_loss_function helper function."""

    def test_create_loss_function_both_true(self):
        """Test create_loss_function with both flags as True."""
        loss_mask = torch.tensor([[1.0, 1.0, 0.0]])

        loss_func = _create_loss_function(loss_mask=loss_mask, check_for_nan_in_loss=True, check_for_spiky_loss=True)

        # Verify it returns a partial function
        assert isinstance(loss_func, partial)
        assert loss_func.func.__name__ == "masked_next_token_loss"

        # Verify the partial has correct arguments
        assert torch.equal(loss_func.args[0], loss_mask)
        assert loss_func.keywords["check_for_nan_in_loss"] == True
        assert loss_func.keywords["check_for_spiky_loss"] == True

    def test_create_loss_function_both_false(self):
        """Test _create_loss_function with both flags as False."""
        loss_mask = torch.tensor([[1.0, 0.0, 1.0]])

        loss_func = _create_loss_function(loss_mask=loss_mask, check_for_nan_in_loss=False, check_for_spiky_loss=False)

        # Verify the partial has correct arguments
        assert torch.equal(loss_func.args[0], loss_mask)
        assert loss_func.keywords["check_for_nan_in_loss"] == False
        assert loss_func.keywords["check_for_spiky_loss"] == False

    def test_create_loss_function_mixed_values(self):
        """Test create_loss_function with mixed flag values."""
        loss_mask = torch.tensor([[0.0, 1.0, 1.0]])

        loss_func = _create_loss_function(loss_mask=loss_mask, check_for_nan_in_loss=True, check_for_spiky_loss=False)

        # Verify the partial has correct mixed values
        assert torch.equal(loss_func.args[0], loss_mask)
        assert loss_func.keywords["check_for_nan_in_loss"] == True
        assert loss_func.keywords["check_for_spiky_loss"] == False

    @patch("megatron.bridge.training.losses.masked_next_token_loss")
    def test_create_loss_function_callable(self, mock_loss_func):
        """Test that the created loss function can be called correctly."""
        loss_mask = torch.tensor([[1.0, 1.0, 1.0]])
        output_tensor = torch.tensor([2.5])

        # Mock return value
        expected_result = (torch.tensor(3.0), torch.tensor(2), {"lm loss": torch.tensor([3.0, 2.0])})
        mock_loss_func.return_value = expected_result

        # Create the loss function
        loss_func = _create_loss_function(loss_mask=loss_mask, check_for_nan_in_loss=True, check_for_spiky_loss=False)

        # Call the partial function
        result = loss_func(output_tensor)

        # Verify the underlying function was called correctly
        mock_loss_func.assert_called_once_with(
            loss_mask, output_tensor, check_for_nan_in_loss=True, check_for_spiky_loss=False
        )

        # Verify the result
        assert result == expected_result


class TestCreateLossFunctionModelopt:
    """Tests for the _create_loss_function_modelopt helper function."""

    def test_create_loss_function_modelopt_regular_model(self):
        """Test _create_loss_function_modelopt with a regular (non-DistillationModel) model."""
        loss_mask = torch.tensor([[1.0, 1.0, 0.0]])
        mock_model = Mock()
        mock_unwrapped_model = Mock()

        with patch("megatron.bridge.training.gpt_step.unwrap_model", return_value=mock_unwrapped_model):
            loss_func = _create_loss_function_modelopt(
                loss_mask=loss_mask,
                model=mock_model,
                check_for_nan_in_loss=True,
                check_for_spiky_loss=True,
            )

            # Verify it returns a partial function for masked_next_token_loss (regular loss)
            assert isinstance(loss_func, partial)
            assert loss_func.func.__name__ == "masked_next_token_loss"

            # Verify the partial has correct arguments
            assert torch.equal(loss_func.args[0], loss_mask)
            assert loss_func.keywords["check_for_nan_in_loss"] == True
            assert loss_func.keywords["check_for_spiky_loss"] == True

    def test_create_loss_function_modelopt_distillation_model(self):
        """Test _create_loss_function_modelopt with a DistillationModel."""
        loss_mask = torch.tensor([[1.0, 0.0, 1.0]])
        mock_model = Mock()
        mock_distillation_model = Mock(spec=mtd.DistillationModel)

        with patch("megatron.bridge.training.gpt_step.unwrap_model", return_value=mock_distillation_model):
            loss_func = _create_loss_function_modelopt(
                loss_mask=loss_mask,
                model=mock_model,
                check_for_nan_in_loss=False,
                check_for_spiky_loss=True,
            )

            # Verify it returns a partial function for loss_func_kd (distillation loss)
            assert isinstance(loss_func, partial)
            assert loss_func.func.__name__ == "loss_func_kd"

            # Verify the partial has correct keyword arguments
            assert torch.equal(loss_func.keywords["loss_mask"], loss_mask)
            assert loss_func.keywords["model"] == mock_distillation_model
            assert isinstance(loss_func.keywords["original_loss_fn"], partial)
            # Verify original_loss_fn is correctly configured
            assert loss_func.keywords["original_loss_fn"].func.__name__ == "masked_next_token_loss"
            assert loss_func.keywords["original_loss_fn"].keywords["check_for_nan_in_loss"] == False
            assert loss_func.keywords["original_loss_fn"].keywords["check_for_spiky_loss"] == True

    def test_create_loss_function_modelopt_both_flags_false(self):
        """Test _create_loss_function_modelopt with both flags as False."""
        loss_mask = torch.tensor([[0.0, 1.0, 1.0]])
        mock_model = Mock()
        mock_unwrapped_model = Mock()

        with patch("megatron.bridge.training.gpt_step.unwrap_model", return_value=mock_unwrapped_model):
            loss_func = _create_loss_function_modelopt(
                loss_mask=loss_mask,
                model=mock_model,
                check_for_nan_in_loss=False,
                check_for_spiky_loss=False,
            )

            # Verify the partial has correct arguments
            assert torch.equal(loss_func.args[0], loss_mask)
            assert loss_func.keywords["check_for_nan_in_loss"] == False
            assert loss_func.keywords["check_for_spiky_loss"] == False
