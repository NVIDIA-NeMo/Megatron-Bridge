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

"""Unit tests for DGPTStep loss helpers and noise application."""

import types
from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.bridge.diffusion.models.common.dgpt_step import (
    DGPTStep,
    _create_loss_function_sbd,
    _masked_loss_sbd_block_diff,
)


pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# _masked_loss_sbd_block_diff
# ---------------------------------------------------------------------------


def _make_rerun_state_machine_mock():
    rsm = MagicMock()
    rsm.validate_result = MagicMock()
    return rsm


class TestMaskedLossSbdBlockDiff:
    """Tests for _masked_loss_sbd_block_diff."""

    def _call(
        self,
        dlm_losses,
        ar_losses,
        num_tokens_ar,
        loss_mask,
        dlm_loss_weight=1.0,
        ar_loss_weight=1.0,
        check_for_nan=False,
        check_for_spiky=False,
        divide_by_masked_tokens=True,
    ):
        output_tensor = (dlm_losses, ar_losses, num_tokens_ar)
        with patch(
            "megatron.bridge.diffusion.models.common.dgpt_step.get_rerun_state_machine",
            return_value=_make_rerun_state_machine_mock(),
        ):
            return _masked_loss_sbd_block_diff(
                loss_mask,
                output_tensor,
                check_for_nan_in_loss=check_for_nan,
                check_for_spiky_loss=check_for_spiky,
                dlm_loss_weight=dlm_loss_weight,
                ar_loss_weight=ar_loss_weight,
                divide_by_masked_tokens=divide_by_masked_tokens,
            )

    def test_returns_three_tuple(self):
        dlm = torch.tensor([1.0, 2.0])
        ar = torch.tensor([0.5, 0.5])
        mask = torch.ones(2, 4, dtype=torch.bool)
        loss, num_tokens, report = self._call(dlm, ar, 4, mask)
        assert isinstance(loss, torch.Tensor)
        assert isinstance(num_tokens, torch.Tensor)
        assert isinstance(report, dict)

    def test_loss_combines_dlm_and_ar(self):
        dlm = torch.tensor([2.0])
        ar = torch.tensor([3.0])
        mask = torch.ones(1, 2, dtype=torch.bool)
        loss, _, _ = self._call(dlm, ar, 2, mask, dlm_loss_weight=1.0, ar_loss_weight=1.0)
        # dlm_loss = sum([2.0]) = 2.0, ar_loss = sum([3.0]) = 3.0, total = 5.0
        assert torch.isclose(loss, torch.tensor(5.0))

    def test_loss_weights_applied(self):
        dlm = torch.tensor([2.0])
        ar = torch.tensor([3.0])
        mask = torch.ones(1, 2, dtype=torch.bool)
        loss, _, _ = self._call(dlm, ar, 2, mask, dlm_loss_weight=0.5, ar_loss_weight=2.0)
        # 0.5*2.0 + 2.0*3.0 = 1.0 + 6.0 = 7.0
        assert torch.isclose(loss, torch.tensor(7.0))

    def test_report_dict_has_expected_keys(self):
        dlm = torch.tensor([1.0])
        ar = torch.tensor([1.0])
        mask = torch.ones(1, 3, dtype=torch.bool)
        _, _, report = self._call(dlm, ar, 3, mask)
        assert "lm loss" in report
        assert "ar loss" in report
        assert "dlm loss" in report
        assert "num_tokens_dlm" in report

    def test_num_tokens_is_sum_of_dlm_and_ar(self):
        dlm = torch.tensor([1.0, 1.0])
        ar = torch.tensor([1.0])
        # loss_mask with 3 True entries -> num_tokens_dlm = 3
        mask = torch.ones(1, 3, dtype=torch.bool)
        _, num_tokens, _ = self._call(dlm, ar, 5, mask)
        # num_tokens_dlm=3, num_tokens_ar=5 => total=8
        assert num_tokens.item() == 8

    def test_zero_dlm_loss(self):
        dlm = torch.tensor([0.0])
        ar = torch.tensor([1.0])
        mask = torch.ones(1, 2, dtype=torch.bool)
        loss, _, _ = self._call(dlm, ar, 2, mask, dlm_loss_weight=1.0, ar_loss_weight=1.0)
        assert torch.isclose(loss, torch.tensor(1.0))

    def test_report_lm_loss_entry_shape(self):
        """lm loss entry should be a 2-element tensor [loss_val, num_tokens]."""
        dlm = torch.tensor([3.0])
        ar = torch.tensor([2.0])
        mask = torch.ones(1, 4, dtype=torch.bool)
        _, _, report = self._call(dlm, ar, 4, mask)
        assert report["lm loss"].shape == (2,)
        assert report["ar loss"].shape == (2,)
        assert report["dlm loss"].shape == (2,)


# ---------------------------------------------------------------------------
# DGPTStep._apply_noise
# ---------------------------------------------------------------------------


class TestDGPTStepApplyNoise:
    """Tests for DGPTStep._apply_noise (noise application and sequence doubling)."""

    def _make_step(self, mask_token_id=100, different_seed_per_dp=False, ar_loss_weight=1.0, dlm_loss_weight=1.0):
        step = DGPTStep(seed=42)
        step.config = types.SimpleNamespace(
            mask_token_id=mask_token_id,
            different_seed_per_dp=different_seed_per_dp,
            ar_loss_weight=ar_loss_weight,
            dlm_loss_weight=dlm_loss_weight,
        )
        step._noise_generator = None
        return step

    def _call_apply_noise(self, step, batch_size=2, seq_len=8):
        tokens = torch.randint(1, 100, (batch_size, seq_len))
        labels = tokens.clone()
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        attention_mask = None
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        return step._apply_noise(tokens, labels, loss_mask, attention_mask, position_ids)

    def test_output_sequence_is_doubled(self):
        """_apply_noise concatenates [noisy | clean], doubling the sequence length."""
        step = self._make_step()
        tokens = torch.randint(1, 100, (2, 8))
        labels = tokens.clone()
        loss_mask = torch.ones(2, 8, dtype=torch.bool)
        position_ids = torch.arange(8).unsqueeze(0).expand(2, -1)

        noisy_inputs, _, _, _, _, masked_indices, p_mask, input_ids_len = step._apply_noise(
            tokens, labels, loss_mask, None, position_ids
        )
        # input_ids_len is the original seq_len
        assert input_ids_len == 8
        # noisy_inputs has doubled length: [xt | x0]
        assert noisy_inputs.shape == (2, 16)

    def test_second_half_equals_original_tokens(self):
        """The second half of noisy_inputs must be the original clean tokens."""
        step = self._make_step()
        tokens = torch.randint(1, 100, (2, 8))
        labels = tokens.clone()
        loss_mask = torch.ones(2, 8, dtype=torch.bool)
        position_ids = torch.arange(8).unsqueeze(0).expand(2, -1)

        noisy_inputs, _, _, _, _, _, _, input_ids_len = step._apply_noise(
            tokens, labels, loss_mask, None, position_ids
        )
        assert torch.equal(noisy_inputs[:, input_ids_len:], tokens)

    def test_masked_indices_boolean_tensor(self):
        """masked_indices must be a boolean tensor of the same shape as original tokens."""
        step = self._make_step(mask_token_id=100)
        tokens = torch.randint(1, 99, (3, 10))
        labels = tokens.clone()
        loss_mask = torch.ones(3, 10, dtype=torch.bool)
        position_ids = torch.arange(10).unsqueeze(0).expand(3, -1)

        _, _, _, _, _, masked_indices, _, _ = step._apply_noise(tokens, labels, loss_mask, None, position_ids)
        assert masked_indices.dtype == torch.bool
        assert masked_indices.shape == (3, 10)

    def test_p_mask_positive_where_masked(self):
        """p_mask values at masked positions should be positive (probabilities)."""
        step = self._make_step(mask_token_id=100)
        tokens = torch.randint(1, 99, (2, 12))
        labels = tokens.clone()
        loss_mask = torch.ones(2, 12, dtype=torch.bool)
        position_ids = torch.arange(12).unsqueeze(0).expand(2, -1)

        _, _, _, _, _, masked_indices, p_mask, _ = step._apply_noise(tokens, labels, loss_mask, None, position_ids)
        if masked_indices.any():
            assert (p_mask[masked_indices] > 0).all()

    def test_mask_token_id_appears_in_noisy_first_half(self):
        """The noisy (first) half must contain the mask_token_id when masking occurs."""
        step = self._make_step(mask_token_id=100)
        # Use tokens that are never 100 to make detection unambiguous
        tokens = torch.randint(1, 99, (4, 16))
        labels = tokens.clone()
        loss_mask = torch.ones(4, 16, dtype=torch.bool)
        position_ids = torch.arange(16).unsqueeze(0).expand(4, -1)

        noisy_inputs, _, _, _, _, masked_indices, _, input_ids_len = step._apply_noise(
            tokens, labels, loss_mask, None, position_ids
        )
        # Noisy positions should match mask_token_id in first half
        noisy_half = noisy_inputs[:, :input_ids_len]
        if masked_indices.any():
            assert (noisy_half[masked_indices] == 100).all()


# ---------------------------------------------------------------------------
# _create_loss_function_sbd
# ---------------------------------------------------------------------------


class TestCreateLossFunctionSbd:
    """Tests for _create_loss_function_sbd partial creation."""

    def test_returns_callable(self):
        mask = torch.ones(2, 4, dtype=torch.bool)
        fn = _create_loss_function_sbd(mask, False, False)
        assert callable(fn)

    def test_partial_encodes_loss_mask(self):
        mask = torch.ones(2, 4, dtype=torch.bool)
        fn = _create_loss_function_sbd(mask, False, False, dlm_loss_weight=0.3, ar_loss_weight=1.0)
        # Calling with dummy tensors (mocked rerun state machine)
        dlm = torch.tensor([1.0])
        ar = torch.tensor([1.0])
        with patch(
            "megatron.bridge.diffusion.models.common.dgpt_step.get_rerun_state_machine",
            return_value=_make_rerun_state_machine_mock(),
        ):
            loss, num_tokens, report = fn((dlm, ar, 4))
        # 0.3*1.0 + 1.0*1.0 = 1.3
        assert torch.isclose(loss, torch.tensor(1.3))


# ---------------------------------------------------------------------------
# Additional imports for new test classes
# ---------------------------------------------------------------------------

from megatron.bridge.diffusion.models.common.dgpt_step import (
    _create_loss_function,
    get_batch,
    get_batch_from_iterator,
)
from megatron.bridge.training.losses import masked_next_token_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tensor_mock():
    t = MagicMock()
    t.cuda.return_value = t
    t.cpu.return_value = t
    return t


# ---------------------------------------------------------------------------
# TestGetBatchFromIterator
# ---------------------------------------------------------------------------


class TestGetBatchFromIterator:
    """Tests for get_batch_from_iterator."""

    def test_basic_batch_first_and_last_stage(self):
        """Both pipeline stages True: tokens, labels, loss_mask, position_ids in required_device_keys."""
        batch = {
            "tokens": _make_tensor_mock(),
            "labels": _make_tensor_mock(),
            "loss_mask": _make_tensor_mock(),
            "position_ids": _make_tensor_mock(),
        }
        with (
            patch(
                "megatron.bridge.diffusion.models.common.dgpt_step.parallel_state.is_pipeline_first_stage",
                return_value=True,
            ),
            patch(
                "megatron.bridge.diffusion.models.common.dgpt_step.parallel_state.is_pipeline_last_stage",
                return_value=True,
            ),
        ):
            result = get_batch_from_iterator(iter([batch]), skip_getting_attention_mask_from_dataset=True)

        for key in ("tokens", "labels", "loss_mask", "position_ids"):
            assert key in result
            assert result[key] is not None

    def test_skip_false_includes_attention_mask(self):
        """skip_getting_attention_mask_from_dataset=False adds attention_mask to required_device_keys."""
        attn_mock = _make_tensor_mock()
        batch = {
            "tokens": _make_tensor_mock(),
            "labels": _make_tensor_mock(),
            "loss_mask": _make_tensor_mock(),
            "position_ids": _make_tensor_mock(),
            "attention_mask": attn_mock,
        }
        with (
            patch(
                "megatron.bridge.diffusion.models.common.dgpt_step.parallel_state.is_pipeline_first_stage",
                return_value=True,
            ),
            patch(
                "megatron.bridge.diffusion.models.common.dgpt_step.parallel_state.is_pipeline_last_stage",
                return_value=True,
            ),
        ):
            result = get_batch_from_iterator(iter([batch]), skip_getting_attention_mask_from_dataset=False)

        assert result.get("attention_mask") is not None

    def test_cu_seqlens_in_batch_adds_host_keys(self):
        """When batch contains cu_seqlens, cu_seqlens_argmin and max_seqlen go through .cpu()."""
        cu_mock = _make_tensor_mock()
        argmin_mock = _make_tensor_mock()
        maxseq_mock = _make_tensor_mock()
        batch = {
            "tokens": _make_tensor_mock(),
            "labels": _make_tensor_mock(),
            "loss_mask": _make_tensor_mock(),
            "position_ids": _make_tensor_mock(),
            "cu_seqlens": cu_mock,
            "cu_seqlens_argmin": argmin_mock,
            "max_seqlen": maxseq_mock,
        }
        with (
            patch(
                "megatron.bridge.diffusion.models.common.dgpt_step.parallel_state.is_pipeline_first_stage",
                return_value=True,
            ),
            patch(
                "megatron.bridge.diffusion.models.common.dgpt_step.parallel_state.is_pipeline_last_stage",
                return_value=True,
            ),
        ):
            result = get_batch_from_iterator(iter([batch]))

        # cu_seqlens goes to device (required_device_keys), argmin/max_seqlen go to host
        assert result["cu_seqlens"] is not None
        argmin_mock.cpu.assert_called()
        maxseq_mock.cpu.assert_called()

    def test_non_required_keys_set_to_none(self):
        """Keys that are neither device nor host keys should be set to None."""
        batch = {
            "tokens": _make_tensor_mock(),
            "labels": _make_tensor_mock(),
            "loss_mask": _make_tensor_mock(),
            "position_ids": _make_tensor_mock(),
            "extra_irrelevant_key": _make_tensor_mock(),
        }
        with (
            patch(
                "megatron.bridge.diffusion.models.common.dgpt_step.parallel_state.is_pipeline_first_stage",
                return_value=True,
            ),
            patch(
                "megatron.bridge.diffusion.models.common.dgpt_step.parallel_state.is_pipeline_last_stage",
                return_value=True,
            ),
        ):
            result = get_batch_from_iterator(iter([batch]))

        assert result["extra_irrelevant_key"] is None

    def test_use_mtp_includes_tokens(self):
        """use_mtp=True ensures tokens/position_ids are in required_device_keys even if not first stage."""
        batch = {
            "tokens": _make_tensor_mock(),
            "labels": _make_tensor_mock(),
            "loss_mask": _make_tensor_mock(),
            "position_ids": _make_tensor_mock(),
        }
        with (
            patch(
                "megatron.bridge.diffusion.models.common.dgpt_step.parallel_state.is_pipeline_first_stage",
                return_value=False,
            ),
            patch(
                "megatron.bridge.diffusion.models.common.dgpt_step.parallel_state.is_pipeline_last_stage",
                return_value=True,
            ),
        ):
            result = get_batch_from_iterator(iter([batch]), use_mtp=True)

        # tokens should be in device keys because use_mtp=True
        assert result.get("tokens") is not None


# ---------------------------------------------------------------------------
# TestGetBatch
# ---------------------------------------------------------------------------


class TestGetBatch:
    """Tests for get_batch."""

    def test_middle_pipeline_stage_returns_nones(self):
        """Middle stage (not first, not last) returns a tuple of 8 Nones."""
        with (
            patch(
                "megatron.bridge.diffusion.models.common.dgpt_step.parallel_state.is_pipeline_first_stage",
                return_value=False,
            ),
            patch(
                "megatron.bridge.diffusion.models.common.dgpt_step.parallel_state.is_pipeline_last_stage",
                return_value=False,
            ),
        ):
            cfg = MagicMock()
            result = get_batch(iter([]), cfg)

        assert result == (None, None, None, None, None, None, None, None)

    def test_first_stage_returns_batch_data(self):
        """First+last stage: get_batch returns the expected tuple from the batch dict."""
        tokens = _make_tensor_mock()
        labels = _make_tensor_mock()
        loss_mask = _make_tensor_mock()
        position_ids = _make_tensor_mock()
        batch_dict = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "attention_mask": None,
            "position_ids": position_ids,
            "cu_seqlens": None,
            "cu_seqlens_argmin": None,
            "max_seqlen": None,
        }
        with (
            patch(
                "megatron.bridge.diffusion.models.common.dgpt_step.parallel_state.is_pipeline_first_stage",
                return_value=True,
            ),
            patch(
                "megatron.bridge.diffusion.models.common.dgpt_step.parallel_state.is_pipeline_last_stage",
                return_value=True,
            ),
            patch(
                "megatron.bridge.diffusion.models.common.dgpt_step.get_batch_from_iterator",
                return_value=batch_dict,
            ),
            patch(
                "megatron.bridge.diffusion.models.common.dgpt_step.get_batch_on_this_cp_rank",
                side_effect=lambda x: x,
            ),
        ):
            cfg = MagicMock()
            cfg.dataset.skip_getting_attention_mask_from_dataset = True
            result = get_batch(iter([]), cfg)

        assert result[0] is tokens
        assert result[1] is labels
        assert result[2] is loss_mask
        assert result[3] is None  # attention_mask
        assert result[4] is position_ids
        assert result[5] is None  # cu_seqlens
        assert result[6] is None  # cu_seqlens_argmin
        assert result[7] is None  # max_seqlen


# ---------------------------------------------------------------------------
# TestMaskedLossSbdBlockDiffNanSpiky
# ---------------------------------------------------------------------------


class TestMaskedLossSbdBlockDiffNanSpiky:
    """Tests for nan/spiky-loss branches in _masked_loss_sbd_block_diff."""

    def test_check_for_nan_calls_validate_result(self):
        """check_for_nan_in_loss=True causes validate_result to be called >= 2 times."""
        dlm = torch.tensor([1.0])
        ar = torch.tensor([1.0])
        mask = torch.ones(1, 2, dtype=torch.bool)
        mock_rsm = MagicMock()
        with patch(
            "megatron.bridge.diffusion.models.common.dgpt_step.get_rerun_state_machine",
            return_value=mock_rsm,
        ):
            _masked_loss_sbd_block_diff(
                mask,
                (dlm, ar, 2),
                check_for_nan_in_loss=True,
                check_for_spiky_loss=False,
            )
        assert mock_rsm.validate_result.call_count >= 2

    def test_check_for_spiky_calls_validate_result(self):
        """check_for_spiky_loss=True causes validate_result to be called >= 1 time."""
        dlm = torch.tensor([1.0])
        ar = torch.tensor([1.0])
        mask = torch.ones(1, 2, dtype=torch.bool)
        mock_rsm = MagicMock()
        with patch(
            "megatron.bridge.diffusion.models.common.dgpt_step.get_rerun_state_machine",
            return_value=mock_rsm,
        ):
            _masked_loss_sbd_block_diff(
                mask,
                (dlm, ar, 2),
                check_for_nan_in_loss=False,
                check_for_spiky_loss=True,
            )
        assert mock_rsm.validate_result.call_count >= 1

    def test_both_checks_enabled(self):
        """Both checks enabled causes validate_result to be called >= 3 times."""
        dlm = torch.tensor([1.0])
        ar = torch.tensor([1.0])
        mask = torch.ones(1, 2, dtype=torch.bool)
        mock_rsm = MagicMock()
        with patch(
            "megatron.bridge.diffusion.models.common.dgpt_step.get_rerun_state_machine",
            return_value=mock_rsm,
        ):
            _masked_loss_sbd_block_diff(
                mask,
                (dlm, ar, 2),
                check_for_nan_in_loss=True,
                check_for_spiky_loss=True,
            )
        assert mock_rsm.validate_result.call_count >= 3

    def test_divide_by_masked_tokens_false_still_computes_loss(self):
        """divide_by_masked_tokens=False runs without error and returns a valid loss."""
        dlm = torch.tensor([2.0])
        ar = torch.tensor([3.0])
        mask = torch.ones(1, 2, dtype=torch.bool)
        mock_rsm = MagicMock()
        with patch(
            "megatron.bridge.diffusion.models.common.dgpt_step.get_rerun_state_machine",
            return_value=mock_rsm,
        ):
            loss, num_tokens, report = _masked_loss_sbd_block_diff(
                mask,
                (dlm, ar, 2),
                check_for_nan_in_loss=False,
                check_for_spiky_loss=False,
                divide_by_masked_tokens=False,
            )
        assert isinstance(loss, torch.Tensor)


# ---------------------------------------------------------------------------
# TestCreateLossFunction
# ---------------------------------------------------------------------------


class TestCreateLossFunction:
    """Tests for _create_loss_function (non-sbd variant)."""

    def test_create_loss_function_returns_callable(self):
        mask = torch.ones(2, 4, dtype=torch.bool)
        fn = _create_loss_function(mask, False, False)
        assert callable(fn)

    def test_create_loss_function_partial_wraps_masked_next_token_loss(self):
        mask = torch.ones(2, 4, dtype=torch.bool)
        fn = _create_loss_function(mask, False, False)
        assert fn.func is masked_next_token_loss


# ---------------------------------------------------------------------------
# TestDGPTStepCall
# ---------------------------------------------------------------------------

import types
from contextlib import ExitStack


_DGPT_MOD = "megatron.bridge.diffusion.models.common.dgpt_step"


def _make_state(check_nan=False, check_spiky=False):
    """Build a minimal GlobalState mock with working timer / straggler_timer."""
    cfg = MagicMock()
    cfg.rerun_state_machine.check_for_nan_in_loss = check_nan
    cfg.rerun_state_machine.check_for_spiky_loss = check_spiky

    # timers("name", ...).start() / .stop() pattern
    timer_instance = MagicMock()
    timer_instance.__enter__ = MagicMock(return_value=None)
    timer_instance.__exit__ = MagicMock(return_value=False)
    timer_mock = MagicMock(return_value=timer_instance)
    timer_mock.__enter__ = MagicMock(return_value=None)
    timer_mock.__exit__ = MagicMock(return_value=False)

    # straggler_timer(bdata=True) -> ctx manager  AND  with straggler_timer: -> ctx manager
    straggler_ctx = MagicMock()
    straggler_ctx.__enter__ = MagicMock(return_value=None)
    straggler_ctx.__exit__ = MagicMock(return_value=False)
    straggler_timer = MagicMock(return_value=straggler_ctx)
    straggler_timer.__enter__ = MagicMock(return_value=None)
    straggler_timer.__exit__ = MagicMock(return_value=False)

    state = MagicMock()
    state.timers = timer_mock
    state.straggler_timer = straggler_timer
    state.cfg = cfg
    return state


def _make_call_context(batch_size=2, seq_len=8, vocab_size=20, different_seed=False):
    """Return a dict of patch kwargs for all Megatron infrastructure needed by DGPTStep.__call__."""
    tokens = torch.randint(1, 90, (batch_size, seq_len))
    labels = tokens.clone()
    loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    # Model returns doubled-sequence logits
    doubled_logits = torch.randn(batch_size, seq_len * 2, vocab_size)
    # per-token loss tensors (indexable with masked_indices of shape [batch, seq])
    per_token_loss = torch.ones(batch_size, seq_len)

    config = types.SimpleNamespace(
        mtp_num_layers=0,
        different_seed_per_dp=different_seed,
        ar_loss_weight=1.0,
        dlm_loss_weight=1.0,
        divide_by_masked_tokens=True,
        mask_token_id=100,
        block_size=4,
        seq_length=seq_len,
        apply_query_key_layer_scaling=False,
        attention_dropout=0.0,
        sequence_parallel=False,
        apply_llama4_style_query_key_layer_scaling=False,
    )

    model = MagicMock()
    model.return_value = doubled_logits
    # language_model doesn't exist as attr (so hasattr check fails gracefully)
    del model.language_model
    model.compute_language_model_loss = MagicMock(return_value=per_token_loss)

    return dict(
        tokens=tokens,
        labels=labels,
        loss_mask=loss_mask,
        position_ids=position_ids,
        doubled_logits=doubled_logits,
        per_token_loss=per_token_loss,
        config=config,
        model=model,
    )


def _run_call(ctx, state=None, first_call=True, different_seed=False):
    """Execute DGPTStep.__call__ with all infrastructure patched."""
    if state is None:
        state = _make_state()

    step = DGPTStep(seed=42)
    step._first_call = first_call

    model = ctx["model"]
    config = ctx["config"]
    config.different_seed_per_dp = different_seed
    tokens = ctx["tokens"]
    labels = ctx["labels"]
    loss_mask = ctx["loss_mask"]
    position_ids = ctx["position_ids"]
    per_token_loss = ctx["per_token_loss"]

    data_iter = iter([])

    with ExitStack() as stack:
        stack.enter_context(patch(f"{_DGPT_MOD}.get_model_config", return_value=config))
        stack.enter_context(
            patch(
                f"{_DGPT_MOD}.get_batch",
                return_value=(tokens, labels, loss_mask, None, position_ids, None, None, None),
            )
        )
        stack.enter_context(patch(f"{_DGPT_MOD}.unwrap_model", side_effect=lambda m: m))
        stack.enter_context(patch(f"{_DGPT_MOD}.get_num_microbatches", return_value=1))
        stack.enter_context(patch(f"{_DGPT_MOD}.get_rerun_state_machine", return_value=MagicMock()))
        mock_empty = stack.enter_context(patch("torch.cuda.empty_cache"))
        stack.enter_context(patch("torch.cuda.is_available", return_value=False))
        if different_seed:
            stack.enter_context(
                patch(
                    f"{_DGPT_MOD}.parallel_state.get_data_parallel_rank",
                    return_value=0,
                )
            )
            real_gen = torch.Generator(device="cpu")
            stack.enter_context(patch("torch.Generator", return_value=real_gen))
        model.compute_language_model_loss.return_value = per_token_loss

        output_tensor, loss_fn = step(state, data_iter, model)

    return step, output_tensor, loss_fn, mock_empty


class TestDGPTStepCall:
    """Tests for DGPTStep.__call__ (lines 129-231)."""

    def test_call_returns_output_tensor_and_loss_fn(self):
        """__call__ returns a 3-tuple output_tensor and a callable loss_function."""
        ctx = _make_call_context()
        _step, output_tensor, loss_fn, _ = _run_call(ctx)
        assert isinstance(output_tensor, tuple)
        assert len(output_tensor) == 3
        assert callable(loss_fn)

    def test_call_first_call_clears_cache(self):
        """When _first_call=True, torch.cuda.empty_cache is called and _first_call becomes False."""
        ctx = _make_call_context()
        step, _ot, _lf, mock_empty = _run_call(ctx, first_call=True)
        mock_empty.assert_called_once()
        assert step._first_call is False

    def test_call_not_first_call_no_cache_clear(self):
        """When _first_call=False, torch.cuda.empty_cache is NOT called."""
        ctx = _make_call_context()
        _step, _ot, _lf, mock_empty = _run_call(ctx, first_call=False)
        mock_empty.assert_not_called()

    def test_call_cu_seqlens_raises(self):
        """If get_batch returns a non-None cu_seqlens, __call__ raises ValueError."""
        ctx = _make_call_context()
        state = _make_state()
        step = DGPTStep(seed=42)

        tokens = ctx["tokens"]
        labels = ctx["labels"]
        loss_mask = ctx["loss_mask"]
        position_ids = ctx["position_ids"]
        config = ctx["config"]
        model = ctx["model"]
        per_token_loss = ctx["per_token_loss"]
        # Return a non-None cu_seqlens
        cu_seqlens_mock = MagicMock()

        with ExitStack() as stack:
            stack.enter_context(patch(f"{_DGPT_MOD}.get_model_config", return_value=config))
            stack.enter_context(
                patch(
                    f"{_DGPT_MOD}.get_batch",
                    return_value=(tokens, labels, loss_mask, None, position_ids, cu_seqlens_mock, None, None),
                )
            )
            stack.enter_context(patch(f"{_DGPT_MOD}.unwrap_model", side_effect=lambda m: m))
            stack.enter_context(patch(f"{_DGPT_MOD}.get_num_microbatches", return_value=1))
            stack.enter_context(patch(f"{_DGPT_MOD}.get_rerun_state_machine", return_value=MagicMock()))
            stack.enter_context(patch("torch.cuda.empty_cache"))
            stack.enter_context(patch("torch.cuda.is_available", return_value=False))
            model.compute_language_model_loss.return_value = per_token_loss

            with pytest.raises(ValueError, match="Packed sequence"):
                step(state, iter([]), model)

    def test_call_microbatch_counter_increments_and_resets(self):
        """After a call with get_num_microbatches=1, _current_microbatch resets to 0."""
        ctx = _make_call_context()
        step, _ot, _lf, _ = _run_call(ctx)
        assert step._current_microbatch == 0

    def test_call_different_seed_per_dp_creates_generator(self):
        """When different_seed_per_dp=True, _noise_generator is set after the call."""
        ctx = _make_call_context(different_seed=True)
        step, _ot, _lf, _ = _run_call(ctx, different_seed=True)
        # Generator was created (mocked) and assigned
        assert step._noise_generator is not None
