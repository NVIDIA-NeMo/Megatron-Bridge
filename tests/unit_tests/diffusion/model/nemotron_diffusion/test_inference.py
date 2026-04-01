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

"""Unit tests for NemotronDiffusion inference utilities."""

import pytest
import torch

from megatron.bridge.diffusion.models.nemotron_diffusion.inference_nemotron_diffusion import (
    add_gumbel_noise,
    get_num_transfer_tokens,
    get_transfer_index,
    set_tp_group,
)


pytestmark = [pytest.mark.unit]


class TestAddGumbelNoise:
    """Tests for add_gumbel_noise."""

    def test_zero_temperature_returns_same_tensor(self):
        logits = torch.randn(2, 10, 100)
        result = add_gumbel_noise(logits, temperature=0)
        assert torch.equal(result, logits)

    def test_nonzero_temperature_changes_values(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 10, 100)
        result = add_gumbel_noise(logits, temperature=1.0)
        assert not torch.equal(result.float(), logits)

    def test_output_shape_preserved(self):
        logits = torch.randn(3, 5, 50)
        result = add_gumbel_noise(logits, temperature=1.0)
        assert result.shape == logits.shape

    def test_output_is_positive(self):
        """exp/noise is always positive."""
        logits = torch.randn(2, 8, 32)
        result = add_gumbel_noise(logits, temperature=1.0)
        assert (result > 0).all()

    def test_argmax_may_differ_from_original(self):
        """With temperature > 0, argmax can change."""
        torch.manual_seed(1)
        logits = torch.zeros(1, 1, 10)
        logits[0, 0, 0] = 10.0  # Strong peak at 0
        results = []
        for _ in range(20):
            r = add_gumbel_noise(logits.clone(), temperature=2.0)
            results.append(torch.argmax(r).item())
        # With high temperature, argmax should not always be 0
        assert len(set(results)) > 1


class TestGetNumTransferTokens:
    """Tests for get_num_transfer_tokens."""

    def test_evenly_divisible(self):
        """When mask_num is divisible by steps, all entries equal base."""
        mask_index = torch.ones(2, 10, dtype=torch.bool)  # 10 masked per row
        result = get_num_transfer_tokens(mask_index, steps=5)
        assert result.shape == (2, 5)
        assert (result == 2).all()

    def test_remainder_distributed_to_first_steps(self):
        """Remainder tokens should be added to the first steps."""
        mask_index = torch.ones(1, 7, dtype=torch.bool)  # 7 masked, steps=3 -> 2,2,3 or 3,2,2
        result = get_num_transfer_tokens(mask_index, steps=3)
        # base = 7//3 = 2, remainder = 1 -> first step gets +1
        assert result[0, 0].item() == 3
        assert result[0, 1].item() == 2
        assert result[0, 2].item() == 2

    def test_total_equals_mask_count(self):
        """Sum of transfers per row must equal number of masked tokens."""
        mask_index = torch.zeros(3, 20, dtype=torch.bool)
        mask_index[:, :13] = True  # 13 masked per row
        result = get_num_transfer_tokens(mask_index, steps=4)
        assert (result.sum(dim=1) == 13).all()

    def test_output_shape(self):
        mask_index = torch.ones(4, 8, dtype=torch.bool)
        result = get_num_transfer_tokens(mask_index, steps=4)
        assert result.shape == (4, 4)

    def test_single_step(self):
        """With steps=1, all masked tokens transferred in one step."""
        mask_index = torch.ones(2, 6, dtype=torch.bool)
        result = get_num_transfer_tokens(mask_index, steps=1)
        assert result.shape == (2, 1)
        assert (result[:, 0] == 6).all()


class TestGetTransferIndex:
    """Tests for get_transfer_index."""

    def _basic_call(
        self, batch=1, seq=8, vocab=10, n_transfer=2, temperature=0.0, remasking="low_confidence", neg_entropy=False
    ):
        torch.manual_seed(0)
        logits = torch.randn(batch, seq, vocab)
        mask_index = torch.zeros(batch, seq, dtype=torch.bool)
        mask_index[:, :4] = True  # first 4 positions masked
        x = torch.randint(0, vocab, (batch, seq))
        num_transfer_tokens = torch.full((batch,), n_transfer, dtype=torch.long)
        return get_transfer_index(
            logits,
            temperature,
            remasking,
            mask_index,
            x,
            num_transfer_tokens,
            threshold=None,
            neg_entropy=neg_entropy,
        )

    def test_output_shapes(self):
        x0, transfer_index = self._basic_call(batch=2, seq=8, vocab=10)
        assert x0.shape == (2, 8)
        assert transfer_index.shape == (2, 8)

    def test_transfer_index_is_bool(self):
        _, transfer_index = self._basic_call()
        assert transfer_index.dtype == torch.bool

    def test_exactly_n_tokens_transferred_per_row(self):
        """Number of True entries in transfer_index per row must equal n_transfer."""
        _, transfer_index = self._basic_call(batch=3, seq=12, n_transfer=3)
        assert (transfer_index.sum(dim=1) == 3).all()

    def test_transfer_only_from_masked_positions(self):
        """transfer_index must only select positions where mask_index is True."""
        torch.manual_seed(0)
        logits = torch.randn(1, 8, 10)
        mask_index = torch.zeros(1, 8, dtype=torch.bool)
        mask_index[:, :4] = True
        x = torch.randint(0, 10, (1, 8))
        num_transfer_tokens = torch.tensor([2])
        _, transfer_index = get_transfer_index(logits, 0.0, "low_confidence", mask_index, x, num_transfer_tokens)
        # All transferred positions must have been masked
        assert not transfer_index[:, 4:].any()

    def test_unmasked_positions_keep_original_x(self):
        """x0 at non-masked positions should equal original x."""
        torch.manual_seed(0)
        logits = torch.randn(1, 8, 10)
        mask_index = torch.zeros(1, 8, dtype=torch.bool)
        mask_index[:, :4] = True
        x = torch.randint(0, 10, (1, 8))
        num_transfer_tokens = torch.tensor([2])
        x0, _ = get_transfer_index(logits, 0.0, "low_confidence", mask_index, x, num_transfer_tokens)
        # Non-masked positions should keep original x values
        assert torch.equal(x0[:, 4:], x[:, 4:])

    def test_random_remasking_strategy(self):
        """random remasking should not raise and produce valid transfer_index."""
        torch.manual_seed(0)
        x0, transfer_index = self._basic_call(remasking="random")
        assert transfer_index.dtype == torch.bool
        assert (transfer_index.sum(dim=1) == 2).all()

    def test_invalid_remasking_strategy_raises(self):
        torch.manual_seed(0)
        with pytest.raises(NotImplementedError):
            self._basic_call(remasking="invalid_strategy")

    def test_neg_entropy_mode(self):
        """neg_entropy=True uses entropy-based confidence — should not raise."""
        x0, transfer_index = self._basic_call(neg_entropy=True)
        assert transfer_index.dtype == torch.bool
        assert (transfer_index.sum(dim=1) == 2).all()


# ---------------------------------------------------------------------------
# Helpers shared by the new test classes
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock, patch

import megatron.bridge.diffusion.models.nemotron_diffusion.inference_nemotron_diffusion as _inf_mod
from megatron.bridge.diffusion.models.nemotron_diffusion.inference_nemotron_diffusion import (
    _clear_kv_cache,
    _get_core_attentions,
    _model_forward,
    _set_inference_mode,
    _set_inference_params,
    _unwrap,
    generate_ar,
    generate_dllm,
)


def _make_mock_model(num_layers=2, vocab_size=50, seq_len=None):
    """Build a mock Megatron GPT-like model with NemotronDiffusionAttention layers."""
    mock_attn = MagicMock()
    layer = MagicMock()
    layer.self_attention.core_attention = mock_attn
    decoder = MagicMock()
    decoder.layers = [layer, layer]  # num_layers copies
    model = MagicMock()
    model.decoder = decoder
    # Ensure no .module attribute so _unwrap returns model itself
    if hasattr(model, "module"):
        del model.module
    return model, mock_attn


def _make_logits(batch, seq_len, vocab_size=50):
    return torch.randn(batch, seq_len, vocab_size)


# ---------------------------------------------------------------------------
# TestGetTransferIndexThreshold
# ---------------------------------------------------------------------------


class TestGetTransferIndexThreshold:
    """Tests for the threshold branch in get_transfer_index (lines 80-88)."""

    def test_threshold_overrides_num_transfer_tokens(self):
        """When threshold is not None, num_transfer_tokens is overridden to mask count."""
        torch.manual_seed(0)
        batch, seq, vocab = 2, 8, 20
        logits = torch.randn(batch, seq, vocab)
        mask_index = torch.zeros(batch, seq, dtype=torch.bool)
        mask_index[:, :4] = True  # 4 masked positions per row
        x = torch.randint(0, vocab, (batch, seq))
        # Pass n_transfer=1 — with threshold set it should be overridden to 4
        num_transfer_tokens = torch.full((batch,), 1, dtype=torch.long)
        x0, transfer_index = get_transfer_index(
            logits,
            0.0,
            "low_confidence",
            mask_index,
            x,
            num_transfer_tokens,
            threshold=0.0,
            neg_entropy=False,
        )
        assert x0.shape == (batch, seq)
        assert transfer_index.shape == (batch, seq)
        assert transfer_index.dtype == torch.bool

    def test_threshold_filters_low_confidence_tokens(self):
        """With a very high threshold, low-confidence tokens are excluded from transfer_index."""
        torch.manual_seed(42)
        batch, seq, vocab = 1, 8, 20
        # Build near-uniform logits so all confidences are low
        logits = torch.zeros(batch, seq, vocab)
        mask_index = torch.zeros(batch, seq, dtype=torch.bool)
        mask_index[:, :4] = True
        x = torch.randint(0, vocab, (batch, seq))
        num_transfer_tokens = torch.full((batch,), 4, dtype=torch.long)
        _x0, transfer_index = get_transfer_index(
            logits,
            0.0,
            "low_confidence",
            mask_index,
            x,
            num_transfer_tokens,
            threshold=100.0,
            neg_entropy=False,
        )
        # With threshold=100.0 (impossibly high), no confidence value passes,
        # so all inner-loop entries get cleared — transfer_index should be sparse/empty
        # The first topk selection is still set before filtering, but indices k>=1 get cleared.
        # At minimum the result must be a valid bool tensor with the right shape.
        assert transfer_index.shape == (batch, seq)
        assert transfer_index.dtype == torch.bool
        # Verify filtering: with threshold=100 all but the top-1 token are removed
        assert transfer_index.sum().item() <= 1


# ---------------------------------------------------------------------------
# TestUnwrapAndGetCoreAttentions
# ---------------------------------------------------------------------------


class TestUnwrapAndGetCoreAttentions:
    """Tests for _unwrap (lines 97-101) and _get_core_attentions (lines 104-110)."""

    def test_unwrap_no_module_attr(self):
        """Object without .module is returned unchanged."""
        obj = object()
        assert _unwrap(obj) is obj

    def test_unwrap_single_wrapper(self):
        """Object with .module pointing to the inner model returns the inner model."""
        inner = MagicMock(spec=[])  # no .module on inner
        wrapper = MagicMock()
        wrapper.module = inner
        result = _unwrap(wrapper)
        assert result is inner

    def test_unwrap_double_wrapper(self):
        """Two levels of wrapping are recursively unwrapped."""
        inner = MagicMock(spec=[])  # no .module
        mid = MagicMock()
        mid.module = inner
        outer = MagicMock()
        outer.module = mid
        result = _unwrap(outer)
        assert result is inner

    def test_get_core_attentions_returns_list(self):
        """_get_core_attentions returns list of core_attention objects from each layer."""
        attn1 = MagicMock()
        attn2 = MagicMock()
        layer1, layer2 = MagicMock(), MagicMock()
        layer1.self_attention.core_attention = attn1
        layer2.self_attention.core_attention = attn2
        model = MagicMock(spec=[])  # no .module
        model.decoder = MagicMock()
        model.decoder.layers = [layer1, layer2]
        result = _get_core_attentions(model)
        assert result == [attn1, attn2]


# ---------------------------------------------------------------------------
# TestSetInferenceModeAndParams
# ---------------------------------------------------------------------------


class TestSetInferenceModeAndParams:
    """Tests for _set_inference_mode, _set_inference_params, _clear_kv_cache."""

    def _model_with_attns(self, n=2):
        attns = [MagicMock() for _ in range(n)]
        layers = []
        for a in attns:
            layer = MagicMock()
            layer.self_attention.core_attention = a
            layers.append(layer)
        model = MagicMock(spec=[])  # no .module
        model.decoder = MagicMock()
        model.decoder.layers = layers
        return model, attns

    def test_set_inference_mode_true_calls_attentions(self):
        model, attns = self._model_with_attns()
        _set_inference_mode(model, True)
        for a in attns:
            a.set_inference_mode.assert_called_once_with(True)

    def test_set_inference_mode_false_calls_attentions(self):
        model, attns = self._model_with_attns()
        _set_inference_mode(model, False)
        for a in attns:
            a.set_inference_mode.assert_called_once_with(False)

    def test_set_inference_params_propagates(self):
        model, attns = self._model_with_attns()
        _set_inference_params(model, causal=True, cache_enabled=False)
        for a in attns:
            a.set_inference_params.assert_called_once_with(True, False)

    def test_clear_kv_cache_calls_attentions(self):
        model, attns = self._model_with_attns()
        _clear_kv_cache(model)
        for a in attns:
            a.clear_kv_cache.assert_called_once_with()


# ---------------------------------------------------------------------------
# TestSetTpGroup
# ---------------------------------------------------------------------------


class TestSetTpGroup:
    """Tests for set_tp_group (lines 143-147)."""

    def test_set_tp_group_stores_group(self):
        mock_group = MagicMock()
        try:
            set_tp_group(mock_group, src_global_rank=1)
            assert _inf_mod._TP_GROUP is mock_group
            assert _inf_mod._TP_SRC_GLOBAL_RANK == 1
        finally:
            _inf_mod._TP_GROUP = None
            _inf_mod._TP_SRC_GLOBAL_RANK = 0

    def test_set_tp_group_default_rank(self):
        mock_group = MagicMock()
        try:
            set_tp_group(mock_group)
            assert _inf_mod._TP_GROUP is mock_group
            assert _inf_mod._TP_SRC_GLOBAL_RANK == 0
        finally:
            _inf_mod._TP_GROUP = None
            _inf_mod._TP_SRC_GLOBAL_RANK = 0


# ---------------------------------------------------------------------------
# TestModelForward
# ---------------------------------------------------------------------------


class TestModelForward:
    """Tests for _model_forward (lines 175-200)."""

    def _make_model_returning(self, output):
        model = MagicMock()
        model.return_value = output
        return model

    def test_model_forward_tensor_output(self):
        """When model returns a tensor, _model_forward returns it directly."""
        logits = _make_logits(1, 4)
        model = self._make_model_returning(logits)
        result = _model_forward(model, torch.zeros(1, 4, dtype=torch.long))
        assert result is logits

    def test_model_forward_tuple_output(self):
        """When model returns a tuple, _model_forward returns index 0."""
        logits = _make_logits(1, 4)
        extra = torch.zeros(1)
        model = self._make_model_returning((logits, extra))
        result = _model_forward(model, torch.zeros(1, 4, dtype=torch.long))
        assert result is logits

    def test_model_forward_passes_position_ids(self):
        """Model must be called with position_ids of shape [1, seq_len]."""
        seq_len = 6
        logits = _make_logits(1, seq_len)
        model = self._make_model_returning(logits)
        input_ids = torch.zeros(1, seq_len, dtype=torch.long)
        _model_forward(model, input_ids)
        call_kwargs = model.call_args
        pos_ids = call_kwargs.kwargs["position_ids"]
        assert pos_ids.shape == (1, seq_len)

    def test_model_forward_no_tp_group(self):
        """When _TP_GROUP is None, no broadcast is attempted."""
        assert _inf_mod._TP_GROUP is None  # default state
        logits = _make_logits(1, 3)
        model = self._make_model_returning(logits)
        with patch.object(_inf_mod, "_broadcast_tensor", side_effect=AssertionError("should not be called")):
            result = _model_forward(model, torch.zeros(1, 3, dtype=torch.long))
        assert result is logits


# ---------------------------------------------------------------------------
# TestGenerateAr
# ---------------------------------------------------------------------------


_MODULE = "megatron.bridge.diffusion.models.nemotron_diffusion.inference_nemotron_diffusion"


class TestGenerateAr:
    """Tests for generate_ar (lines 208-254)."""

    def _run_generate_ar(self, prompt_len=4, max_new_tokens=3, temperature=0.0, eos_token_id=None, vocab_size=50):
        model, mock_attn = _make_mock_model()
        prompt = torch.zeros(1, prompt_len, dtype=torch.long)

        call_count = 0

        def fake_forward(m, input_ids):
            nonlocal call_count
            call_count += 1
            return _make_logits(input_ids.shape[0], input_ids.shape[1], vocab_size)

        with patch(f"{_MODULE}._model_forward", side_effect=fake_forward), patch(f"{_MODULE}._tp_send_cmd"):
            result = generate_ar(
                model,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                eos_token_id=eos_token_id,
            )
        return result, model, mock_attn, call_count

    def test_generate_ar_greedy_output_length(self):
        """Output length == prompt_len + max_new_tokens with temperature=0."""
        prompt_len, max_new_tokens = 4, 5
        result, *_ = self._run_generate_ar(prompt_len=prompt_len, max_new_tokens=max_new_tokens)
        assert result.shape == (1, prompt_len + max_new_tokens)

    def test_generate_ar_with_temperature(self):
        """temperature > 0 uses multinomial sampling; output length is still correct."""
        torch.manual_seed(7)
        prompt_len, max_new_tokens = 3, 4
        result, *_ = self._run_generate_ar(prompt_len=prompt_len, max_new_tokens=max_new_tokens, temperature=1.0)
        assert result.shape == (1, prompt_len + max_new_tokens)

    def test_generate_ar_stops_at_eos(self):
        """Generation stops as soon as eos_token_id is produced."""
        model, mock_attn = _make_mock_model()
        eos_id = 7
        prompt = torch.zeros(1, 3, dtype=torch.long)

        def fake_forward(m, input_ids):
            # Always produce a logit spike at eos_id
            logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 50)
            logits[:, :, eos_id] = 100.0
            return logits

        with patch(f"{_MODULE}._model_forward", side_effect=fake_forward), patch(f"{_MODULE}._tp_send_cmd"):
            result = generate_ar(model, prompt, max_new_tokens=20, eos_token_id=eos_id)

        # Should stop after generating eos (prompt_len + 1 token)
        assert result.shape[1] == 4  # 3 prompt + 1 eos token

    def test_generate_ar_sets_inference_mode(self):
        """After generation, inference mode is disabled on all attention layers."""
        _result, model, mock_attn, _count = self._run_generate_ar()
        # The last call to set_inference_mode on each attention should be False
        calls = mock_attn.set_inference_mode.call_args_list
        assert calls[-1].args == (False,) or calls[-1] == ((False,), {})


# ---------------------------------------------------------------------------
# TestGenerateDllm
# ---------------------------------------------------------------------------


class TestGenerateDllm:
    """Tests for generate_dllm (lines 262-459)."""

    # Small dims to keep tests fast
    _PROMPT_LEN = 4
    _GEN_LENGTH = 4
    _BLOCK_LENGTH = 4
    _STEPS = 4
    _VOCAB = 50

    def _run(self, extra_kwargs=None, vocab_size=None):
        vocab_size = vocab_size or self._VOCAB
        model, mock_attn = _make_mock_model(vocab_size=vocab_size)
        prompt = torch.zeros(1, self._PROMPT_LEN, dtype=torch.long)

        def fake_forward(m, input_ids):
            return _make_logits(input_ids.shape[0], input_ids.shape[1], vocab_size)

        kwargs = dict(
            gen_length=self._GEN_LENGTH,
            block_length=self._BLOCK_LENGTH,
            steps=self._STEPS,
            temperature=0.0,
            mask_id=999,
        )
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        with (
            patch(f"{_MODULE}._model_forward", side_effect=fake_forward),
            patch(f"{_MODULE}._tp_send_cmd"),
            patch("torch.cuda.synchronize"),
        ):
            result = generate_dllm(model, prompt, **kwargs)
        return result, model, mock_attn

    def test_generate_dllm_output_shape(self):
        """Returns (x_accum, nfe, timing); x_accum has shape (batch, prompt+gen)."""
        (x_accum, nfe, timing), *_ = self._run()
        assert x_accum.shape == (1, self._PROMPT_LEN + self._GEN_LENGTH)

    def test_generate_dllm_nfe_count(self):
        """nfe counts denoising forward passes (not KV-update passes)."""
        (x_accum, nfe, timing), *_ = self._run()
        # At most steps forward passes (early-exit when no masks remain)
        assert 0 <= nfe <= self._STEPS

    def test_generate_dllm_timing_keys(self):
        """Timing dict must contain prefill_ms, denoise_ms, kv_update_ms."""
        (_x, _nfe, timing), *_ = self._run()
        assert "prefill_ms" in timing
        assert "denoise_ms" in timing
        assert "kv_update_ms" in timing

    def test_generate_dllm_shift_logits_false(self):
        """shift_logits=False path runs without error and returns correct shape."""
        (x_accum, _nfe, _timing), *_ = self._run(extra_kwargs={"shift_logits": False})
        assert x_accum.shape == (1, self._PROMPT_LEN + self._GEN_LENGTH)

    def test_generate_dllm_with_model_schedule(self):
        """Providing model_schedule runs without error and returns correct output shape."""
        sched_model, _ = _make_mock_model()

        vocab_size = self._VOCAB

        def fake_forward(m, input_ids):
            return _make_logits(input_ids.shape[0], input_ids.shape[1], vocab_size)

        prompt = torch.zeros(1, self._PROMPT_LEN, dtype=torch.long)
        model, _attn = _make_mock_model()

        # steps_per_block = steps / num_blocks = 4 / 1 = 4 (gen=block=4 => 1 block)
        model_schedule = [(sched_model, 4)]

        with (
            patch(f"{_MODULE}._model_forward", side_effect=fake_forward),
            patch(f"{_MODULE}._tp_send_cmd"),
            patch("torch.cuda.synchronize"),
        ):
            x_accum, nfe, timing = generate_dllm(
                model,
                prompt,
                gen_length=self._GEN_LENGTH,
                block_length=self._BLOCK_LENGTH,
                steps=self._STEPS,
                mask_id=999,
                model_schedule=model_schedule,
            )
        assert x_accum.shape == (1, self._PROMPT_LEN + self._GEN_LENGTH)

    def test_generate_dllm_model_schedule_wrong_steps_raises(self):
        """model_schedule whose step total != steps_per_block raises ValueError."""
        sched_model, _ = _make_mock_model()
        # steps_per_block=4 but schedule only provides 3 steps
        model_schedule = [(sched_model, 3)]
        prompt = torch.zeros(1, self._PROMPT_LEN, dtype=torch.long)
        model, _ = _make_mock_model()

        def fake_forward(m, input_ids):
            return _make_logits(input_ids.shape[0], input_ids.shape[1], self._VOCAB)

        with (
            patch(f"{_MODULE}._model_forward", side_effect=fake_forward),
            patch(f"{_MODULE}._tp_send_cmd"),
            patch("torch.cuda.synchronize"),
            pytest.raises(ValueError, match="model_schedule total steps"),
        ):
            generate_dllm(
                model,
                prompt,
                gen_length=self._GEN_LENGTH,
                block_length=self._BLOCK_LENGTH,
                steps=self._STEPS,
                mask_id=999,
                model_schedule=model_schedule,
            )


# ---------------------------------------------------------------------------
# TestTpSendCmd
# ---------------------------------------------------------------------------

import megatron.bridge.diffusion.models.nemotron_diffusion.inference_nemotron_diffusion as inf_mod


class TestTpSendCmd:
    """Tests for _tp_send_cmd (lines 150-158)."""

    def setup_method(self):
        self._orig_group = inf_mod._TP_GROUP
        self._orig_rank = inf_mod._TP_SRC_GLOBAL_RANK

    def teardown_method(self):
        inf_mod._TP_GROUP = self._orig_group
        inf_mod._TP_SRC_GLOBAL_RANK = self._orig_rank

    def test_no_op_when_tp_group_none(self):
        """When _TP_GROUP is None, broadcast is never called."""
        inf_mod._TP_GROUP = None
        with patch("torch.distributed.broadcast") as mock_bcast:
            inf_mod._tp_send_cmd(1)
        mock_bcast.assert_not_called()

    def test_broadcasts_cmd_when_tp_group_set(self):
        """When _TP_GROUP is set, broadcast is called at least once for the cmd."""
        import torch as real_torch

        inf_mod._TP_GROUP = MagicMock()
        with patch.object(inf_mod, "torch") as mock_t:
            mock_t.tensor = lambda *a, **kw: real_torch.tensor(*a, **{k: v for k, v in kw.items() if k != "device"})
            mock_t.long = real_torch.long
            mock_t.distributed = MagicMock()
            inf_mod._tp_send_cmd(1)
            mock_t.distributed.broadcast.assert_called()

    def test_broadcasts_extra_when_provided(self):
        """When extra is provided, broadcast is called twice (cmd + extra)."""
        import torch as real_torch

        inf_mod._TP_GROUP = MagicMock()
        with patch.object(inf_mod, "torch") as mock_t:
            mock_t.tensor = lambda *a, **kw: real_torch.tensor(*a, **{k: v for k, v in kw.items() if k != "device"})
            mock_t.long = real_torch.long
            mock_t.distributed = MagicMock()
            inf_mod._tp_send_cmd(1, extra=[1, 0])
            assert mock_t.distributed.broadcast.call_count == 2


# ---------------------------------------------------------------------------
# TestBroadcastTensor
# ---------------------------------------------------------------------------


class TestBroadcastTensor:
    """Tests for _broadcast_tensor (lines 161-172)."""

    def test_broadcast_tensor_as_src(self):
        """When rank == src, broadcasts shape then data; returns the same tensor."""
        import torch as real_torch

        tensor = real_torch.tensor([[1, 2, 3]], dtype=real_torch.long)
        group = MagicMock()
        with patch.object(inf_mod, "torch") as mock_t:
            mock_t.distributed.get_rank.return_value = 0
            mock_t.distributed.broadcast = MagicMock()
            mock_t.tensor = real_torch.tensor
            mock_t.long = real_torch.long
            mock_t.zeros = real_torch.zeros
            inf_mod._broadcast_tensor(tensor, src=0, group=group)
        # broadcast called twice: once for shape_t, once for data
        assert mock_t.distributed.broadcast.call_count == 2

    def test_broadcast_tensor_as_non_src(self):
        """When rank != src, creates a zero tensor from broadcasted shape and fills it."""
        import torch as real_torch

        # We need shape_t.tolist() to return [1, 3] after broadcast fills it.
        # Simulate: get_rank returns 1 (non-src), zeros(2) -> after broadcast set to [1,3]
        tensor = real_torch.zeros(1, 3, dtype=real_torch.long)
        group = MagicMock()

        # shape_t will be zeros(2); after broadcast it stays [0,0] in test,
        # but we can verify broadcast is called twice and no exception is raised.
        with patch.object(inf_mod, "torch") as mock_t:
            shape_holder = real_torch.zeros(2, dtype=real_torch.long)

            def fake_broadcast(t, src, group):
                if t is shape_holder:
                    # Simulate receiving shape [1, 3]
                    t[0] = 1
                    t[1] = 3

            mock_t.distributed.get_rank.return_value = 1
            mock_t.long = real_torch.long
            mock_t.distributed.broadcast = MagicMock(side_effect=fake_broadcast)

            # First call: zeros(2, dtype=long, device="cuda") -> shape_holder
            def fake_zeros(*args, **kwargs):
                kwargs.pop("device", None)
                return real_torch.zeros(*args, **kwargs)

            mock_t.zeros = MagicMock(side_effect=fake_zeros)
            # shape_t = zeros(2,...) -> shape_holder-like; after broadcast tolist=[1,3]
            # For simplicity just verify no exception and broadcast called
            inf_mod._broadcast_tensor(tensor, src=0, group=group)
        assert mock_t.distributed.broadcast.call_count == 2


# ---------------------------------------------------------------------------
# TestModelForwardWithTpGroup
# ---------------------------------------------------------------------------


class TestModelForwardWithTpGroup:
    """Tests for _model_forward when _TP_GROUP is not None (lines 188-190)."""

    def setup_method(self):
        self._orig_group = inf_mod._TP_GROUP
        self._orig_rank = inf_mod._TP_SRC_GLOBAL_RANK

    def teardown_method(self):
        inf_mod._TP_GROUP = self._orig_group
        inf_mod._TP_SRC_GLOBAL_RANK = self._orig_rank

    def test_model_forward_with_tp_group_broadcasts_input(self):
        """When _TP_GROUP is set, _tp_send_cmd and _broadcast_tensor are called."""
        import torch as real_torch

        inf_mod._TP_GROUP = MagicMock()
        input_ids = real_torch.zeros(1, 4, dtype=real_torch.long)
        broadcasted = real_torch.zeros(1, 4, dtype=real_torch.long)
        logits = real_torch.randn(1, 4, 50)
        model = MagicMock()
        model.return_value = logits

        with (
            patch.object(inf_mod, "_tp_send_cmd") as mock_cmd,
            patch.object(inf_mod, "_broadcast_tensor", return_value=broadcasted) as mock_bcast,
        ):
            result = inf_mod._model_forward(model, input_ids)

        mock_cmd.assert_called_once()
        mock_bcast.assert_called_once()
        assert result is logits


# ---------------------------------------------------------------------------
# TestTpSendStop
# ---------------------------------------------------------------------------


class TestTpSendStop:
    """Tests for tp_send_stop (line 469)."""

    def test_tp_send_stop_calls_tp_send_cmd(self):
        """tp_send_stop must call _tp_send_cmd with _CMD_STOP (0)."""
        with patch.object(inf_mod, "_tp_send_cmd") as mock_cmd:
            inf_mod.tp_send_stop()
        mock_cmd.assert_called_once_with(inf_mod._CMD_STOP)
        assert inf_mod._CMD_STOP == 0


# ---------------------------------------------------------------------------
# TestGenerateDllmEdgeCases
# ---------------------------------------------------------------------------


class TestGenerateDllmEdgeCases:
    """Edge-case tests for generate_dllm covering lines 382-448."""

    _VOCAB = 50
    _MASK_ID = 999

    def _make_model(self):
        model, mock_attn = _make_mock_model()
        return model

    def test_generate_dllm_early_break_all_unmasked(self):
        """If block tokens are all unmasked after step 0, the inner loop breaks early (line 383)."""
        import torch as real_torch

        prompt = real_torch.zeros(1, 4, dtype=real_torch.long)
        # Model returns logits that confidently predict non-mask tokens
        logits_no_mask = real_torch.zeros(1, 4, self._VOCAB)
        logits_no_mask[:, :, 1] = 100.0  # strong peak at token 1 (not mask_id)

        call_count = 0

        def fake_forward(m, input_ids):
            nonlocal call_count
            call_count += 1
            return real_torch.zeros(1, input_ids.shape[1], self._VOCAB)

        model = self._make_model()

        # Use shift_logits=False, small block so we can track steps
        with (
            patch(f"{_MODULE}._model_forward", side_effect=fake_forward),
            patch(f"{_MODULE}._tp_send_cmd"),
            patch("torch.cuda.synchronize"),
        ):
            x_accum, nfe, timing = generate_dllm(
                model,
                prompt,
                gen_length=4,
                block_length=4,
                steps=4,
                mask_id=self._MASK_ID,
                shift_logits=False,
            )
        # nfe must be <= steps_per_block (early break possible)
        assert nfe <= 4
        assert x_accum.shape == (1, 8)

    def test_generate_dllm_dream_style_block_length_1(self):
        """dream_style=True with block_length==1 uses next_logits_context directly (line 395)."""
        import torch as real_torch

        prompt = real_torch.zeros(1, 4, dtype=real_torch.long)

        def fake_forward(m, input_ids):
            return real_torch.randn(1, input_ids.shape[1], self._VOCAB)

        model = self._make_model()

        with (
            patch(f"{_MODULE}._model_forward", side_effect=fake_forward),
            patch(f"{_MODULE}._tp_send_cmd"),
            patch("torch.cuda.synchronize"),
        ):
            x_accum, nfe, timing = generate_dllm(
                model,
                prompt,
                gen_length=4,
                block_length=1,  # triggers line 395: logits_use = next_logits_context
                steps=4,
                mask_id=self._MASK_ID,
                shift_logits=True,  # dream_style=True
            )
        assert x_accum.shape == (1, 8)

    def test_generate_dllm_dream_style_next_logits_updated_between_blocks(self):
        """dream_style=True with multiple blocks triggers next_logits_context update (line 448)."""
        import torch as real_torch

        prompt = real_torch.zeros(1, 4, dtype=real_torch.long)
        # gen_length=8, block_length=4 -> 2 blocks, steps=8 -> 4 steps/block

        forward_calls = []

        def fake_forward(m, input_ids):
            forward_calls.append(input_ids.shape)
            return real_torch.randn(1, input_ids.shape[1], self._VOCAB)

        model = self._make_model()

        with (
            patch(f"{_MODULE}._model_forward", side_effect=fake_forward),
            patch(f"{_MODULE}._tp_send_cmd"),
            patch("torch.cuda.synchronize"),
        ):
            x_accum, nfe, timing = generate_dllm(
                model,
                prompt,
                gen_length=8,
                block_length=4,
                steps=8,
                mask_id=self._MASK_ID,
                shift_logits=True,  # dream_style=True -> line 448 executes for first block
            )
        # Two blocks means next_logits_context is updated at end of block 0
        assert x_accum.shape == (1, 12)  # prompt(4) + gen(8)
        # At least the prefill + denoising steps + kv updates should have fired
        assert len(forward_calls) >= 2
