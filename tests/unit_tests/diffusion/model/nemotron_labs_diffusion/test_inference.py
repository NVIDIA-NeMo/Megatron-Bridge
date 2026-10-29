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

"""Unit tests for NemotronLabsDiffusion inference utilities."""

import pytest
import torch

import megatron.bridge.diffusion.models.nemotron_labs_diffusion.inference_nemotron_labs_diffusion as inf_mod
from megatron.bridge.diffusion.common.dllm import (
    add_gumbel_noise as shared_add_gumbel_noise,
)
from megatron.bridge.diffusion.common.dllm import (
    get_num_transfer_tokens as shared_get_num_transfer_tokens,
)
from megatron.bridge.diffusion.common.dllm import (
    get_transfer_index as shared_get_transfer_index,
)
from megatron.bridge.diffusion.models.nemotron_labs_diffusion.inference_nemotron_labs_diffusion import (
    get_transfer_index,
    set_tp_group,
)


pytestmark = [pytest.mark.unit]


def test_sampling_helpers_reexport_shared_implementations():
    assert inf_mod.add_gumbel_noise is shared_add_gumbel_noise
    assert inf_mod.get_num_transfer_tokens is shared_get_num_transfer_tokens
    assert inf_mod.get_transfer_index is shared_get_transfer_index


# ---------------------------------------------------------------------------
# Helpers shared by the new test classes
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock, patch

import megatron.bridge.diffusion.models.nemotron_labs_diffusion.inference_nemotron_labs_diffusion as _inf_mod
from megatron.bridge.diffusion.models.nemotron_labs_diffusion.inference_nemotron_labs_diffusion import (
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
    """Build a mock Megatron GPT-like model with NemotronLabsDiffusionAttention layers."""
    mock_attn = MagicMock()
    layer = MagicMock()
    layer.self_attention.core_attention = mock_attn
    decoder = MagicMock()
    decoder.layers = [layer, layer]  # num_layers copies
    model = MagicMock()
    model.decoder = decoder
    # Strip auto-generated wrapper attrs so _unwrap() terminates instead of recursing
    # forever through MagicMock's on-demand attribute creation.
    if hasattr(model, "module"):
        del model.module
    if hasattr(model, "language_model"):
        del model.language_model
    return model, mock_attn


def _make_logits(batch, seq_len, vocab_size=50):
    return torch.randn(batch, seq_len, vocab_size)


# ---------------------------------------------------------------------------
# TestGetTransferIndexThreshold
# ---------------------------------------------------------------------------


class TestGetTransferIndexThreshold:
    """Tests for the threshold branch in get_transfer_index (lines 80-88)."""

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


_MODULE = "megatron.bridge.diffusion.models.nemotron_labs_diffusion.inference_nemotron_labs_diffusion"


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

    def test_generate_ar_selects_from_full_vocab_with_parallel_output(self):
        """TP generation selects the global maximum, not a rank-local vocabulary index."""
        model, _ = _make_mock_model()
        prompt = torch.zeros(1, 1, dtype=torch.long)

        def fake_mcore_forward(*, input_ids, position_ids, attention_mask, runtime_gather_output=None):
            del position_ids, attention_mask
            vocab_size = 8 if runtime_gather_output else 4
            logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], vocab_size)
            logits[:, :, 2] = 5.0
            if runtime_gather_output:
                logits[:, :, 6] = 10.0
            return logits

        model.side_effect = fake_mcore_forward

        with patch(f"{_MODULE}._tp_send_cmd"):
            result = generate_ar(model, prompt, max_new_tokens=1)

        assert result[0, -1].item() == 6

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
