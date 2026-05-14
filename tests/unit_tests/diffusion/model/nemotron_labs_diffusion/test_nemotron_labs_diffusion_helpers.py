# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""CPU-only unit tests for NemotronLabsDiffusion inference helper utilities.

Covers the pure-tensor and pure-Python helpers that don't require GPU or
torch.distributed. The actual generate_ar / generate_dllm functions need
a real GPU model and are exercised in functional tests.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import megatron.bridge.diffusion.models.nemotron_labs_diffusion.inference_nemotron_labs_diffusion as inf


pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# add_gumbel_noise
# ---------------------------------------------------------------------------


class TestAddGumbelNoise:
    def test_temperature_zero_returns_logits_unchanged(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        out = inf.add_gumbel_noise(logits, temperature=0)
        assert torch.equal(out, logits)

    def test_temperature_zero_returns_same_object(self):
        logits = torch.tensor([[1.0, 2.0]])
        out = inf.add_gumbel_noise(logits, temperature=0)
        assert out is logits

    def test_temperature_nonzero_changes_shape_dtype(self):
        torch.manual_seed(0)
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        out = inf.add_gumbel_noise(logits, temperature=1.0)
        assert out.shape == logits.shape
        assert out.dtype == torch.float64

    def test_temperature_nonzero_produces_finite_values(self):
        torch.manual_seed(0)
        logits = torch.tensor([[0.5, 1.0, 1.5, 2.0]])
        out = inf.add_gumbel_noise(logits, temperature=1.0)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# get_num_transfer_tokens
# ---------------------------------------------------------------------------


class TestGetNumTransferTokens:
    def test_exact_division(self):
        mask = torch.tensor([[True, True, True, True]])
        out = inf.get_num_transfer_tokens(mask, steps=4)
        # 4 masks over 4 steps -> 1 per step
        assert out.shape == (1, 4)
        assert torch.equal(out, torch.tensor([[1, 1, 1, 1]], dtype=torch.int64))

    def test_remainder_distributed_to_first_steps(self):
        mask = torch.tensor([[True, True, True, True, True]])  # 5 masks
        out = inf.get_num_transfer_tokens(mask, steps=2)
        # 5/2 = 2 base, 1 remainder added to step 0
        assert torch.equal(out, torch.tensor([[3, 2]], dtype=torch.int64))

    def test_sum_equals_mask_count(self):
        mask = torch.tensor([[True, True, False, True, True, True, False]])  # 5 masks
        out = inf.get_num_transfer_tokens(mask, steps=3)
        assert int(out.sum().item()) == 5

    def test_zero_masks(self):
        mask = torch.zeros(1, 4, dtype=torch.bool)
        out = inf.get_num_transfer_tokens(mask, steps=2)
        assert torch.equal(out, torch.zeros(1, 2, dtype=torch.int64))

    def test_batch_dim_preserved(self):
        mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
        out = inf.get_num_transfer_tokens(mask, steps=2)
        assert out.shape == (2, 2)
        # batch 0: 4 masks / 2 = 2,2; batch 1: 2/2 = 1,1
        assert torch.equal(out, torch.tensor([[2, 2], [1, 1]], dtype=torch.int64))


# ---------------------------------------------------------------------------
# get_transfer_index
# ---------------------------------------------------------------------------


class TestGetTransferIndex:
    def _basic(self):
        # 1 batch, seq_len=4, vocab=3
        logits = torch.tensor([[[5.0, 1.0, 1.0], [1.0, 5.0, 1.0], [1.0, 1.0, 5.0], [2.0, 1.0, 1.0]]])
        mask_index = torch.tensor([[True, True, False, True]])
        x = torch.tensor([[0, 0, 99, 0]])  # 99 is the existing token at unmasked position
        num_transfer_tokens = torch.tensor([[2]])  # transfer 2 tokens
        return logits, mask_index, x, num_transfer_tokens

    def test_returns_two_tensors(self):
        logits, mask_index, x, num_transfer_tokens = self._basic()
        x0, transfer_index = inf.get_transfer_index(
            logits,
            temperature=0,
            remasking="low_confidence",
            mask_index=mask_index,
            x=x,
            num_transfer_tokens=num_transfer_tokens,
        )
        assert x0.shape == x.shape
        assert transfer_index.shape == x.shape
        assert transfer_index.dtype == torch.bool

    def test_x0_preserves_unmasked_positions(self):
        logits, mask_index, x, num_transfer_tokens = self._basic()
        x0, _ = inf.get_transfer_index(
            logits,
            temperature=0,
            remasking="low_confidence",
            mask_index=mask_index,
            x=x,
            num_transfer_tokens=num_transfer_tokens,
        )
        # position 2 is unmasked, should keep x value (99)
        assert x0[0, 2].item() == 99

    def test_x0_predicts_argmax_for_masked_positions(self):
        logits, mask_index, x, num_transfer_tokens = self._basic()
        x0, _ = inf.get_transfer_index(
            logits,
            temperature=0,
            remasking="low_confidence",
            mask_index=mask_index,
            x=x,
            num_transfer_tokens=num_transfer_tokens,
        )
        # logits argmax over vocab for masked positions: pos0->0, pos1->1, pos3->0
        assert x0[0, 0].item() == 0
        assert x0[0, 1].item() == 1
        assert x0[0, 3].item() == 0

    def test_transfer_count_equals_num_transfer_tokens(self):
        logits, mask_index, x, num_transfer_tokens = self._basic()
        _, transfer_index = inf.get_transfer_index(
            logits,
            temperature=0,
            remasking="low_confidence",
            mask_index=mask_index,
            x=x,
            num_transfer_tokens=num_transfer_tokens,
        )
        # Exactly num_transfer_tokens[0] positions transfered
        assert int(transfer_index.sum().item()) == 2

    def test_transfer_only_at_masked_positions(self):
        logits, mask_index, x, num_transfer_tokens = self._basic()
        _, transfer_index = inf.get_transfer_index(
            logits,
            temperature=0,
            remasking="low_confidence",
            mask_index=mask_index,
            x=x,
            num_transfer_tokens=num_transfer_tokens,
        )
        # Unmasked position (idx 2) should never be selected
        assert transfer_index[0, 2].item() is False

    def test_random_remasking_works(self):
        logits, mask_index, x, num_transfer_tokens = self._basic()
        torch.manual_seed(0)
        x0, transfer_index = inf.get_transfer_index(
            logits,
            temperature=0,
            remasking="random",
            mask_index=mask_index,
            x=x,
            num_transfer_tokens=num_transfer_tokens,
        )
        assert int(transfer_index.sum().item()) == 2

    def test_unknown_remasking_raises(self):
        logits, mask_index, x, num_transfer_tokens = self._basic()
        with pytest.raises(NotImplementedError):
            inf.get_transfer_index(
                logits,
                temperature=0,
                remasking="unknown",
                mask_index=mask_index,
                x=x,
                num_transfer_tokens=num_transfer_tokens,
            )

    def test_neg_entropy_path(self):
        logits, mask_index, x, num_transfer_tokens = self._basic()
        x0, transfer_index = inf.get_transfer_index(
            logits,
            temperature=0,
            remasking="low_confidence",
            mask_index=mask_index,
            x=x,
            num_transfer_tokens=num_transfer_tokens,
            neg_entropy=True,
        )
        assert int(transfer_index.sum().item()) == 2


# ---------------------------------------------------------------------------
# _unwrap
# ---------------------------------------------------------------------------


class TestUnwrap:
    def test_returns_plain_model_unchanged(self):
        # A plain object with neither .module nor .language_model
        m = SimpleNamespace()
        assert inf._unwrap(m) is m

    def test_unwraps_module_wrapper(self):
        inner = SimpleNamespace()
        outer = SimpleNamespace(module=inner)
        assert inf._unwrap(outer) is inner

    def test_unwraps_language_model_wrapper(self):
        inner = SimpleNamespace()
        outer = SimpleNamespace(language_model=inner)
        assert inf._unwrap(outer) is inner

    def test_recursive_unwrap(self):
        leaf = SimpleNamespace()
        mid = SimpleNamespace(language_model=leaf)
        root = SimpleNamespace(module=mid)
        assert inf._unwrap(root) is leaf


# ---------------------------------------------------------------------------
# _get_core_attentions
# ---------------------------------------------------------------------------


class TestGetCoreAttentions:
    def test_extracts_one_per_layer(self):
        attn_a = MagicMock(name="attn_a")
        attn_b = MagicMock(name="attn_b")
        layer_a = SimpleNamespace(self_attention=SimpleNamespace(core_attention=attn_a))
        layer_b = SimpleNamespace(self_attention=SimpleNamespace(core_attention=attn_b))
        model = SimpleNamespace(decoder=SimpleNamespace(layers=[layer_a, layer_b]))
        out = inf._get_core_attentions(model)
        assert out == [attn_a, attn_b]

    def test_unwraps_before_extraction(self):
        attn = MagicMock(name="attn")
        layer = SimpleNamespace(self_attention=SimpleNamespace(core_attention=attn))
        inner = SimpleNamespace(decoder=SimpleNamespace(layers=[layer]))
        wrapped = SimpleNamespace(module=inner)
        out = inf._get_core_attentions(wrapped)
        assert out == [attn]


# ---------------------------------------------------------------------------
# set_tp_group + _tp_send_cmd no-op path
# ---------------------------------------------------------------------------


class TestSetTPGroup:
    def teardown_method(self):
        # restore to None to keep test isolation
        inf.set_tp_group(None, src_global_rank=0)

    def test_set_tp_group_stores_globals(self):
        sentinel_group = object()
        inf.set_tp_group(sentinel_group, src_global_rank=7)
        assert inf._TP_GROUP is sentinel_group
        assert inf._TP_SRC_GLOBAL_RANK == 7

    def test_set_tp_group_to_none(self):
        inf.set_tp_group(None, src_global_rank=0)
        assert inf._TP_GROUP is None
        assert inf._TP_SRC_GLOBAL_RANK == 0


class TestTpSendCmdNoop:
    def setup_method(self):
        inf.set_tp_group(None, src_global_rank=0)

    def test_returns_quickly_when_tp_group_is_none(self):
        # If TP_GROUP is None the function should be a no-op and not touch torch.distributed.
        # Calling without a CUDA device must not raise.
        inf._tp_send_cmd(inf._CMD_FORWARD)
        inf._tp_send_cmd(inf._CMD_SET_PARAMS, extra=[1, 0])
        inf._tp_send_cmd(inf._CMD_STOP)


# ---------------------------------------------------------------------------
# get_transfer_index — threshold branch
# ---------------------------------------------------------------------------


class TestGetTransferIndexThreshold:
    """Cover the 'threshold is not None' branch."""

    def _basic(self):
        logits = torch.tensor([[[10.0, 0.0, 0.0], [5.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])
        mask_index = torch.tensor([[True, True, True, True]])
        x = torch.tensor([[0, 0, 0, 0]])
        num_transfer_tokens = torch.tensor([[4]])
        return logits, mask_index, x, num_transfer_tokens

    def test_threshold_demotes_low_confidence_picks(self):
        logits, mask_index, x, num_transfer_tokens = self._basic()
        # neg_entropy=False, so confidence == softmax max prob — keep tokens above threshold
        _, transfer_index = inf.get_transfer_index(
            logits,
            temperature=0,
            remasking="low_confidence",
            mask_index=mask_index,
            x=x,
            num_transfer_tokens=num_transfer_tokens,
            threshold=0.95,
            neg_entropy=False,
        )
        # With strong logit at pos 0 (logit 10), softmax ~ 1.0 -> kept.
        # Other positions much lower -> demoted to False.
        assert transfer_index[0, 0].item() is True
        # At least one of the weaker-confidence ones is demoted
        assert int(transfer_index.sum().item()) < 4

    def test_threshold_keeps_first_top_pick(self):
        logits, mask_index, x, num_transfer_tokens = self._basic()
        # Threshold loop starts at k=1, so index 0 of topk is always preserved
        _, transfer_index = inf.get_transfer_index(
            logits,
            temperature=0,
            remasking="low_confidence",
            mask_index=mask_index,
            x=x,
            num_transfer_tokens=num_transfer_tokens,
            threshold=999.0,  # impossibly high, demotes everything but the first
            neg_entropy=False,
        )
        assert int(transfer_index.sum().item()) == 1


# ---------------------------------------------------------------------------
# _set_inference_mode / _set_inference_params / _clear_kv_cache
# ---------------------------------------------------------------------------


def _model_with_n_attns(n):
    attns = [MagicMock(name=f"attn_{i}") for i in range(n)]
    layers = [SimpleNamespace(self_attention=SimpleNamespace(core_attention=a)) for a in attns]
    return SimpleNamespace(decoder=SimpleNamespace(layers=layers)), attns


class TestSetInferenceMode:
    def setup_method(self):
        inf.set_tp_group(None, src_global_rank=0)

    def test_sets_mode_on_all_attentions(self):
        model, attns = _model_with_n_attns(3)
        inf._set_inference_mode(model, enabled=True)
        for a in attns:
            a.set_inference_mode.assert_called_once_with(True)

    def test_sets_mode_off_on_all_attentions(self):
        model, attns = _model_with_n_attns(2)
        inf._set_inference_mode(model, enabled=False)
        for a in attns:
            a.set_inference_mode.assert_called_once_with(False)


class TestSetInferenceParams:
    def setup_method(self):
        inf.set_tp_group(None, src_global_rank=0)

    def test_forwards_causal_and_cache_enabled(self):
        model, attns = _model_with_n_attns(2)
        inf._set_inference_params(model, causal=True, cache_enabled=False)
        for a in attns:
            a.set_inference_params.assert_called_once_with(True, False)

    def test_forwards_all_false(self):
        model, attns = _model_with_n_attns(2)
        inf._set_inference_params(model, causal=False, cache_enabled=False)
        for a in attns:
            a.set_inference_params.assert_called_once_with(False, False)


class TestClearKvCache:
    def setup_method(self):
        inf.set_tp_group(None, src_global_rank=0)

    def test_calls_clear_on_all_attentions(self):
        model, attns = _model_with_n_attns(4)
        inf._clear_kv_cache(model)
        for a in attns:
            a.clear_kv_cache.assert_called_once_with()


# ---------------------------------------------------------------------------
# _model_forward (no-TP path)
# ---------------------------------------------------------------------------


class TestModelForwardNoTP:
    def setup_method(self):
        inf.set_tp_group(None, src_global_rank=0)

    def test_passes_input_ids_and_position_ids(self):
        captured = {}

        def fake_forward(*, input_ids, position_ids, attention_mask):
            captured["input_ids"] = input_ids
            captured["position_ids"] = position_ids
            captured["attention_mask"] = attention_mask
            return torch.zeros(input_ids.shape[0], input_ids.shape[1], 5)

        model = MagicMock(side_effect=fake_forward)

        input_ids = torch.tensor([[7, 8, 9, 10]])
        out = inf._model_forward(model, input_ids)

        assert torch.equal(captured["input_ids"], input_ids)
        # position_ids = arange(seq_len) broadcast to input shape
        assert torch.equal(captured["position_ids"], torch.tensor([[0, 1, 2, 3]]))
        assert captured["attention_mask"] is None
        assert out.shape == (1, 4, 5)

    def test_unwraps_tuple_output(self):
        # Some Megatron forwards return (logits, *_)
        logits = torch.zeros(1, 2, 5)
        model = MagicMock(return_value=(logits, "aux"))
        out = inf._model_forward(model, torch.tensor([[1, 2]]))
        assert torch.equal(out, logits)


# ---------------------------------------------------------------------------
# generate_ar
# ---------------------------------------------------------------------------


class _StubModel:
    """Callable stub model with explicit .decoder.layers — avoids MagicMock's
    auto-generation of .module / .language_model which would cause _unwrap()
    to infinitely recurse."""

    def __init__(self, forward_fn, n_attns=1):
        self.forward_fn = forward_fn
        self.attns = [MagicMock(name=f"attn_{i}") for i in range(n_attns)]
        self.decoder = SimpleNamespace(
            layers=[SimpleNamespace(self_attention=SimpleNamespace(core_attention=a)) for a in self.attns]
        )

    def __call__(self, **kwargs):
        return self.forward_fn(**kwargs)


class TestGenerateAR:
    def setup_method(self):
        inf.set_tp_group(None, src_global_rank=0)

    def _make_model(self, sequence):
        """Build a stub model whose argmax over the last position of returned logits
        equals the next token from  (one per forward call)."""
        vocab = 5
        call_count = {"n": 0}

        def forward(*, input_ids, position_ids, attention_mask):
            seq_len = input_ids.shape[1]
            logits = torch.full((1, seq_len, vocab), -1e9)
            next_tok = sequence[call_count["n"]]
            call_count["n"] += 1
            logits[0, -1, next_tok] = 1.0
            return logits

        return _StubModel(forward, n_attns=1)

    def test_appends_max_new_tokens(self):
        model = self._make_model([2, 3, 4])  # prefill + 2 decode calls
        prompt = torch.tensor([[0, 1]])
        out = inf.generate_ar(model, prompt, max_new_tokens=2, temperature=0.0)
        assert out.shape == (1, 4)
        # First two are the prompt, then 3 (from second forward) and 4 (third)
        # NOTE: prefill produces 2, but next_token reads the LAST position of the prefill's logits.
        # So the appended tokens come from the prefill's last-pos argmax (=2) and the first decode call's
        # last-pos argmax (=3). The third call (4) isn't appended because we stopped at max_new_tokens=2.
        assert out[0].tolist() == [0, 1, 2, 3]

    def test_stops_on_eos(self):
        # prefill returns 3 (will be appended), first decode returns EOS=4 (will be appended then stop)
        # EOS chosen within vocab=5 range
        model = self._make_model([3, 4, 1])  # the last shouldn't be reached
        prompt = torch.tensor([[0, 0]])
        out = inf.generate_ar(model, prompt, max_new_tokens=10, temperature=0.0, eos_token_id=4)
        # We append 3 then 4 and stop
        assert out[0].tolist() == [0, 0, 3, 4]

    def test_calls_inference_mode_helpers(self):
        def fake_forward(*, input_ids, position_ids, attention_mask):
            return torch.zeros(input_ids.shape[0], input_ids.shape[1], 3)

        model = _StubModel(fake_forward, n_attns=1)
        attns = model.attns

        inf.generate_ar(model, torch.tensor([[1, 2]]), max_new_tokens=1, temperature=0.0)
        # set_inference_mode called twice (once True at start, once False at end)
        assert attns[0].set_inference_mode.call_count == 2
        attns[0].set_inference_mode.assert_any_call(True)
        attns[0].set_inference_mode.assert_any_call(False)
        attns[0].set_inference_params.assert_called_once_with(True, True)
        attns[0].clear_kv_cache.assert_called_once_with()


# ---------------------------------------------------------------------------
# tp_send_stop
# ---------------------------------------------------------------------------


class TestTpSendStop:
    def setup_method(self):
        inf.set_tp_group(None, src_global_rank=0)

    def test_is_noop_when_tp_group_none(self):
        # No exception, no torch.distributed required
        inf.tp_send_stop()
