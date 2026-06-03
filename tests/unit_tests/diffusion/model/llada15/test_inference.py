# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for LLaDA1.5 inference helpers and the block-diffusion loop.

Uses ``MagicMock`` models (no real GPTModel, no checkpoint, no GPU). The mock
attention modules are spec'd against ``LLaDA15TEDotProductAttention`` so the
``isinstance`` filter in ``_iter_llada15_attentions`` passes without
constructing a Transformer Engine module.
"""

from unittest.mock import MagicMock

import pytest
import torch

from megatron.bridge.diffusion.models.llada15.inference_llada15 import (
    _clear_attention_state,
    _iter_llada15_attentions,
    _set_padding_mask,
    _unwrap,
    generate_block_diffusion,
)
from megatron.bridge.diffusion.models.llada15.llada15_attention import LLaDA15TEDotProductAttention


pytestmark = [pytest.mark.unit]


def _make_mock_model(num_layers=2, vocab_size=16):
    """Build a callable mock GPTModel with spec'd LLaDA15 attention layers.

    Calling the model returns random logits ``[B, S, vocab]`` shaped to the
    input, so the block-diffusion loop can run end-to-end on CPU.
    """
    attns = [MagicMock(spec=LLaDA15TEDotProductAttention) for _ in range(num_layers)]
    layers = []
    for a in attns:
        layer = MagicMock()
        layer.self_attention.core_attention = a
        layers.append(layer)

    model = MagicMock()
    # _unwrap stops when there is no .module / .language_model.
    del model.module
    del model.language_model
    model.decoder.layers = layers

    def _forward(input_ids=None, position_ids=None, attention_mask=None):
        b, s = input_ids.shape
        return torch.randn(b, s, vocab_size)

    model.side_effect = _forward
    model.__call__ = _forward
    return model, attns


def _make_planned_model(plan, num_layers=2, vocab_size=16):
    """Build a mock model whose logits are dictated by ``plan(input_ids)``.

    ``plan`` returns ``(tgt, peak)`` where ``tgt[B, S]`` is the argmax token at
    each position and ``peak[B, S]`` is the logit height assigned to it. A higher
    peak yields a higher softmax confidence, so the diffusion sampler unmasks
    that position earlier — this lets a test pin down *which* token lands at
    *which* position and *in what order*, making the block-diffusion loop fully
    deterministic without a real model.
    """
    attns = [MagicMock(spec=LLaDA15TEDotProductAttention) for _ in range(num_layers)]
    layers = []
    for a in attns:
        layer = MagicMock()
        layer.self_attention.core_attention = a
        layers.append(layer)
    model = MagicMock()
    del model.module
    del model.language_model
    model.decoder.layers = layers

    def _forward(input_ids=None, position_ids=None, attention_mask=None):
        tgt, peak = plan(input_ids)
        b, s = input_ids.shape
        logits = torch.zeros(b, s, vocab_size)
        logits.scatter_(2, tgt.unsqueeze(-1), peak.unsqueeze(-1))
        return logits

    model.side_effect = _forward
    model.__call__ = _forward
    return model, attns


class TestUnwrap:
    def test_unwrap_plain_model(self):
        m = MagicMock()
        del m.module
        del m.language_model
        assert _unwrap(m) is m

    def test_unwrap_module_wrapper(self):
        inner = MagicMock()
        del inner.module
        del inner.language_model
        wrapper = MagicMock()
        wrapper.module = inner
        assert _unwrap(wrapper) is inner

    def test_unwrap_language_model_wrapper(self):
        inner = MagicMock()
        del inner.module
        del inner.language_model
        wrapper = MagicMock()
        del wrapper.module
        wrapper.language_model = inner
        assert _unwrap(wrapper) is inner


class TestAttentionHelpers:
    def test_iter_yields_spec_attentions(self):
        model, attns = _make_mock_model(num_layers=3)
        found = list(_iter_llada15_attentions(model))
        assert len(found) == 3
        assert found == attns

    def test_iter_skips_non_llada15_attention(self):
        model, _ = _make_mock_model(num_layers=2)
        # Replace one core_attention with a plain mock (fails isinstance).
        model.decoder.layers[0].self_attention.core_attention = MagicMock()
        found = list(_iter_llada15_attentions(model))
        assert len(found) == 1

    def test_set_padding_mask_broadcasts(self):
        model, attns = _make_mock_model(num_layers=2)
        mask = torch.zeros(1, 5, dtype=torch.bool)
        _set_padding_mask(model, mask)
        for a in attns:
            a.set_padding_mask.assert_called_once_with(mask)

    def test_clear_attention_state_calls_reset(self):
        model, attns = _make_mock_model(num_layers=2)
        _clear_attention_state(model)
        for a in attns:
            a.reset_inference_state.assert_called_once()


class TestGenerateBlockDiffusion:
    def test_output_shape_and_prompt_preserved(self):
        torch.manual_seed(0)
        model, _ = _make_mock_model(num_layers=2, vocab_size=16)
        prompt = torch.tensor([[3, 4, 5]])  # [1, 3]
        out = generate_block_diffusion(
            model,
            prompt,
            gen_length=4,
            block_length=2,
            steps=4,
            mask_token_id=999,  # outside vocab so it never gets re-predicted as itself
        )
        assert out.shape == (1, 3 + 4)
        # Prompt prefix is preserved verbatim.
        assert out[0, :3].tolist() == [3, 4, 5]

    def test_all_masks_filled(self):
        torch.manual_seed(0)
        model, _ = _make_mock_model(num_layers=2, vocab_size=16)
        prompt = torch.tensor([[1, 2]])
        out = generate_block_diffusion(model, prompt, gen_length=4, block_length=2, steps=2, mask_token_id=999)
        # No mask tokens should remain in the generated region.
        assert int((out[:, 2:] == 999).sum()) == 0

    def test_cleanup_called_on_attention(self):
        model, attns = _make_mock_model(num_layers=2, vocab_size=16)
        prompt = torch.tensor([[1, 2]])
        generate_block_diffusion(model, prompt, gen_length=2, block_length=2, steps=2, mask_token_id=999)
        # try/finally must always clear stored mask state.
        for a in attns:
            a.reset_inference_state.assert_called()

    def test_padding_mask_installed_when_pad_present(self):
        model, attns = _make_mock_model(num_layers=2, vocab_size=16)
        # Left position is padding (pad id 0); prompt has a pad token.
        prompt = torch.tensor([[0, 7, 8]])
        generate_block_diffusion(
            model, prompt, gen_length=2, block_length=2, steps=2, mask_token_id=999, pad_token_id=0
        )
        # set_padding_mask should have been called with a non-None mask.
        for a in attns:
            calls = [c for c in a.set_padding_mask.call_args_list if c.args and c.args[0] is not None]
            assert calls, "expected a non-None padding mask to be installed"


# Token ids used by the deterministic early-stop scenarios below.
_EOS = 7
_FILLER = 3
_MASK = 999
_PAD = 0


def _first_eos_idx(region):
    """Index of the first EOS in a list, or ``len(region)`` if none."""
    return region.index(_EOS) if _EOS in region else len(region)


def _no_mask_before_eos(out, prompt_len):
    """The invariant both bugs violate: for every row, the generated region
    before that row's first EOS contains no ``mask_token_id``.

    Trailing mask ids *after* a row's first EOS are expected (the function
    returns full width and the caller trims at the first EOS), so only the
    pre-EOS slice is checked.
    """
    gen = out[:, prompt_len:]
    for r in range(gen.shape[0]):
        region = gen[r].tolist()
        if _MASK in region[: _first_eos_idx(region)]:
            return False
    return True


class TestEosEarlyStop:
    """Regression tests for the batched block-diffusion EOS early-stop fix.

    Each scenario drives the real ``generate_block_diffusion`` with a
    deterministic planned-logits model. Scenarios A and B fail on the pre-fix
    per-step, batch-global ``.any()`` early stop and pass on the block-boundary,
    per-sample ``.any(dim=1).all()`` fix; C and D pin the surrounding contract.
    """

    def test_short_row_does_not_collapse_batch(self):
        """Bug A: a short row that emits EOS in block 0 must not halt the whole
        batch and return a slower row still full of mask ids."""

        def plan(input_ids):
            b, s = input_ids.shape
            tgt = torch.full((b, s), _FILLER, dtype=torch.long)
            tgt[0] = _EOS  # row 0 emits EOS wherever it is unmasked
            peak = torch.full((b, s), 8.0)
            return tgt, peak

        model, _ = _make_planned_model(plan)
        prompt = torch.tensor([[1, 2], [1, 2]])  # B=2, prompt_len=2
        out = generate_block_diffusion(
            model,
            prompt,
            gen_length=4,
            block_length=2,
            steps=4,
            mask_token_id=_MASK,
            eos_token_id=_EOS,
            eos_early_stop=True,
        )
        assert _no_mask_before_eos(out, 2)
        # The long row (never emits EOS) must be fully generated, not truncated.
        assert _MASK not in out[1, 2:].tolist()

    def test_left_padded_mixed_length_batch(self):
        """Bug A on the padded path: mixed-length prompts are left-padded to a
        common width and ``pad_token_id`` is passed (as a real caller would).
        The short row emits EOS in block 0; the early-stop must not collapse the
        batch, the long row must finish, the prompt+pad prefix must be preserved,
        and the key-padding mask must be installed on every attention layer.
        """

        def plan(input_ids):
            b, s = input_ids.shape
            tgt = torch.full((b, s), _FILLER, dtype=torch.long)
            tgt[0] = _EOS  # short row (row 0) emits EOS wherever unmasked
            peak = torch.full((b, s), 8.0)
            return tgt, peak

        model, attns = _make_planned_model(plan)
        # Row 0 prompt = [1, 2] (short), row 1 = [1, 2, 4, 5] (long); left-pad
        # row 0 to width 4 with _PAD so the generation region [4:] is aligned.
        prompt = torch.tensor([[_PAD, _PAD, 1, 2], [1, 2, 4, 5]])  # B=2, prompt_len=4
        out = generate_block_diffusion(
            model,
            prompt,
            gen_length=4,
            block_length=2,
            steps=4,
            mask_token_id=_MASK,
            eos_token_id=_EOS,
            eos_early_stop=True,
            pad_token_id=_PAD,
        )
        assert _no_mask_before_eos(out, 4)
        # Long row (never emits EOS) is fully generated, not returned full of mask.
        assert _MASK not in out[1, 4:].tolist()
        # Prompt + left-padding prefix is preserved verbatim for both rows.
        assert out[0, :4].tolist() == [_PAD, _PAD, 1, 2]
        assert out[1, :4].tolist() == [1, 2, 4, 5]
        # The padded path installs a non-None key-padding mask on every layer.
        for a in attns:
            calls = [c for c in a.set_padding_mask.call_args_list if c.args and c.args[0] is not None]
            assert calls, "expected a non-None padding mask to be installed"

    def test_no_stray_mask_before_eos_single_row(self):
        """Bug B: even at B==1, EOS committed at a later position (higher
        confidence) must not be returned with an earlier position still masked."""

        def plan(input_ids):
            b, s = input_ids.shape
            prompt_len = 1
            tgt = torch.full((b, s), _FILLER, dtype=torch.long)
            peak = torch.full((b, s), 4.0)
            eos_pos = prompt_len + 2  # last gen position is the EOS...
            tgt[0, eos_pos] = _EOS
            peak[0, eos_pos] = 12.0  # ...and has the highest confidence -> unmasked first
            peak[0, prompt_len + 0] = 6.0
            peak[0, prompt_len + 1] = 5.0
            return tgt, peak

        model, _ = _make_planned_model(plan)
        prompt = torch.tensor([[1]])  # B=1, prompt_len=1
        out = generate_block_diffusion(
            model,
            prompt,
            gen_length=3,
            block_length=3,
            steps=3,
            mask_token_id=_MASK,
            eos_token_id=_EOS,
            eos_early_stop=True,
        )
        region = out[0, 1:].tolist()
        assert _no_mask_before_eos(out, 1)
        assert _MASK not in region[: _first_eos_idx(region)]

    def test_early_stop_fires_at_block_boundary(self):
        """Positive control: once every row has emitted EOS in a block, the loop
        stops at that block boundary, leaving later blocks untouched (mask) for
        the caller to trim — and never with a mask before the EOS."""

        def plan(input_ids):
            b, s = input_ids.shape
            return torch.full((b, s), _EOS, dtype=torch.long), torch.full((b, s), 8.0)

        model, _ = _make_planned_model(plan)
        prompt = torch.tensor([[1, 2]])  # B=1
        out = generate_block_diffusion(
            model,
            prompt,
            gen_length=4,
            block_length=2,
            steps=4,
            mask_token_id=_MASK,
            eos_token_id=_EOS,
            eos_early_stop=True,
        )
        region = out[0, 2:].tolist()
        assert _no_mask_before_eos(out, 2)
        assert region[0] == _EOS  # block 0 unmasked
        assert all(t == _MASK for t in region[2:])  # block 1 left untouched after stop

    def test_early_stop_disabled_runs_to_full_length(self):
        """With ``eos_early_stop=False`` (the default), an early EOS must not
        stop generation — every block is filled regardless."""

        def plan(input_ids):
            b, s = input_ids.shape
            return torch.full((b, s), _EOS, dtype=torch.long), torch.full((b, s), 8.0)

        model, _ = _make_planned_model(plan)
        prompt = torch.tensor([[1, 2]])
        out = generate_block_diffusion(
            model,
            prompt,
            gen_length=4,
            block_length=2,
            steps=4,
            mask_token_id=_MASK,
            eos_token_id=_EOS,
            eos_early_stop=False,
        )
        assert _MASK not in out[0, 2:].tolist()  # no mask left anywhere

    def test_per_sample_reduction_semantics(self):
        """The core of the fix, isolated: ``.any()`` halts as soon as one row has
        an EOS (the bug); ``.any(dim=1).all()`` waits for every row."""
        gen_one_row = torch.tensor([[_EOS, _FILLER], [_MASK, _MASK]])
        gen_all_rows = torch.tensor([[_EOS, _FILLER], [_FILLER, _EOS]])
        gen_no_rows = torch.tensor([[_MASK, _FILLER], [_MASK, _MASK]])

        # Old, buggy predicate would stop on the first finished row.
        assert bool((gen_one_row == _EOS).any()) is True
        # Fixed predicate waits until every row is done.
        assert bool((gen_one_row == _EOS).any(dim=1).all()) is False
        assert bool((gen_all_rows == _EOS).any(dim=1).all()) is True
        assert bool((gen_no_rows == _EOS).any(dim=1).all()) is False
