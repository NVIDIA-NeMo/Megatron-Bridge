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
