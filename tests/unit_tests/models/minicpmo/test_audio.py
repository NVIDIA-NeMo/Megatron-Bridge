# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperAttention

from megatron.bridge.models.minicpmo.audio import MiniCPMOAudioAttention


def test_attention_preserves_state_outputs_and_gradients():
    config = WhisperConfig(d_model=16, encoder_attention_heads=2, attention_dropout=0.0)
    config._attn_implementation = "eager"
    kwargs = dict(embed_dim=16, num_heads=2, dropout=0.0, config=config, layer_idx=0)
    reference = WhisperAttention(**kwargs)
    actual = MiniCPMOAudioAttention(**kwargs)
    actual.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(2, 5, 16, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    mask = torch.zeros(2, 1, 5, 5)
    mask[:, :, :, -1] = torch.finfo(mask.dtype).min
    expected, expected_weights = reference(x, attention_mask=mask, output_attentions=True)
    observed, observed_weights, cache = actual(y, attention_mask=mask, output_attentions=True)
    assert cache is None
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    torch.testing.assert_close(observed_weights, expected_weights, rtol=0, atol=0)
    expected.square().sum().backward()
    observed.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
    for (name, p), (other_name, q) in zip(reference.named_parameters(), actual.named_parameters(), strict=True):
        assert name == other_name
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


@pytest.mark.parametrize("argument", ["past_key_value", "past_key_values", "layer_head_mask"])
def test_unsupported_state_is_rejected(argument):
    config = WhisperConfig(d_model=16, encoder_attention_heads=2)
    layer = MiniCPMOAudioAttention(embed_dim=16, num_heads=2, config=config)
    with pytest.raises(NotImplementedError):
        layer(torch.randn(1, 3, 16), **{argument: torch.ones(1)})
