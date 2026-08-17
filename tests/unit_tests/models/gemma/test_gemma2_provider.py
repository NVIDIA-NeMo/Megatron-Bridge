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

import pytest
import torch
from megatron.core.activations import fast_gelu

from megatron.bridge.models.gemma.gemma2_provider import (
    Gemma2ModelProvider,
    _get_softcap_score_mod,
    get_swa,
    logit_softcapping,
)


pytestmark = pytest.mark.unit


class TestLogitSoftcapping:
    def test_none_scale_returns_input_unchanged(self):
        logits = torch.tensor([1.0, -2.0, 3.0])

        assert logit_softcapping(logits, None) is logits

    def test_zero_scale_returns_input_unchanged(self):
        logits = torch.tensor([1.0, -2.0, 3.0])

        assert logit_softcapping(logits, 0.0) is logits

    def test_softcapping_applies_scaled_tanh(self):
        logits = torch.tensor([0.0, 10.0, -10.0, 100.0])
        scale = 30.0

        capped = logit_softcapping(logits, scale)

        expected = scale * torch.tanh(logits / scale)
        assert torch.allclose(capped, expected)

    def test_softcapping_bounds_output_by_scale(self):
        logits = torch.linspace(-1e4, 1e4, steps=101)
        scale = 50.0

        capped = logit_softcapping(logits, scale)

        assert capped.abs().max() <= scale


@pytest.mark.skipif(not torch.cuda.is_available(), reason="get_swa builds its mask on the CUDA device")
class TestGetSwa:
    def test_square_causal_window_mask(self):
        window = (2, 0)
        mask = get_swa(4, 4, window)

        assert mask.shape == (4, 4)
        # True marks masked-out positions: outside the sliding window or future tokens.
        allowed = ~mask
        for q in range(4):
            for k in range(4):
                inside_window = q - window[0] <= k <= q + window[1]
                assert allowed[q, k].item() == inside_window

    def test_rectangular_mask_offsets_by_key_length(self):
        mask = get_swa(2, 4, (1, 0))

        allowed = ~mask
        # Query q attends keys aligned to the end of the key sequence.
        offset = 4 - 2
        for q in range(2):
            for k in range(4):
                inside_window = q + offset - 1 <= k <= q + offset
                assert allowed[q, k].item() == inside_window


class TestSoftcapScoreMod:
    def test_score_mod_is_cached_per_softcap(self):
        assert _get_softcap_score_mod(50.0) is _get_softcap_score_mod(50.0)

    def test_distinct_softcaps_get_distinct_mods(self):
        assert _get_softcap_score_mod(50.0) is not _get_softcap_score_mod(30.0)

    def test_zero_softcap_is_identity(self):
        score = torch.tensor(3.0)

        assert _get_softcap_score_mod(0.0)(score, 0, 0, 0, 0) is score

    def test_nonzero_softcap_applies_tanh(self):
        score = torch.tensor(100.0)
        softcap = 50.0

        result = _get_softcap_score_mod(softcap)(score, 0, 0, 0, 0)

        assert torch.allclose(result, softcap * torch.tanh(score / softcap))


class TestGemma2ModelProvider:
    def test_default_configuration(self):
        provider = Gemma2ModelProvider(num_layers=2, hidden_size=64, num_attention_heads=4)

        assert provider.normalization == "RMSNorm"
        assert provider.activation_func is fast_gelu
        assert provider.gated_linear_unit is True
        assert provider.position_embedding_type == "rope"
        assert provider.add_bias_linear is False
        assert provider.share_embeddings_and_output_weights is True
        assert provider.layernorm_zero_centered_gamma is True
        assert provider.layernorm_epsilon == 1e-6
        assert provider.rotary_base == 10000
        assert provider.window_size == (4095, 0)
        assert provider.vocab_size == 256000
        assert provider.kv_channels == 256

    def test_softcapping_defaults(self):
        provider = Gemma2ModelProvider(num_layers=2, hidden_size=64, num_attention_heads=4)

        assert provider.query_pre_attn_scalar == 224
        assert provider.attn_logit_softcapping == 50.0
        assert provider.final_logit_softcapping == 30.0

    def test_query_pre_attn_scalar_override(self):
        provider = Gemma2ModelProvider(num_layers=2, hidden_size=64, num_attention_heads=4, query_pre_attn_scalar=128)

        assert provider.query_pre_attn_scalar == 128
