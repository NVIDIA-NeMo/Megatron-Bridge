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

"""CPU-only unit tests for Gemma 4 modeling helpers."""

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.gemma.modeling_gemma4 import (
    Gemma4DenseRotaryEmbedding,
    Gemma4DenseSelfAttention,
    Gemma4DenseTransformerLayer,
    Gemma4MoEExperts,
    Gemma4MoERouter,
    Gemma4OutputLayer,
    Gemma4RMSNorm,
    Gemma4RotaryEmbedding,
    _compute_per_layer_inputs,
    _gemma4_layer_input,
    _is_gemma4_sliding_layer,
    _logit_softcapping,
    get_gemma4_layer_spec,
)


def _config(**kwargs):
    defaults = {
        "hidden_size": 4,
        "num_experts": 3,
        "top_k_experts": 2,
        "layernorm_epsilon": 1e-6,
        "moe_intermediate_size": 3,
        "window_size": (511, 0),
        "window_attn_skip_freq": ["sliding_attention", "full_attention"],
        "kv_channels": 8,
        "global_kv_channels": 8,
        "rotary_interleaved": False,
        "sliding_window_rope_base": 10_000.0,
        "full_attention_rope_base": 1_000_000.0,
        "full_attention_rope_partial_factor": 0.5,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestGemma4RMSNorm:
    def test_matches_hf_style_rms_norm(self):
        norm = Gemma4RMSNorm(_config(), hidden_size=2, eps=1e-6)
        with torch.no_grad():
            norm.weight.copy_(torch.tensor([2.0, 3.0]))
        hidden_states = torch.tensor([[[3.0, 4.0]]], dtype=torch.float32)

        out = norm(hidden_states)

        expected = hidden_states * torch.pow(hidden_states.pow(2).mean(-1, keepdim=True) + 1e-6, -0.5)
        expected = expected * torch.tensor([2.0, 3.0])
        torch.testing.assert_close(out, expected)

    def test_without_scale_has_no_weight(self):
        norm = Gemma4RMSNorm(_config(), hidden_size=2, with_scale=False)

        assert not hasattr(norm, "weight")


class TestGemma4MoE:
    def test_router_returns_normalized_topk_weights(self):
        router = Gemma4MoERouter(_config(hidden_size=4, num_experts=3, top_k_experts=2))
        with torch.no_grad():
            router.proj.weight.zero_()
            router.per_expert_scale.fill_(1.0)
        hidden_states = torch.ones(5, 4)

        router_probs, top_k_weights, top_k_index = router(hidden_states)

        assert router_probs.shape == (5, 3)
        assert top_k_weights.shape == (5, 2)
        assert top_k_index.shape == (5, 2)
        torch.testing.assert_close(top_k_weights.sum(dim=-1), torch.ones(5))

    def test_experts_return_hidden_shape(self):
        experts = Gemma4MoEExperts(_config(hidden_size=4, num_experts=2, moe_intermediate_size=3))
        hidden_states = torch.ones(2, 4)
        top_k_index = torch.tensor([[0], [1]])
        top_k_weights = torch.ones(2, 1)

        out = experts(hidden_states, top_k_index, top_k_weights)

        assert out.shape == hidden_states.shape


class TestGemma4LayerSpec:
    @pytest.mark.parametrize(
        ("skip_freq", "layer_number", "expected"),
        [
            (["sliding_attention", "full_attention"], 1, True),
            (["sliding_attention", "full_attention"], 2, False),
            ([1, 0], 1, True),
            ([1, 0], 2, False),
        ],
    )
    def test_is_gemma4_sliding_layer_from_list(self, skip_freq, layer_number, expected):
        cfg = _config(window_attn_skip_freq=skip_freq)

        assert _is_gemma4_sliding_layer(cfg, layer_number) is expected

    def test_is_gemma4_sliding_layer_returns_false_without_window(self):
        cfg = _config(window_size=None)

        assert _is_gemma4_sliding_layer(cfg, 1) is False

    def test_get_gemma4_layer_spec_uses_dense_components(self):
        layer_spec = get_gemma4_layer_spec()

        assert layer_spec.module is Gemma4DenseTransformerLayer
        assert layer_spec.submodules.self_attention.module is Gemma4DenseSelfAttention
        assert layer_spec.submodules.post_self_attn_layernorm is Gemma4RMSNorm
        assert layer_spec.submodules.post_mlp_layernorm is Gemma4RMSNorm


class TestGemma4RotaryEmbeddings:
    def test_dense_rotary_uses_full_attention_partial_factor(self):
        rotary = Gemma4DenseRotaryEmbedding(_config(), use_cpu_initialization=True)

        assert rotary.rope_full.inv_freq.numel() == 4
        torch.testing.assert_close(rotary.rope_full.inv_freq[-2:], torch.zeros(2))

    def test_moe_rotary_builds_local_and_global_embeddings(self):
        rotary = Gemma4RotaryEmbedding(
            kv_channels=8,
            rotary_percent=1.0,
            rotary_base=1_000_000,
            rotary_base_local=10_000,
            global_kv_channels=16,
            global_rotary_percent=0.25,
            use_cpu_initialization=True,
        )

        assert rotary.inv_freq.numel() == 2
        assert rotary.rope_local.inv_freq.numel() == 4


class TestGemma4PLEHelpers:
    def test_compute_per_layer_inputs_combines_token_and_model_projections(self):
        class FakeEmbedding(torch.nn.Module):
            def forward(self, input_ids):
                batch, seq = input_ids.shape
                return torch.ones(batch, seq, 6)

        class FakeProjection(torch.nn.Module):
            def forward(self, hidden_states):
                batch, seq, _ = hidden_states.shape
                return torch.full((batch, seq, 6), 4.0), None

        model = SimpleNamespace(
            config=SimpleNamespace(
                per_layer_embed_dim=3,
                num_layers=2,
                hidden_size=4,
                sequence_parallel=False,
            ),
            per_layer_embedding=FakeEmbedding(),
            per_layer_model_proj=FakeProjection(),
            per_layer_proj_norm=torch.nn.Identity(),
        )
        input_ids = torch.ones(2, 3, dtype=torch.long)
        decoder_input = torch.zeros(3, 2, 4)

        out = _compute_per_layer_inputs(model, input_ids, decoder_input)

        assert out.shape == (2, 3, 2, 3)
        expected_value = (4.0 * (4**-0.5) + (3**0.5)) * (2.0**-0.5)
        torch.testing.assert_close(out, torch.full_like(out, expected_value))

    def test_compute_per_layer_inputs_returns_none_without_modules(self):
        model = SimpleNamespace(per_layer_embedding=None)

        assert _compute_per_layer_inputs(model, torch.ones(1, 2, dtype=torch.long), torch.ones(2, 1, 4)) is None

    def test_gemma4_layer_input_selects_global_layer(self):
        per_layer_inputs = torch.arange(1 * 2 * 3 * 4, dtype=torch.float32).view(1, 2, 3, 4)
        layer = SimpleNamespace(layer_number=2)

        out = _gemma4_layer_input(per_layer_inputs, layer)

        torch.testing.assert_close(out, per_layer_inputs[:, :, 1, :].transpose(0, 1))


class TestGemma4OutputHelpers:
    def test_logit_softcapping_applies_tanh_scale(self):
        logits = torch.tensor([-4.0, 0.0, 4.0])

        out = _logit_softcapping(logits, 2.0)

        torch.testing.assert_close(out, 2.0 * torch.tanh(logits / 2.0))

    def test_logit_softcapping_returns_input_without_scale(self):
        logits = torch.tensor([1.0])

        assert _logit_softcapping(logits, None) is logits

    def test_output_layer_applies_final_softcap(self):
        class BaseOutput(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(final_logit_softcapping=2.0)

            def forward(self, x):
                return x, None

        class OutputLayer(Gemma4OutputLayer, BaseOutput):
            pass

        layer = OutputLayer()
        logits = torch.tensor([[-4.0, 0.0, 4.0]])

        out, bias = layer(logits)

        torch.testing.assert_close(out, 2.0 * torch.tanh(logits / 2.0))
        assert bias is None
