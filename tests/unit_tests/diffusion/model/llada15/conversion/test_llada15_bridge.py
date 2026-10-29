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

"""Unit tests for LLaDA15Bridge: provider mapping and weight-name registry.

Uses a flat ``types.SimpleNamespace`` mock of the OLMo-style ``LLaDAConfig``
(LLaDA1.5 has no ``text_config``). No HuggingFace download, no safetensors, no
GPU — the tests validate config→provider field mapping and the *structure* of
the weight-name registry (not numeric weight conversion, which is covered by
the GPU round-trip script).
"""

import types

import pytest

from megatron.bridge.diffusion.conversion.llada15.llada15_bridge import LLaDA15Bridge
from megatron.bridge.diffusion.models.llada15.llada15_provider import LLaDA15ModelProvider
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping, QKVMapping


pytestmark = [pytest.mark.unit]


def _make_hf_config(
    d_model=4096,
    n_heads=32,
    n_kv_heads=32,
    n_layers=32,
    mlp_hidden_size=12288,
    vocab_size=126464,
    embedding_size=126464,
    rope_theta=500000.0,
    rms_norm_eps=1e-5,
    weight_tying=False,
    attention_layer_norm=False,
):
    """Flat OLMo-style LLaDA config mock (no text_config)."""
    return types.SimpleNamespace(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        n_layers=n_layers,
        mlp_hidden_size=mlp_hidden_size,
        mlp_ratio=4,
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        max_sequence_length=4096,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        weight_tying=weight_tying,
        include_bias=False,
        include_qkv_bias=False,
        attention_layer_norm=attention_layer_norm,
        residual_dropout=0.0,
        attention_dropout=0.0,
    )


class DummyHFPretrained:
    def __init__(self, hf_config):
        self.config = hf_config


class TestProviderBridge:
    def _provider(self, **kw):
        return LLaDA15Bridge().provider_bridge(DummyHFPretrained(_make_hf_config(**kw)))

    def test_returns_llada15_provider(self):
        assert isinstance(self._provider(), LLaDA15ModelProvider)

    def test_num_layers(self):
        assert self._provider(n_layers=16).num_layers == 16

    def test_hidden_size(self):
        assert self._provider(d_model=2048).hidden_size == 2048

    def test_attention_heads(self):
        p = self._provider(n_heads=16, n_kv_heads=16)
        assert p.num_attention_heads == 16
        assert p.num_query_groups == 16

    def test_ffn_hidden_size(self):
        assert self._provider(mlp_hidden_size=8192).ffn_hidden_size == 8192

    def test_vocab_prefers_embedding_size(self):
        # embedding_size (padded vocab) takes precedence over vocab_size.
        assert self._provider(vocab_size=120000, embedding_size=126464).vocab_size == 126464

    def test_rotary_base_from_rope_theta(self):
        assert self._provider(rope_theta=500000.0).rotary_base == 500000.0

    def test_position_embedding_type_is_rope(self):
        # Full RoPE handled by Megatron; must not be "none" or "learned_absolute".
        assert self._provider().position_embedding_type == "rope"

    def test_normalization_is_rmsnorm(self):
        assert self._provider().normalization == "RMSNorm"

    def test_share_embeddings_follows_weight_tying(self):
        assert self._provider(weight_tying=False).share_embeddings_and_output_weights is False
        assert self._provider(weight_tying=True).share_embeddings_and_output_weights is True

    def test_qk_layernorm_follows_config(self):
        assert self._provider(attention_layer_norm=False).qk_layernorm is False
        assert self._provider(attention_layer_norm=True).qk_layernorm is True


class TestMappingRegistry:
    def setup_method(self):
        self.registry = LLaDA15Bridge().mapping_registry()
        self.mappings = list(self.registry)

    def test_registry_not_empty(self):
        assert len(self.mappings) > 0

    def test_megatron_keys_are_bare(self):
        # Targets a bare GPTModel, not a VLM wrapper.
        for m in self.mappings:
            mp = getattr(m, "megatron_param", "")
            assert not mp.startswith("language_model."), f"unexpected prefix: {mp}"

    def test_qkv_mapping_separate_qkv(self):
        qkv = [m for m in self.mappings if isinstance(m, QKVMapping)]
        assert len(qkv) == 1
        assert "linear_qkv" in qkv[0].megatron_param
        hf = qkv[0].hf_param
        assert hf["q"].endswith("q_proj.weight")
        assert hf["k"].endswith("k_proj.weight")
        assert hf["v"].endswith("v_proj.weight")

    def test_gated_mlp_gate_is_ff_proj_up_is_up_proj(self):
        gated = [m for m in self.mappings if isinstance(m, GatedMLPMapping)]
        assert len(gated) == 1
        assert "linear_fc1" in gated[0].megatron_param
        hf = gated[0].hf_param
        # SwiGLU: ff_proj is the gate (gets SiLU), up_proj is the linear up.
        assert hf["gate"].endswith("ff_proj.weight")
        assert hf["up"].endswith("up_proj.weight")

    def test_word_embeddings_mapping(self):
        m = [
            x
            for x in self.mappings
            if isinstance(x, AutoMapping) and getattr(x, "megatron_param", "") == "embedding.word_embeddings.weight"
        ]
        assert len(m) == 1
        assert m[0].hf_param == "model.transformer.wte.weight"

    def test_output_layer_mapping(self):
        m = [
            x
            for x in self.mappings
            if isinstance(x, AutoMapping) and getattr(x, "megatron_param", "") == "output_layer.weight"
        ]
        assert len(m) == 1
        assert m[0].hf_param == "model.transformer.ff_out.weight"

    def test_final_layernorm_mapping(self):
        m = [
            x
            for x in self.mappings
            if isinstance(x, AutoMapping) and getattr(x, "megatron_param", "") == "decoder.final_layernorm.weight"
        ]
        assert len(m) == 1
        assert m[0].hf_param == "model.transformer.ln_f.weight"
