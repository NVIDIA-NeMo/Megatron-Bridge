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

"""Unit tests for NemotronDiffusionBridge mapping registry and provider bridge."""

import types

import pytest

from megatron.bridge.diffusion.conversion.nemotron_diffusion.nemotron_diffusion_bridge import NemotronDiffusionBridge


pytestmark = [pytest.mark.unit]


def _make_hf_config(
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=8,
    tie_word_embeddings=False,
    rope_theta=10000.0,
    vocab_size=32000,
):
    text_cfg = types.SimpleNamespace(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        tie_word_embeddings=tie_word_embeddings,
        rope_parameters={"rope_theta": rope_theta},
        vocab_size=vocab_size,
    )
    hf_cfg = types.SimpleNamespace(text_config=text_cfg)
    return hf_cfg


class DummyHFPretrained:
    def __init__(self, hf_config):
        self.config = hf_config


class TestNemotronDiffusionBridgeMappingRegistry:
    """Tests for NemotronDiffusionBridge.mapping_registry()."""

    def setup_method(self):
        self.bridge = NemotronDiffusionBridge()
        self.registry = self.bridge.mapping_registry()

    def test_registry_is_not_none(self):
        assert self.registry is not None

    def test_has_language_model_prefix_on_megatron_keys(self):
        """All LM megatron param keys must use language_model. prefix."""
        mappings = list(self.registry)
        lm_mappings = [
            m for m in mappings if hasattr(m, "megatron_param") and "decoder" in getattr(m, "megatron_param", "")
        ]
        assert len(lm_mappings) > 0
        for m in lm_mappings:
            assert m.megatron_param.startswith("language_model."), (
                f"Expected 'language_model.' prefix, got: {m.megatron_param}"
            )

    def test_has_vision_tower_replicated_mapping(self):
        """Registry must contain a ReplicatedMapping for vision_tower.**"""
        from megatron.bridge.models.conversion.param_mapping import ReplicatedMapping

        mappings = list(self.registry)
        vision_mappings = [
            m
            for m in mappings
            if isinstance(m, ReplicatedMapping) and "vision_tower" in getattr(m, "megatron_param", "")
        ]
        assert len(vision_mappings) == 1
        assert vision_mappings[0].megatron_param == "vision_tower.**"
        assert vision_mappings[0].hf_param == "vision_tower.**"

    def test_has_multi_modal_projector_replicated_mapping(self):
        """Registry must contain a ReplicatedMapping for multi_modal_projector.**"""
        from megatron.bridge.models.conversion.param_mapping import ReplicatedMapping

        mappings = list(self.registry)
        proj_mappings = [
            m
            for m in mappings
            if isinstance(m, ReplicatedMapping) and "multi_modal_projector" in getattr(m, "megatron_param", "")
        ]
        assert len(proj_mappings) == 1
        assert proj_mappings[0].megatron_param == "multi_modal_projector.**"
        assert proj_mappings[0].hf_param == "multi_modal_projector.**"

    def test_has_qkv_mapping(self):
        """Registry must contain a QKVMapping for the attention QKV."""
        from megatron.bridge.models.conversion.param_mapping import QKVMapping

        mappings = list(self.registry)
        qkv_mappings = [m for m in mappings if isinstance(m, QKVMapping)]
        assert len(qkv_mappings) == 1
        assert "linear_qkv" in qkv_mappings[0].megatron_param

    def test_has_gated_mlp_mapping(self):
        """Registry must contain a GatedMLPMapping for the MLP."""
        from megatron.bridge.models.conversion.param_mapping import GatedMLPMapping

        mappings = list(self.registry)
        gated_mappings = [m for m in mappings if isinstance(m, GatedMLPMapping)]
        assert len(gated_mappings) == 1
        assert "linear_fc1" in gated_mappings[0].megatron_param

    def test_embedding_mapping_present(self):
        """word_embeddings mapping must be present with correct HF key."""
        from megatron.bridge.models.conversion.param_mapping import AutoMapping

        mappings = list(self.registry)
        embed_mappings = [
            m for m in mappings if isinstance(m, AutoMapping) and "word_embeddings" in getattr(m, "megatron_param", "")
        ]
        assert len(embed_mappings) == 1
        assert embed_mappings[0].hf_param == "language_model.model.embed_tokens.weight"

    def test_output_layer_mapping_present(self):
        """output_layer mapping must be present with correct HF key."""
        from megatron.bridge.models.conversion.param_mapping import AutoMapping

        mappings = list(self.registry)
        out_mappings = [
            m for m in mappings if isinstance(m, AutoMapping) and "output_layer" in getattr(m, "megatron_param", "")
        ]
        assert len(out_mappings) == 1
        assert out_mappings[0].hf_param == "language_model.lm_head.weight"


class TestNemotronDiffusionBridgeProviderBridge:
    """Tests for NemotronDiffusionBridge.provider_bridge()."""

    def test_returns_nemotron_diffusion_model_provider(self):
        from megatron.bridge.diffusion.models.nemotron_diffusion.nemotron_diffusion_provider import (
            NemotronDiffusionModelProvider,
        )

        bridge = NemotronDiffusionBridge()
        hf = DummyHFPretrained(_make_hf_config())
        provider = bridge.provider_bridge(hf)
        assert isinstance(provider, NemotronDiffusionModelProvider)

    def test_provider_has_correct_hidden_size(self):
        bridge = NemotronDiffusionBridge()
        hf = DummyHFPretrained(_make_hf_config(hidden_size=2048))
        provider = bridge.provider_bridge(hf)
        assert provider.hidden_size == 2048

    def test_provider_has_correct_num_layers(self):
        bridge = NemotronDiffusionBridge()
        hf = DummyHFPretrained(_make_hf_config(num_hidden_layers=16))
        provider = bridge.provider_bridge(hf)
        assert provider.num_layers == 16

    def test_provider_has_correct_vocab_size(self):
        bridge = NemotronDiffusionBridge()
        hf = DummyHFPretrained(_make_hf_config(vocab_size=65536))
        provider = bridge.provider_bridge(hf)
        assert provider.vocab_size == 65536

    def test_provider_share_embeddings_false_by_default(self):
        bridge = NemotronDiffusionBridge()
        hf = DummyHFPretrained(_make_hf_config(tie_word_embeddings=False))
        provider = bridge.provider_bridge(hf)
        assert provider.share_embeddings_and_output_weights is False

    def test_provider_share_embeddings_true_when_tied(self):
        bridge = NemotronDiffusionBridge()
        hf = DummyHFPretrained(_make_hf_config(tie_word_embeddings=True))
        provider = bridge.provider_bridge(hf)
        assert provider.share_embeddings_and_output_weights is True

    def test_provider_rotary_base_from_config(self):
        bridge = NemotronDiffusionBridge()
        hf = DummyHFPretrained(_make_hf_config(rope_theta=500000.0))
        provider = bridge.provider_bridge(hf)
        assert provider.rotary_base == 500000.0

    def test_provider_uses_text_config_when_nested(self):
        """provider_bridge must read from text_config when it exists."""
        bridge = NemotronDiffusionBridge()
        hf = DummyHFPretrained(_make_hf_config(hidden_size=512, num_hidden_layers=4))
        provider = bridge.provider_bridge(hf)
        assert provider.hidden_size == 512
        assert provider.num_layers == 4

    def test_provider_falls_back_to_flat_config(self):
        """provider_bridge must fall back to flat config when text_config is absent."""
        bridge = NemotronDiffusionBridge()
        flat_cfg = types.SimpleNamespace(
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=6,
            tie_word_embeddings=False,
            rope_parameters={"rope_theta": 10000.0},
            vocab_size=32000,
        )
        hf = DummyHFPretrained(flat_cfg)
        # SimpleNamespace doesn't have text_config, getattr falls back to hf_config itself
        provider = bridge.provider_bridge(hf)
        assert provider.hidden_size == 768
