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

"""Unit tests for NemotronLabsDiffusionBridge mapping registry and provider bridge."""

import types

import pytest

from megatron.bridge.diffusion.conversion.nemotron_labs_diffusion.nemotron_labs_diffusion_bridge import (
    NemotronLabsDiffusionBridge,
)


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


class TestNemotronLabsDiffusionBridgeMappingRegistry:
    """Tests for NemotronLabsDiffusionBridge.mapping_registry()."""

    def setup_method(self):
        self.bridge = NemotronLabsDiffusionBridge()
        self.registry = self.bridge.mapping_registry()

    def test_registry_is_not_none(self):
        assert self.registry is not None

    def test_megatron_keys_are_bare(self):
        """Megatron-side keys must be bare (no 'language_model.' prefix).

        NemotronLabsDiffusion targets a bare GPTModel, not a VLM wrapper, so Megatron keys are
        'embedding.*', 'decoder.*', 'output_layer.*'.
        """
        mappings = list(self.registry)
        lm_mappings = [m for m in mappings if hasattr(m, "megatron_param")]
        assert len(lm_mappings) > 0
        for m in lm_mappings:
            assert not m.megatron_param.startswith("language_model."), (
                f"Unexpected 'language_model.' prefix on Megatron key: {m.megatron_param}"
            )

    def test_no_vision_tower_mapping(self):
        """Vision tower weights are intentionally not mapped (Megatron is text-only)."""
        mappings = list(self.registry)
        vision_mappings = [m for m in mappings if "vision_tower" in getattr(m, "megatron_param", "")]
        assert len(vision_mappings) == 0

    def test_no_multi_modal_projector_mapping(self):
        """Multi-modal projector weights are intentionally not mapped (Megatron is text-only)."""
        mappings = list(self.registry)
        proj_mappings = [m for m in mappings if "multi_modal_projector" in getattr(m, "megatron_param", "")]
        assert len(proj_mappings) == 0

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
        assert embed_mappings[0].hf_param == "encoder.embed_tokens.weight"

    def test_output_layer_mapping_present(self):
        """output_layer mapping must be present with correct HF key."""
        from megatron.bridge.models.conversion.param_mapping import AutoMapping

        mappings = list(self.registry)
        out_mappings = [
            m for m in mappings if isinstance(m, AutoMapping) and "output_layer" in getattr(m, "megatron_param", "")
        ]
        assert len(out_mappings) == 1
        assert out_mappings[0].hf_param == "diffusion_head.weight"


class TestNemotronLabsDiffusionBridgeProviderBridge:
    """Tests for NemotronLabsDiffusionBridge.provider_bridge()."""

    def test_returns_nemotron_labs_diffusion_model_provider(self):
        from megatron.bridge.diffusion.models.nemotron_labs_diffusion.nemotron_labs_diffusion_provider import (
            NemotronLabsDiffusionModelProvider,
        )

        bridge = NemotronLabsDiffusionBridge()
        hf = DummyHFPretrained(_make_hf_config())
        provider = bridge.provider_bridge(hf)
        assert isinstance(provider, NemotronLabsDiffusionModelProvider)

    def test_provider_has_correct_hidden_size(self):
        bridge = NemotronLabsDiffusionBridge()
        hf = DummyHFPretrained(_make_hf_config(hidden_size=2048))
        provider = bridge.provider_bridge(hf)
        assert provider.hidden_size == 2048

    def test_provider_has_correct_num_layers(self):
        bridge = NemotronLabsDiffusionBridge()
        hf = DummyHFPretrained(_make_hf_config(num_hidden_layers=16))
        provider = bridge.provider_bridge(hf)
        assert provider.num_layers == 16

    def test_provider_has_correct_vocab_size(self):
        bridge = NemotronLabsDiffusionBridge()
        hf = DummyHFPretrained(_make_hf_config(vocab_size=65536))
        provider = bridge.provider_bridge(hf)
        assert provider.vocab_size == 65536

    def test_provider_share_embeddings_false_by_default(self):
        bridge = NemotronLabsDiffusionBridge()
        hf = DummyHFPretrained(_make_hf_config(tie_word_embeddings=False))
        provider = bridge.provider_bridge(hf)
        assert provider.share_embeddings_and_output_weights is False

    def test_provider_share_embeddings_true_when_tied(self):
        bridge = NemotronLabsDiffusionBridge()
        hf = DummyHFPretrained(_make_hf_config(tie_word_embeddings=True))
        provider = bridge.provider_bridge(hf)
        assert provider.share_embeddings_and_output_weights is True

    def test_provider_rotary_base_from_config(self):
        bridge = NemotronLabsDiffusionBridge()
        hf = DummyHFPretrained(_make_hf_config(rope_theta=500000.0))
        provider = bridge.provider_bridge(hf)
        assert provider.rotary_base == 500000.0

    def test_provider_uses_text_config_when_nested(self):
        """provider_bridge must read from text_config when it exists."""
        bridge = NemotronLabsDiffusionBridge()
        hf = DummyHFPretrained(_make_hf_config(hidden_size=512, num_hidden_layers=4))
        provider = bridge.provider_bridge(hf)
        assert provider.hidden_size == 512
        assert provider.num_layers == 4

    def test_provider_falls_back_to_flat_config(self):
        """provider_bridge must fall back to flat config when text_config is absent."""
        bridge = NemotronLabsDiffusionBridge()
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


class TestNemotronLabsDiffusionBridgeMappingRegistryVLM:
    """Tests for the VLM-format mapping registry (HF keys use language_model.* prefix)."""

    def setup_method(self):
        self.bridge = NemotronLabsDiffusionBridge()
        # provider_bridge sets _is_text_only based on whether hf_config has text_config
        hf = DummyHFPretrained(_make_hf_config())  # has text_config -> VLM mode
        self.bridge.provider_bridge(hf)
        self.registry = self.bridge.mapping_registry()

    def test_is_vlm_mode(self):
        assert self.bridge._is_text_only is False

    def test_vlm_embedding_mapping_uses_language_model_prefix(self):
        from megatron.bridge.models.conversion.param_mapping import AutoMapping

        mappings = list(self.registry)
        embed_mappings = [
            m for m in mappings if isinstance(m, AutoMapping) and "word_embeddings" in getattr(m, "megatron_param", "")
        ]
        assert len(embed_mappings) == 1
        assert embed_mappings[0].hf_param == "language_model.model.embed_tokens.weight"

    def test_vlm_output_layer_mapping_uses_lm_head(self):
        from megatron.bridge.models.conversion.param_mapping import AutoMapping

        mappings = list(self.registry)
        out_mappings = [
            m for m in mappings if isinstance(m, AutoMapping) and "output_layer" in getattr(m, "megatron_param", "")
        ]
        assert len(out_mappings) == 1
        assert out_mappings[0].hf_param == "language_model.lm_head.weight"


class _MockConfig:
    """A mutable config class with to_dict, simulating a trust_remote_code PretrainedConfig."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class TestToCfgDictMonkeyPatch:
    """Tests for the to_cfg_dict monkey-patch in provider_bridge()."""

    def _make_mock_hf_config(self):
        text_cfg = _MockConfig(
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=8,
            tie_word_embeddings=False,
            rope_parameters={"rope_theta": 10000.0},
            vocab_size=32000,
        )
        hf_cfg = _MockConfig(text_config=text_cfg)
        return hf_cfg

    def test_to_cfg_dict_added_when_config_has_to_dict(self):
        """provider_bridge adds to_cfg_dict to config classes that have to_dict."""
        bridge = NemotronLabsDiffusionBridge()
        hf_cfg = self._make_mock_hf_config()
        hf = DummyHFPretrained(hf_cfg)

        assert not hasattr(_MockConfig, "to_cfg_dict")
        bridge.provider_bridge(hf)
        assert hasattr(_MockConfig, "to_cfg_dict")

        # Clean up monkey-patch so it doesn't leak to other tests
        delattr(_MockConfig, "to_cfg_dict")

    def test_to_cfg_dict_returns_correct_target(self):
        """to_cfg_dict must produce a _target_ using cls.__module__ and cls.__qualname__."""
        bridge = NemotronLabsDiffusionBridge()
        hf_cfg = self._make_mock_hf_config()
        hf = DummyHFPretrained(hf_cfg)
        bridge.provider_bridge(hf)

        result = hf_cfg.to_cfg_dict()
        expected_target = f"{_MockConfig.__module__}.{_MockConfig.__qualname__}.from_dict"
        assert result["_target_"] == expected_target
        assert result["_call_"] is True
        assert "config_dict" in result

        delattr(_MockConfig, "to_cfg_dict")

    def test_to_cfg_dict_preserves_dynamic_attributes(self):
        """to_cfg_dict must capture dynamic attributes like rope_parameters via to_dict."""
        bridge = NemotronLabsDiffusionBridge()
        hf_cfg = self._make_mock_hf_config()
        hf_cfg.llama_4_scaling_beta = 0.7  # dynamic attribute
        hf = DummyHFPretrained(hf_cfg)
        bridge.provider_bridge(hf)

        result = hf_cfg.to_cfg_dict()
        assert result["config_dict"]["llama_4_scaling_beta"] == 0.7

        delattr(_MockConfig, "to_cfg_dict")

    def test_to_cfg_dict_not_added_to_simplenamespace(self):
        """SimpleNamespace has no to_dict, so to_cfg_dict must not be added."""
        bridge = NemotronLabsDiffusionBridge()
        hf_cfg = _make_hf_config()  # uses SimpleNamespace
        hf = DummyHFPretrained(hf_cfg)
        bridge.provider_bridge(hf)

        assert not hasattr(types.SimpleNamespace, "to_cfg_dict")

    def test_to_cfg_dict_not_added_twice(self):
        """If to_cfg_dict already exists, provider_bridge must not overwrite it."""
        bridge = NemotronLabsDiffusionBridge()
        hf_cfg = self._make_mock_hf_config()
        hf = DummyHFPretrained(hf_cfg)

        sentinel = lambda self: {"sentinel": True}
        _MockConfig.to_cfg_dict = sentinel

        bridge.provider_bridge(hf)
        assert _MockConfig.to_cfg_dict is sentinel

        delattr(_MockConfig, "to_cfg_dict")
