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

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.nemotron_vl.model_config import NemotronVLModelBuilder, NemotronVLModelConfig
from megatron.bridge.models.nemotron_vl.nemotron_vl_bridge import NemotronVLBridge
from megatron.bridge.models.nemotron_vl.nemotron_vl_provider import (
    NemotronVLModelProvider,
)


@pytest.fixture
def mock_llm_config():
    # Use spec=[] so hasattr() only returns True for explicitly-set attributes,
    # matching real HF config behaviour (Nemotron config has no MLA fields
    # like q_lora_rank, so they must not appear in the provider kwargs).
    cfg = Mock(spec=[])
    cfg.hybrid_override_pattern = "M-M-M-M*-M-M-M-M*-M-M-M-M-M*"
    cfg.hidden_size = 5120
    cfg.intermediate_size = 20480
    cfg.num_attention_heads = 40
    cfg.num_key_value_heads = 8
    cfg.initializer_range = 0.02
    cfg.rms_norm_eps = 1e-5  # CONFIG_MAPPING uses rms_norm_eps -> layernorm_epsilon
    cfg.vocab_size = 262144
    cfg.max_position_embeddings = 131072
    cfg.hidden_act = "relu2"
    cfg.rope_scaling = None
    cfg.torch_dtype = "bfloat16"
    return cfg


@pytest.fixture
def mock_hf_config(mock_llm_config):
    cfg = Mock()
    cfg.llm_config = mock_llm_config
    return cfg


@pytest.fixture
def mock_hf_pretrained(mock_hf_config):
    pretrained = Mock(spec=PreTrainedCausalLM)
    pretrained.config = mock_hf_config
    return pretrained


@pytest.fixture
def nemotron_vl_bridge():
    return NemotronVLBridge()


class TestNemotronVLBridgeInitialization:
    def test_bridge_initialization(self, nemotron_vl_bridge):
        assert isinstance(nemotron_vl_bridge, NemotronVLBridge)

    def test_bridge_has_required_methods(self, nemotron_vl_bridge):
        assert hasattr(nemotron_vl_bridge, "provider_bridge")
        assert callable(nemotron_vl_bridge.provider_bridge)
        assert hasattr(nemotron_vl_bridge, "mapping_registry")
        assert callable(nemotron_vl_bridge.mapping_registry)

    def test_model_config_bridge_is_serializable(self, nemotron_vl_bridge, mock_hf_pretrained):
        result = nemotron_vl_bridge.model_config_bridge(mock_hf_pretrained)

        assert isinstance(result, NemotronVLModelConfig)
        assert type(result.transformer) is TransformerConfig
        assert result.hybrid_layer_pattern == mock_hf_pretrained.config.llm_config.hybrid_override_pattern
        assert result.get_builder_cls() is NemotronVLModelBuilder
        restored = type(result).from_dict(result.as_dict())
        assert isinstance(restored, NemotronVLModelConfig)

    def test_builder_forwards_pipeline_construction_state(self, nemotron_vl_bridge, mock_hf_pretrained, monkeypatch):
        config = nemotron_vl_bridge.model_config_bridge(mock_hf_pretrained)
        pg_collection = SimpleNamespace(pp=object())
        captured = {}
        built_model = object()

        def fake_llava_model(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(config=kwargs["language_transformer_config"])

        monkeypatch.setattr("megatron.bridge.models.nemotron_vl.model_config.LLaVAModel", fake_llava_model)
        monkeypatch.setattr(
            "megatron.bridge.models.nemotron_vl.model_config.NemotronVLModel",
            lambda **kwargs: built_model,
        )

        result = NemotronVLModelBuilder(config).build_model(
            pg_collection,
            pre_process=False,
            post_process=True,
            vp_stage=3,
        )

        assert result is built_model
        assert captured["add_encoder"] is False
        assert captured["pg_collection"] is pg_collection
        assert captured["vp_stage"] == 3
        assert captured["language_transformer_config"].language_model_type == "nemotron5-hybrid-12b"
        assert captured["vision_transformer_config"].vision_model_type == "radio"
        assert not hasattr(config.transformer, "language_model_type")
        assert not hasattr(config.transformer, "vision_model_type")


class TestNemotronVLBridgeProviderBridge:
    def test_provider_bridge_basic_config(self, nemotron_vl_bridge, mock_hf_pretrained):
        provider = nemotron_vl_bridge.provider_bridge(mock_hf_pretrained)
        provider.finalize()

        assert isinstance(provider, NemotronVLModelProvider)

        assert provider.num_layers == 28
        assert provider.hidden_size == 5120
        assert provider.ffn_hidden_size == 20480
        assert provider.num_attention_heads == 40
        assert provider.num_query_groups == 8
        assert provider.init_method_std == 0.02
        assert provider.layernorm_epsilon == 1e-5
        assert provider.vocab_size == 262144
        assert provider.seq_length == 131072

    @patch.object(NemotronVLBridge, "dtype_from_hf")
    def test_provider_bridge_dtype_fp16(self, mock_dtype_from_hf, nemotron_vl_bridge, mock_hf_pretrained):
        mock_dtype_from_hf.return_value = torch.float16
        provider = nemotron_vl_bridge.provider_bridge(mock_hf_pretrained)
        provider.finalize()
        assert provider.fp16 is True
        assert provider.bf16 is False
        assert provider.params_dtype == torch.float16

    @patch.object(NemotronVLBridge, "dtype_from_hf")
    def test_provider_bridge_dtype_bf16(self, mock_dtype_from_hf, nemotron_vl_bridge, mock_hf_pretrained):
        mock_dtype_from_hf.return_value = torch.bfloat16
        provider = nemotron_vl_bridge.provider_bridge(mock_hf_pretrained)
        provider.finalize()
        assert provider.fp16 is False
        assert provider.bf16 is True
        assert provider.params_dtype == torch.bfloat16

    @patch.object(NemotronVLBridge, "dtype_from_hf")
    def test_provider_bridge_dtype_fp32(self, mock_dtype_from_hf, nemotron_vl_bridge, mock_hf_pretrained):
        mock_dtype_from_hf.return_value = torch.float32
        provider = nemotron_vl_bridge.provider_bridge(mock_hf_pretrained)
        provider.finalize()
        assert provider.fp16 is False
        assert provider.bf16 is False
        assert provider.params_dtype == torch.float32


class TestNemotronVLBridgeMappingRegistry:
    def test_mapping_registry_returns_registry(self, nemotron_vl_bridge):
        registry = nemotron_vl_bridge.mapping_registry()
        assert isinstance(registry, MegatronMappingRegistry)
        assert len(registry.mappings) > 0

    def test_mapping_registry_contains_expected_groups(self, nemotron_vl_bridge):
        registry = nemotron_vl_bridge.mapping_registry()
        names = []
        for m in registry.mappings:
            if hasattr(m, "megatron_param"):
                names.append(str(getattr(m, "megatron_param")))
            hf = getattr(m, "hf_param", None)
            if isinstance(hf, dict):
                names.extend([str(v) for v in hf.values()])
            elif isinstance(hf, str):
                names.append(hf)

        # Vision model mappings (RADIO)
        assert any("vision_model" in n for n in names)
        # Language model mappings
        assert any("language_model" in n for n in names)
        # QKV mappings should be present for both language and vision
        assert any("linear_qkv" in n for n in names)

        assert "llava_model.language_model.decoder.layers.*.mixer.conv1d_weight" in names
        assert "llava_model.language_model.decoder.layers.*.mixer.conv1d_bias" in names
        assert "llava_model.language_model.decoder.layers.*.mixer.conv1d.weight" in names
        assert "llava_model.language_model.decoder.layers.*.mixer.conv1d.bias" in names

        reverse_weight = registry.hf_to_megatron_lookup("language_model.backbone.layers.0.mixer.conv1d.weight")
        reverse_bias = registry.hf_to_megatron_lookup("language_model.backbone.layers.0.mixer.conv1d.bias")
        assert reverse_weight.megatron_param == "llava_model.language_model.decoder.layers.0.mixer.conv1d_weight"
        assert reverse_bias.megatron_param == "llava_model.language_model.decoder.layers.0.mixer.conv1d_bias"
