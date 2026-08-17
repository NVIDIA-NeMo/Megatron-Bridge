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

from unittest.mock import Mock

import pytest
import torch
from megatron.core.activations import fast_gelu
from transformers import Gemma2Config

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.gemma.gemma2_bridge import Gemma2Bridge
from megatron.bridge.models.gemma.gemma2_provider import Gemma2ModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


pytestmark = pytest.mark.unit


def _hf_config(**overrides):
    kwargs = dict(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=1024,
        vocab_size=512,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        query_pre_attn_scalar=224,
        attn_logit_softcapping=50.0,
        final_logit_softcapping=30.0,
        sliding_window=512,
        torch_dtype=torch.bfloat16,
    )
    kwargs.update(overrides)
    return Gemma2Config(**kwargs)


def _pretrained(config):
    pretrained = Mock(spec=PreTrainedCausalLM)
    pretrained.config = config
    return pretrained


@pytest.fixture
def bridge():
    return Gemma2Bridge()


class TestGemma2ProviderBridge:
    def test_returns_gemma2_provider(self, bridge):
        provider = bridge.provider_bridge(_pretrained(_hf_config()))

        assert isinstance(provider, Gemma2ModelProvider)

    def test_basic_dimensions(self, bridge):
        provider = bridge.provider_bridge(_pretrained(_hf_config()))

        assert provider.num_layers == 2
        assert provider.hidden_size == 64
        assert provider.ffn_hidden_size == 128
        assert provider.num_attention_heads == 4
        assert provider.vocab_size == 512

    def test_softcapping_fields_from_hf_config(self, bridge):
        config = _hf_config(attn_logit_softcapping=42.0, final_logit_softcapping=21.0, query_pre_attn_scalar=128)
        provider = bridge.provider_bridge(_pretrained(config))

        assert provider.attn_logit_softcapping == 42.0
        assert provider.final_logit_softcapping == 21.0
        assert provider.query_pre_attn_scalar == 128

    def test_sliding_window_maps_to_window_size(self, bridge):
        provider = bridge.provider_bridge(_pretrained(_hf_config(sliding_window=512)))

        assert provider.window_size == (511, 0)

    def test_gemma2_architecture_invariants(self, bridge):
        provider = bridge.provider_bridge(_pretrained(_hf_config()))

        assert provider.normalization == "RMSNorm"
        assert provider.activation_func is fast_gelu
        assert provider.gated_linear_unit is True
        assert provider.add_bias_linear is False
        assert provider.attention_dropout == 0.0
        assert provider.hidden_dropout == 0.0
        assert provider.share_embeddings_and_output_weights is True
        assert provider.layernorm_zero_centered_gamma is True


class TestGemma2MegatronToHfConfig:
    def test_roundtrips_softcapping_and_window(self, bridge):
        provider = bridge.provider_bridge(_pretrained(_hf_config(sliding_window=512)))

        hf_config = Gemma2Bridge.megatron_to_hf_config(provider)

        assert hf_config["query_pre_attn_scalar"] == 224
        assert hf_config["attn_logit_softcapping"] == 50.0
        assert hf_config["final_logit_softcapping"] == 30.0
        assert hf_config["sliding_window"] == 512


class TestGemma2MappingRegistry:
    def test_registry_type(self, bridge):
        assert isinstance(bridge.mapping_registry(), MegatronMappingRegistry)

    @pytest.mark.parametrize(
        "megatron_param",
        [
            "embedding.word_embeddings.weight",
            "decoder.final_layernorm.weight",
            "decoder.layers.*.self_attention.linear_qkv.weight",
            "decoder.layers.*.mlp.linear_fc1.weight",
            "decoder.layers.*.mlp.linear_fc2.weight",
            # Gemma2-specific post-layernorm locations
            "decoder.layers.*.mlp.linear_fc2.post_layernorm.weight",
            "decoder.layers.*.self_attention.linear_proj.post_layernorm.weight",
        ],
    )
    def test_expected_megatron_params_are_mapped(self, bridge, megatron_param):
        mapped = {mapping.megatron_param for mapping in bridge.mapping_registry().get_all_mappings()}

        assert megatron_param in mapped

    def test_no_lm_head_mapping_for_tied_embeddings(self, bridge):
        mapped = {mapping.megatron_param for mapping in bridge.mapping_registry().get_all_mappings()}

        assert "output_layer.weight" not in mapped
