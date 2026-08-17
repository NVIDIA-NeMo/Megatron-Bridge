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
import torch.nn.functional as F
from transformers import MistralConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mistral.mistral_bridge import MistralBridge
from megatron.bridge.models.mistral.mistral_provider import MistralModelProvider


pytestmark = pytest.mark.unit


def _hf_config(**overrides):
    kwargs = dict(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        vocab_size=512,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        rope_theta=100000.0,
        torch_dtype=torch.bfloat16,
    )
    kwargs.update(overrides)
    return MistralConfig(**kwargs)


def _pretrained(config):
    pretrained = Mock(spec=PreTrainedCausalLM)
    pretrained.config = config
    return pretrained


@pytest.fixture
def bridge():
    return MistralBridge()


class TestMistralProviderBridge:
    def test_basic_config_fields(self, bridge):
        provider = bridge.provider_bridge(_pretrained(_hf_config()))

        assert isinstance(provider, MistralModelProvider)
        assert provider.num_layers == 2
        assert provider.hidden_size == 64
        assert provider.ffn_hidden_size == 128
        assert provider.num_attention_heads == 4
        assert provider.num_query_groups == 2
        assert provider.seq_length == 1024
        assert provider.vocab_size == 512
        assert provider.layernorm_epsilon == 1e-6
        assert provider.init_method_std == 0.02
        assert provider.rotary_base == 100000.0
        assert provider.gated_linear_unit is True

    def test_dtype_from_hf_config(self, bridge):
        provider = bridge.provider_bridge(_pretrained(_hf_config(torch_dtype=torch.bfloat16)))

        assert provider.bf16 is True
        assert provider.fp16 is False
        assert provider.params_dtype == torch.bfloat16

    def test_kv_channels_from_head_dim(self, bridge):
        config = _hf_config()
        provider = bridge.provider_bridge(_pretrained(config))

        assert provider.kv_channels == config.head_dim

    def test_sliding_window_maps_to_window_size(self, bridge):
        provider = bridge.provider_bridge(_pretrained(_hf_config(sliding_window=4096)))

        assert provider.window_size == [4095, 0]
        assert provider.cp_comm_type == "a2a"

    def test_no_sliding_window(self, bridge):
        provider = bridge.provider_bridge(_pretrained(_hf_config(sliding_window=None)))

        assert provider.window_size is None
        assert provider.cp_comm_type is None

    @pytest.mark.parametrize("tie_word_embeddings", [True, False])
    def test_tied_embeddings(self, bridge, tie_word_embeddings):
        provider = bridge.provider_bridge(_pretrained(_hf_config(tie_word_embeddings=tie_word_embeddings)))

        assert provider.share_embeddings_and_output_weights is tie_word_embeddings

    def test_default_rope_is_not_treated_as_yarn(self, bridge):
        # transformers 5.x always exposes rope_scaling as a dict with rope_type "default"
        provider = bridge.provider_bridge(_pretrained(_hf_config()))

        assert provider.position_embedding_type == "rope"
        assert provider.yarn_rotary_scaling_factor is None

    def test_yarn_rope_scaling_sets_yarn_fields(self, bridge):
        config = _hf_config(
            rope_scaling={
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 256,
            }
        )
        provider = bridge.provider_bridge(_pretrained(config))

        assert provider.position_embedding_type == "yarn"
        assert provider.yarn_rotary_scaling_factor == 4.0
        assert provider.yarn_original_max_position_embeddings == 256

    def test_yarn_truncate_maps_to_correction_range_rounding(self, bridge):
        config = _hf_config(
            rope_scaling={
                "rope_type": "yarn",
                "factor": 4.0,
                "truncate": True,
            }
        )
        provider = bridge.provider_bridge(_pretrained(config))

        assert provider.yarn_correction_range_round_to_int is True

    def test_yarn_rope_scaling_with_legacy_type_key(self, bridge):
        config = _hf_config()
        # Legacy checkpoints use "type" instead of "rope_type".
        config.rope_scaling = {"type": "yarn", "factor": 2.0}
        provider = bridge.provider_bridge(_pretrained(config))

        assert provider.position_embedding_type == "yarn"
        assert provider.yarn_rotary_scaling_factor == 2.0

    def test_provider_defaults(self):
        provider = MistralModelProvider(num_layers=2, hidden_size=64, num_attention_heads=4)

        assert provider.normalization == "RMSNorm"
        assert provider.activation_func is F.silu
        assert provider.position_embedding_type == "rope"
        assert provider.add_bias_linear is False
        assert provider.gated_linear_unit is True
        assert provider.share_embeddings_and_output_weights is False


class TestMistralMappingRegistry:
    def test_registry_type_and_size(self, bridge):
        registry = bridge.mapping_registry()

        assert isinstance(registry, MegatronMappingRegistry)
        # 7 direct mappings + QKV + GatedMLP, plus any registry-added derived mappings.
        assert len(registry) >= 9

    @pytest.mark.parametrize(
        "megatron_param",
        [
            "embedding.word_embeddings.weight",
            "output_layer.weight",
            "decoder.final_layernorm.weight",
            "decoder.layers.*.self_attention.linear_qkv.weight",
            "decoder.layers.*.mlp.linear_fc1.weight",
            "decoder.layers.*.mlp.linear_fc2.weight",
        ],
    )
    def test_expected_megatron_params_are_mapped(self, bridge, megatron_param):
        registry = bridge.mapping_registry()
        mapped = {mapping.megatron_param for mapping in registry.get_all_mappings()}

        assert megatron_param in mapped
