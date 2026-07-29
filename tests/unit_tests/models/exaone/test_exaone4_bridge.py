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
from unittest.mock import Mock

import torch
import torch.nn.functional as F

from megatron.bridge.models.exaone.exaone4.exaone4_bridge import Exaone4Bridge
from megatron.bridge.models.exaone.exaone4.exaone4_provider import exaone4_layer_spec
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


def make_exaone4_config(**overrides):
    config = {
        "architectures": ["Exaone4ForCausalLM"],
        "hidden_size": 2048,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "max_position_embeddings": 65536,
        "model_type": "exaone4",
        "num_attention_heads": 32,
        "num_hidden_layers": 30,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-5,
        "rope_scaling": {
            "rope_type": "llama3",
            "factor": 16.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
        "rope_theta": 1000000.0,
        "tie_word_embeddings": True,
        "torch_dtype": torch.bfloat16,
        "vocab_size": 102400,
        "head_dim": 64,
    }
    config.update(overrides)
    return SimpleNamespace(**config)


def make_pretrained(config):
    pretrained = Mock(spec=PreTrainedCausalLM)
    pretrained.config = config
    return pretrained


class TestExaone4Bridge:
    """Test cases for EXAONE 4.0 bridge config and mappings."""

    def test_provider_bridge_basic_mapping(self):
        config = make_exaone4_config()
        provider = Exaone4Bridge().provider_bridge(make_pretrained(config))

        assert isinstance(provider, GPTModelProvider)
        assert provider.num_layers == config.num_hidden_layers
        assert provider.hidden_size == config.hidden_size
        assert provider.ffn_hidden_size == config.intermediate_size
        assert provider.num_attention_heads == config.num_attention_heads
        assert provider.num_query_groups == config.num_key_value_heads
        assert provider.seq_length == config.max_position_embeddings
        assert provider.init_method_std == config.initializer_range
        assert provider.layernorm_epsilon == config.rms_norm_eps
        assert provider.rotary_base == config.rope_theta
        assert provider.kv_channels == config.head_dim
        assert provider.vocab_size == config.vocab_size
        assert provider.share_embeddings_and_output_weights is True
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func == F.silu
        assert provider.gated_linear_unit is True
        assert provider.position_embedding_type == "rope"
        assert provider.add_bias_linear is False
        assert provider.add_qkv_bias is False
        assert provider.qk_layernorm is True
        assert provider.hidden_dropout == 0.0
        assert provider.attention_dropout == 0.0
        assert provider.transformer_layer_spec == exaone4_layer_spec
        assert provider.autocast_dtype == torch.bfloat16

    def test_provider_bridge_rope_scaling_mapping(self):
        rope_scaling = {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 2.0,
            "high_freq_factor": 6.0,
            "original_max_position_embeddings": 4096,
        }
        provider = Exaone4Bridge().provider_bridge(make_pretrained(make_exaone4_config(rope_scaling=rope_scaling)))

        assert provider.rope_scaling is True
        assert provider.rope_scaling_factor == rope_scaling["factor"]
        assert provider.rope_scaling_low_freq_factor == rope_scaling["low_freq_factor"]
        assert provider.rope_scaling_high_freq_factor == rope_scaling["high_freq_factor"]
        assert (
            provider.rope_scaling_original_max_position_embeddings == rope_scaling["original_max_position_embeddings"]
        )

    def test_provider_bridge_dtype_handling(self):
        provider = Exaone4Bridge().provider_bridge(make_pretrained(make_exaone4_config(torch_dtype=torch.float16)))

        assert provider.params_dtype == torch.float16
        assert provider.fp16 is True
        assert provider.bf16 is False

    def test_megatron_to_hf_config_preserves_rope_scaling(self):
        provider = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            ffn_hidden_size=256,
            num_attention_heads=4,
            num_query_groups=2,
            seq_length=1024,
            vocab_size=32000,
            rope_scaling=True,
            rope_scaling_factor=8.0,
        )
        # llama3-style scaling fields are not GPTModelProvider dataclass fields;
        # provider_bridge sets them as plain attributes, so mirror that here.
        provider.rope_scaling_low_freq_factor = 2.0
        provider.rope_scaling_high_freq_factor = 6.0
        provider.rope_scaling_original_max_position_embeddings = 4096

        hf_config = Exaone4Bridge.megatron_to_hf_config(provider)

        assert hf_config["model_type"] == "exaone4"
        assert hf_config["tie_word_embeddings"] is True
        assert hf_config["rope_scaling"] == {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 2.0,
            "high_freq_factor": 6.0,
            "original_max_position_embeddings": 4096,
        }

    def test_mapping_registry_contains_exaone_weights(self):
        registry = Exaone4Bridge().mapping_registry()

        assert (
            registry.megatron_to_hf_lookup("embedding.word_embeddings.weight").hf_param == "model.embed_tokens.weight"
        )
        assert registry.megatron_to_hf_lookup("decoder.final_layernorm.weight").hf_param == "model.norm.weight"
        assert (
            registry.megatron_to_hf_lookup("decoder.layers.0.self_attention.linear_proj.weight").hf_param
            == "model.layers.0.self_attn.o_proj.weight"
        )
        assert (
            registry.megatron_to_hf_lookup("decoder.layers.0.self_attention.q_layernorm.weight").hf_param
            == "model.layers.0.self_attn.q_norm.weight"
        )
        assert (
            registry.megatron_to_hf_lookup("decoder.layers.0.self_attention.k_layernorm.weight").hf_param
            == "model.layers.0.self_attn.k_norm.weight"
        )
        post_attn_key = "decoder.layers.0.self_attention.linear_proj.post_layernorm.weight"
        assert (
            registry.megatron_to_hf_lookup(post_attn_key).hf_param == "model.layers.0.post_attention_layernorm.weight"
        )
        assert (
            registry.megatron_to_hf_lookup("decoder.layers.0.mlp.linear_fc2.post_layernorm.weight").hf_param
            == "model.layers.0.post_feedforward_layernorm.weight"
        )

        qkv_mapping = registry.megatron_to_hf_lookup("decoder.layers.0.self_attention.linear_qkv.weight")
        assert qkv_mapping.hf_param == {
            "q": "model.layers.0.self_attn.q_proj.weight",
            "k": "model.layers.0.self_attn.k_proj.weight",
            "v": "model.layers.0.self_attn.v_proj.weight",
        }

        gated_mlp_mapping = registry.megatron_to_hf_lookup("decoder.layers.0.mlp.linear_fc1.weight")
        assert gated_mlp_mapping.hf_param == {
            "gate": "model.layers.0.mlp.gate_proj.weight",
            "up": "model.layers.0.mlp.up_proj.weight",
        }
