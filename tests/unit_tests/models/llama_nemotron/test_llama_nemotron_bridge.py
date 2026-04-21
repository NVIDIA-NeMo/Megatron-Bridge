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

import json
from unittest.mock import Mock

import pytest
import torch
from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.llama_nemotron.llama_nemotron_bridge import LlamaNemotronBridge
from megatron.bridge.models.llama_nemotron.llama_nemotron_provider import (
    Llama31Nemotron70BProvider,
    Llama31NemotronNano8BProvider,
    Llama31NemotronUltra253BProvider,
    Llama33NemotronSuper49BProvider,
    LlamaNemotronHeterogeneousProvider,
)


class TestLlamaNemotronBridge:
    """Test cases for LlamaNemotronBridge class."""

    @pytest.fixture
    def llama_nemotron_nano_config_dict(self):
        """Create a sample Llama-Nemotron Nano configuration."""
        return {
            "architectures": ["LlamaForCausalLM"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 128000,
            "eos_token_id": 128001,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 14336,
            "max_position_embeddings": 131072,
            "mlp_bias": False,
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-05,
            "rope_scaling": {
                "factor": 8.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            },
            "rope_theta": 500000.0,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "use_cache": True,
            "vocab_size": 128256,
        }

    @pytest.fixture
    def llama_nemotron_super_config_dict(self):
        """Create a sample Llama-Nemotron Super 49B configuration (heterogeneous)."""
        return {
            "architectures": ["DeciLMForCausalLM"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 128000,
            "eos_token_id": [128001, 128008, 128009],
            "hidden_act": "silu",
            "hidden_size": 8192,
            "initializer_range": 0.02,
            "intermediate_size": None,  # Varies per layer in heterogeneous config
            "max_position_embeddings": 131072,
            "mlp_bias": False,
            "model_type": "nemotron-nas",
            "num_attention_heads": 64,
            "num_hidden_layers": 80,
            "num_key_value_heads": None,  # Varies per layer
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-05,
            "rope_scaling": {
                "factor": 8.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            },
            "rope_theta": 500000.0,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "use_cache": True,
            "vocab_size": 128256,
            "block_configs": [
                {"attention": {"n_heads_in_group": 8, "no_op": False}, "ffn": {"ffn_mult": 2.625, "no_op": False}},
                {"attention": {"n_heads_in_group": None, "no_op": True}, "ffn": {"ffn_mult": 1.0, "no_op": False}},
            ],
        }

    @pytest.fixture
    def mock_pretrained_nano(self, llama_nemotron_nano_config_dict):
        """Create a mock PreTrainedCausalLM with Llama-Nemotron Nano config."""
        config = LlamaConfig(**llama_nemotron_nano_config_dict)
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = config
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)
        mock_pretrained.model = Mock(spec=LlamaForCausalLM)
        mock_pretrained.model.dtype = torch.bfloat16
        return mock_pretrained

    @pytest.fixture
    def mock_pretrained_super(self, llama_nemotron_super_config_dict):
        """Create a mock PreTrainedCausalLM with Llama-Nemotron Super config (heterogeneous)."""
        # Create a mock config object since we can't use LlamaConfig for DeciLM
        mock_config = Mock()
        for key, value in llama_nemotron_super_config_dict.items():
            setattr(mock_config, key, value)

        # Add missing attributes that the bridge expects
        mock_config.head_dim = 128  # Set actual value, not Mock

        # Serialize the heterogeneous config from the provided dict for the bridge to consume
        mock_config.to_json_string = Mock(return_value=json.dumps(llama_nemotron_super_config_dict))

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mock_config
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)
        mock_pretrained.model = Mock()
        mock_pretrained.model.dtype = torch.bfloat16
        return mock_pretrained

    def test_bridge_registration(self):
        """Test that LlamaNemotronBridge is properly registered."""
        assert issubclass(LlamaNemotronBridge, MegatronModelBridge)

    def test_provider_bridge_nano_8b_should_use_llama_bridge(self, mock_pretrained_nano):
        """Test that Nano 8B models should use LlamaBridge, not LlamaNemotronBridge."""
        bridge = LlamaNemotronBridge()

        # This should raise an error because homogeneous models should use LlamaBridge
        with pytest.raises(ValueError, match="only handles heterogeneous models"):
            bridge.provider_bridge(mock_pretrained_nano)

    def test_provider_bridge_heterogeneous_super(self, mock_pretrained_super, llama_nemotron_super_config_dict):
        """Test provider_bridge functionality for heterogeneous Super 49B model."""
        bridge = LlamaNemotronBridge()
        provider = bridge.provider_bridge(mock_pretrained_super)

        # Verify that provider is of correct type (generic heterogeneous provider)
        assert isinstance(provider, LlamaNemotronHeterogeneousProvider)

        # Check that key configuration values are correctly mapped
        config = llama_nemotron_super_config_dict
        assert provider.num_layers == config["num_hidden_layers"]
        assert provider.hidden_size == config["hidden_size"]
        assert provider.num_attention_heads == config["num_attention_heads"]
        assert provider.kv_channels == 128  # Nemotron-specific override

        # Check that heterogeneous config is passed
        assert hasattr(provider, "heterogeneous_layers_config_encoded_json")
        assert provider.heterogeneous_layers_config_encoded_json != ""

        # Check data types
        assert provider.bf16 is True
        assert provider.params_dtype == torch.bfloat16

    def test_mapping_registry(self):
        """Test that mapping_registry returns proper mappings."""
        bridge = LlamaNemotronBridge()
        registry = bridge.mapping_registry()

        # Verify that registry is not None and has mappings
        assert registry is not None
        assert len(registry.mappings) > 0

        # Check for key mappings
        mapping_dict = {}
        for mapping in registry.mappings:
            if hasattr(mapping, "megatron_param") and hasattr(mapping, "hf_param"):
                mapping_dict[mapping.megatron_param] = mapping.hf_param

        # Verify key parameter mappings exist
        assert "embedding.word_embeddings.weight" in mapping_dict
        assert "output_layer.weight" in mapping_dict
        assert "decoder.final_layernorm.weight" in mapping_dict

    def test_dtype_configuration(self, mock_pretrained_super):
        """Test that data type configuration works correctly for heterogeneous models."""
        bridge = LlamaNemotronBridge()
        provider = bridge.provider_bridge(mock_pretrained_super)

        # Check that bfloat16 is correctly detected from config
        assert provider.bf16 is True
        assert provider.fp16 is False
        assert provider.params_dtype == torch.bfloat16


class TestLlamaNemotronSpecificProviders:
    """Test cases for specific Llama-Nemotron model provider configurations."""

    def test_nano_8b_config(self):
        """Test Llama3.1-Nemotron-Nano-8B provider configuration."""
        provider = Llama31NemotronNano8BProvider()
        provider.finalize()

        assert provider.hidden_size == 4096
        assert provider.num_layers == 32
        assert provider.num_attention_heads == 32
        assert provider.num_query_groups == 8
        assert provider.ffn_hidden_size == 14336
        assert provider.kv_channels == 128
        assert provider.seq_length == 131072
        assert provider.scale_factor == 8.0
        assert provider.gated_linear_unit is True

    def test_70b_config(self):
        """Test Llama3.1-Nemotron-70B provider configuration."""
        provider = Llama31Nemotron70BProvider()
        provider.finalize()

        assert provider.hidden_size == 8192
        assert provider.num_layers == 80
        assert provider.num_attention_heads == 64
        assert provider.num_query_groups == 8
        assert provider.ffn_hidden_size == 28672
        assert provider.kv_channels == 128
        assert provider.seq_length == 131072
        assert provider.scale_factor == 8.0

    def test_super_49b_config(self):
        """Test Llama3.3-Nemotron-Super-49B provider configuration."""
        provider = Llama33NemotronSuper49BProvider()
        # Provide minimal heterogeneous config JSON to avoid file access
        provider.heterogeneous_layers_config_encoded_json = json.dumps(
            {
                "block_configs": [
                    {
                        "attention": {"n_heads_in_group": 8, "no_op": False},
                        "ffn": {"ffn_mult": 2.625, "no_op": False},
                    }
                ],
                "rope_scaling": {
                    "factor": 8.0,
                    "low_freq_factor": 1.0,
                    "high_freq_factor": 4.0,
                    "original_max_position_embeddings": 8192,
                    "rope_type": "llama3",
                },
            }
        )
        provider.finalize()

        assert provider.hidden_size == 8192
        assert provider.num_layers == 80
        assert provider.num_attention_heads == 64
        assert provider.kv_channels == 128
        assert provider.seq_length == 131072

        # The ffn_hidden_size comes from the base Llama31ModelProvider70B
        # This is correct since heterogeneous models have varying FFN sizes per layer
        assert provider.ffn_hidden_size == 28672  # From base class

        # Should have heterogeneous config processed
        assert hasattr(provider, "per_block_parameters")
        assert hasattr(provider, "heterogeneous_layers_config_encoded_json")

    def test_ultra_253b_config(self):
        """Test Llama3.1-Nemotron-Ultra-253B provider configuration."""
        provider = Llama31NemotronUltra253BProvider()
        # Provide minimal heterogeneous config JSON to avoid file access
        provider.heterogeneous_layers_config_encoded_json = json.dumps(
            {
                "block_configs": [
                    {
                        "attention": {"n_heads_in_group": 16, "no_op": False},
                        "ffn": {"ffn_mult": 1.0, "no_op": False},
                    }
                ],
                "rope_scaling": {
                    "factor": 16.0,
                    "low_freq_factor": 1.0,
                    "high_freq_factor": 4.0,
                    "original_max_position_embeddings": 8192,
                    "rope_type": "llama3",
                },
            }
        )
        provider.finalize()

        assert provider.hidden_size == 16384
        assert provider.num_layers == 162
        assert provider.num_attention_heads == 128
        assert provider.kv_channels == 128
        assert provider.seq_length == 131072

        # The ffn_hidden_size comes from the base Llama31ModelProvider405B (53248)
        # This is correct since heterogeneous models have varying FFN sizes per layer
        assert provider.ffn_hidden_size == 53248  # From base class, not override

        # Should have heterogeneous config processed in finalize
        assert hasattr(provider, "per_block_parameters")
        assert hasattr(provider, "heterogeneous_layers_config_encoded_json")
