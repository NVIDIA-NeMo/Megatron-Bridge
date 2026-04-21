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

from unittest.mock import Mock

import pytest
import torch
from transformers import GenerationConfig, NemotronConfig, NemotronForCausalLM

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.nemotron.nemotron_bridge import NemotronBridge
from megatron.bridge.models.nemotron.nemotron_provider import (
    NemotronModelProvider,
)


class TestNemotronBridge:
    """Test cases for NemotronBridge class."""

    @pytest.fixture
    def nemotron_config_dict(self):
        """Create a sample Nemotron configuration."""
        return {
            "architectures": ["NemotronForCausalLM"],
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "relu2",
            "hidden_size": 3072,
            "initializer_range": 0.0134,
            "intermediate_size": 9216,
            "max_position_embeddings": 4096,
            "model_type": "nemotron",
            "norm_eps": 1e-05,
            "num_attention_heads": 24,
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "partial_rotary_factor": 0.5,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "use_cache": True,
            "vocab_size": 256000,
        }

    @pytest.fixture
    def nemotron_config(self, nemotron_config_dict):
        """Create a NemotronConfig instance."""
        return NemotronConfig(**nemotron_config_dict)

    @pytest.fixture
    def mock_nemotron_model(self, nemotron_config):
        """Create a mock NemotronForCausalLM model."""
        mock_model = Mock(spec=NemotronForCausalLM)
        mock_model.config = nemotron_config
        mock_model.dtype = torch.bfloat16
        return mock_model

    @pytest.fixture
    def mock_pretrained_nemotron(self, nemotron_config):
        """Create a mock PreTrainedCausalLM with Nemotron model."""
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = nemotron_config
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)
        mock_pretrained.model = Mock(spec=NemotronForCausalLM)
        mock_pretrained.model.dtype = torch.bfloat16
        return mock_pretrained

    def test_bridge_registration(self):
        """Test that NemotronBridge is properly registered."""
        # The @MegatronModelBridge.register_bridge decorator should register the bridge
        # Check that the class exists and has the expected base class
        assert issubclass(NemotronBridge, MegatronModelBridge)

    def test_provider_bridge_basic(self, mock_pretrained_nemotron, nemotron_config):
        """Test basic provider_bridge functionality."""
        bridge = NemotronBridge()
        provider = bridge.provider_bridge(mock_pretrained_nemotron)

        # Verify that provider is of correct type
        assert isinstance(provider, NemotronModelProvider)

        # Check that key configuration values are correctly mapped
        assert provider.num_layers == nemotron_config.num_hidden_layers
        assert provider.hidden_size == nemotron_config.hidden_size
        assert provider.ffn_hidden_size == nemotron_config.intermediate_size
        assert provider.num_attention_heads == nemotron_config.num_attention_heads
        assert provider.num_query_groups == nemotron_config.num_key_value_heads
        assert provider.seq_length == nemotron_config.max_position_embeddings
        assert provider.layernorm_epsilon == nemotron_config.norm_eps
        assert provider.rotary_base == nemotron_config.rope_theta
        assert provider.rotary_percent == nemotron_config.partial_rotary_factor
        assert provider.vocab_size == nemotron_config.vocab_size
        assert provider.share_embeddings_and_output_weights == nemotron_config.tie_word_embeddings

    def test_mapping_registry(self, mock_pretrained_nemotron):
        """Test that mapping_registry returns proper mappings."""
        bridge = NemotronBridge()
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
        assert "decoder.final_layernorm.bias" in mapping_dict

    def test_dtype_configuration(self, mock_pretrained_nemotron):
        """Test that data type configuration works correctly."""
        bridge = NemotronBridge()
        provider = bridge.provider_bridge(mock_pretrained_nemotron)

        # Check that bfloat16 is correctly detected from config
        assert provider.bf16 is True
        assert provider.fp16 is False
        assert provider.params_dtype == torch.bfloat16
