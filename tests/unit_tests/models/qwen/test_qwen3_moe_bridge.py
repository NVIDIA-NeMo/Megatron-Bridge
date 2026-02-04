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

"""
Unit tests for Qwen3 MoE bridge functionality.
"""

from unittest.mock import Mock

import pytest
import torch
from transformers import GenerationConfig

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.qwen.qwen3_moe_bridge import Qwen3MoEBridge


class TestQwen3MoEBridge:
    """Test cases for Qwen3MoEBridge class."""

    @pytest.fixture
    def qwen3_moe_30b_config_dict(self):
        """Create a sample Qwen3 MoE 30B configuration matching the expected model structure."""
        return {
            "architectures": ["Qwen3MoeForCausalLM"],
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "hidden_act": "silu",
            "hidden_size": 2048,
            "initializer_range": 0.02,
            "intermediate_size": 6144,
            "max_position_embeddings": 40960,
            "model_type": "qwen3_moe",
            "num_attention_heads": 32,
            "num_hidden_layers": 48,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-06,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "use_cache": True,
            "vocab_size": 151936,
            # MoE specific parameters
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 768,
        }

    @pytest.fixture
    def qwen3_moe_235b_config_dict(self):
        """Create a sample Qwen3 MoE 235B configuration matching the expected model structure."""
        return {
            "architectures": ["Qwen3MoeForCausalLM"],
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 12288,
            "max_position_embeddings": 40960,
            "model_type": "qwen3_moe",
            "num_attention_heads": 64,
            "num_hidden_layers": 94,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-06,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "use_cache": True,
            "vocab_size": 151936,
            # MoE specific parameters
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 1536,
        }

    @pytest.fixture
    def mock_qwen3_moe_config(self, qwen3_moe_30b_config_dict):
        """Create a mock Qwen3 MoE configuration.

        Uses spec=list to make getattr return None for undefined attributes
        instead of Mock objects, which would incorrectly be passed to GPTModelProvider.
        """
        config = Mock(spec=[])
        for key, value in qwen3_moe_30b_config_dict.items():
            setattr(config, key, value)
        return config

    @pytest.fixture
    def mock_pretrained_qwen3_moe(self, mock_qwen3_moe_config):
        """Create a mock PreTrainedCausalLM with Qwen3 MoE model."""
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mock_qwen3_moe_config
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)
        mock_pretrained.model = Mock()
        mock_pretrained.model.dtype = torch.bfloat16
        return mock_pretrained

    def test_bridge_registration(self):
        """Test that Qwen3MoEBridge is properly registered."""
        # The @MegatronModelBridge.register_bridge decorator should register the bridge
        # Check that the class exists and has the expected base class
        assert issubclass(Qwen3MoEBridge, MegatronModelBridge)

    def test_provider_bridge_basic(self, mock_pretrained_qwen3_moe, mock_qwen3_moe_config):
        """Test basic provider_bridge functionality."""
        bridge = Qwen3MoEBridge()

        # Call provider_bridge
        result = bridge.provider_bridge(mock_pretrained_qwen3_moe)

        # Check that it returns a GPTModelProvider instance (after refactoring)
        assert isinstance(result, GPTModelProvider)

        # Check basic configuration mapping
        assert result.num_layers == mock_qwen3_moe_config.num_hidden_layers
        assert result.hidden_size == mock_qwen3_moe_config.hidden_size
        assert result.num_attention_heads == mock_qwen3_moe_config.num_attention_heads
        assert result.seq_length == mock_qwen3_moe_config.max_position_embeddings
        assert result.rotary_base == mock_qwen3_moe_config.rope_theta

    def test_provider_bridge_vocabulary(self, mock_pretrained_qwen3_moe, mock_qwen3_moe_config):
        """Test vocabulary size mapping."""
        bridge = Qwen3MoEBridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_moe)

        # Check vocabulary configuration
        assert result.vocab_size == mock_qwen3_moe_config.vocab_size
        assert result.share_embeddings_and_output_weights == mock_qwen3_moe_config.tie_word_embeddings

    def test_provider_bridge_attention_config(self, mock_pretrained_qwen3_moe, mock_qwen3_moe_config):
        """Test attention configuration mapping."""
        bridge = Qwen3MoEBridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_moe)

        # Check attention configuration
        assert result.num_attention_heads == mock_qwen3_moe_config.num_attention_heads
        assert result.num_query_groups == mock_qwen3_moe_config.num_key_value_heads
        assert result.qk_layernorm is True  # Qwen3 MoE uses QK layernorm

    def test_provider_bridge_mlp_config(self, mock_pretrained_qwen3_moe, mock_qwen3_moe_config):
        """Test MLP configuration mapping."""
        bridge = Qwen3MoEBridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_moe)

        # Check MLP configuration
        assert result.ffn_hidden_size == mock_qwen3_moe_config.intermediate_size
        assert result.moe_ffn_hidden_size == mock_qwen3_moe_config.moe_intermediate_size
        assert result.gated_linear_unit is True  # Qwen3 MoE uses gated linear units

    def test_provider_bridge_moe_config(self, mock_pretrained_qwen3_moe, mock_qwen3_moe_config):
        """Test MoE-specific configuration mapping."""
        bridge = Qwen3MoEBridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_moe)

        # Check MoE-specific configuration
        assert result.num_moe_experts == mock_qwen3_moe_config.num_experts
        assert result.moe_router_topk == mock_qwen3_moe_config.num_experts_per_tok
        assert result.moe_grouped_gemm is True

    def test_provider_bridge_normalization(self, mock_pretrained_qwen3_moe, mock_qwen3_moe_config):
        """Test normalization configuration."""
        bridge = Qwen3MoEBridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_moe)

        # Check normalization settings
        assert result.layernorm_epsilon == mock_qwen3_moe_config.rms_norm_eps
        assert result.init_method_std == mock_qwen3_moe_config.initializer_range

    def test_provider_bridge_position_embedding(self, mock_pretrained_qwen3_moe, mock_qwen3_moe_config):
        """Test position embedding configuration."""
        bridge = Qwen3MoEBridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_moe)

        # Check position embedding
        assert result.rotary_base == mock_qwen3_moe_config.rope_theta
        assert result.position_embedding_type == "rope"

    def test_provider_bridge_dtype_handling(self, qwen3_moe_30b_config_dict):
        """Test dtype handling in provider_bridge."""
        # Test with bfloat16
        config = Mock(spec=[])
        for key, value in qwen3_moe_30b_config_dict.items():
            setattr(config, key, value)
        config.torch_dtype = "bfloat16"

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = config
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)

        bridge = Qwen3MoEBridge()
        result = bridge.provider_bridge(mock_pretrained)

        assert result.bf16 is True
        assert result.fp16 is False
        assert result.params_dtype == torch.bfloat16

        # Test with float16
        config.torch_dtype = "float16"
        result = bridge.provider_bridge(mock_pretrained)

        assert result.fp16 is True
        assert result.bf16 is False
        assert result.params_dtype == torch.float16

    def test_provider_bridge_generation_config(self, mock_pretrained_qwen3_moe):
        """Test generation config mapping."""
        bridge = Qwen3MoEBridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_moe)

        # Check that generation config is passed through
        assert result.generation_config == mock_pretrained_qwen3_moe.generation_config

    def test_provider_bridge_tie_word_embeddings_true(self, mock_qwen3_moe_config):
        """Test provider_bridge with tie_word_embeddings=True."""
        mock_qwen3_moe_config.tie_word_embeddings = True

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mock_qwen3_moe_config
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)

        bridge = Qwen3MoEBridge()
        result = bridge.provider_bridge(mock_pretrained)

        assert result.share_embeddings_and_output_weights is True

    def test_provider_bridge_tie_word_embeddings_false(self, mock_qwen3_moe_config):
        """Test provider_bridge with tie_word_embeddings=False."""
        mock_qwen3_moe_config.tie_word_embeddings = False

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mock_qwen3_moe_config
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)

        bridge = Qwen3MoEBridge()
        result = bridge.provider_bridge(mock_pretrained)

        assert result.share_embeddings_and_output_weights is False

    def test_provider_bridge_missing_tie_word_embeddings(self, mock_qwen3_moe_config):
        """Test provider_bridge when tie_word_embeddings is missing."""
        # Remove tie_word_embeddings attribute
        if hasattr(mock_qwen3_moe_config, "tie_word_embeddings"):
            delattr(mock_qwen3_moe_config, "tie_word_embeddings")

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mock_qwen3_moe_config
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)

        bridge = Qwen3MoEBridge()
        result = bridge.provider_bridge(mock_pretrained)

        # GPTModelProvider defaults share_embeddings_and_output_weights to True
        assert result.share_embeddings_and_output_weights is True

    def test_provider_bridge_235b_config(self, qwen3_moe_235b_config_dict):
        """Test provider_bridge with Qwen3 MoE 235B configuration."""
        config = Mock(spec=[])
        for key, value in qwen3_moe_235b_config_dict.items():
            setattr(config, key, value)

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = config
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)

        bridge = Qwen3MoEBridge()
        result = bridge.provider_bridge(mock_pretrained)

        # Check 235B-specific configuration
        assert result.num_layers == 94
        assert result.hidden_size == 4096
        assert result.num_attention_heads == 64
        assert result.ffn_hidden_size == 12288
        assert result.moe_ffn_hidden_size == 1536

    def test_mapping_registry(self):
        """Test mapping_registry returns valid mappings."""
        bridge = Qwen3MoEBridge()

        registry = bridge.mapping_registry()

        # Check that registry is not None and has mappings
        assert registry is not None
        assert len(registry.mappings) > 0

        # Check for expected mapping types
        mapping_types = [type(mapping).__name__ for mapping in registry.mappings]
        assert "AutoMapping" in mapping_types
        assert "QKVMapping" in mapping_types
        assert "GatedMLPMapping" in mapping_types

    def test_mapping_registry_parameter_mappings(self):
        """Test that mapping_registry contains expected parameter mappings."""
        bridge = Qwen3MoEBridge()

        registry = bridge.mapping_registry()

        # Extract all AutoMapping instances
        auto_mappings = [m for m in registry.mappings if type(m).__name__ == "AutoMapping"]

        # Check for critical parameter mappings
        hf_params = [mapping.hf_param for mapping in auto_mappings]
        megatron_params = [mapping.megatron_param for mapping in auto_mappings]

        # Should have embedding mappings
        assert "model.embed_tokens.weight" in hf_params
        assert "embedding.word_embeddings.weight" in megatron_params

        # Should have output layer mappings
        assert "lm_head.weight" in hf_params
        assert "output_layer.weight" in megatron_params

        # Should have layer norm mappings
        assert "model.norm.weight" in hf_params
        assert "decoder.final_layernorm.weight" in megatron_params

    def test_mapping_registry_qkv_mapping(self):
        """Test that mapping_registry contains QKV mapping."""
        bridge = Qwen3MoEBridge()

        registry = bridge.mapping_registry()

        # Extract QKVMapping instances
        qkv_mappings = [m for m in registry.mappings if type(m).__name__ == "QKVMapping"]

        # Should have at least one QKV mapping
        assert len(qkv_mappings) > 0

        # Check the QKV mapping structure
        qkv_mapping = qkv_mappings[0]
        assert hasattr(qkv_mapping, "hf_param")
        assert isinstance(qkv_mapping.hf_param, dict)
        assert "q" in qkv_mapping.hf_param
        assert "k" in qkv_mapping.hf_param
        assert "v" in qkv_mapping.hf_param
        assert hasattr(qkv_mapping, "megatron_param")

    def test_mapping_registry_moe_mappings(self):
        """Test that mapping_registry contains MoE-specific mappings."""
        bridge = Qwen3MoEBridge()

        registry = bridge.mapping_registry()

        # Extract all mappings
        auto_mappings = [m for m in registry.mappings if type(m).__name__ == "AutoMapping"]
        gated_mlp_mappings = [m for m in registry.mappings if type(m).__name__ == "GatedMLPMapping"]

        # Check for MoE router mapping
        hf_params = [mapping.hf_param for mapping in auto_mappings]
        assert "model.layers.*.mlp.gate.weight" in hf_params

        # Check for expert mappings in GatedMLPMapping
        assert len(gated_mlp_mappings) > 0

        # Check expert down projection mapping
        expert_down_params = [
            mapping.hf_param
            for mapping in auto_mappings
            if "experts" in mapping.hf_param and "down_proj" in mapping.hf_param
        ]
        assert len(expert_down_params) > 0
