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

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mixtral.mixtral_bridge import MixtralBridge
from megatron.bridge.models.mixtral.mixtral_provider import MixtralModelProvider


@pytest.fixture
def mixtral_8x7b_config_dict():
    """Create a sample Mixtral 8x7B configuration."""
    return {
        "architectures": ["MixtralForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 14336,
        "max_position_embeddings": 32768,
        "model_type": "mixtral",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "rms_norm_eps": 1e-05,
        "rope_theta": 1000000.0,
        "router_aux_loss_coef": 0.01,
        "sliding_window": None,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.40.0",
        "use_cache": True,
        "vocab_size": 32000,
    }


@pytest.fixture
def mixtral_8x22b_config_dict():
    """Create a sample Mixtral 8x22B configuration."""
    return {
        "architectures": ["MixtralForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 6144,
        "initializer_range": 0.02,
        "intermediate_size": 16384,
        "max_position_embeddings": 65536,
        "model_type": "mixtral",
        "num_attention_heads": 48,
        "num_hidden_layers": 56,
        "num_key_value_heads": 8,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "rms_norm_eps": 1e-05,
        "rope_theta": 1000000.0,
        "router_aux_loss_coef": 0.01,
        "sliding_window": None,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.40.0",
        "use_cache": True,
        "vocab_size": 32768,
    }


class TestMixtralBridge:
    """Test cases for MixtralBridge class."""

    @pytest.fixture
    def mixtral_8x7b_config(self, mixtral_8x7b_config_dict):
        """Create a Mixtral config instance for 8x7B model."""
        config = Mock()
        for key, value in mixtral_8x7b_config_dict.items():
            setattr(config, key, value)
        return config

    @pytest.fixture
    def mixtral_8x22b_config(self, mixtral_8x22b_config_dict):
        """Create a Mixtral config instance for 8x22B model."""
        config = Mock()
        for key, value in mixtral_8x22b_config_dict.items():
            setattr(config, key, value)
        return config

    @pytest.fixture
    def mock_pretrained_mixtral_8x7b(self, mixtral_8x7b_config):
        """Create a mock PreTrainedCausalLM with Mixtral 8x7B model."""
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mixtral_8x7b_config
        mock_pretrained.generation_config = Mock()
        return mock_pretrained

    @pytest.fixture
    def mock_pretrained_mixtral_8x22b(self, mixtral_8x22b_config):
        """Create a mock PreTrainedCausalLM with Mixtral 8x22B model."""
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mixtral_8x22b_config
        mock_pretrained.generation_config = Mock()
        return mock_pretrained

    def test_bridge_registration(self):
        """Test that MixtralBridge is properly registered."""
        assert issubclass(MixtralBridge, MegatronModelBridge)

    def test_provider_bridge_basic_8x7b(self, mock_pretrained_mixtral_8x7b, mixtral_8x7b_config):
        """Test basic provider_bridge functionality for Mixtral 8x7B."""
        bridge = MixtralBridge()

        result = bridge.provider_bridge(mock_pretrained_mixtral_8x7b)

        assert isinstance(result, MixtralModelProvider)

        # Check basic configuration mapping
        assert result.num_layers == mixtral_8x7b_config.num_hidden_layers
        assert result.hidden_size == mixtral_8x7b_config.hidden_size
        assert result.num_attention_heads == mixtral_8x7b_config.num_attention_heads
        assert result.seq_length == mixtral_8x7b_config.max_position_embeddings
        assert result.rotary_base == mixtral_8x7b_config.rope_theta

    def test_provider_bridge_basic_8x22b(self, mock_pretrained_mixtral_8x22b, mixtral_8x22b_config):
        """Test basic provider_bridge functionality for Mixtral 8x22B."""
        bridge = MixtralBridge()

        result = bridge.provider_bridge(mock_pretrained_mixtral_8x22b)

        assert isinstance(result, MixtralModelProvider)

        # Check basic configuration mapping
        assert result.num_layers == mixtral_8x22b_config.num_hidden_layers
        assert result.hidden_size == mixtral_8x22b_config.hidden_size
        assert result.num_attention_heads == mixtral_8x22b_config.num_attention_heads

    def test_provider_bridge_vocabulary(self, mock_pretrained_mixtral_8x7b, mixtral_8x7b_config):
        """Test vocabulary size mapping."""
        bridge = MixtralBridge()

        result = bridge.provider_bridge(mock_pretrained_mixtral_8x7b)

        assert result.vocab_size == mixtral_8x7b_config.vocab_size
        assert result.share_embeddings_and_output_weights == mixtral_8x7b_config.tie_word_embeddings

    def test_provider_bridge_attention_config(self, mock_pretrained_mixtral_8x7b, mixtral_8x7b_config):
        """Test attention configuration mapping."""
        bridge = MixtralBridge()

        result = bridge.provider_bridge(mock_pretrained_mixtral_8x7b)

        assert result.num_attention_heads == mixtral_8x7b_config.num_attention_heads
        assert result.num_query_groups == mixtral_8x7b_config.num_key_value_heads
        assert result.add_qkv_bias == mixtral_8x7b_config.attention_bias

    def test_provider_bridge_mlp_config(self, mock_pretrained_mixtral_8x7b, mixtral_8x7b_config):
        """Test MLP configuration mapping."""
        bridge = MixtralBridge()

        result = bridge.provider_bridge(mock_pretrained_mixtral_8x7b)

        assert result.ffn_hidden_size == mixtral_8x7b_config.intermediate_size
        assert result.moe_ffn_hidden_size == mixtral_8x7b_config.intermediate_size
        assert result.gated_linear_unit is True  # Mixtral uses gated MLP

    def test_provider_bridge_moe_config(self, mock_pretrained_mixtral_8x7b, mixtral_8x7b_config):
        """Test MoE configuration mapping."""
        bridge = MixtralBridge()

        result = bridge.provider_bridge(mock_pretrained_mixtral_8x7b)

        assert result.num_moe_experts == mixtral_8x7b_config.num_local_experts
        assert result.moe_router_topk == mixtral_8x7b_config.num_experts_per_tok
        assert result.moe_aux_loss_coeff == mixtral_8x7b_config.router_aux_loss_coef

    def test_provider_bridge_normalization(self, mock_pretrained_mixtral_8x7b, mixtral_8x7b_config):
        """Test normalization configuration."""
        bridge = MixtralBridge()

        result = bridge.provider_bridge(mock_pretrained_mixtral_8x7b)

        assert result.layernorm_epsilon == mixtral_8x7b_config.rms_norm_eps

    def test_provider_bridge_position_embedding(self, mock_pretrained_mixtral_8x7b, mixtral_8x7b_config):
        """Test position embedding configuration."""
        bridge = MixtralBridge()

        result = bridge.provider_bridge(mock_pretrained_mixtral_8x7b)

        assert result.rotary_base == mixtral_8x7b_config.rope_theta

    def test_provider_bridge_dtype_handling(self, mixtral_8x7b_config):
        """Test dtype handling in provider_bridge."""
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mixtral_8x7b_config
        mock_pretrained.generation_config = Mock()

        bridge = MixtralBridge()
        result = bridge.provider_bridge(mock_pretrained)

        assert result.params_dtype == torch.bfloat16
        assert result.bf16 is True
        assert result.fp16 is False

    def test_provider_bridge_fp16_dtype_handling(self, mixtral_8x7b_config):
        """Test FP16 dtype handling in provider_bridge."""
        mixtral_8x7b_config.torch_dtype = "float16"

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mixtral_8x7b_config
        mock_pretrained.generation_config = Mock()

        bridge = MixtralBridge()
        result = bridge.provider_bridge(mock_pretrained)

        assert result.params_dtype == torch.float16
        assert result.fp16 is True
        assert result.bf16 is False

    def test_provider_bridge_without_tie_embeddings(self, mixtral_8x7b_config):
        """Test provider_bridge when tie_word_embeddings is not present."""
        delattr(mixtral_8x7b_config, "tie_word_embeddings")

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mixtral_8x7b_config
        mock_pretrained.generation_config = None

        bridge = MixtralBridge()
        result = bridge.provider_bridge(mock_pretrained)

        # Should default to False when tie_word_embeddings is not present
        assert result.share_embeddings_and_output_weights is False

    def test_provider_bridge_without_attention_bias(self, mixtral_8x7b_config):
        """Test provider_bridge when attention_bias is not present."""
        delattr(mixtral_8x7b_config, "attention_bias")

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mixtral_8x7b_config
        mock_pretrained.generation_config = None

        bridge = MixtralBridge()
        result = bridge.provider_bridge(mock_pretrained)

        # Should default to False when attention_bias is not present
        assert result.add_qkv_bias is False

    def test_mapping_registry_implementation(self):
        """Test that mapping_registry returns a proper MegatronMappingRegistry."""
        bridge = MixtralBridge()

        mapping_registry = bridge.mapping_registry()

        assert mapping_registry is not None

    def test_provider_bridge_generation_config(self, mock_pretrained_mixtral_8x7b):
        """Test that generation config is passed through."""
        bridge = MixtralBridge()

        result = bridge.provider_bridge(mock_pretrained_mixtral_8x7b)

        assert result.generation_config == mock_pretrained_mixtral_8x7b.generation_config

    def test_provider_bridge_init_method_std(self, mock_pretrained_mixtral_8x7b, mixtral_8x7b_config):
        """Test initializer range configuration."""
        bridge = MixtralBridge()

        result = bridge.provider_bridge(mock_pretrained_mixtral_8x7b)

        assert result.init_method_std == mixtral_8x7b_config.initializer_range


class TestMixtralBridgeMoEFeatures:
    """Test Mixtral-specific MoE features in the bridge."""

    @pytest.fixture
    def mixtral_config_with_custom_moe(self):
        """Create a Mixtral config with custom MoE settings."""
        config = Mock()
        config.hidden_size = 4096
        config.num_attention_heads = 32
        config.num_hidden_layers = 32
        config.num_key_value_heads = 8
        config.intermediate_size = 14336
        config.max_position_embeddings = 32768
        config.num_local_experts = 16  # Custom
        config.num_experts_per_tok = 4  # Custom
        config.router_aux_loss_coef = 0.02  # Custom
        config.vocab_size = 32000
        config.initializer_range = 0.02
        config.rms_norm_eps = 1e-5
        config.rope_theta = 1000000.0
        config.torch_dtype = "bfloat16"
        config.tie_word_embeddings = False
        config.attention_bias = False
        return config

    def test_provider_bridge_custom_moe_config(self, mixtral_config_with_custom_moe):
        """Test provider_bridge with custom MoE configuration."""
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mixtral_config_with_custom_moe
        mock_pretrained.generation_config = Mock()

        bridge = MixtralBridge()
        result = bridge.provider_bridge(mock_pretrained)

        assert result.num_moe_experts == 16
        assert result.moe_router_topk == 4
        assert result.moe_aux_loss_coeff == 0.02

    def test_provider_bridge_moe_default_settings(self, mixtral_config_with_custom_moe):
        """Test that MoE default settings are correctly applied."""
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mixtral_config_with_custom_moe
        mock_pretrained.generation_config = Mock()

        bridge = MixtralBridge()
        result = bridge.provider_bridge(mock_pretrained)

        # Check inherited MoE defaults from MixtralModelProvider
        assert result.moe_token_dispatcher_type == "alltoall"
        assert result.moe_grouped_gemm is True
        assert result.moe_permute_fusion is True


class TestAutoBridgeIntegration:
    """Integration tests for AutoBridge with Mixtral models."""

    def test_supports_mixtral_architectures(self, mixtral_8x7b_config_dict, mixtral_8x22b_config_dict):
        """Test that AutoBridge.supports correctly identifies Mixtral models."""
        from megatron.bridge.models import AutoBridge

        # Test Mixtral 8x7B
        config_8x7b = Mock()
        for key, value in mixtral_8x7b_config_dict.items():
            setattr(config_8x7b, key, value)
        assert AutoBridge.supports(config_8x7b)

        # Test Mixtral 8x22B
        config_8x22b = Mock()
        for key, value in mixtral_8x22b_config_dict.items():
            setattr(config_8x22b, key, value)
        assert AutoBridge.supports(config_8x22b)

        # Test non-causal LM architecture
        non_causal_config = Mock()
        non_causal_config.architectures = ["MixtralModel"]  # Not ForCausalLM
        assert not AutoBridge.supports(non_causal_config)
