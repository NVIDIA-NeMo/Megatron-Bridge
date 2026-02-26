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
Unit tests for Qwen3 Next bridge functionality.
"""

from unittest.mock import Mock

import pytest
import torch

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.qwen.qwen3_next_bridge import Qwen3NextBridge
from megatron.bridge.models.qwen.qwen_provider import Qwen3NextModelProvider


class TestQwen3NextBridge:
    """Test cases for Qwen3NextBridge class."""

    @pytest.fixture
    def qwen3_next_80b_a3b_config_dict(self):
        """Create a sample Qwen3 Next 80B configuration matching the expected model structure."""
        return {
            "architectures": ["Qwen3NextForCausalLM"],
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "decoder_sparse_step": 1,
            "eos_token_id": 151645,
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": 2048,
            "initializer_range": 0.02,
            "intermediate_size": 5120,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "max_position_embeddings": 262144,
            "mlp_only_layers": [],
            "model_type": "qwen3_next",
            "moe_intermediate_size": 512,
            "norm_topk_prob": True,
            "num_attention_heads": 16,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "num_hidden_layers": 48,
            "num_key_value_heads": 2,
            "output_router_logits": False,
            "partial_rotary_factor": 0.25,
            "rms_norm_eps": 1e-06,
            "rope_scaling": None,
            "rope_theta": 10000000,
            "router_aux_loss_coef": 0.001,
            "shared_expert_intermediate_size": 512,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "use_cache": True,
            "use_sliding_window": False,
            "vocab_size": 151936,
        }

    @pytest.fixture
    def mock_qwen3_next_config(self, qwen3_next_80b_a3b_config_dict):
        """Create a mock Qwen3 Next configuration."""
        config = Mock()
        for key, value in qwen3_next_80b_a3b_config_dict.items():
            setattr(config, key, value)
        return config

    @pytest.fixture
    def mock_pretrained_qwen3_next(self, mock_qwen3_next_config):
        """Create a mock PreTrainedCausalLM with Qwen3 Next model."""
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mock_qwen3_next_config
        mock_pretrained.model = Mock()
        mock_pretrained.model.dtype = torch.bfloat16
        return mock_pretrained

    def test_bridge_registration(self):
        """Test that Qwen3NextBridge is properly registered."""
        # The @MegatronModelBridge.register_bridge decorator should register the bridge
        # Check that the class exists and has the expected base class
        assert issubclass(Qwen3NextBridge, MegatronModelBridge)

    def test_provider_bridge_basic(self, mock_pretrained_qwen3_next, mock_qwen3_next_config):
        """Test basic provider_bridge functionality."""
        bridge = Qwen3NextBridge()

        # Call provider_bridge
        result = bridge.provider_bridge(mock_pretrained_qwen3_next)

        # Check that it returns a Qwen3NextModelProvider instance
        assert isinstance(result, Qwen3NextModelProvider)

        # Check basic configuration mapping
        assert result.num_layers == mock_qwen3_next_config.num_hidden_layers
        assert result.hidden_size == mock_qwen3_next_config.hidden_size
        assert result.num_attention_heads == mock_qwen3_next_config.num_attention_heads
        assert result.seq_length == mock_qwen3_next_config.max_position_embeddings
        assert result.rotary_base == mock_qwen3_next_config.rope_theta

    def test_provider_bridge_vocabulary(self, mock_pretrained_qwen3_next, mock_qwen3_next_config):
        """Test vocabulary size mapping."""
        bridge = Qwen3NextBridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_next)

        # Check vocabulary configuration
        assert result.vocab_size == mock_qwen3_next_config.vocab_size
        assert result.share_embeddings_and_output_weights == mock_qwen3_next_config.tie_word_embeddings

    def test_provider_bridge_attention_config(self, mock_pretrained_qwen3_next, mock_qwen3_next_config):
        """Test attention configuration mapping."""
        bridge = Qwen3NextBridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_next)

        # Check attention configuration
        assert result.num_attention_heads == mock_qwen3_next_config.num_attention_heads
        assert result.num_query_groups == mock_qwen3_next_config.num_key_value_heads
        assert result.qk_layernorm is True  # Qwen3 Next uses QK layernorm
        assert result.layernorm_zero_centered_gamma is True
        assert result.attention_output_gate is True

    def test_provider_bridge_linear_attention_config(self, mock_pretrained_qwen3_next, mock_qwen3_next_config):
        """Test linear attention configuration mapping."""
        bridge = Qwen3NextBridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_next)

        # Check linear attention configuration
        if isinstance(result.linear_attention_freq, int):
            assert result.linear_attention_freq == mock_qwen3_next_config.full_attention_interval
        else:
            assert isinstance(result.linear_attention_freq, list)
            for i in range(result.num_layers):
                if (i + 1) % mock_qwen3_next_config.full_attention_interval == 0:
                    assert result.linear_attention_freq[i] == 0
                else:
                    assert result.linear_attention_freq[i] == 1
        assert result.linear_conv_kernel_dim == mock_qwen3_next_config.linear_conv_kernel_dim
        assert result.linear_key_head_dim == mock_qwen3_next_config.linear_key_head_dim
        assert result.linear_value_head_dim == mock_qwen3_next_config.linear_value_head_dim
        assert result.linear_num_key_heads == mock_qwen3_next_config.linear_num_key_heads
        assert result.linear_num_value_heads == mock_qwen3_next_config.linear_num_value_heads
        assert result.experimental_attention_variant == "gated_delta_net"

    def test_provider_bridge_mlp_config(self, mock_pretrained_qwen3_next, mock_qwen3_next_config):
        """Test MLP configuration mapping."""
        bridge = Qwen3NextBridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_next)

        # Check MLP configuration
        assert result.ffn_hidden_size == mock_qwen3_next_config.intermediate_size
        assert result.moe_ffn_hidden_size == mock_qwen3_next_config.moe_intermediate_size
        assert result.gated_linear_unit is True  # Qwen3 Next uses gated linear units

    def test_provider_bridge_moe_config(self, mock_pretrained_qwen3_next, mock_qwen3_next_config):
        """Test MoE-specific configuration mapping."""
        bridge = Qwen3NextBridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_next)

        # Check MoE-specific configuration
        assert result.num_moe_experts == mock_qwen3_next_config.num_experts
        assert result.moe_router_topk == mock_qwen3_next_config.num_experts_per_tok
        assert result.moe_grouped_gemm is True
        assert result.moe_shared_expert_intermediate_size == mock_qwen3_next_config.shared_expert_intermediate_size
        assert result.moe_shared_expert_gate is True

    def test_provider_bridge_normalization(self, mock_pretrained_qwen3_next, mock_qwen3_next_config):
        """Test normalization configuration."""
        bridge = Qwen3NextBridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_next)

        # Check normalization settings
        assert result.layernorm_epsilon == mock_qwen3_next_config.rms_norm_eps
        assert result.init_method_std == mock_qwen3_next_config.initializer_range

    def test_provider_bridge_position_embedding(self, mock_pretrained_qwen3_next, mock_qwen3_next_config):
        """Test position embedding configuration."""
        bridge = Qwen3NextBridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_next)

        # Check position embedding
        assert result.rotary_base == mock_qwen3_next_config.rope_theta
        assert result.rotary_percent == mock_qwen3_next_config.partial_rotary_factor
        assert result.position_embedding_type == "rope"

    def test_provider_bridge_mtp_config(self, mock_pretrained_qwen3_next, mock_qwen3_next_config):
        """Test MTP configuration mapping."""
        bridge = Qwen3NextBridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_next)

        # Check MTP configuration
        assert not result.mtp_num_layers

    def test_provider_bridge_dtype_handling(self, qwen3_next_80b_a3b_config_dict):
        """Test dtype handling in provider_bridge."""
        # Test with bfloat16
        config = Mock()
        for key, value in qwen3_next_80b_a3b_config_dict.items():
            setattr(config, key, value)
        config.torch_dtype = "bfloat16"

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = config

        bridge = Qwen3NextBridge()
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

    def test_provider_bridge_tie_word_embeddings_true(self, mock_qwen3_next_config):
        """Test provider_bridge with tie_word_embeddings=True."""
        mock_qwen3_next_config.tie_word_embeddings = True

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mock_qwen3_next_config

        bridge = Qwen3NextBridge()
        result = bridge.provider_bridge(mock_pretrained)

        assert result.share_embeddings_and_output_weights is True

    def test_provider_bridge_tie_word_embeddings_false(self, mock_qwen3_next_config):
        """Test provider_bridge with tie_word_embeddings=False."""
        mock_qwen3_next_config.tie_word_embeddings = False

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mock_qwen3_next_config

        bridge = Qwen3NextBridge()
        result = bridge.provider_bridge(mock_pretrained)

        assert result.share_embeddings_and_output_weights is False

    def test_provider_bridge_missing_tie_word_embeddings(self, mock_qwen3_next_config):
        """Test provider_bridge when tie_word_embeddings is missing."""
        # Remove tie_word_embeddings attribute
        if hasattr(mock_qwen3_next_config, "tie_word_embeddings"):
            delattr(mock_qwen3_next_config, "tie_word_embeddings")

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mock_qwen3_next_config

        bridge = Qwen3NextBridge()
        result = bridge.provider_bridge(mock_pretrained)

        # Should default to False when missing
        assert result.share_embeddings_and_output_weights is False

    def test_provider_bridge_80b_a3b_config(self, qwen3_next_80b_a3b_config_dict):
        """Test provider_bridge with Qwen3 Next 80B A3B configuration."""
        config = Mock()
        for key, value in qwen3_next_80b_a3b_config_dict.items():
            setattr(config, key, value)

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = config

        bridge = Qwen3NextBridge()
        result = bridge.provider_bridge(mock_pretrained)

        # Check 80B-A3B-specific configuration
        assert result.num_layers == 48
        assert result.hidden_size == 2048
        assert result.num_attention_heads == 16
        assert result.ffn_hidden_size == 5120
        assert result.moe_ffn_hidden_size == 512

    def test_mapping_registry(self):
        """Test mapping_registry returns valid mappings."""
        bridge = Qwen3NextBridge()

        registry = bridge.mapping_registry()

        # Check that registry is not None and has mappings
        assert registry is not None
        assert len(registry.mappings) > 0

        # Check for expected mapping types
        mapping_types = [type(mapping).__name__ for mapping in registry.mappings]
        assert "AutoMapping" in mapping_types
        assert "QKVMapping" in mapping_types
        assert "GatedMLPMapping" in mapping_types
        assert "GDNLinearMapping" in mapping_types
        assert "RMSNorm2ZeroCenteredRMSNormMapping" in mapping_types

    def test_mapping_registry_parameter_mappings(self):
        """Test that mapping_registry contains expected parameter mappings."""
        bridge = Qwen3NextBridge()

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

    def test_mapping_registry_mtp_mapping(self):
        """Test that mapping_registry contains MTP mapping."""
        bridge = Qwen3NextBridge()

        registry = bridge.mapping_registry()

        # Extract MTP mappings
        auto_mappings = [m for m in registry.mappings if type(m).__name__ == "AutoMapping"]

        # Check for critical parameter mappings
        hf_params = [mapping.hf_param for mapping in auto_mappings]
        megatron_params = [mapping.megatron_param for mapping in auto_mappings]

        # Should have embedding and hidden projection
        assert "mtp.fc.weight" in hf_params
        assert "mtp.layers.0.eh_proj.weight" in megatron_params

        # Should have pre-fc norms for embedding and hidden
        assert "mtp.pre_fc_norm_embedding.weight" in hf_params
        assert "mtp.pre_fc_norm_hidden.weight" in hf_params
        assert "mtp.layers.0.enorm.weight" in megatron_params
        assert "mtp.layers.0.hnorm.weight" in megatron_params

        # Should have final layernorm
        assert "mtp.norm.weight" in hf_params
        assert "mtp.layers.0.final_layernorm.weight" in megatron_params

    def test_mapping_registry_qkv_mapping(self):
        """Test that mapping_registry contains QKV mapping."""
        bridge = Qwen3NextBridge()

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

    def test_mapping_registry_gdn_linear_mapping(self):
        """Test that mapping_registry contains GDN linear mapping."""
        bridge = Qwen3NextBridge()

        registry = bridge.mapping_registry()

        # Extract GDNLinearMapping instances
        gdn_linear_mappings = [m for m in registry.mappings if type(m).__name__ == "GDNLinearMapping"]
        assert len(gdn_linear_mappings) > 0

        # Check the GDN linear mapping structure
        gdn_linear_mapping = gdn_linear_mappings[0]
        assert hasattr(gdn_linear_mapping, "hf_param")
        assert isinstance(gdn_linear_mapping.hf_param, dict)
        assert "qkvz" in gdn_linear_mapping.hf_param
        assert "ba" in gdn_linear_mapping.hf_param
        assert hasattr(gdn_linear_mapping, "megatron_param")

    def test_mapping_registry_moe_mappings(self):
        """Test that mapping_registry contains MoE-specific mappings."""
        bridge = Qwen3NextBridge()

        registry = bridge.mapping_registry()

        # Extract all mappings
        auto_mappings = [m for m in registry.mappings if type(m).__name__ == "AutoMapping"]
        replicated_mappings = [m for m in registry.mappings if type(m).__name__ == "ReplicatedMapping"]
        gated_mlp_mappings = [m for m in registry.mappings if type(m).__name__ == "GatedMLPMapping"]

        # Check for MoE router mapping
        hf_params = [mapping.hf_param for mapping in auto_mappings]
        assert "model.layers.*.mlp.gate.weight" in hf_params
        # shared_expert_gate is represented via ReplicatedMapping in bridge
        replicated_hf_params = [mapping.hf_param for mapping in replicated_mappings]
        assert "model.layers.*.mlp.shared_expert_gate.weight" in replicated_hf_params

        # Check for expert mappings in GatedMLPMapping
        assert len(gated_mlp_mappings) > 0

        # Check expert down projection mapping
        expert_down_params = [
            mapping.hf_param
            for mapping in auto_mappings
            if "experts" in mapping.hf_param and "down_proj" in mapping.hf_param
        ]
        assert len(expert_down_params) > 0
