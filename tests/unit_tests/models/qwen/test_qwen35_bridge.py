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

"""
Unit tests for Qwen3.5 bridge functionality.
"""

from copy import deepcopy
from dataclasses import fields
from unittest.mock import Mock, patch

import pytest
import torch
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from transformers import Qwen3_5Config, Qwen3_5MoeConfig, Qwen3_5MoeTextConfig, Qwen3_5TextConfig

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.qwen.qwen35_bridge import Qwen35Bridge, Qwen35MoEBridge


_NULL_ATTRS = (
    "q_lora_rank",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    "n_routed_experts",
    "num_local_experts",
    "num_nextn_predict_layers",
    "mtp_num_hidden_layers",
)


class TestQwen35DenseBridge:
    """Test cases for Qwen35Bridge (dense) class."""

    @pytest.fixture
    def qwen3_5_27b_config_dict(self):
        """Create a sample Qwen3.5-27B dense configuration matching the expected model structure."""
        return {
            "architectures": ["Qwen3_5ForCausalLM"],
            "attention_dropout": 0.0,
            "bos_token_id": 248045,
            "eos_token_id": 248046,
            "full_attention_interval": 4,
            "hidden_act": "silu",
            "hidden_size": 5120,
            "initializer_range": 0.02,
            "intermediate_size": 17408,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_value_head_dim": 128,
            "max_position_embeddings": 262144,
            "model_type": "qwen3_5",
            "num_attention_heads": 24,
            "num_hidden_layers": 64,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {"partial_rotary_factor": 0.25, "rope_theta": 10000000},
            "rope_theta": 10000000,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "use_cache": True,
            "vocab_size": 248320,
            "attention_bias": False,
        }

    @pytest.fixture
    def mock_qwen3_5_config(self, qwen3_5_27b_config_dict):
        """Create a mock Qwen3.5 dense configuration."""
        config = Mock()
        for key, value in qwen3_5_27b_config_dict.items():
            setattr(config, key, value)
        for null_attr in _NULL_ATTRS:
            setattr(config, null_attr, None)
        return config

    @pytest.fixture
    def mock_pretrained_qwen3_5(self, mock_qwen3_5_config):
        """Create a mock PreTrainedCausalLM with Qwen3.5 dense model."""
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mock_qwen3_5_config
        mock_pretrained.model = Mock()
        mock_pretrained.model.dtype = torch.bfloat16
        return mock_pretrained

    def test_bridge_registration(self):
        """Test that Qwen35Bridge is properly registered."""
        assert issubclass(Qwen35Bridge, MegatronModelBridge)

    def test_provider_bridge_basic(self, mock_pretrained_qwen3_5, mock_qwen3_5_config):
        """Test basic provider_bridge functionality."""
        bridge = Qwen35Bridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_5)

        assert isinstance(result, GPTModelProvider)
        assert result.num_layers == mock_qwen3_5_config.num_hidden_layers
        assert result.hidden_size == mock_qwen3_5_config.hidden_size
        assert result.num_attention_heads == mock_qwen3_5_config.num_attention_heads
        assert result.seq_length == mock_qwen3_5_config.max_position_embeddings
        assert result.rotary_base == mock_qwen3_5_config.rope_theta

    def test_provider_bridge_vocabulary(self, mock_pretrained_qwen3_5, mock_qwen3_5_config):
        """Test vocabulary size mapping."""
        bridge = Qwen35Bridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_5)

        assert result.vocab_size == mock_qwen3_5_config.vocab_size
        assert result.share_embeddings_and_output_weights == mock_qwen3_5_config.tie_word_embeddings

    def test_provider_bridge_attention_config(self, mock_pretrained_qwen3_5, mock_qwen3_5_config):
        """Test attention configuration mapping."""
        bridge = Qwen35Bridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_5)

        assert result.num_attention_heads == mock_qwen3_5_config.num_attention_heads
        assert result.num_query_groups == mock_qwen3_5_config.num_key_value_heads
        assert result.qk_layernorm is True
        assert result.layernorm_zero_centered_gamma is True
        assert result.attention_output_gate is True

    def test_provider_bridge_linear_attention_config(self, mock_pretrained_qwen3_5, mock_qwen3_5_config):
        """Test linear attention (GDN) configuration mapping."""
        bridge = Qwen35Bridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_5)

        if isinstance(result.linear_attention_freq, int):
            assert result.linear_attention_freq == mock_qwen3_5_config.full_attention_interval
        else:
            assert isinstance(result.linear_attention_freq, list)
            for i in range(result.num_layers):
                if (i + 1) % mock_qwen3_5_config.full_attention_interval == 0:
                    assert result.linear_attention_freq[i] == 0
                else:
                    assert result.linear_attention_freq[i] == 1
        assert result.linear_conv_kernel_dim == mock_qwen3_5_config.linear_conv_kernel_dim
        assert result.linear_key_head_dim == mock_qwen3_5_config.linear_key_head_dim
        assert result.linear_value_head_dim == mock_qwen3_5_config.linear_value_head_dim
        assert result.linear_num_key_heads == mock_qwen3_5_config.linear_num_key_heads
        assert result.linear_num_value_heads == mock_qwen3_5_config.linear_num_value_heads
        assert result.experimental_attention_variant == "gdn"

    def test_provider_bridge_mlp_config(self, mock_pretrained_qwen3_5, mock_qwen3_5_config):
        """Test MLP configuration mapping."""
        bridge = Qwen35Bridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_5)

        assert result.ffn_hidden_size == mock_qwen3_5_config.intermediate_size
        assert result.gated_linear_unit is True

    def test_provider_bridge_normalization(self, mock_pretrained_qwen3_5, mock_qwen3_5_config):
        """Test normalization configuration."""
        bridge = Qwen35Bridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_5)

        assert result.layernorm_epsilon == mock_qwen3_5_config.rms_norm_eps
        assert result.init_method_std == mock_qwen3_5_config.initializer_range

    def test_provider_bridge_position_embedding(self, mock_pretrained_qwen3_5, mock_qwen3_5_config):
        """Test position embedding configuration."""
        bridge = Qwen35Bridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_5)

        # Check position embedding
        assert result.rotary_base == mock_qwen3_5_config.rope_theta
        assert result.rotary_percent == mock_qwen3_5_config.rope_parameters["partial_rotary_factor"]
        assert result.position_embedding_type == "rope"

    def test_provider_bridge_mtp_config(self, mock_pretrained_qwen3_5, mock_qwen3_5_config):
        """Test MTP configuration mapping."""
        bridge = Qwen35Bridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_5)

        # Check MTP configuration
        assert not result.mtp_num_layers

    def test_provider_bridge_dtype_handling(self, qwen3_5_27b_config_dict):
        """Test dtype handling in provider_bridge."""
        config = Mock()
        for key, value in qwen3_5_27b_config_dict.items():
            setattr(config, key, value)
        for null_attr in _NULL_ATTRS:
            setattr(config, null_attr, None)
        config.torch_dtype = "bfloat16"

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = config

        bridge = Qwen35Bridge()
        result = bridge.provider_bridge(mock_pretrained)

        assert result.bf16 is True
        assert result.fp16 is False
        assert result.params_dtype == torch.bfloat16

        config.torch_dtype = "float16"
        result = bridge.provider_bridge(mock_pretrained)

        assert result.fp16 is True
        assert result.bf16 is False
        assert result.params_dtype == torch.float16

    def test_provider_bridge_tie_word_embeddings_true(self, mock_qwen3_5_config):
        """Test provider_bridge with tie_word_embeddings=True."""
        mock_qwen3_5_config.tie_word_embeddings = True

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mock_qwen3_5_config

        bridge = Qwen35Bridge()
        result = bridge.provider_bridge(mock_pretrained)

        assert result.share_embeddings_and_output_weights is True

    def test_provider_bridge_tie_word_embeddings_false(self, mock_qwen3_5_config):
        """Test provider_bridge with tie_word_embeddings=False."""
        mock_qwen3_5_config.tie_word_embeddings = False

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mock_qwen3_5_config

        bridge = Qwen35Bridge()
        result = bridge.provider_bridge(mock_pretrained)

        assert result.share_embeddings_and_output_weights is False

    def test_provider_bridge_27b_config(self, qwen3_5_27b_config_dict):
        """Test provider_bridge with Qwen3.5-27B dense configuration."""
        config = Mock()
        for key, value in qwen3_5_27b_config_dict.items():
            setattr(config, key, value)
        for null_attr in _NULL_ATTRS:
            setattr(config, null_attr, None)

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = config

        bridge = Qwen35Bridge()
        result = bridge.provider_bridge(mock_pretrained)

        assert result.num_layers == 64
        assert result.hidden_size == 5120
        assert result.num_attention_heads == 24
        assert result.ffn_hidden_size == 17408
        assert result.hetereogenous_dist_checkpoint is True

    def test_mapping_registry(self):
        """Test mapping_registry returns valid mappings."""
        bridge = Qwen35Bridge()

        registry = bridge.mapping_registry()

        # Check that registry is not None and has mappings
        assert registry is not None
        assert len(registry.mappings) > 0

        # Check for expected mapping types
        mapping_types = [type(mapping).__name__ for mapping in registry.mappings]
        assert "AutoMapping" in mapping_types
        assert "QKVMapping" in mapping_types
        assert "GatedMLPMapping" in mapping_types
        assert "GDNLinearMappingSeparate" in mapping_types
        assert "RMSNorm2ZeroCenteredRMSNormMapping" in mapping_types

    def test_mapping_registry_parameter_mappings(self):
        """Test that mapping_registry contains expected parameter mappings."""
        bridge = Qwen35Bridge()

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

    def test_mapping_registry_mtp_layer_spellings(self):
        """Every dense MTP sub-layer mapping is registered under both Megatron-Core spellings."""
        bridge = Qwen35Bridge()

        registry = bridge.mapping_registry()

        megatron_params = [mapping.megatron_param for mapping in registry.mappings]
        for layer_attr in ("mtp_model_layer", "transformer_layer"):
            for suffix in (
                "mlp.linear_fc1.layer_norm_weight",
                "mlp.linear_fc2.weight",
                "mlp.linear_fc1.weight",
                "self_attention.linear_qkv.layer_norm_weight",
                "self_attention.q_layernorm.weight",
                "self_attention.k_layernorm.weight",
                "self_attention.linear_proj.weight",
            ):
                assert f"mtp.layers.0.{layer_attr}.{suffix}" in megatron_params
            assert f"mtp.layers.*.{layer_attr}.self_attention.linear_qkv.weight" in megatron_params

    def test_mapping_registry_mtp_mapping(self):
        """Test that mapping_registry contains MTP mapping."""
        bridge = Qwen35Bridge()

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
        bridge = Qwen35Bridge()

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
        bridge = Qwen35Bridge()

        registry = bridge.mapping_registry()

        # Extract GDNLinearMappingSeparate instances
        gdn_linear_mappings = [m for m in registry.mappings if type(m).__name__ == "GDNLinearMappingSeparate"]
        assert len(gdn_linear_mappings) > 0

        # Check the GDN linear mapping structure
        gdn_linear_mapping = gdn_linear_mappings[0]
        assert hasattr(gdn_linear_mapping, "hf_param")
        assert isinstance(gdn_linear_mapping.hf_param, dict)
        assert "qkv" in gdn_linear_mapping.hf_param
        assert "z" in gdn_linear_mapping.hf_param
        assert "b" in gdn_linear_mapping.hf_param
        assert "a" in gdn_linear_mapping.hf_param
        assert hasattr(gdn_linear_mapping, "megatron_param")

    def test_mapping_registry_no_moe_mappings(self):
        """Test that dense bridge does not contain MoE-specific mappings."""
        bridge = Qwen35Bridge()

        registry = bridge.mapping_registry()

        auto_mappings = [m for m in registry.mappings if type(m).__name__ == "AutoMapping"]

        hf_params = [mapping.hf_param for mapping in auto_mappings]
        megatron_params = [mapping.megatron_param for mapping in auto_mappings]

        # Check for no expert and router mappings
        assert not any("router" in p or "experts" in p for p in hf_params + megatron_params)


class TestQwen35MoEBridge:
    """Test cases for Qwen3_5MoEBridge (MoE) class."""

    @pytest.fixture
    def qwen3_5_397b_a17b_config_dict(self):
        """Create a sample Qwen3.5-397B-A17B MoE configuration matching the expected model structure."""
        return {
            "architectures": ["Qwen3_5MoeForCausalLM"],
            "attention_dropout": 0.0,
            "bos_token_id": 248045,
            "eos_token_id": 248046,
            "full_attention_interval": 4,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 1024,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 64,
            "linear_value_head_dim": 128,
            "max_position_embeddings": 262144,
            "model_type": "qwen3_5_moe",
            "moe_intermediate_size": 1024,
            "norm_topk_prob": True,
            "num_attention_heads": 32,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "num_hidden_layers": 60,
            "num_key_value_heads": 2,
            "output_router_logits": False,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {"partial_rotary_factor": 0.25, "rope_theta": 10000000},
            "rope_theta": 10000000,
            "router_aux_loss_coef": 0.001,
            "shared_expert_intermediate_size": 4096,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "use_cache": True,
            "vocab_size": 248320,
            "attention_bias": False,
        }

    @pytest.fixture
    def mock_qwen3_5_moe_config(self, qwen3_5_397b_a17b_config_dict):
        """Create a mock Qwen3.5 MoE configuration."""
        config = Mock()
        for key, value in qwen3_5_397b_a17b_config_dict.items():
            setattr(config, key, value)
        for null_attr in _NULL_ATTRS:
            setattr(config, null_attr, None)
        return config

    @pytest.fixture
    def mock_pretrained_qwen3_5_moe(self, mock_qwen3_5_moe_config):
        """Create a mock PreTrainedCausalLM with Qwen3.5 MoE model."""
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mock_qwen3_5_moe_config
        mock_pretrained.model = Mock()
        mock_pretrained.model.dtype = torch.bfloat16
        return mock_pretrained

    def test_bridge_registration(self):
        """Test that Qwen3_5MoEBridge is properly registered."""
        assert issubclass(Qwen35MoEBridge, MegatronModelBridge)

    def test_provider_bridge_basic(self, mock_pretrained_qwen3_5_moe, mock_qwen3_5_moe_config):
        """Test basic provider_bridge functionality."""
        bridge = Qwen35MoEBridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_5_moe)

        assert isinstance(result, GPTModelProvider)
        assert result.num_layers == mock_qwen3_5_moe_config.num_hidden_layers
        assert result.hidden_size == mock_qwen3_5_moe_config.hidden_size
        assert result.num_attention_heads == mock_qwen3_5_moe_config.num_attention_heads
        assert result.seq_length == mock_qwen3_5_moe_config.max_position_embeddings
        assert result.rotary_base == mock_qwen3_5_moe_config.rope_theta

    def test_provider_bridge_vocabulary(self, mock_pretrained_qwen3_5_moe, mock_qwen3_5_moe_config):
        """Test vocabulary size mapping."""
        bridge = Qwen35MoEBridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_5_moe)

        assert result.vocab_size == mock_qwen3_5_moe_config.vocab_size
        assert result.share_embeddings_and_output_weights == mock_qwen3_5_moe_config.tie_word_embeddings

    def test_provider_bridge_attention_config(self, mock_pretrained_qwen3_5_moe, mock_qwen3_5_moe_config):
        """Test attention configuration mapping."""
        bridge = Qwen35MoEBridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_5_moe)

        assert result.num_attention_heads == mock_qwen3_5_moe_config.num_attention_heads
        assert result.num_query_groups == mock_qwen3_5_moe_config.num_key_value_heads
        assert result.qk_layernorm is True
        assert result.layernorm_zero_centered_gamma is True
        assert result.attention_output_gate is True

    def test_provider_bridge_linear_attention_config(self, mock_pretrained_qwen3_5_moe, mock_qwen3_5_moe_config):
        """Test linear attention (GDN) configuration mapping."""
        bridge = Qwen35MoEBridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_5_moe)

        if isinstance(result.linear_attention_freq, int):
            assert result.linear_attention_freq == mock_qwen3_5_moe_config.full_attention_interval
        else:
            assert isinstance(result.linear_attention_freq, list)
        assert result.linear_conv_kernel_dim == mock_qwen3_5_moe_config.linear_conv_kernel_dim
        assert result.linear_key_head_dim == mock_qwen3_5_moe_config.linear_key_head_dim
        assert result.linear_value_head_dim == mock_qwen3_5_moe_config.linear_value_head_dim
        assert result.linear_num_key_heads == mock_qwen3_5_moe_config.linear_num_key_heads
        assert result.linear_num_value_heads == mock_qwen3_5_moe_config.linear_num_value_heads
        assert result.experimental_attention_variant == "gdn"

    def test_provider_bridge_moe_config(self, mock_pretrained_qwen3_5_moe, mock_qwen3_5_moe_config):
        """Test MoE-specific configuration mapping."""
        bridge = Qwen35MoEBridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_5_moe)

        assert result.num_moe_experts == mock_qwen3_5_moe_config.num_experts
        assert result.moe_router_topk == mock_qwen3_5_moe_config.num_experts_per_tok
        assert result.moe_ffn_hidden_size == mock_qwen3_5_moe_config.moe_intermediate_size
        assert result.moe_grouped_gemm is True
        assert result.moe_shared_expert_intermediate_size == mock_qwen3_5_moe_config.shared_expert_intermediate_size
        assert result.moe_shared_expert_gate is True
        assert result.moe_router_dtype == "fp32"

    def test_provider_bridge_normalization(self, mock_pretrained_qwen3_5_moe, mock_qwen3_5_moe_config):
        """Test normalization configuration."""
        bridge = Qwen35MoEBridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_5_moe)

        assert result.layernorm_epsilon == mock_qwen3_5_moe_config.rms_norm_eps
        assert result.init_method_std == mock_qwen3_5_moe_config.initializer_range

    def test_provider_bridge_position_embedding(self, mock_pretrained_qwen3_5_moe, mock_qwen3_5_moe_config):
        """Test position embedding configuration."""
        bridge = Qwen35MoEBridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_5_moe)

        assert result.rotary_base == mock_qwen3_5_moe_config.rope_theta
        assert result.rotary_percent == mock_qwen3_5_moe_config.rope_parameters["partial_rotary_factor"]
        assert result.position_embedding_type == "rope"

    def test_provider_bridge_mtp_config(self, mock_pretrained_qwen3_5_moe, mock_qwen3_5_moe_config):
        """Test MTP configuration mapping."""
        bridge = Qwen35MoEBridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3_5_moe)

        assert not result.mtp_num_layers

    def test_provider_bridge_dtype_handling(self, qwen3_5_397b_a17b_config_dict):
        """Test dtype handling in provider_bridge."""
        config = Mock()
        for key, value in qwen3_5_397b_a17b_config_dict.items():
            setattr(config, key, value)
        for null_attr in _NULL_ATTRS:
            setattr(config, null_attr, None)
        config.torch_dtype = "bfloat16"

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = config

        bridge = Qwen35MoEBridge()
        result = bridge.provider_bridge(mock_pretrained)

        assert result.bf16 is True
        assert result.fp16 is False
        assert result.params_dtype == torch.bfloat16

        config.torch_dtype = "float16"
        result = bridge.provider_bridge(mock_pretrained)

        assert result.fp16 is True
        assert result.bf16 is False
        assert result.params_dtype == torch.float16

    def test_provider_bridge_tie_word_embeddings_true(self, mock_qwen3_5_moe_config):
        """Test provider_bridge with tie_word_embeddings=True."""
        mock_qwen3_5_moe_config.tie_word_embeddings = True

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mock_qwen3_5_moe_config

        bridge = Qwen35MoEBridge()
        result = bridge.provider_bridge(mock_pretrained)

        assert result.share_embeddings_and_output_weights is True

    def test_provider_bridge_tie_word_embeddings_false(self, mock_qwen3_5_moe_config):
        """Test provider_bridge with tie_word_embeddings=False."""
        mock_qwen3_5_moe_config.tie_word_embeddings = False

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mock_qwen3_5_moe_config

        bridge = Qwen35MoEBridge()
        result = bridge.provider_bridge(mock_pretrained)

        assert result.share_embeddings_and_output_weights is False

    def test_provider_bridge_397b_a17b_config(self, qwen3_5_397b_a17b_config_dict):
        """Test provider_bridge with Qwen3.5-397B-A17B MoE configuration."""
        config = Mock()
        for key, value in qwen3_5_397b_a17b_config_dict.items():
            setattr(config, key, value)
        for null_attr in _NULL_ATTRS:
            setattr(config, null_attr, None)

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = config

        bridge = Qwen35MoEBridge()
        result = bridge.provider_bridge(mock_pretrained)

        assert result.num_layers == 60
        assert result.hidden_size == 4096
        assert result.num_attention_heads == 32
        assert result.moe_ffn_hidden_size == 1024
        assert result.hetereogenous_dist_checkpoint is True

    def test_mapping_registry(self):
        """Test mapping_registry returns valid mappings."""
        bridge = Qwen35MoEBridge()

        registry = bridge.mapping_registry()

        assert registry is not None
        assert len(registry.mappings) > 0

        mapping_types = [type(mapping).__name__ for mapping in registry.mappings]
        assert "AutoMapping" in mapping_types
        assert "QKVMapping" in mapping_types
        assert "GatedMLPMapping" in mapping_types
        assert "GDNLinearMappingSeparate" in mapping_types
        # Routed-expert mappings default to per-expert without a checkpoint; the fused vs per-expert
        # selection is covered by test_moe_expert_mappings_fused_vs_per_expert.
        assert "ReplicatedMapping" in mapping_types
        assert "RMSNorm2ZeroCenteredRMSNormMapping" in mapping_types

    def test_mapping_registry_parameter_mappings(self):
        """Test that mapping_registry contains expected parameter mappings."""
        bridge = Qwen35MoEBridge()

        registry = bridge.mapping_registry()

        auto_mappings = [m for m in registry.mappings if type(m).__name__ == "AutoMapping"]

        hf_params = [mapping.hf_param for mapping in auto_mappings]
        megatron_params = [mapping.megatron_param for mapping in auto_mappings]

        assert "model.embed_tokens.weight" in hf_params
        assert "embedding.word_embeddings.weight" in megatron_params

        assert "lm_head.weight" in hf_params
        assert "output_layer.weight" in megatron_params

        assert "model.norm.weight" in hf_params
        assert "decoder.final_layernorm.weight" in megatron_params

    def test_mapping_registry_mtp_mapping(self):
        """Test that mapping_registry contains MTP mapping."""
        bridge = Qwen35MoEBridge()

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

    def test_mapping_registry_mtp_layer_spellings(self):
        """Every MTP sub-layer mapping is registered under both Megatron-Core spellings."""
        bridge = Qwen35MoEBridge()

        registry = bridge.mapping_registry()

        megatron_params = [mapping.megatron_param for mapping in registry.mappings]
        for layer_attr in ("mtp_model_layer", "transformer_layer"):
            for suffix in (
                "mlp.router.weight",
                "pre_mlp_layernorm.weight",
                "self_attention.linear_qkv.layer_norm_weight",
                "self_attention.q_layernorm.weight",
                "self_attention.k_layernorm.weight",
                "self_attention.linear_proj.weight",
                "mlp.shared_experts.gate_weight",
            ):
                assert f"mtp.layers.0.{layer_attr}.{suffix}" in megatron_params
            for suffix in (
                "self_attention.linear_qkv.weight",
                "mlp.shared_experts.linear_fc1.weight",
                "mlp.shared_experts.linear_fc2.weight",
                "mlp.experts.linear_fc1.weight*",
                "mlp.experts.linear_fc2.weight*",
            ):
                assert f"mtp.layers.*.{layer_attr}.{suffix}" in megatron_params

    def test_mapping_registry_qkv_mapping(self):
        """Test that mapping_registry contains QKV mapping."""
        bridge = Qwen35MoEBridge()

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
        bridge = Qwen35MoEBridge()

        registry = bridge.mapping_registry()

        # Extract GDNLinearMappingSeparate instances
        gdn_linear_mappings = [m for m in registry.mappings if type(m).__name__ == "GDNLinearMappingSeparate"]
        assert len(gdn_linear_mappings) > 0

        # Check the GDN linear mapping structure
        gdn_linear_mapping = gdn_linear_mappings[0]
        assert hasattr(gdn_linear_mapping, "hf_param")
        assert isinstance(gdn_linear_mapping.hf_param, dict)
        assert "qkv" in gdn_linear_mapping.hf_param
        assert "z" in gdn_linear_mapping.hf_param
        assert "b" in gdn_linear_mapping.hf_param
        assert "a" in gdn_linear_mapping.hf_param
        assert hasattr(gdn_linear_mapping, "megatron_param")

    def test_mapping_registry_moe_mappings(self):
        """Test that mapping_registry contains MoE-specific mappings."""
        bridge = Qwen35MoEBridge()

        registry = bridge.mapping_registry()

        # Extract all mappings
        auto_mappings = [m for m in registry.mappings if type(m).__name__ == "AutoMapping"]
        replicated_mappings = [m for m in registry.mappings if type(m).__name__ == "ReplicatedMapping"]

        # Check for MoE router mapping
        hf_params = [mapping.hf_param for mapping in auto_mappings]
        assert "model.layers.*.mlp.gate.weight" in hf_params
        # shared_expert_gate is represented via ReplicatedMapping in bridge
        replicated_hf_params = [mapping.hf_param for mapping in replicated_mappings]
        assert "model.layers.*.mlp.shared_expert_gate.weight" in replicated_hf_params

        # Routed-expert mappings exist for both the grouped-GEMM (experts.linear_fc*.weight*) and
        # SequentialMLP (local_experts.*.linear_fc*) layouts. Their fused-vs-per-expert form depends
        # on the checkpoint and is covered by test_moe_expert_mappings_fused_vs_per_expert.
        megatron_params = [getattr(m, "megatron_param", "") for m in registry.mappings]
        assert any(p.endswith("mlp.experts.linear_fc1.weight*") for p in megatron_params)
        assert any(p.endswith("local_experts.*.linear_fc1.weight") for p in megatron_params)

    def test_moe_expert_mappings_fused_vs_per_expert(self):
        """Routed-expert mappings match the detected checkpoint layout, for grouped and sequential."""

        def mapping_types(experts_packed):
            mappings = Qwen35MoEBridge._get_moe_lm_mappings(experts_packed=experts_packed)

            def _type(suffix):
                return type(next(m for m in mappings if getattr(m, "megatron_param", "").endswith(suffix))).__name__

            return {
                "grouped_fc1": _type("mlp.experts.linear_fc1.weight*"),
                "grouped_fc2": _type("mlp.experts.linear_fc2.weight*"),
                "seq_fc1": _type("local_experts.*.linear_fc1.weight"),
                "seq_fc2": _type("local_experts.*.linear_fc2.weight"),
            }

        # Fused checkpoint (real Qwen3.5/3.6): read the stacked experts.gate_up_proj / down_proj.
        assert mapping_types(experts_packed=True) == {
            "grouped_fc1": "FusedGatedExpertMapping",
            "grouped_fc2": "FusedExpertMapping",
            "seq_fc1": "FusedGatedExpertMapping",
            "seq_fc2": "FusedExpertMapping",
        }
        # Per-expert checkpoint (save_pretrained output): read experts.*.gate_proj / up_proj / down_proj.
        assert mapping_types(experts_packed=False) == {
            "grouped_fc1": "GatedMLPMapping",
            "grouped_fc2": "AutoMapping",
            "seq_fc1": "GatedMLPMapping",
            "seq_fc2": "AutoMapping",
        }


_SMALL_GDN_TEXT_CONFIG = dict(
    hidden_size=64,
    num_hidden_layers=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    max_position_embeddings=4096,
    vocab_size=128,
    # Released Qwen3.5 and Qwen3.6 text configs carry one MTP layer.
    mtp_num_hidden_layers=1,
    dtype="bfloat16",
)
_SMALL_MOE_CONFIG = dict(
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    num_experts=4,
    num_experts_per_tok=2,
)
# Multimodal checkpoints nest the language config and keep the mRoPE sections there.
_MULTIMODAL_ROPE_PARAMETERS = {
    "rope_type": "default",
    "rope_theta": 10000000,
    "partial_rotary_factor": 0.25,
    "mrope_section": [11, 11, 10],
    "mrope_interleaved": True,
}
_YARN_ROPE_PARAMETERS = {
    "rope_type": "yarn",
    "rope_theta": 10000000,
    "partial_rotary_factor": 0.25,
    "factor": 4.0,
    "original_max_position_embeddings": 1024,
}


def _qwen35_text_config(**overrides) -> Qwen3_5TextConfig:
    """Return a small Qwen3.5 dense text config with the released 3 GDN : 1 attention layer layout."""
    config = Qwen3_5TextConfig(**{**_SMALL_GDN_TEXT_CONFIG, "intermediate_size": 128, **overrides})
    config.architectures = ["Qwen3_5ForCausalLM"]
    return config


def _qwen35_moe_text_config(**overrides) -> Qwen3_5MoeTextConfig:
    """Return a small Qwen3.5 MoE text config with the released 3 GDN : 1 attention layer layout."""
    config = Qwen3_5MoeTextConfig(**{**_SMALL_GDN_TEXT_CONFIG, **_SMALL_MOE_CONFIG, **overrides})
    config.architectures = ["Qwen3_5MoeForCausalLM"]
    return config


def _text_config_from_multimodal(multimodal_config, architecture: str):
    """Extract the language config of a multimodal checkpoint, as the Qwen3.5 text recipes do."""
    text_config = deepcopy(multimodal_config.text_config)
    text_config.architectures = [architecture]
    return text_config


def _dense_from_multimodal():
    text_config = {**_SMALL_GDN_TEXT_CONFIG, "intermediate_size": 128, "rope_parameters": _MULTIMODAL_ROPE_PARAMETERS}
    multimodal = Qwen3_5Config(text_config=text_config, tie_word_embeddings=False)
    return _text_config_from_multimodal(multimodal, "Qwen3_5ForCausalLM")


def _moe_from_multimodal():
    text_config = {**_SMALL_GDN_TEXT_CONFIG, **_SMALL_MOE_CONFIG, "rope_parameters": _MULTIMODAL_ROPE_PARAMETERS}
    multimodal = Qwen3_5MoeConfig(text_config=text_config, tie_word_embeddings=False)
    return _text_config_from_multimodal(multimodal, "Qwen3_5MoeForCausalLM")


_DENSE_CONFIGS = {
    "default": lambda: _qwen35_text_config(),
    "tied_fp16": lambda: _qwen35_text_config(tie_word_embeddings=True, dtype="float16"),
    "no_mtp": lambda: _qwen35_text_config(mtp_num_hidden_layers=0),
    "yarn": lambda: _qwen35_text_config(rope_parameters=_YARN_ROPE_PARAMETERS),
    "from_multimodal": _dense_from_multimodal,
}
_MOE_CONFIGS = {
    "default": lambda: _qwen35_moe_text_config(),
    "tied_fp16": lambda: _qwen35_moe_text_config(tie_word_embeddings=True, dtype="float16"),
    "yarn": lambda: _qwen35_moe_text_config(rope_parameters=_YARN_ROPE_PARAMETERS),
    "from_multimodal": _moe_from_multimodal,
}
_PARITY_CASES = [
    *((Qwen35Bridge, make_config) for make_config in _DENSE_CONFIGS.values()),
    *((Qwen35MoEBridge, make_config) for make_config in _MOE_CONFIGS.values()),
]
_PARITY_IDS = [*(f"dense-{name}" for name in _DENSE_CONFIGS), *(f"moe-{name}" for name in _MOE_CONFIGS)]


class TestQwen35BuilderConfig:
    """Builder-backed ModelConfig path for the Qwen3.5 text bridges."""

    @pytest.mark.parametrize("bridge_class", [Qwen35Bridge, Qwen35MoEBridge])
    def test_provider_bridge_is_inherited_compatibility_only(self, bridge_class):
        assert "provider_bridge" not in bridge_class.__dict__

    @pytest.mark.parametrize("bridge_class", [Qwen35Bridge, Qwen35MoEBridge])
    def test_conversion_uses_builder_config(self, bridge_class):
        assert bridge_class.MODEL_CONFIG_CLASS is BridgeGPTModelConfig
        assert bridge_class.USE_MODEL_CONFIG_FOR_CONVERSION is True

    @pytest.mark.parametrize(
        ("bridge_class", "config"),
        [(Qwen35Bridge, _qwen35_text_config()), (Qwen35MoEBridge, _qwen35_moe_text_config())],
        ids=["dense", "moe"],
    )
    def test_hf_config_to_model_config_uses_direct_mapping(self, bridge_class, config):
        bridge = bridge_class()
        with (
            patch.object(bridge, "provider_bridge", side_effect=AssertionError("provider path used")),
            patch.object(bridge, "hf_config_to_provider_kwargs", side_effect=AssertionError("provider kwargs used")),
        ):
            model_config = bridge.hf_config_to_model_config(config)

        assert isinstance(model_config, BridgeGPTModelConfig)
        assert model_config.num_layers == 8
        assert model_config.normalization == "RMSNorm"
        assert model_config.qk_layernorm is True
        assert model_config.layernorm_zero_centered_gamma is True
        assert model_config.attention_output_gate is True
        assert model_config.experimental_attention_variant == "gdn"
        assert model_config.linear_attention_freq == [1, 1, 1, 0, 1, 1, 1, 0]
        assert model_config.linear_num_key_heads == 2
        assert model_config.linear_num_value_heads == 4
        assert model_config.rotary_percent == 0.25
        assert model_config.position_embedding_type == "rope"
        assert model_config.share_embeddings_and_output_weights is False
        assert model_config.mtp_num_layers == 1
        assert model_config.mtp_loss_scaling_factor == 0.1
        assert model_config.hetereogenous_dist_checkpoint is True
        # The builder derives the hybrid block spec from experimental_attention_variant.
        assert model_config.transformer_layer_spec is None
        if bridge_class is Qwen35MoEBridge:
            assert model_config.num_moe_experts == 4
            assert model_config.moe_router_topk == 2
            assert model_config.moe_ffn_hidden_size == 32
            assert model_config.moe_shared_expert_intermediate_size == 32
            assert model_config.moe_shared_expert_gate is True
            assert model_config.moe_router_load_balancing_type == "global_aux_loss"
            assert model_config.moe_router_dtype == "fp32"
        else:
            assert model_config.num_moe_experts is None
            assert model_config.ffn_hidden_size == 128

    @pytest.mark.parametrize(("bridge_class", "make_config"), _PARITY_CASES, ids=_PARITY_IDS)
    def test_model_config_matches_provider_runtime_config(self, bridge_class, make_config):
        config = make_config()
        bridge = bridge_class()
        model_config = bridge.hf_config_to_model_config(config)
        pretrained = Mock(spec=PreTrainedCausalLM)
        pretrained.config = config
        with pytest.warns(FutureWarning, match=r"deprecated.*get_model_config.*get_model"):
            provider = bridge.provider_bridge(pretrained)

        model_config_fields = {field.name for field in fields(model_config)}
        model_config_fields.update(field.name for field in fields(model_config.transformer))
        shared = ({field.name for field in fields(provider)} & model_config_fields) - {"transformer_layer_spec"}
        for field_name in sorted(shared):
            assert getattr(model_config, field_name) == getattr(provider, field_name), field_name

        # The provider's default layer spec builds homogeneous layers, so it keeps the hybrid block spec.
        assert provider.transformer_layer_spec is get_transformer_block_with_experimental_attention_variant_spec
        assert model_config.transformer_layer_spec is None

    @pytest.mark.parametrize(
        ("bridge_class", "make_config"),
        [(Qwen35Bridge, _qwen35_text_config), (Qwen35MoEBridge, _qwen35_moe_text_config)],
        ids=["dense", "moe"],
    )
    def test_yarn_checkpoint_maps_to_yarn_embeddings(self, bridge_class, make_config):
        model_config = bridge_class().hf_config_to_model_config(make_config(rope_parameters=_YARN_ROPE_PARAMETERS))

        assert model_config.position_embedding_type == "yarn"
        assert model_config.transformer.yarn_rotary_scaling_factor == 4.0
        assert model_config.transformer.yarn_original_max_position_embeddings == 1024

    @pytest.mark.parametrize(
        ("bridge_class", "config"),
        [(Qwen35Bridge, _qwen35_text_config()), (Qwen35MoEBridge, _qwen35_moe_text_config())],
        ids=["dense", "moe"],
    )
    def test_builder_default_layer_spec_builds_gated_delta_net_block(self, bridge_class, config, monkeypatch):
        from megatron.training.models import gpt as mcore_gpt

        calls = []

        def block_spec(config, vp_stage=None):
            calls.append((config, vp_stage))
            return "hybrid-block-spec"

        monkeypatch.setattr(mcore_gpt, "get_transformer_block_with_experimental_attention_variant_spec", block_spec)
        model_config = bridge_class().hf_config_to_model_config(config)

        assert mcore_gpt.default_layer_spec(model_config, vp_stage=None) == "hybrid-block-spec"
        assert calls == [(model_config.transformer, None)]
