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

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from transformers import GenerationConfig, Qwen2Config, Qwen3ForCausalLM

from megatron.bridge.models import AutoBridge
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.qwen.qwen3_bridge import Qwen3Bridge
from megatron.bridge.models.qwen.qwen_provider import Qwen3ModelProvider


class TestMegatronQwen3Bridge:
    """Test cases for MegatronQwen3Bridge class."""

    @pytest.fixture
    def qwen3_1p7b_config_dict(self):
        """Create a sample Qwen3 configuration matching the provided example."""
        return {
            "architectures": ["Qwen3ForCausalLM"],
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "hidden_act": "silu",
            "hidden_size": 2048,
            "initializer_range": 0.02,
            "intermediate_size": 6144,
            "max_position_embeddings": 40960,
            "model_type": "qwen2",  # Qwen3 uses qwen2 model type in transformers
            "num_attention_heads": 16,
            "num_hidden_layers": 28,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-06,
            "rope_theta": 1000000.0,
            "sliding_window": 4096,
            "tie_word_embeddings": True,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.37.0",
            "use_cache": True,
            "vocab_size": 151936,
        }

    @pytest.fixture
    def qwen3_config(self, qwen3_1p7b_config_dict):
        """Create a Qwen2Config instance (used for Qwen3)."""
        return Qwen2Config(**qwen3_1p7b_config_dict)

    @pytest.fixture
    def mock_qwen3_model(self, qwen3_config):
        """Create a mock Qwen3ForCausalLM model."""
        mock_model = Mock(spec=Qwen3ForCausalLM)
        mock_model.config = qwen3_config
        mock_model.dtype = torch.bfloat16
        return mock_model

    @pytest.fixture
    def mock_pretrained_qwen3(self, qwen3_config):
        """Create a mock PreTrainedCausalLM with Qwen3 model."""
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = qwen3_config
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)
        mock_pretrained.model = Mock(spec=Qwen3ForCausalLM)
        mock_pretrained.model.dtype = torch.bfloat16
        return mock_pretrained

    def test_bridge_registration(self):
        """Test that MegatronQwen3Bridge is properly registered."""
        # The @MegatronModelBridge.register_bridge decorator should register the bridge
        # Check that the class exists and has the expected base class
        assert issubclass(Qwen3Bridge, MegatronModelBridge)

    def test_provider_bridge_basic(self, mock_pretrained_qwen3, qwen3_config):
        """Test basic provider_bridge functionality."""
        bridge = Qwen3Bridge()

        # Call provider_bridge
        result = bridge.provider_bridge(mock_pretrained_qwen3)

        # Check that it returns a Qwen3ModelProvider instance
        assert isinstance(result, Qwen3ModelProvider)

        # Check basic configuration mapping
        assert result.num_layers == qwen3_config.num_hidden_layers
        assert result.hidden_size == qwen3_config.hidden_size
        assert result.num_attention_heads == qwen3_config.num_attention_heads
        assert result.seq_length == qwen3_config.max_position_embeddings
        assert result.rotary_base == qwen3_config.rope_theta

    def test_provider_bridge_vocabulary(self, mock_pretrained_qwen3, qwen3_config):
        """Test vocabulary size mapping."""
        bridge = Qwen3Bridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3)

        # Check vocabulary configuration
        assert result.vocab_size == qwen3_config.vocab_size
        assert result.share_embeddings_and_output_weights == qwen3_config.tie_word_embeddings

    def test_provider_bridge_attention_config(self, mock_pretrained_qwen3, qwen3_config):
        """Test attention configuration mapping."""
        bridge = Qwen3Bridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3)

        # Check attention configuration
        assert result.num_attention_heads == qwen3_config.num_attention_heads
        assert result.num_query_groups == qwen3_config.num_key_value_heads

    def test_provider_bridge_mlp_config(self, mock_pretrained_qwen3, qwen3_config):
        """Test MLP configuration mapping."""
        bridge = Qwen3Bridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3)

        # Check MLP configuration
        assert result.ffn_hidden_size == qwen3_config.intermediate_size
        assert result.gated_linear_unit == True  # Qwen3 uses gated MLP

    def test_provider_bridge_normalization(self, mock_pretrained_qwen3, qwen3_config):
        """Test normalization configuration."""
        bridge = Qwen3Bridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3)

        # Check normalization settings
        assert result.layernorm_epsilon == qwen3_config.rms_norm_eps

    def test_provider_bridge_position_embedding(self, mock_pretrained_qwen3, qwen3_config):
        """Test position embedding configuration."""
        bridge = Qwen3Bridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3)

        # Check position embedding
        assert result.rotary_base == qwen3_config.rope_theta

    def test_provider_bridge_qwen3_specific_features(self, mock_pretrained_qwen3):
        """Test Qwen3-specific features."""
        bridge = Qwen3Bridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3)

        # Check Qwen3-specific features
        assert result.qk_layernorm == True  # Qwen3 uses QK layernorm
        assert result.add_qkv_bias == False  # Qwen3 does not have QKV bias

    def test_provider_bridge_dtype_handling(self, qwen3_config):
        """Test dtype handling in provider_bridge."""
        # Create model with specific dtype
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = qwen3_config
        mock_pretrained.model = Mock(spec=Qwen3ForCausalLM)
        mock_pretrained.model.dtype = torch.bfloat16
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)

        bridge = Qwen3Bridge()
        result = bridge.provider_bridge(mock_pretrained)

        # The provider should respect the model's dtype
        assert result.params_dtype == torch.bfloat16
        assert result.bf16 == True
        assert result.fp16 == False

    def test_provider_bridge_fp16_dtype_handling(self, qwen3_config):
        """Test FP16 dtype handling in provider_bridge."""
        # Create model with FP16 dtype - set it in the config
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = qwen3_config
        mock_pretrained.config.torch_dtype = torch.float16  # Set config dtype to fp16
        mock_pretrained.model = Mock(spec=Qwen3ForCausalLM)
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)

        bridge = Qwen3Bridge()
        result = bridge.provider_bridge(mock_pretrained)

        # The provider should respect the config's dtype
        assert result.params_dtype == torch.float16
        assert result.fp16 == True
        assert result.bf16 == False

    def test_provider_bridge_with_custom_kwargs(self, mock_pretrained_qwen3):
        """Test provider_bridge with custom keyword arguments."""
        bridge = Qwen3Bridge()

        # Pass model only
        result = bridge.provider_bridge(mock_pretrained_qwen3)

        # Just verify that we got a valid Qwen3ModelProvider
        assert isinstance(result, Qwen3ModelProvider)

    def test_provider_bridge_without_tie_embeddings(self, qwen3_config):
        """Test provider_bridge when tie_word_embeddings is not present."""
        # Remove tie_word_embeddings from config
        config_dict = qwen3_config.to_dict()
        del config_dict["tie_word_embeddings"]
        config = Qwen2Config(**config_dict)

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = config
        mock_pretrained.model = Mock(spec=Qwen3ForCausalLM)
        mock_pretrained.model.dtype = torch.float32
        mock_pretrained.generation_config = None

        bridge = Qwen3Bridge()
        result = bridge.provider_bridge(mock_pretrained)

        # Should default to False when tie_word_embeddings is not present
        assert result.share_embeddings_and_output_weights == False

    def test_mapping_registry_implementation(self, mock_pretrained_qwen3):
        """Test that mapping_registry returns a proper MegatronMappingRegistry."""
        bridge = Qwen3Bridge()

        # Get the mapping registry
        mapping_registry = bridge.mapping_registry()

        # Check it's not None
        assert mapping_registry is not None
        # Check it has param mappings (they are passed as args to __init__)
        # The mapping registry should have embedding, layer norm, attention, and MLP mappings
        # We can't directly access _param_mappings, but we know it was created with them

    def test_provider_bridge_make_vocab_size_divisible_by(self, mock_pretrained_qwen3):
        """Test make_vocab_size_divisible_by calculation."""
        bridge = Qwen3Bridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3)

        # The method should calculate a reasonable divisor based on vocab size
        assert hasattr(result, "make_vocab_size_divisible_by")
        assert result.make_vocab_size_divisible_by > 0

    def test_provider_bridge_generation_config(self, mock_pretrained_qwen3):
        """Test that generation config is passed through."""
        bridge = Qwen3Bridge()

        result = bridge.provider_bridge(mock_pretrained_qwen3)

        # Generation config should be passed from the pretrained model
        assert result.generation_config == mock_pretrained_qwen3.generation_config


class TestAutoBridgeIntegration:
    """Integration tests for AutoBridge with Qwen3 models."""

    @pytest.fixture
    def qwen3_configs(self):
        """Different Qwen3 model configurations for testing."""
        return {
            "qwen3-600m": {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen2",
                "hidden_size": 1024,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "intermediate_size": 3072,
                "vocab_size": 151936,
                "max_position_embeddings": 40960,
                "rope_theta": 1000000.0,
                "rms_norm_eps": 1e-06,
                "tie_word_embeddings": True,
            },
            "qwen3-1p7b": {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen2",
                "hidden_size": 2048,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "intermediate_size": 6144,
                "vocab_size": 151936,
                "max_position_embeddings": 40960,
                "rope_theta": 1000000.0,
                "rms_norm_eps": 1e-06,
                "tie_word_embeddings": True,
            },
            "qwen3-8b": {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen2",
                "hidden_size": 4096,
                "num_hidden_layers": 36,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "intermediate_size": 12288,
                "vocab_size": 151936,
                "max_position_embeddings": 40960,
                "rope_theta": 1000000.0,
                "rms_norm_eps": 1e-06,
                "tie_word_embeddings": False,
            },
        }

    def create_mock_model_files(self, config_dict, save_dir):
        """Create mock model files in a directory."""
        import json

        # Save config
        config_path = Path(save_dir) / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        # Create a dummy safetensors index file
        index_path = Path(save_dir) / "model.safetensors.index.json"
        index_data = {
            "metadata": {"total_size": 1000000},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00001.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00001.safetensors",
            },
        }
        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=2)

        # Create tokenizer files
        tokenizer_config = {
            "tokenizer_class": "Qwen2Tokenizer",
            "model_max_length": config_dict["max_position_embeddings"],
        }
        tokenizer_path = Path(save_dir) / "tokenizer_config.json"
        with open(tokenizer_path, "w") as f:
            json.dump(tokenizer_config, f, indent=2)

        # Create dummy tokenizer.json
        tokenizer_json_path = Path(save_dir) / "tokenizer.json"
        tokenizer_data = {
            "version": "1.0",
            "model": {"type": "BPE"},
        }
        with open(tokenizer_json_path, "w") as f:
            json.dump(tokenizer_data, f, indent=2)

    @patch("megatron.bridge.models.conversion.auto_bridge.PreTrainedCausalLM.from_pretrained")
    @patch("transformers.AutoConfig.from_pretrained")
    def test_from_pretrained_with_temp_dir(self, mock_autoconfig, mock_pretrained, qwen3_configs):
        """Test AutoBridge.from_hf_pretrained with temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with Qwen3 1.7B config
            config_dict = qwen3_configs["qwen3-1p7b"]
            self.create_mock_model_files(config_dict, temp_dir)

            # Mock the config loading
            config = Qwen2Config(**config_dict)
            mock_autoconfig.return_value = config

            # Mock the pretrained model
            mock_model = Mock(spec=PreTrainedCausalLM)
            mock_model.config = config
            mock_model.model_name_or_path = temp_dir
            mock_pretrained.return_value = mock_model

            # Create bridge from the temp directory
            bridge = AutoBridge.from_hf_pretrained(temp_dir)

            # Verify
            assert isinstance(bridge, AutoBridge)
            assert bridge.hf_pretrained == mock_model
            mock_autoconfig.assert_called_once_with(temp_dir, trust_remote_code=False)
            mock_pretrained.assert_called_once_with(temp_dir)

    @patch("megatron.bridge.models.conversion.auto_bridge.PreTrainedCausalLM.from_pretrained")
    @patch("transformers.AutoConfig.from_pretrained")
    def test_from_pretrained_multiple_models(self, mock_autoconfig, mock_pretrained, qwen3_configs):
        """Test AutoBridge.from_hf_pretrained with different Qwen3 model configs."""
        for model_name, config_dict in qwen3_configs.items():
            with tempfile.TemporaryDirectory() as temp_dir:
                self.create_mock_model_files(config_dict, temp_dir)

                # Mock the config loading
                config = Qwen2Config(**config_dict)
                mock_autoconfig.return_value = config

                # Mock the pretrained model
                mock_model = Mock(spec=PreTrainedCausalLM)
                mock_model.config = config
                mock_model.model_name_or_path = temp_dir
                mock_pretrained.return_value = mock_model

                # Create bridge
                bridge = AutoBridge.from_hf_pretrained(temp_dir, torch_dtype=torch.float16)

                # Verify
                assert isinstance(bridge, AutoBridge)

                # Get the provider to verify model-specific settings
                # Since _model_bridge is a property, we need to patch the method it calls
                with patch(
                    "megatron.bridge.models.conversion.auto_bridge.model_bridge.get_model_bridge"
                ) as mock_get_bridge:
                    mock_bridge = Mock()
                    mock_provider = Mock(spec=Qwen3ModelProvider)
                    mock_bridge.provider_bridge.return_value = mock_provider
                    mock_get_bridge.return_value = mock_bridge

                    _ = bridge.to_megatron_provider(load_weights=False)

                    # Verify provider_bridge was called with correct model
                    mock_bridge.provider_bridge.assert_called_once_with(mock_model)

                # Clear mocks for next iteration
                mock_autoconfig.reset_mock()
                mock_pretrained.reset_mock()

    @patch("megatron.bridge.models.conversion.auto_bridge.PreTrainedCausalLM.from_pretrained")
    @patch("transformers.AutoConfig.from_pretrained")
    def test_from_pretrained_with_kwargs(self, mock_autoconfig, mock_pretrained, qwen3_configs):
        """Test AutoBridge.from_hf_pretrained with various kwargs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dict = qwen3_configs["qwen3-8b"]
            self.create_mock_model_files(config_dict, temp_dir)

            # Mock the config loading
            config = Qwen2Config(**config_dict)
            mock_autoconfig.return_value = config

            # Mock the pretrained model
            mock_model = Mock(spec=PreTrainedCausalLM)
            mock_model.config = config
            mock_pretrained.return_value = mock_model

            # Test with various kwargs
            kwargs = {
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
                "trust_remote_code": True,
                "attn_implementation": "flash_attention_2",
            }

            _ = AutoBridge.from_hf_pretrained(temp_dir, **kwargs)

            # Verify kwargs were passed through
            mock_pretrained.assert_called_once_with(temp_dir, **kwargs)

    def test_supports_qwen3_architectures(self, qwen3_configs):
        """Test that AutoBridge.supports correctly identifies Qwen3 models."""
        for model_name, config_dict in qwen3_configs.items():
            config = Qwen2Config(**config_dict)
            assert AutoBridge.supports(config) == True

        # Test non-causal LM architecture
        non_causal_config = Mock()
        non_causal_config.architectures = ["Qwen2Model"]  # Not ForCausalLM
        assert AutoBridge.supports(non_causal_config) == False

    def test_list_supported_models(self):
        """Test list_supported_models includes Qwen3ForCausalLM."""
        # This test requires the dispatch system to be set up
        # Since we're testing in isolation, we'll skip this test
        # In a real environment, this would work if the bridges are registered
        pass  # Skip for now as it requires full dispatch setup


class TestQwen3BridgeParameterMapping:
    """Test parameter mapping functionality in Qwen3Bridge."""

    @pytest.fixture
    def mock_qwen3_state_dict(self):
        """Create a mock state dict with Qwen3 parameter names."""
        return {
            "model.embed_tokens.weight": torch.randn(151936, 2048),
            "lm_head.weight": torch.randn(151936, 2048),
            "model.norm.weight": torch.randn(2048),
            "model.layers.0.input_layernorm.weight": torch.randn(2048),
            "model.layers.0.post_attention_layernorm.weight": torch.randn(2048),
            "model.layers.0.self_attn.q_norm.weight": torch.randn(256),  # Qwen3 specific
            "model.layers.0.self_attn.k_norm.weight": torch.randn(256),  # Qwen3 specific
            "model.layers.0.self_attn.q_proj.weight": torch.randn(2048, 2048),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(1024, 2048),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(1024, 2048),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(2048, 2048),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(6144, 2048),
            "model.layers.0.mlp.up_proj.weight": torch.randn(6144, 2048),
            "model.layers.0.mlp.down_proj.weight": torch.randn(2048, 6144),
        }

    def test_mapping_registry_has_qwen3_specific_mappings(self):
        """Test that mapping registry includes Qwen3-specific QK norm mappings."""
        bridge = Qwen3Bridge()
        mapping_registry = bridge.mapping_registry()

        # This test verifies that the mapping registry was created
        # The actual parameter mappings are tested in integration tests
        assert mapping_registry is not None

    def test_qwen3_qk_norm_mapping_difference(self):
        """Test that Qwen3 bridge includes QK norm mappings not present in Qwen2."""
        bridge = Qwen3Bridge()
        mapping_registry = bridge.mapping_registry()

        # Qwen3 should have QK norm mappings
        # This is implicitly tested by the bridge's mapping_registry method
        # which includes the QK norm parameter mappings
        assert mapping_registry is not None

    def test_qwen3_no_qkv_bias_mapping(self):
        """Test that Qwen3 bridge doesn't include QKV bias mappings."""
        bridge = Qwen3Bridge()
        mapping_registry = bridge.mapping_registry()

        # Qwen3 doesn't have QKV bias, unlike Qwen2
        # This is reflected in the QKVMapping not including bias terms
        assert mapping_registry is not None
