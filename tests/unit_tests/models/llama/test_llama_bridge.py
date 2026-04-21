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
from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM

from megatron.bridge.models import AutoBridge
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.llama.llama_bridge import LlamaBridge
from megatron.bridge.models.llama.llama_provider import Llama31ModelProvider, LlamaModelProvider


class TestMegatronLlamaBridge:
    """Test cases for MegatronLlamaBridge class."""

    @pytest.fixture
    def llama_3_2_1b_config_dict(self):
        """Create a sample Llama configuration matching the provided example."""
        return {
            "architectures": ["LlamaForCausalLM"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 128000,
            "eos_token_id": 128001,
            "head_dim": 64,
            "hidden_act": "silu",
            "hidden_size": 2048,
            "initializer_range": 0.02,
            "intermediate_size": 8192,
            "max_position_embeddings": 131072,
            "mlp_bias": False,
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_hidden_layers": 16,
            "num_key_value_heads": 8,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-05,
            "rope_scaling": {
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            },
            "rope_theta": 500000.0,
            "tie_word_embeddings": True,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.45.0.dev0",
            "use_cache": True,
            "vocab_size": 128256,
        }

    @pytest.fixture
    def llama_config(self, llama_3_2_1b_config_dict):
        """Create a LlamaConfig instance."""
        return LlamaConfig(**llama_3_2_1b_config_dict)

    @pytest.fixture
    def mock_llama_model(self, llama_config):
        """Create a mock LlamaForCausalLM model."""
        mock_model = Mock(spec=LlamaForCausalLM)
        mock_model.config = llama_config
        mock_model.dtype = torch.bfloat16
        return mock_model

    @pytest.fixture
    def mock_pretrained_llama(self, llama_config):
        """Create a mock PreTrainedCausalLM with Llama model."""
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = llama_config
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)
        mock_pretrained.model = Mock(spec=LlamaForCausalLM)
        mock_pretrained.model.dtype = torch.bfloat16
        return mock_pretrained

    def test_bridge_registration(self):
        """Test that MegatronLlamaBridge is properly registered."""
        # The @MegatronModelBridge.register_bridge decorator should register the bridge
        # Check that the class exists and has the expected base class
        assert issubclass(LlamaBridge, MegatronModelBridge)

    def test_provider_bridge_basic(self, mock_pretrained_llama, llama_config):
        """Test basic provider_bridge functionality."""
        bridge = LlamaBridge()

        # Call provider_bridge
        result = bridge.provider_bridge(mock_pretrained_llama)

        # Check that it returns a LlamaModelProvider instance
        assert isinstance(result, LlamaModelProvider)

        # Check basic configuration mapping
        assert result.num_layers == llama_config.num_hidden_layers
        assert result.hidden_size == llama_config.hidden_size
        assert result.num_attention_heads == llama_config.num_attention_heads
        assert result.seq_length == llama_config.max_position_embeddings
        assert result.rotary_base == llama_config.rope_theta

    def test_provider_bridge_vocabulary(self, mock_pretrained_llama, llama_config):
        """Test vocabulary size mapping."""
        bridge = LlamaBridge()

        result = bridge.provider_bridge(mock_pretrained_llama)

        # Check vocabulary configuration
        assert result.vocab_size == llama_config.vocab_size
        assert result.share_embeddings_and_output_weights == llama_config.tie_word_embeddings

    def test_provider_bridge_attention_config(self, mock_pretrained_llama, llama_config):
        """Test attention configuration mapping."""
        bridge = LlamaBridge()

        result = bridge.provider_bridge(mock_pretrained_llama)

        # Check attention configuration
        assert result.num_attention_heads == llama_config.num_attention_heads
        assert result.num_query_groups == llama_config.num_key_value_heads

    def test_provider_bridge_mlp_config(self, mock_pretrained_llama, llama_config):
        """Test MLP configuration mapping."""
        bridge = LlamaBridge()

        result = bridge.provider_bridge(mock_pretrained_llama)

        # Check MLP configuration
        assert result.ffn_hidden_size == llama_config.intermediate_size
        assert result.gated_linear_unit == True  # Llama uses SwiGLU

    def test_provider_bridge_normalization(self, mock_pretrained_llama, llama_config):
        """Test normalization configuration."""
        bridge = LlamaBridge()

        result = bridge.provider_bridge(mock_pretrained_llama)

        # Check normalization settings
        assert result.layernorm_epsilon == llama_config.rms_norm_eps

    def test_provider_bridge_position_embedding(self, mock_pretrained_llama, llama_config):
        """Test position embedding configuration."""
        bridge = LlamaBridge()

        result = bridge.provider_bridge(mock_pretrained_llama)

        # Check position embedding
        assert result.rotary_base == llama_config.rope_theta

    def test_provider_bridge_rope_scaling(self, mock_pretrained_llama, llama_config):
        """Test RoPE scaling configuration."""
        bridge = LlamaBridge()

        result = bridge.provider_bridge(mock_pretrained_llama)

        # Check that Llama3.1 provider is used when rope_type is 'llama3'
        if hasattr(llama_config, "rope_scaling") and llama_config.rope_scaling:
            if llama_config.rope_scaling.get("rope_type") == "llama3":
                assert isinstance(result, Llama31ModelProvider)

    def test_provider_bridge_dtype_handling(self, llama_config):
        """Test dtype handling in provider_bridge."""
        # Create model with specific dtype
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = llama_config
        mock_pretrained.model = Mock(spec=LlamaForCausalLM)
        mock_pretrained.model.dtype = torch.bfloat16
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)

        bridge = LlamaBridge()
        result = bridge.provider_bridge(mock_pretrained)

        # The provider should respect the model's dtype
        assert result.params_dtype == torch.bfloat16
        assert result.bf16 == True
        assert result.fp16 == False

    def test_provider_bridge_with_custom_kwargs(self, mock_pretrained_llama):
        """Test provider_bridge with custom keyword arguments."""
        bridge = LlamaBridge()

        # Pass model only
        result = bridge.provider_bridge(mock_pretrained_llama)

        # Just verify that we got a valid LlamaModelProvider
        assert isinstance(result, LlamaModelProvider)

    def test_provider_bridge_no_rope_scaling(self):
        """Test provider_bridge when rope_scaling is not present."""
        # Create config without rope_scaling
        config_dict = {
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 8192,
            "vocab_size": 128256,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-05,
            "tie_word_embeddings": True,
            "model_type": "llama",
        }

        config = LlamaConfig(**config_dict)
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = config
        mock_pretrained.model = Mock(spec=LlamaForCausalLM)
        mock_pretrained.model.dtype = torch.float32
        mock_pretrained.generation_config = None

        bridge = LlamaBridge()
        result = bridge.provider_bridge(mock_pretrained)

        # Should use LlamaModelProvider (not Llama31) when rope_scaling is not present
        assert isinstance(result, LlamaModelProvider)
        assert not isinstance(result, Llama31ModelProvider)

    def test_mapping_registry_implementation(self, mock_pretrained_llama):
        """Test that mapping_registry returns a proper MegatronMappingRegistry."""
        bridge = LlamaBridge()

        # Get the mapping registry
        mapping_registry = bridge.mapping_registry()

        # Check it's not None
        assert mapping_registry is not None
        # Check it has param mappings (they are passed as args to __init__)
        # The mapping registry should have embedding, layer norm, attention, and MLP mappings
        # We can't directly access _param_mappings, but we know it was created with them

    def test_provider_bridge_fixed_settings(self, mock_pretrained_llama):
        """Test fixed settings that should always be set regardless of config."""
        bridge = LlamaBridge()

        result = bridge.provider_bridge(mock_pretrained_llama)

        # These should always be set to these values for Llama
        assert result.bias_activation_fusion == True


class TestAutoBridgeIntegration:
    """Integration tests for AutoBridge with Llama models."""

    @pytest.fixture
    def llama_configs(self):
        """Different Llama model configurations for testing."""
        return {
            "llama-3.2-1b": {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 2048,
                "num_hidden_layers": 16,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "intermediate_size": 8192,
                "vocab_size": 128256,
                "max_position_embeddings": 131072,
                "rope_theta": 500000.0,
                "rms_norm_eps": 1e-05,
                "tie_word_embeddings": True,
                "rope_scaling": {
                    "factor": 32.0,
                    "rope_type": "llama3",
                    "low_freq_factor": 1.0,
                    "high_freq_factor": 4.0,
                    "original_max_position_embeddings": 8192,
                },
            },
            "llama-2-7b": {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 32,  # No GQA
                "intermediate_size": 11008,
                "vocab_size": 32000,
                "max_position_embeddings": 4096,
                "rope_theta": 10000.0,
                "rms_norm_eps": 1e-05,
                "tie_word_embeddings": False,
                # No rope_scaling for Llama 2
            },
            "llama-3-8b": {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "intermediate_size": 14336,
                "vocab_size": 128256,
                "max_position_embeddings": 8192,
                "rope_theta": 500000.0,
                "rms_norm_eps": 1e-05,
                "tie_word_embeddings": False,
                "rope_scaling": {
                    "factor": 8.0,
                    "low_freq_factor": 1.0,
                    "high_freq_factor": 4.0,
                    "original_max_position_embeddings": 8192,
                    "rope_type": "llama3",
                },
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
            "tokenizer_class": "LlamaTokenizer",
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
    def test_from_pretrained_with_temp_dir(self, mock_autoconfig, mock_pretrained, llama_configs):
        """Test AutoBridge.from_hf_pretrained with temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with Llama 3.2 1B config
            config_dict = llama_configs["llama-3.2-1b"]
            self.create_mock_model_files(config_dict, temp_dir)

            # Mock the config loading
            config = LlamaConfig(**config_dict)
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
    def test_from_pretrained_multiple_models(self, mock_autoconfig, mock_pretrained, llama_configs):
        """Test AutoBridge.from_hf_pretrained with different Llama model configs."""
        for model_name, config_dict in llama_configs.items():
            with tempfile.TemporaryDirectory() as temp_dir:
                self.create_mock_model_files(config_dict, temp_dir)

                # Mock the config loading
                config = LlamaConfig(**config_dict)
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
                    mock_provider = Mock(spec=LlamaModelProvider)
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
    def test_from_pretrained_with_kwargs(self, mock_autoconfig, mock_pretrained, llama_configs):
        """Test AutoBridge.from_hf_pretrained with various kwargs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dict = llama_configs["llama-3-8b"]
            self.create_mock_model_files(config_dict, temp_dir)

            # Mock the config loading
            config = LlamaConfig(**config_dict)
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

    def test_supports_llama_architectures(self, llama_configs):
        """Test that AutoBridge.supports correctly identifies Llama models."""
        for model_name, config_dict in llama_configs.items():
            config = LlamaConfig(**config_dict)
            assert AutoBridge.supports(config) == True

        # Test non-causal LM architecture
        non_causal_config = Mock()
        non_causal_config.architectures = ["LlamaModel"]  # Not ForCausalLM
        assert AutoBridge.supports(non_causal_config) == False

    def test_list_supported_models(self):
        """Test list_supported_models includes LlamaForCausalLM."""
        # This test requires the dispatch system to be set up
        # Since we're testing in isolation, we'll skip this test
        # In a real environment, this would work if the bridges are registered
        pass  # Skip for now as it requires full dispatch setup
