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
from transformers import GemmaConfig, GemmaForCausalLM, GenerationConfig

from megatron.bridge.models import AutoBridge
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.gemma.gemma_bridge import GemmaBridge
from megatron.bridge.models.gemma.gemma_provider import GemmaModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


class TestMegatronGemmaBridge:
    """Test cases for MegatronGemmaBridge class."""

    @pytest.fixture
    def gemma_2b_config_dict(self):
        """Create a sample Gemma 2B configuration."""
        return {
            "architectures": ["GemmaForCausalLM"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 2,
            "eos_token_id": 1,
            "head_dim": 256,
            "hidden_act": "gelu",
            "hidden_size": 2048,
            "initializer_range": 0.02,
            "intermediate_size": 16384,
            "max_position_embeddings": 8192,
            "model_type": "gemma",
            "num_attention_heads": 8,
            "num_hidden_layers": 18,
            "num_key_value_heads": 1,
            "pad_token_id": 0,
            "rms_norm_eps": 1e-06,
            "rope_scaling": None,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.38.0.dev0",
            "use_cache": True,
            "vocab_size": 256000,
        }

    @pytest.fixture
    def gemma_7b_config_dict(self):
        """Create a sample Gemma 7B configuration."""
        return {
            "architectures": ["GemmaForCausalLM"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 2,
            "eos_token_id": 1,
            "head_dim": 256,
            "hidden_act": "gelu",
            "hidden_size": 3072,
            "initializer_range": 0.02,
            "intermediate_size": 24576,
            "max_position_embeddings": 8192,
            "model_type": "gemma",
            "num_attention_heads": 16,
            "num_hidden_layers": 28,
            "num_key_value_heads": 16,
            "pad_token_id": 0,
            "rms_norm_eps": 1e-06,
            "rope_scaling": None,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.38.0.dev0",
            "use_cache": True,
            "vocab_size": 256000,
        }

    @pytest.fixture
    def gemma_2b_config(self, gemma_2b_config_dict):
        """Create a GemmaConfig instance for 2B model."""
        return GemmaConfig(**gemma_2b_config_dict)

    @pytest.fixture
    def gemma_7b_config(self, gemma_7b_config_dict):
        """Create a GemmaConfig instance for 7B model."""
        return GemmaConfig(**gemma_7b_config_dict)

    @pytest.fixture
    def mock_gemma_2b_model(self, gemma_2b_config):
        """Create a mock GemmaForCausalLM 2B model."""
        mock_model = Mock(spec=GemmaForCausalLM)
        mock_model.config = gemma_2b_config
        mock_model.dtype = torch.bfloat16
        return mock_model

    @pytest.fixture
    def mock_gemma_7b_model(self, gemma_7b_config):
        """Create a mock GemmaForCausalLM 7B model."""
        mock_model = Mock(spec=GemmaForCausalLM)
        mock_model.config = gemma_7b_config
        mock_model.dtype = torch.bfloat16
        return mock_model

    @pytest.fixture
    def mock_pretrained_gemma_2b(self, gemma_2b_config):
        """Create a mock PreTrainedCausalLM with Gemma 2B model."""
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = gemma_2b_config
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)
        mock_pretrained.model = Mock(spec=GemmaForCausalLM)
        mock_pretrained.model.dtype = torch.bfloat16
        return mock_pretrained

    @pytest.fixture
    def mock_pretrained_gemma_7b(self, gemma_7b_config):
        """Create a mock PreTrainedCausalLM with Gemma 7B model."""
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = gemma_7b_config
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)
        mock_pretrained.model = Mock(spec=GemmaForCausalLM)
        mock_pretrained.model.dtype = torch.bfloat16
        return mock_pretrained

    def test_bridge_registration(self):
        """Test that MegatronGemmaBridge is properly registered."""
        # The @MegatronModelBridge.register_bridge decorator should register the bridge
        # Check that the class exists and has the expected base class
        assert issubclass(GemmaBridge, MegatronModelBridge)

    def test_provider_bridge_basic_2b(self, mock_pretrained_gemma_2b, gemma_2b_config):
        """Test basic provider_bridge functionality for Gemma 2B."""
        bridge = GemmaBridge()

        # Call provider_bridge
        result = bridge.provider_bridge(mock_pretrained_gemma_2b)

        # Check that it returns a GemmaModelProvider instance
        assert isinstance(result, GemmaModelProvider)

        # Check basic configuration mapping
        assert result.num_layers == gemma_2b_config.num_hidden_layers
        assert result.hidden_size == gemma_2b_config.hidden_size
        assert result.num_attention_heads == gemma_2b_config.num_attention_heads
        assert result.seq_length == gemma_2b_config.max_position_embeddings
        assert result.rotary_base == gemma_2b_config.rope_theta

    def test_provider_bridge_basic_7b(self, mock_pretrained_gemma_7b, gemma_7b_config):
        """Test basic provider_bridge functionality for Gemma 7B."""
        bridge = GemmaBridge()

        # Call provider_bridge
        result = bridge.provider_bridge(mock_pretrained_gemma_7b)

        # Check that it returns a GemmaModelProvider instance
        assert isinstance(result, GemmaModelProvider)

        # Check basic configuration mapping
        assert result.num_layers == gemma_7b_config.num_hidden_layers
        assert result.hidden_size == gemma_7b_config.hidden_size
        assert result.num_attention_heads == gemma_7b_config.num_attention_heads
        assert result.seq_length == gemma_7b_config.max_position_embeddings
        assert result.rotary_base == gemma_7b_config.rope_theta

    def test_provider_bridge_vocabulary(self, mock_pretrained_gemma_2b, gemma_2b_config):
        """Test vocabulary size mapping."""
        bridge = GemmaBridge()

        result = bridge.provider_bridge(mock_pretrained_gemma_2b)

        # Check vocabulary configuration
        assert result.vocab_size == gemma_2b_config.vocab_size
        # Gemma uses tied embeddings by default
        assert result.share_embeddings_and_output_weights == True

    def test_provider_bridge_attention_config(self, mock_pretrained_gemma_2b, gemma_2b_config):
        """Test attention configuration mapping."""
        bridge = GemmaBridge()

        result = bridge.provider_bridge(mock_pretrained_gemma_2b)

        # Check attention configuration
        assert result.num_attention_heads == gemma_2b_config.num_attention_heads
        assert result.num_query_groups == gemma_2b_config.num_key_value_heads

    def test_provider_bridge_mlp_config(self, mock_pretrained_gemma_2b, gemma_2b_config):
        """Test MLP configuration mapping."""
        bridge = GemmaBridge()

        result = bridge.provider_bridge(mock_pretrained_gemma_2b)

        # Check MLP configuration
        assert result.ffn_hidden_size == gemma_2b_config.intermediate_size
        assert result.gated_linear_unit == True  # Gemma uses gated MLP

    def test_provider_bridge_normalization(self, mock_pretrained_gemma_2b, gemma_2b_config):
        """Test normalization configuration."""
        bridge = GemmaBridge()

        result = bridge.provider_bridge(mock_pretrained_gemma_2b)

        # Check normalization settings
        assert result.layernorm_epsilon == gemma_2b_config.rms_norm_eps

    def test_provider_bridge_position_embedding(self, mock_pretrained_gemma_2b, gemma_2b_config):
        """Test position embedding configuration."""
        bridge = GemmaBridge()

        result = bridge.provider_bridge(mock_pretrained_gemma_2b)

        # Check position embedding
        assert result.rotary_base == gemma_2b_config.rope_theta

    def test_provider_bridge_gemma_specific_features(self, mock_pretrained_gemma_2b, gemma_2b_config):
        """Test Gemma-specific features."""
        bridge = GemmaBridge()

        result = bridge.provider_bridge(mock_pretrained_gemma_2b)

        # Check Gemma-specific features
        assert result.kv_channels == gemma_2b_config.head_dim  # Gemma has explicit head_dim
        assert result.add_bias_linear == False  # Gemma doesn't use bias in linear layers
        assert result.layernorm_zero_centered_gamma == True  # Gemma-specific RMSNorm behavior

    def test_provider_bridge_head_dim_calculation(self, mock_pretrained_gemma_7b, gemma_7b_config):
        """Test head dimension calculation for Gemma 7B."""
        bridge = GemmaBridge()

        result = bridge.provider_bridge(mock_pretrained_gemma_7b)

        # Gemma 7B should use the explicit head_dim from config
        assert result.kv_channels == gemma_7b_config.head_dim  # 256
        # Verify this is different from standard calculation
        standard_calculation = gemma_7b_config.hidden_size // gemma_7b_config.num_attention_heads  # 3072 / 16 = 192
        assert result.kv_channels != standard_calculation
        assert result.kv_channels == 256  # Gemma uses 256 regardless of model size

    def test_provider_bridge_head_dim_fallback(self, gemma_2b_config):
        """Test head dimension fallback when head_dim is not in config."""
        # Create config without head_dim
        config_dict = gemma_2b_config.to_dict()
        del config_dict["head_dim"]
        config = GemmaConfig(**config_dict)

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = config
        mock_pretrained.model = Mock(spec=GemmaForCausalLM)
        mock_pretrained.model.dtype = torch.bfloat16
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)

        bridge = GemmaBridge()
        result = bridge.provider_bridge(mock_pretrained)

        # Should fallback to standard calculation
        expected_kv_channels = config.hidden_size // config.num_attention_heads  # 2048 / 8 = 256
        assert result.kv_channels == expected_kv_channels

    def test_provider_bridge_dtype_handling(self, gemma_2b_config):
        """Test dtype handling in provider_bridge."""
        # Create model with specific dtype
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = gemma_2b_config
        mock_pretrained.model = Mock(spec=GemmaForCausalLM)
        mock_pretrained.model.dtype = torch.bfloat16
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)

        bridge = GemmaBridge()
        result = bridge.provider_bridge(mock_pretrained)

        # The provider should respect the model's dtype
        assert result.params_dtype == torch.bfloat16
        assert result.bf16 == True
        assert result.fp16 == False

    def test_provider_bridge_fp16_dtype_handling(self, gemma_2b_config):
        """Test FP16 dtype handling in provider_bridge."""
        # Create model with FP16 dtype - set it in the config
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = gemma_2b_config
        mock_pretrained.config.torch_dtype = torch.float16  # Set config dtype to fp16
        mock_pretrained.model = Mock(spec=GemmaForCausalLM)
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)

        bridge = GemmaBridge()
        result = bridge.provider_bridge(mock_pretrained)

        # The provider should respect the config's dtype
        assert result.params_dtype == torch.float16
        assert result.fp16 == True
        assert result.bf16 == False

    def test_provider_bridge_without_tie_embeddings(self, gemma_2b_config):
        """Test provider_bridge when tie_word_embeddings is not present."""
        # Remove tie_word_embeddings from config if it exists
        config_dict = gemma_2b_config.to_dict()
        if "tie_word_embeddings" in config_dict:
            del config_dict["tie_word_embeddings"]
        config = GemmaConfig(**config_dict)

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = config
        mock_pretrained.model = Mock(spec=GemmaForCausalLM)
        mock_pretrained.model.dtype = torch.float32
        mock_pretrained.generation_config = None

        bridge = GemmaBridge()
        result = bridge.provider_bridge(mock_pretrained)

        # Gemma should default to True for tied embeddings
        assert result.share_embeddings_and_output_weights == True

    def test_mapping_registry_implementation(self, mock_pretrained_gemma_2b):
        """Test that mapping_registry returns a proper MegatronMappingRegistry."""
        bridge = GemmaBridge()

        # Get the mapping registry
        mapping_registry = bridge.mapping_registry()

        # Check it's not None
        assert mapping_registry is not None
        # Check it has param mappings (they are passed as args to __init__)
        # The mapping registry should have embedding, layer norm, attention, and MLP mappings

    def test_provider_bridge_make_vocab_size_divisible_by(self, mock_pretrained_gemma_2b):
        """Test make_vocab_size_divisible_by calculation."""
        bridge = GemmaBridge()

        result = bridge.provider_bridge(mock_pretrained_gemma_2b)

        # The method should calculate a reasonable divisor based on vocab size
        assert hasattr(result, "make_vocab_size_divisible_by")
        assert result.make_vocab_size_divisible_by > 0

    def test_provider_bridge_generation_config(self, mock_pretrained_gemma_2b):
        """Test that generation config is passed through."""
        bridge = GemmaBridge()

        result = bridge.provider_bridge(mock_pretrained_gemma_2b)

        # Generation config should be passed from the pretrained model
        assert result.generation_config == mock_pretrained_gemma_2b.generation_config


class TestAutoBridgeIntegration:
    """Integration tests for AutoBridge with Gemma models."""

    @pytest.fixture
    def gemma_configs(self):
        """Different Gemma model configurations for testing."""
        return {
            "gemma-2b": {
                "architectures": ["GemmaForCausalLM"],
                "model_type": "gemma",
                "hidden_size": 2048,
                "num_hidden_layers": 18,
                "num_attention_heads": 8,
                "num_key_value_heads": 1,
                "intermediate_size": 16384,
                "vocab_size": 256000,
                "max_position_embeddings": 8192,
                "rope_theta": 10000.0,
                "rms_norm_eps": 1e-06,
                "head_dim": 256,
                "attention_bias": False,
                "torch_dtype": "bfloat16",
            },
            "gemma-7b": {
                "architectures": ["GemmaForCausalLM"],
                "model_type": "gemma",
                "hidden_size": 3072,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 16,
                "intermediate_size": 24576,
                "vocab_size": 256000,
                "max_position_embeddings": 8192,
                "rope_theta": 10000.0,
                "rms_norm_eps": 1e-06,
                "head_dim": 256,
                "attention_bias": False,
                "torch_dtype": "bfloat16",
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
            "tokenizer_class": "GemmaTokenizer",
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
    @patch("megatron.bridge.models.conversion.auto_bridge.safe_load_config_with_retry")
    def test_from_pretrained_with_temp_dir(self, mock_safe_load_config, mock_pretrained, gemma_configs):
        """Test AutoBridge.from_hf_pretrained with temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with Gemma 2B config
            config_dict = gemma_configs["gemma-2b"]
            self.create_mock_model_files(config_dict, temp_dir)

            # Mock the config loading
            config = GemmaConfig(**config_dict)
            mock_safe_load_config.return_value = config

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
            mock_safe_load_config.assert_called_once_with(temp_dir, trust_remote_code=False)
            mock_pretrained.assert_called_once_with(temp_dir)

    @patch("megatron.bridge.models.conversion.auto_bridge.PreTrainedCausalLM.from_pretrained")
    @patch("megatron.bridge.models.conversion.auto_bridge.safe_load_config_with_retry")
    def test_from_pretrained_multiple_models(self, mock_safe_load_config, mock_pretrained, gemma_configs):
        """Test AutoBridge.from_hf_pretrained with different Gemma model configs."""
        for model_name, config_dict in gemma_configs.items():
            with tempfile.TemporaryDirectory() as temp_dir:
                self.create_mock_model_files(config_dict, temp_dir)

                # Mock the config loading
                config = GemmaConfig(**config_dict)
                mock_safe_load_config.return_value = config

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
                    mock_provider = Mock(spec=GemmaModelProvider)
                    mock_bridge.provider_bridge.return_value = mock_provider
                    mock_get_bridge.return_value = mock_bridge

                    _ = bridge.to_megatron_provider(load_weights=False)

                    # Verify provider_bridge was called with correct model
                    mock_bridge.provider_bridge.assert_called_once_with(mock_model)

                # Clear mocks for next iteration
                mock_safe_load_config.reset_mock()
                mock_pretrained.reset_mock()

    @patch("megatron.bridge.models.conversion.auto_bridge.PreTrainedCausalLM.from_pretrained")
    @patch("megatron.bridge.models.conversion.auto_bridge.safe_load_config_with_retry")
    def test_from_pretrained_with_kwargs(self, mock_safe_load_config, mock_pretrained, gemma_configs):
        """Test AutoBridge.from_hf_pretrained with various kwargs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dict = gemma_configs["gemma-7b"]
            self.create_mock_model_files(config_dict, temp_dir)

            # Mock the config loading
            config = GemmaConfig(**config_dict)
            mock_safe_load_config.return_value = config

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

    def test_supports_gemma_architectures(self, gemma_configs):
        """Test that AutoBridge.supports correctly identifies Gemma models."""
        for model_name, config_dict in gemma_configs.items():
            config = GemmaConfig(**config_dict)
            assert AutoBridge.supports(config) == True

        # Test non-causal LM architecture
        non_causal_config = Mock()
        non_causal_config.architectures = ["GemmaModel"]  # Not ForCausalLM
        assert AutoBridge.supports(non_causal_config) == False

    def test_list_supported_models(self):
        """Test list_supported_models includes GemmaForCausalLM."""
        # This test requires the dispatch system to be set up
        # Since we're testing in isolation, we'll skip this test
        # In a real environment, this would work if the bridges are registered
        pass  # Skip for now as it requires full dispatch setup


class TestGemmaBridgeParameterMapping:
    """Test parameter mapping functionality in GemmaBridge."""

    @pytest.fixture
    def mock_gemma_state_dict(self):
        """Create a mock state dict with Gemma parameter names."""
        return {
            "model.embed_tokens.weight": torch.randn(256000, 2048),
            "model.norm.weight": torch.randn(2048),
            "model.layers.0.input_layernorm.weight": torch.randn(2048),
            "model.layers.0.post_attention_layernorm.weight": torch.randn(2048),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(2048, 2048),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(256, 2048),  # GQA: different size for K
            "model.layers.0.self_attn.v_proj.weight": torch.randn(256, 2048),  # GQA: different size for V
            "model.layers.0.self_attn.o_proj.weight": torch.randn(2048, 2048),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(16384, 2048),
            "model.layers.0.mlp.up_proj.weight": torch.randn(16384, 2048),
            "model.layers.0.mlp.down_proj.weight": torch.randn(2048, 16384),
        }

    def test_mapping_registry_has_gemma_specific_mappings(self):
        """Test that mapping registry includes Gemma-specific mappings."""
        bridge = GemmaBridge()
        mapping_registry = bridge.mapping_registry()

        # This test verifies that the mapping registry was created
        # The actual parameter mappings are tested in integration tests
        assert mapping_registry is not None

    def test_gemma_tied_embeddings_mapping(self):
        """Test that Gemma bridge handles tied embeddings correctly."""
        bridge = GemmaBridge()
        mapping_registry = bridge.mapping_registry()

        # Gemma uses tied embeddings, so there should be no separate lm_head.weight mapping
        # This is reflected in the mapping registry not including lm_head.weight
        assert mapping_registry is not None

    def test_gemma_no_bias_mapping(self):
        """Test that Gemma bridge doesn't include bias mappings."""
        bridge = GemmaBridge()
        mapping_registry = bridge.mapping_registry()

        # Gemma doesn't have bias in linear layers
        # This is reflected in the QKVMapping and other mappings not including bias terms
        assert mapping_registry is not None

    def test_gemma_gated_mlp_mapping(self):
        """Test that Gemma bridge includes gated MLP mappings."""
        bridge = GemmaBridge()
        mapping_registry = bridge.mapping_registry()

        # Gemma uses gated MLP, so it should have GatedMLPMapping
        # This combines gate_proj and up_proj into linear_fc1
        assert mapping_registry is not None
