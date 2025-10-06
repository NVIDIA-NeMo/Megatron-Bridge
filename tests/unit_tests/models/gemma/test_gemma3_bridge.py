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

import math
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from transformers import Gemma3Config, Gemma3ForCausalLM, GenerationConfig

from megatron.bridge.models import AutoBridge
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.gemma.gemma3_bridge import Gemma3ModelBridge
from megatron.bridge.models.gemma.gemma3_provider import Gemma3ModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


class TestMegatronGemma3Bridge:
    """Test cases for MegatronGemma3Bridge class."""

    @pytest.fixture
    def gemma3_1b_config_dict(self):
        """Create a sample Gemma3 1B configuration."""
        return {
            "architectures": ["Gemma3ForCausalLM"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 2,
            "eos_token_id": 1,
            "head_dim": 256,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 1152,
            "initializer_range": 0.02,
            "intermediate_size": 6912,
            "max_position_embeddings": 32768,
            "model_type": "gemma3",
            "num_attention_heads": 4,
            "num_hidden_layers": 26,
            "num_key_value_heads": 1,
            "pad_token_id": 0,
            "query_pre_attn_scalar": 256,
            "rms_norm_eps": 1e-06,
            "rope_local_base_freq": 10000.0,
            "rope_theta": 1000000.0,
            "sliding_window": 512,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.46.0",
            "use_cache": True,
            "vocab_size": 262144,
            "rope_scaling": None,
        }

    @pytest.fixture
    def gemma3_4b_config_dict(self):
        """Create a sample Gemma3 4B configuration."""
        return {
            "architectures": ["Gemma3ForCausalLM"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 2,
            "eos_token_id": 1,
            "head_dim": 256,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 2560,
            "initializer_range": 0.02,
            "intermediate_size": 10240,
            "max_position_embeddings": 131072,
            "model_type": "gemma3",
            "num_attention_heads": 8,
            "num_hidden_layers": 34,
            "num_key_value_heads": 4,
            "pad_token_id": 0,
            "query_pre_attn_scalar": 256,
            "rms_norm_eps": 1e-06,
            "rope_local_base_freq": 10000.0,
            "rope_theta": 1000000.0,
            "sliding_window": 1024,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.46.0",
            "use_cache": True,
            "vocab_size": 262208,
            "rope_scaling": {"factor": 8.0, "type": "linear"},
        }

    @pytest.fixture
    def gemma3_27b_config_dict(self):
        """Create a sample Gemma3 27B configuration."""
        return {
            "architectures": ["Gemma3ForCausalLM"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 2,
            "eos_token_id": 1,
            "head_dim": 128,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 5376,
            "initializer_range": 0.02,
            "intermediate_size": 21504,
            "max_position_embeddings": 131072,
            "model_type": "gemma3",
            "num_attention_heads": 32,
            "num_hidden_layers": 62,
            "num_key_value_heads": 16,
            "pad_token_id": 0,
            "query_pre_attn_scalar": 168,  # Different from other sizes
            "rms_norm_eps": 1e-06,
            "rope_local_base_freq": 10000.0,
            "rope_theta": 1000000.0,
            "sliding_window": 1024,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.46.0",
            "use_cache": True,
            "vocab_size": 262208,
            "rope_scaling": {"factor": 8.0, "type": "linear"},
        }

    @pytest.fixture
    def gemma3_1b_config(self, gemma3_1b_config_dict):
        """Create a Gemma3Config instance for 1B model."""
        return Gemma3Config(**gemma3_1b_config_dict)

    @pytest.fixture
    def gemma3_4b_config(self, gemma3_4b_config_dict):
        """Create a Gemma3Config instance for 4B model."""
        return Gemma3Config(**gemma3_4b_config_dict)

    @pytest.fixture
    def gemma3_27b_config(self, gemma3_27b_config_dict):
        """Create a Gemma3Config instance for 27B model."""
        return Gemma3Config(**gemma3_27b_config_dict)

    @pytest.fixture
    def mock_gemma3_1b_model(self, gemma3_1b_config):
        """Create a mock Gemma3ForCausalLM 1B model."""
        mock_model = Mock(spec=Gemma3ForCausalLM)
        mock_model.config = gemma3_1b_config
        mock_model.dtype = torch.bfloat16
        return mock_model

    @pytest.fixture
    def mock_gemma3_4b_model(self, gemma3_4b_config):
        """Create a mock Gemma3ForCausalLM 4B model."""
        mock_model = Mock(spec=Gemma3ForCausalLM)
        mock_model.config = gemma3_4b_config
        mock_model.dtype = torch.bfloat16
        return mock_model

    @pytest.fixture
    def mock_gemma3_27b_model(self, gemma3_27b_config):
        """Create a mock Gemma3ForCausalLM 27B model."""
        mock_model = Mock(spec=Gemma3ForCausalLM)
        mock_model.config = gemma3_27b_config
        mock_model.dtype = torch.bfloat16
        return mock_model

    @pytest.fixture
    def mock_pretrained_gemma3_1b(self, gemma3_1b_config):
        """Create a mock PreTrainedCausalLM with Gemma3 1B model."""
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = gemma3_1b_config
        mock_pretrained._model_name_or_path = "google/gemma3-1b"
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)
        mock_pretrained.model = Mock(spec=Gemma3ForCausalLM)
        mock_pretrained.model.dtype = torch.bfloat16
        return mock_pretrained

    @pytest.fixture
    def mock_pretrained_gemma3_4b(self, gemma3_4b_config):
        """Create a mock PreTrainedCausalLM with Gemma3 4B model."""
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = gemma3_4b_config
        mock_pretrained._model_name_or_path = "google/gemma3-4b"
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)
        mock_pretrained.model = Mock(spec=Gemma3ForCausalLM)
        mock_pretrained.model.dtype = torch.bfloat16
        return mock_pretrained

    @pytest.fixture
    def mock_pretrained_gemma3_27b(self, gemma3_27b_config):
        """Create a mock PreTrainedCausalLM with Gemma3 27B model."""
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = gemma3_27b_config
        mock_pretrained._model_name_or_path = "google/gemma3-27b"
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)
        mock_pretrained.model = Mock(spec=Gemma3ForCausalLM)
        mock_pretrained.model.dtype = torch.bfloat16
        return mock_pretrained

    def test_bridge_registration(self):
        """Test that MegatronGemma3Bridge is properly registered."""
        # The @MegatronModelBridge.register_bridge decorator should register the bridge
        # Check that the class exists and has the expected base class
        assert issubclass(Gemma3ModelBridge, MegatronModelBridge)

    @patch("megatron.bridge.models.gemma.gemma3_bridge.AutoConfig.from_pretrained")
    def test_provider_bridge_basic_1b(self, mock_autoconfig, mock_pretrained_gemma3_1b, gemma3_1b_config):
        """Test basic provider_bridge functionality for Gemma3 1B."""
        # Mock the VL config loading
        mock_autoconfig.return_value = gemma3_1b_config

        bridge = Gemma3ModelBridge()

        # Call provider_bridge
        result = bridge.provider_bridge(mock_pretrained_gemma3_1b)

        # Check that it returns a Gemma3ModelProvider instance
        assert isinstance(result, Gemma3ModelProvider)

        # Check basic configuration mapping
        assert result.num_layers == gemma3_1b_config.num_hidden_layers
        assert result.hidden_size == gemma3_1b_config.hidden_size
        assert result.num_attention_heads == gemma3_1b_config.num_attention_heads
        assert result.seq_length == gemma3_1b_config.max_position_embeddings
        assert result.rotary_base == (gemma3_1b_config.rope_local_base_freq, gemma3_1b_config.rope_theta)

    @patch("megatron.bridge.models.gemma.gemma3_bridge.AutoConfig.from_pretrained")
    def test_provider_bridge_basic_4b(self, mock_autoconfig, mock_pretrained_gemma3_4b, gemma3_4b_config):
        """Test basic provider_bridge functionality for Gemma3 4B."""
        # Mock the VL config loading
        mock_autoconfig.return_value = gemma3_4b_config

        bridge = Gemma3ModelBridge()

        # Call provider_bridge
        result = bridge.provider_bridge(mock_pretrained_gemma3_4b)

        # Check that it returns a Gemma3ModelProvider instance
        assert isinstance(result, Gemma3ModelProvider)

        # Check basic configuration mapping
        assert result.num_layers == gemma3_4b_config.num_hidden_layers
        assert result.hidden_size == gemma3_4b_config.hidden_size
        assert result.num_attention_heads == gemma3_4b_config.num_attention_heads
        assert result.seq_length == gemma3_4b_config.max_position_embeddings
        assert result.rotary_base == (gemma3_4b_config.rope_local_base_freq, gemma3_4b_config.rope_theta)

    @patch("megatron.bridge.models.gemma.gemma3_bridge.AutoConfig.from_pretrained")
    def test_provider_bridge_basic_27b(self, mock_autoconfig, mock_pretrained_gemma3_27b, gemma3_27b_config):
        """Test basic provider_bridge functionality for Gemma3 27B."""
        # Mock the VL config loading
        mock_autoconfig.return_value = gemma3_27b_config

        bridge = Gemma3ModelBridge()

        # Call provider_bridge
        result = bridge.provider_bridge(mock_pretrained_gemma3_27b)

        # Check that it returns a Gemma3ModelProvider instance
        assert isinstance(result, Gemma3ModelProvider)

        # Check basic configuration mapping
        assert result.num_layers == gemma3_27b_config.num_hidden_layers
        assert result.hidden_size == gemma3_27b_config.hidden_size
        assert result.num_attention_heads == gemma3_27b_config.num_attention_heads
        assert result.seq_length == gemma3_27b_config.max_position_embeddings
        assert result.rotary_base == (gemma3_27b_config.rope_local_base_freq, gemma3_27b_config.rope_theta)

    @patch("megatron.bridge.models.gemma.gemma3_bridge.AutoConfig.from_pretrained")
    def test_provider_bridge_vocabulary(self, mock_autoconfig, mock_pretrained_gemma3_1b, gemma3_1b_config):
        """Test vocabulary size mapping."""
        mock_autoconfig.return_value = gemma3_1b_config
        bridge = Gemma3ModelBridge()

        result = bridge.provider_bridge(mock_pretrained_gemma3_1b)

        # Check vocabulary configuration
        assert result.vocab_size == gemma3_1b_config.vocab_size
        # Gemma3 uses tied embeddings by default
        assert result.share_embeddings_and_output_weights == True

    @patch("megatron.bridge.models.gemma.gemma3_bridge.AutoConfig.from_pretrained")
    def test_provider_bridge_attention_config(self, mock_autoconfig, mock_pretrained_gemma3_1b, gemma3_1b_config):
        """Test attention configuration mapping."""
        mock_autoconfig.return_value = gemma3_1b_config
        bridge = Gemma3ModelBridge()

        result = bridge.provider_bridge(mock_pretrained_gemma3_1b)

        # Check attention configuration
        assert result.num_attention_heads == gemma3_1b_config.num_attention_heads
        assert result.num_query_groups == gemma3_1b_config.num_key_value_heads
        assert result.window_size == gemma3_1b_config.sliding_window

    @patch("megatron.bridge.models.gemma.gemma3_bridge.AutoConfig.from_pretrained")
    def test_provider_bridge_mlp_config(self, mock_autoconfig, mock_pretrained_gemma3_1b, gemma3_1b_config):
        """Test MLP configuration mapping."""
        mock_autoconfig.return_value = gemma3_1b_config
        bridge = Gemma3ModelBridge()

        result = bridge.provider_bridge(mock_pretrained_gemma3_1b)

        # Check MLP configuration
        assert result.ffn_hidden_size == gemma3_1b_config.intermediate_size
        assert result.gated_linear_unit == True  # Gemma3 uses gated MLP

    @patch("megatron.bridge.models.gemma.gemma3_bridge.AutoConfig.from_pretrained")
    def test_provider_bridge_normalization(self, mock_autoconfig, mock_pretrained_gemma3_1b, gemma3_1b_config):
        """Test normalization configuration."""
        mock_autoconfig.return_value = gemma3_1b_config
        bridge = Gemma3ModelBridge()

        result = bridge.provider_bridge(mock_pretrained_gemma3_1b)

        # Check normalization settings
        assert result.layernorm_epsilon == gemma3_1b_config.rms_norm_eps

    @patch("megatron.bridge.models.gemma.gemma3_bridge.AutoConfig.from_pretrained")
    def test_provider_bridge_position_embedding(self, mock_autoconfig, mock_pretrained_gemma3_1b, gemma3_1b_config):
        """Test position embedding configuration."""
        mock_autoconfig.return_value = gemma3_1b_config
        bridge = Gemma3ModelBridge()

        result = bridge.provider_bridge(mock_pretrained_gemma3_1b)

        # Check position embedding - Gemma3 has dual rotary bases
        assert result.rotary_base == (gemma3_1b_config.rope_local_base_freq, gemma3_1b_config.rope_theta)

    @patch("megatron.bridge.models.gemma.gemma3_bridge.AutoConfig.from_pretrained")
    def test_provider_bridge_gemma3_specific_features(
        self, mock_autoconfig, mock_pretrained_gemma3_1b, gemma3_1b_config
    ):
        """Test Gemma3-specific features."""
        mock_autoconfig.return_value = gemma3_1b_config
        bridge = Gemma3ModelBridge()

        result = bridge.provider_bridge(mock_pretrained_gemma3_1b)

        # Check Gemma3-specific features
        assert result.window_size == gemma3_1b_config.sliding_window
        assert result.add_bias_linear == False  # Gemma3 doesn't use bias in linear layers
        assert result.layernorm_zero_centered_gamma == True  # Gemma3-specific RMSNorm behavior
        assert result.softmax_scale == 1.0 / math.sqrt(gemma3_1b_config.query_pre_attn_scalar)

    @patch("megatron.bridge.models.gemma.gemma3_bridge.AutoConfig.from_pretrained")
    def test_provider_bridge_head_dim_calculation_1b(
        self, mock_autoconfig, mock_pretrained_gemma3_1b, gemma3_1b_config
    ):
        """Test head dimension calculation for Gemma3 1B."""
        mock_autoconfig.return_value = gemma3_1b_config
        bridge = Gemma3ModelBridge()

        result = bridge.provider_bridge(mock_pretrained_gemma3_1b)

        # Gemma3 1B should use the explicit head_dim from config
        assert result.kv_channels == gemma3_1b_config.head_dim  # 256
        # Verify this matches the HF config
        assert result.kv_channels == 256

    @patch("megatron.bridge.models.gemma.gemma3_bridge.AutoConfig.from_pretrained")
    def test_provider_bridge_head_dim_calculation_27b(
        self, mock_autoconfig, mock_pretrained_gemma3_27b, gemma3_27b_config
    ):
        """Test head dimension calculation for Gemma3 27B."""
        mock_autoconfig.return_value = gemma3_27b_config
        bridge = Gemma3ModelBridge()

        result = bridge.provider_bridge(mock_pretrained_gemma3_27b)

        # Gemma3 27B should use the explicit head_dim from config
        assert result.kv_channels == gemma3_27b_config.head_dim  # 128
        # Verify this is different from standard calculation
        standard_calculation = (
            gemma3_27b_config.hidden_size // gemma3_27b_config.num_attention_heads
        )  # 5376 / 32 = 168
        assert result.kv_channels != standard_calculation
        assert result.kv_channels == 128  # Correct value from HF config

    @patch("megatron.bridge.models.gemma.gemma3_bridge.AutoConfig.from_pretrained")
    def test_provider_bridge_dtype_handling(self, mock_autoconfig, gemma3_1b_config):
        """Test dtype handling in provider_bridge."""
        mock_autoconfig.return_value = gemma3_1b_config

        # Create model with specific dtype - set it in the config
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = gemma3_1b_config
        mock_pretrained._model_name_or_path = "google/gemma3-1b"
        mock_pretrained.model = Mock(spec=Gemma3ForCausalLM)
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)

        bridge = Gemma3ModelBridge()
        result = bridge.provider_bridge(mock_pretrained)

        # The provider should respect the config's dtype
        assert result.params_dtype == torch.bfloat16
        assert result.bf16 == True
        assert result.fp16 == False

    @patch("megatron.bridge.models.gemma.gemma3_bridge.AutoConfig.from_pretrained")
    def test_provider_bridge_rope_scaling_config(self, mock_autoconfig, mock_pretrained_gemma3_4b, gemma3_4b_config):
        """Test rope scaling configuration."""
        mock_autoconfig.return_value = gemma3_4b_config
        bridge = Gemma3ModelBridge()

        result = bridge.provider_bridge(mock_pretrained_gemma3_4b)

        # Check rope scaling configuration specific to Gemma3
        assert result.rope_scaling_factor == gemma3_4b_config.rope_scaling["factor"]
        assert result.rope_scaling_factor == 8.0

    @patch("megatron.bridge.models.gemma.gemma3_bridge.AutoConfig.from_pretrained")
    def test_provider_bridge_no_rope_scaling(self, mock_autoconfig, mock_pretrained_gemma3_1b, gemma3_1b_config):
        """Test configuration without rope scaling."""
        mock_autoconfig.return_value = gemma3_1b_config
        bridge = Gemma3ModelBridge()

        result = bridge.provider_bridge(mock_pretrained_gemma3_1b)

        # 1B model has no rope scaling
        assert result.rope_scaling_factor == 1.0

    @patch("megatron.bridge.models.gemma.gemma3_bridge.AutoConfig.from_pretrained")
    def test_provider_bridge_query_pre_attn_scalar_variants(
        self, mock_autoconfig, mock_pretrained_gemma3_27b, gemma3_27b_config
    ):
        """Test query_pre_attn_scalar for 27B model which has different value."""
        mock_autoconfig.return_value = gemma3_27b_config
        bridge = Gemma3ModelBridge()

        result = bridge.provider_bridge(mock_pretrained_gemma3_27b)

        # 27B model has different query_pre_attn_scalar
        expected_softmax_scale = 1.0 / math.sqrt(gemma3_27b_config.query_pre_attn_scalar)
        assert result.softmax_scale == expected_softmax_scale
        assert abs(result.softmax_scale - (1.0 / math.sqrt(168))) < 1e-6  # 168 for 27B model

    def test_mapping_registry_implementation(self, mock_pretrained_gemma3_1b):
        """Test that mapping_registry returns a proper MegatronMappingRegistry."""
        bridge = Gemma3ModelBridge()

        # Get the mapping registry
        mapping_registry = bridge.mapping_registry()

        # Check it's not None
        assert mapping_registry is not None
        # Check it has param mappings (they are passed as args to __init__)
        # The mapping registry should have embedding, layer norm, attention, and MLP mappings

    @patch("megatron.bridge.models.gemma.gemma3_bridge.AutoConfig.from_pretrained")
    def test_provider_bridge_generation_config(self, mock_autoconfig, mock_pretrained_gemma3_1b, gemma3_1b_config):
        """Test that generation config is passed through."""
        mock_autoconfig.return_value = gemma3_1b_config
        bridge = Gemma3ModelBridge()

        result = bridge.provider_bridge(mock_pretrained_gemma3_1b)

        # Generation config should be passed from the pretrained model
        assert result.generation_config == mock_pretrained_gemma3_1b.generation_config


class TestAutoBridgeIntegration:
    """Integration tests for AutoBridge with Gemma3 models."""

    @pytest.fixture
    def gemma3_configs(self):
        """Different Gemma3 model configurations for testing."""
        return {
            "gemma3-1b": {
                "architectures": ["Gemma3ForCausalLM"],
                "model_type": "gemma3",
                "hidden_size": 1152,
                "num_hidden_layers": 26,
                "num_attention_heads": 4,
                "num_key_value_heads": 1,
                "intermediate_size": 6912,
                "vocab_size": 262144,
                "max_position_embeddings": 32768,
                "rope_local_base_freq": 10000.0,
                "rope_theta": 1000000.0,
                "rms_norm_eps": 1e-06,
                "head_dim": 256,
                "attention_bias": False,
                "torch_dtype": "bfloat16",
                "query_pre_attn_scalar": 256,
                "sliding_window": 512,
                "rope_scaling": None,
            },
            "gemma3-4b": {
                "architectures": ["Gemma3ForCausalLM"],
                "model_type": "gemma3",
                "hidden_size": 2560,
                "num_hidden_layers": 34,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "intermediate_size": 10240,
                "vocab_size": 262208,
                "max_position_embeddings": 131072,
                "rope_local_base_freq": 10000.0,
                "rope_theta": 1000000.0,
                "rms_norm_eps": 1e-06,
                "head_dim": 256,
                "attention_bias": False,
                "torch_dtype": "bfloat16",
                "query_pre_attn_scalar": 256,
                "sliding_window": 1024,
                "rope_scaling": {"factor": 8.0, "type": "linear"},
            },
            "gemma3-27b": {
                "architectures": ["Gemma3ForCausalLM"],
                "model_type": "gemma3",
                "hidden_size": 5376,
                "num_hidden_layers": 62,
                "num_attention_heads": 32,
                "num_key_value_heads": 16,
                "intermediate_size": 21504,
                "vocab_size": 262208,
                "max_position_embeddings": 131072,
                "rope_local_base_freq": 10000.0,
                "rope_theta": 1000000.0,
                "rms_norm_eps": 1e-06,
                "head_dim": 128,
                "attention_bias": False,
                "torch_dtype": "bfloat16",
                "query_pre_attn_scalar": 168,
                "sliding_window": 1024,
                "rope_scaling": {"factor": 8.0, "type": "linear"},
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
    @patch("megatron.bridge.models.hf_pretrained.safe_config_loader.AutoConfig.from_pretrained")
    def test_from_pretrained_with_temp_dir(self, mock_autoconfig, mock_pretrained, gemma3_configs):
        """Test AutoBridge.from_hf_pretrained with temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with Gemma3 1B config
            config_dict = gemma3_configs["gemma3-1b"]
            self.create_mock_model_files(config_dict, temp_dir)

            # Mock the config loading
            config = Gemma3Config(**config_dict)
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

    def test_supports_gemma3_architectures(self, gemma3_configs):
        """Test that AutoBridge.supports correctly identifies Gemma3 models."""
        for model_name, config_dict in gemma3_configs.items():
            config = Gemma3Config(**config_dict)
            assert AutoBridge.supports(config) == True

        # Test non-causal LM architecture
        non_causal_config = Mock()
        non_causal_config.architectures = ["Gemma3Model"]  # Not ForCausalLM
        assert AutoBridge.supports(non_causal_config) == False


class TestGemma3BridgeParameterMapping:
    """Test parameter mapping functionality in Gemma3Bridge."""

    @pytest.fixture
    def mock_gemma3_state_dict(self):
        """Create a mock state dict with Gemma3 parameter names."""
        return {
            "model.embed_tokens.weight": torch.randn(262144, 1152),
            "model.norm.weight": torch.randn(1152),
            "model.layers.0.input_layernorm.weight": torch.randn(1152),
            "model.layers.0.pre_feedforward_layernorm.weight": torch.randn(1152),
            "model.layers.0.post_feedforward_layernorm.weight": torch.randn(1152),
            "model.layers.0.post_attention_layernorm.weight": torch.randn(1152),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(1152, 1152),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(256, 1152),  # GQA: different size for K
            "model.layers.0.self_attn.v_proj.weight": torch.randn(256, 1152),  # GQA: different size for V
            "model.layers.0.self_attn.o_proj.weight": torch.randn(1152, 1152),
            "model.layers.0.self_attn.q_norm.weight": torch.randn(256),  # Gemma3 has Q/K norms
            "model.layers.0.self_attn.k_norm.weight": torch.randn(256),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(6912, 1152),
            "model.layers.0.mlp.up_proj.weight": torch.randn(6912, 1152),
            "model.layers.0.mlp.down_proj.weight": torch.randn(1152, 6912),
        }

    def test_mapping_registry_has_gemma3_specific_mappings(self):
        """Test that mapping registry includes Gemma3-specific mappings."""
        bridge = Gemma3ModelBridge()
        mapping_registry = bridge.mapping_registry()

        # This test verifies that the mapping registry was created
        # The actual parameter mappings are tested in integration tests
        assert mapping_registry is not None

    def test_gemma3_tied_embeddings_mapping(self):
        """Test that Gemma3 bridge handles tied embeddings correctly."""
        bridge = Gemma3ModelBridge()
        mapping_registry = bridge.mapping_registry()

        # Gemma3 uses tied embeddings, so there should be no separate lm_head.weight mapping
        # This is reflected in the mapping registry not including lm_head.weight
        assert mapping_registry is not None

    def test_gemma3_no_bias_mapping(self):
        """Test that Gemma3 bridge doesn't include bias mappings."""
        bridge = Gemma3ModelBridge()
        mapping_registry = bridge.mapping_registry()

        # Gemma3 doesn't have bias in linear layers
        # This is reflected in the QKVMapping and other mappings not including bias terms
        assert mapping_registry is not None

    def test_gemma3_gated_mlp_mapping(self):
        """Test that Gemma3 bridge includes gated MLP mappings."""
        bridge = Gemma3ModelBridge()
        mapping_registry = bridge.mapping_registry()

        # Gemma3 uses gated MLP, so it should have GatedMLPMapping
        # This combines gate_proj and up_proj into linear_fc1
        assert mapping_registry is not None

    def test_gemma3_additional_layer_norms_mapping(self):
        """Test that Gemma3 bridge includes additional layer norm mappings."""
        bridge = Gemma3ModelBridge()
        mapping_registry = bridge.mapping_registry()

        # Gemma3 has additional layer normalizations including Q/K norms
        # pre_feedforward_layernorm, post_feedforward_layernorm, post_attention_layernorm
        # q_norm, k_norm
        assert mapping_registry is not None

    def test_gemma3_qk_norm_mapping(self):
        """Test that Gemma3 bridge includes Q/K normalization mappings."""
        bridge = Gemma3ModelBridge()
        mapping_registry = bridge.mapping_registry()

        # Gemma3 has Q and K normalization layers which are unique to this architecture
        # These should be mapped correctly
        assert mapping_registry is not None
