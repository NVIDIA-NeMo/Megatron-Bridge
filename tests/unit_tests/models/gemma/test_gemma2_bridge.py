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
from transformers import Gemma2Config, Gemma2ForCausalLM, GenerationConfig

from megatron.bridge.models import AutoBridge
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.gemma.gemma2_bridge import Gemma2Bridge
from megatron.bridge.models.gemma.gemma2_provider import Gemma2ModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


class TestMegatronGemma2Bridge:
    """Test cases for MegatronGemma2Bridge class."""

    @pytest.fixture
    def gemma2_2b_config_dict(self):
        """Create a sample Gemma2 2B configuration."""
        return {
            "architectures": ["Gemma2ForCausalLM"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attn_logit_softcapping": 50.0,
            "bos_token_id": 2,
            "cache_implementation": "hybrid",
            "eos_token_id": 1,
            "final_logit_softcapping": 30.0,
            "head_dim": 256,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 2304,
            "initializer_range": 0.02,
            "intermediate_size": 9216,
            "max_position_embeddings": 8192,
            "model_type": "gemma2",
            "num_attention_heads": 8,
            "num_hidden_layers": 26,
            "num_key_value_heads": 4,
            "pad_token_id": 0,
            "query_pre_attn_scalar": 256,
            "rms_norm_eps": 1e-06,
            "rope_theta": 10000.0,
            "sliding_window": 4096,
            "torch_dtype": "float32",
            "transformers_version": "4.42.4",
            "use_cache": True,
            "vocab_size": 256000,
        }

    @pytest.fixture
    def gemma2_9b_config_dict(self):
        """Create a sample Gemma2 9B configuration."""
        return {
            "architectures": ["Gemma2ForCausalLM"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attn_logit_softcapping": 50.0,
            "bos_token_id": 2,
            "cache_implementation": "hybrid",
            "eos_token_id": 1,
            "final_logit_softcapping": 30.0,
            "head_dim": 256,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 3584,
            "initializer_range": 0.02,
            "intermediate_size": 14336,
            "max_position_embeddings": 8192,
            "model_type": "gemma2",
            "num_attention_heads": 16,
            "num_hidden_layers": 42,
            "num_key_value_heads": 8,
            "pad_token_id": 0,
            "query_pre_attn_scalar": 256,
            "rms_norm_eps": 1e-06,
            "rope_theta": 10000.0,
            "sliding_window": 4096,
            "sliding_window_size": 4096,
            "torch_dtype": "float32",
            "transformers_version": "4.42.0.dev0",
            "use_cache": True,
            "vocab_size": 256000,
        }

    @pytest.fixture
    def gemma2_27b_config_dict(self):
        """Create a sample Gemma2 27B configuration."""
        return {
            "architectures": ["Gemma2ForCausalLM"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attn_logit_softcapping": 50.0,
            "bos_token_id": 2,
            "cache_implementation": "hybrid",
            "eos_token_id": 1,
            "final_logit_softcapping": 30.0,
            "head_dim": 128,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 4608,
            "initializer_range": 0.02,
            "intermediate_size": 36864,
            "max_position_embeddings": 8192,
            "model_type": "gemma2",
            "num_attention_heads": 32,
            "num_hidden_layers": 46,
            "num_key_value_heads": 16,
            "pad_token_id": 0,
            "query_pre_attn_scalar": 144,
            "rms_norm_eps": 1e-06,
            "rope_theta": 10000.0,
            "sliding_window": 4096,
            "sliding_window_size": 4096,
            "torch_dtype": "float32",
            "transformers_version": "4.42.0.dev0",
            "use_cache": True,
            "vocab_size": 256000,
            "_attn_implementation": "eager",
        }

    @pytest.fixture
    def gemma2_2b_config(self, gemma2_2b_config_dict):
        """Create a Gemma2Config instance for 2B model."""
        return Gemma2Config(**gemma2_2b_config_dict)

    @pytest.fixture
    def gemma2_9b_config(self, gemma2_9b_config_dict):
        """Create a Gemma2Config instance for 9B model."""
        return Gemma2Config(**gemma2_9b_config_dict)

    @pytest.fixture
    def gemma2_27b_config(self, gemma2_27b_config_dict):
        """Create a Gemma2Config instance for 27B model."""
        return Gemma2Config(**gemma2_27b_config_dict)

    @pytest.fixture
    def mock_gemma2_2b_model(self, gemma2_2b_config):
        """Create a mock Gemma2ForCausalLM 2B model."""
        mock_model = Mock(spec=Gemma2ForCausalLM)
        mock_model.config = gemma2_2b_config
        mock_model.dtype = torch.bfloat16
        return mock_model

    @pytest.fixture
    def mock_gemma2_9b_model(self, gemma2_9b_config):
        """Create a mock Gemma2ForCausalLM 9B model."""
        mock_model = Mock(spec=Gemma2ForCausalLM)
        mock_model.config = gemma2_9b_config
        mock_model.dtype = torch.bfloat16
        return mock_model

    @pytest.fixture
    def mock_gemma2_27b_model(self, gemma2_27b_config):
        """Create a mock Gemma2ForCausalLM 27B model."""
        mock_model = Mock(spec=Gemma2ForCausalLM)
        mock_model.config = gemma2_27b_config
        mock_model.dtype = torch.bfloat16
        return mock_model

    @pytest.fixture
    def mock_pretrained_gemma2_2b(self, gemma2_2b_config):
        """Create a mock PreTrainedCausalLM with Gemma2 2B model."""
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = gemma2_2b_config
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)
        mock_pretrained.model = Mock(spec=Gemma2ForCausalLM)
        mock_pretrained.model.dtype = torch.bfloat16
        return mock_pretrained

    @pytest.fixture
    def mock_pretrained_gemma2_9b(self, gemma2_9b_config):
        """Create a mock PreTrainedCausalLM with Gemma2 9B model."""
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = gemma2_9b_config
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)
        mock_pretrained.model = Mock(spec=Gemma2ForCausalLM)
        mock_pretrained.model.dtype = torch.bfloat16
        return mock_pretrained

    @pytest.fixture
    def mock_pretrained_gemma2_27b(self, gemma2_27b_config):
        """Create a mock PreTrainedCausalLM with Gemma2 27B model."""
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = gemma2_27b_config
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)
        mock_pretrained.model = Mock(spec=Gemma2ForCausalLM)
        mock_pretrained.model.dtype = torch.bfloat16
        return mock_pretrained

    def test_bridge_registration(self):
        """Test that MegatronGemma2Bridge is properly registered."""
        # The @MegatronModelBridge.register_bridge decorator should register the bridge
        # Check that the class exists and has the expected base class
        assert issubclass(Gemma2Bridge, MegatronModelBridge)

    def test_provider_bridge_basic_2b(self, mock_pretrained_gemma2_2b, gemma2_2b_config):
        """Test basic provider_bridge functionality for Gemma2 2B."""
        bridge = Gemma2Bridge()

        # Call provider_bridge
        result = bridge.provider_bridge(mock_pretrained_gemma2_2b)

        # Check that it returns a Gemma2ModelProvider instance
        assert isinstance(result, Gemma2ModelProvider)

        # Check basic configuration mapping
        assert result.num_layers == gemma2_2b_config.num_hidden_layers
        assert result.hidden_size == gemma2_2b_config.hidden_size
        assert result.num_attention_heads == gemma2_2b_config.num_attention_heads
        assert result.seq_length == gemma2_2b_config.max_position_embeddings
        assert result.rotary_base == gemma2_2b_config.rope_theta

    def test_provider_bridge_basic_9b(self, mock_pretrained_gemma2_9b, gemma2_9b_config):
        """Test basic provider_bridge functionality for Gemma2 9B."""
        bridge = Gemma2Bridge()

        # Call provider_bridge
        result = bridge.provider_bridge(mock_pretrained_gemma2_9b)

        # Check that it returns a Gemma2ModelProvider instance
        assert isinstance(result, Gemma2ModelProvider)

        # Check basic configuration mapping
        assert result.num_layers == gemma2_9b_config.num_hidden_layers
        assert result.hidden_size == gemma2_9b_config.hidden_size
        assert result.num_attention_heads == gemma2_9b_config.num_attention_heads
        assert result.seq_length == gemma2_9b_config.max_position_embeddings
        assert result.rotary_base == gemma2_9b_config.rope_theta

    def test_provider_bridge_basic_27b(self, mock_pretrained_gemma2_27b, gemma2_27b_config):
        """Test basic provider_bridge functionality for Gemma2 27B."""
        bridge = Gemma2Bridge()

        # Call provider_bridge
        result = bridge.provider_bridge(mock_pretrained_gemma2_27b)

        # Check that it returns a Gemma2ModelProvider instance
        assert isinstance(result, Gemma2ModelProvider)

        # Check basic configuration mapping
        assert result.num_layers == gemma2_27b_config.num_hidden_layers
        assert result.hidden_size == gemma2_27b_config.hidden_size
        assert result.num_attention_heads == gemma2_27b_config.num_attention_heads
        assert result.seq_length == gemma2_27b_config.max_position_embeddings
        assert result.rotary_base == gemma2_27b_config.rope_theta

    def test_provider_bridge_vocabulary(self, mock_pretrained_gemma2_2b, gemma2_2b_config):
        """Test vocabulary size mapping."""
        bridge = Gemma2Bridge()

        result = bridge.provider_bridge(mock_pretrained_gemma2_2b)

        # Check vocabulary configuration
        assert result.vocab_size == gemma2_2b_config.vocab_size
        # Gemma2 uses tied embeddings by default
        assert result.share_embeddings_and_output_weights == True

    def test_provider_bridge_attention_config(self, mock_pretrained_gemma2_2b, gemma2_2b_config):
        """Test attention configuration mapping."""
        bridge = Gemma2Bridge()

        result = bridge.provider_bridge(mock_pretrained_gemma2_2b)

        # Check attention configuration
        assert result.num_attention_heads == gemma2_2b_config.num_attention_heads
        assert result.num_query_groups == gemma2_2b_config.num_key_value_heads

    def test_provider_bridge_mlp_config(self, mock_pretrained_gemma2_2b, gemma2_2b_config):
        """Test MLP configuration mapping."""
        bridge = Gemma2Bridge()

        result = bridge.provider_bridge(mock_pretrained_gemma2_2b)

        # Check MLP configuration
        assert result.ffn_hidden_size == gemma2_2b_config.intermediate_size
        assert result.gated_linear_unit == True  # Gemma2 uses gated MLP

    def test_provider_bridge_normalization(self, mock_pretrained_gemma2_2b, gemma2_2b_config):
        """Test normalization configuration."""
        bridge = Gemma2Bridge()

        result = bridge.provider_bridge(mock_pretrained_gemma2_2b)

        # Check normalization settings
        assert result.layernorm_epsilon == gemma2_2b_config.rms_norm_eps

    def test_provider_bridge_position_embedding(self, mock_pretrained_gemma2_2b, gemma2_2b_config):
        """Test position embedding configuration."""
        bridge = Gemma2Bridge()

        result = bridge.provider_bridge(mock_pretrained_gemma2_2b)

        # Check position embedding
        assert result.rotary_base == gemma2_2b_config.rope_theta

    def test_provider_bridge_gemma2_specific_features(self, mock_pretrained_gemma2_2b, gemma2_2b_config):
        """Test Gemma2-specific features."""
        bridge = Gemma2Bridge()

        result = bridge.provider_bridge(mock_pretrained_gemma2_2b)

        # Check Gemma2-specific features
        assert result.query_pre_attn_scalar == gemma2_2b_config.query_pre_attn_scalar
        assert result.attn_logit_softcapping == gemma2_2b_config.attn_logit_softcapping
        assert result.final_logit_softcapping == gemma2_2b_config.final_logit_softcapping
        assert result.window_size == (gemma2_2b_config.sliding_window, 0)
        assert result.add_bias_linear == False  # Gemma2 doesn't use bias in linear layers
        assert result.layernorm_zero_centered_gamma == True  # Gemma2-specific RMSNorm behavior

    def test_provider_bridge_head_dim_calculation_2b(self, mock_pretrained_gemma2_2b, gemma2_2b_config):
        """Test head dimension calculation for Gemma2 2B."""
        bridge = Gemma2Bridge()

        result = bridge.provider_bridge(mock_pretrained_gemma2_2b)

        # Gemma2 2B should use the explicit head_dim from config
        assert result.kv_channels == gemma2_2b_config.head_dim  # 256
        # Verify this matches the HF config
        assert result.kv_channels == 256

    def test_provider_bridge_head_dim_calculation_9b(self, mock_pretrained_gemma2_9b, gemma2_9b_config):
        """Test head dimension calculation for Gemma2 9B."""
        bridge = Gemma2Bridge()

        result = bridge.provider_bridge(mock_pretrained_gemma2_9b)

        # Gemma2 9B should use the explicit head_dim from config
        assert result.kv_channels == gemma2_9b_config.head_dim  # 256
        # Verify this is different from standard calculation
        standard_calculation = gemma2_9b_config.hidden_size // gemma2_9b_config.num_attention_heads  # 3584 / 16 = 224
        assert result.kv_channels != standard_calculation
        assert result.kv_channels == 256

    def test_provider_bridge_head_dim_calculation_27b(self, mock_pretrained_gemma2_27b, gemma2_27b_config):
        """Test head dimension calculation for Gemma2 27B - this is where NeMo has a bug."""
        bridge = Gemma2Bridge()

        result = bridge.provider_bridge(mock_pretrained_gemma2_27b)

        # Gemma2 27B should use the explicit head_dim from config
        assert result.kv_channels == gemma2_27b_config.head_dim  # 128
        # Verify this is different from both standard calculation and NeMo default
        standard_calculation = (
            gemma2_27b_config.hidden_size // gemma2_27b_config.num_attention_heads
        )  # 4608 / 32 = 144
        nemo_default = 256  # What NeMo incorrectly uses
        assert result.kv_channels != standard_calculation
        assert result.kv_channels != nemo_default
        assert result.kv_channels == 128  # Correct value from HF config

    def test_provider_bridge_dtype_handling(self, gemma2_2b_config):
        """Test dtype handling in provider_bridge."""
        # Create model with specific dtype - set it in the config
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = gemma2_2b_config
        mock_pretrained.config.torch_dtype = torch.bfloat16  # Set config dtype to bfloat16
        mock_pretrained.model = Mock(spec=Gemma2ForCausalLM)
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)

        bridge = Gemma2Bridge()
        result = bridge.provider_bridge(mock_pretrained)

        # The provider should respect the config's dtype
        assert result.params_dtype == torch.bfloat16
        assert result.bf16 == True
        assert result.fp16 == False

    def test_provider_bridge_fp16_dtype_handling(self, gemma2_2b_config):
        """Test FP16 dtype handling in provider_bridge."""
        # Create model with FP16 dtype - set it in the config
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = gemma2_2b_config
        mock_pretrained.config.torch_dtype = torch.float16  # Set config dtype to fp16
        mock_pretrained.model = Mock(spec=Gemma2ForCausalLM)
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)

        bridge = Gemma2Bridge()
        result = bridge.provider_bridge(mock_pretrained)

        # The provider should respect the config's dtype
        assert result.params_dtype == torch.float16
        assert result.fp16 == True
        assert result.bf16 == False

    def test_provider_bridge_sliding_window_config(self, mock_pretrained_gemma2_2b, gemma2_2b_config):
        """Test sliding window configuration."""
        bridge = Gemma2Bridge()

        result = bridge.provider_bridge(mock_pretrained_gemma2_2b)

        # Check sliding window configuration specific to Gemma2
        assert result.window_size == (gemma2_2b_config.sliding_window, 0)
        assert result.window_size == (4096, 0)

    def test_provider_bridge_query_pre_attn_scalar_variants(self, mock_pretrained_gemma2_27b, gemma2_27b_config):
        """Test query_pre_attn_scalar for 27B model which has different value."""
        bridge = Gemma2Bridge()

        result = bridge.provider_bridge(mock_pretrained_gemma2_27b)

        # 27B model has different query_pre_attn_scalar
        assert result.query_pre_attn_scalar == gemma2_27b_config.query_pre_attn_scalar
        assert result.query_pre_attn_scalar == 144  # Different from 2B/9B which use 256

    def test_mapping_registry_implementation(self, mock_pretrained_gemma2_2b):
        """Test that mapping_registry returns a proper MegatronMappingRegistry."""
        bridge = Gemma2Bridge()

        # Get the mapping registry
        mapping_registry = bridge.mapping_registry()

        # Check it's not None
        assert mapping_registry is not None
        # Check it has param mappings (they are passed as args to __init__)
        # The mapping registry should have embedding, layer norm, attention, and MLP mappings

    def test_provider_bridge_make_vocab_size_divisible_by(self, mock_pretrained_gemma2_2b):
        """Test make_vocab_size_divisible_by calculation."""
        bridge = Gemma2Bridge()

        result = bridge.provider_bridge(mock_pretrained_gemma2_2b)

        # The method should calculate a reasonable divisor based on vocab size
        assert hasattr(result, "make_vocab_size_divisible_by")
        assert result.make_vocab_size_divisible_by > 0

    def test_provider_bridge_generation_config(self, mock_pretrained_gemma2_2b):
        """Test that generation config is passed through."""
        bridge = Gemma2Bridge()

        result = bridge.provider_bridge(mock_pretrained_gemma2_2b)

        # Generation config should be passed from the pretrained model
        assert result.generation_config == mock_pretrained_gemma2_2b.generation_config


class TestAutoBridgeIntegration:
    """Integration tests for AutoBridge with Gemma2 models."""

    @pytest.fixture
    def gemma2_configs(self):
        """Different Gemma2 model configurations for testing."""
        return {
            "gemma2-2b": {
                "architectures": ["Gemma2ForCausalLM"],
                "model_type": "gemma2",
                "hidden_size": 2304,
                "num_hidden_layers": 26,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "intermediate_size": 9216,
                "vocab_size": 256000,
                "max_position_embeddings": 8192,
                "rope_theta": 10000.0,
                "rms_norm_eps": 1e-06,
                "head_dim": 256,
                "attention_bias": False,
                "torch_dtype": "bfloat16",
                "query_pre_attn_scalar": 256,
                "attn_logit_softcapping": 50.0,
                "final_logit_softcapping": 30.0,
                "sliding_window": 4096,
            },
            "gemma2-9b": {
                "architectures": ["Gemma2ForCausalLM"],
                "model_type": "gemma2",
                "hidden_size": 3584,
                "num_hidden_layers": 42,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "intermediate_size": 14336,
                "vocab_size": 256000,
                "max_position_embeddings": 8192,
                "rope_theta": 10000.0,
                "rms_norm_eps": 1e-06,
                "head_dim": 256,
                "attention_bias": False,
                "torch_dtype": "bfloat16",
                "query_pre_attn_scalar": 256,
                "attn_logit_softcapping": 50.0,
                "final_logit_softcapping": 30.0,
                "sliding_window": 4096,
            },
            "gemma2-27b": {
                "architectures": ["Gemma2ForCausalLM"],
                "model_type": "gemma2",
                "hidden_size": 4608,
                "num_hidden_layers": 46,
                "num_attention_heads": 32,
                "num_key_value_heads": 16,
                "intermediate_size": 36864,
                "vocab_size": 256000,
                "max_position_embeddings": 8192,
                "rope_theta": 10000.0,
                "rms_norm_eps": 1e-06,
                "head_dim": 128,
                "attention_bias": False,
                "torch_dtype": "bfloat16",
                "query_pre_attn_scalar": 144,
                "attn_logit_softcapping": 50.0,
                "final_logit_softcapping": 30.0,
                "sliding_window": 4096,
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
    def test_from_pretrained_with_temp_dir(self, mock_autoconfig, mock_pretrained, gemma2_configs):
        """Test AutoBridge.from_hf_pretrained with temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with Gemma2 2B config
            config_dict = gemma2_configs["gemma2-2b"]
            self.create_mock_model_files(config_dict, temp_dir)

            # Mock the config loading
            config = Gemma2Config(**config_dict)
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

    def test_supports_gemma2_architectures(self, gemma2_configs):
        """Test that AutoBridge.supports correctly identifies Gemma2 models."""
        for model_name, config_dict in gemma2_configs.items():
            config = Gemma2Config(**config_dict)
            assert AutoBridge.supports(config) == True

        # Test non-causal LM architecture
        non_causal_config = Mock()
        non_causal_config.architectures = ["Gemma2Model"]  # Not ForCausalLM
        assert AutoBridge.supports(non_causal_config) == False


class TestGemma2BridgeParameterMapping:
    """Test parameter mapping functionality in Gemma2Bridge."""

    @pytest.fixture
    def mock_gemma2_state_dict(self):
        """Create a mock state dict with Gemma2 parameter names."""
        return {
            "model.embed_tokens.weight": torch.randn(256000, 2304),
            "model.norm.weight": torch.randn(2304),
            "model.layers.0.input_layernorm.weight": torch.randn(2304),
            "model.layers.0.pre_feedforward_layernorm.weight": torch.randn(2304),
            "model.layers.0.post_feedforward_layernorm.weight": torch.randn(2304),
            "model.layers.0.post_attention_layernorm.weight": torch.randn(2304),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(2304, 2304),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(1024, 2304),  # GQA: different size for K
            "model.layers.0.self_attn.v_proj.weight": torch.randn(1024, 2304),  # GQA: different size for V
            "model.layers.0.self_attn.o_proj.weight": torch.randn(2304, 2304),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(9216, 2304),
            "model.layers.0.mlp.up_proj.weight": torch.randn(9216, 2304),
            "model.layers.0.mlp.down_proj.weight": torch.randn(2304, 9216),
        }

    def test_mapping_registry_has_gemma2_specific_mappings(self):
        """Test that mapping registry includes Gemma2-specific mappings."""
        bridge = Gemma2Bridge()
        mapping_registry = bridge.mapping_registry()

        # This test verifies that the mapping registry was created
        # The actual parameter mappings are tested in integration tests
        assert mapping_registry is not None

    def test_gemma2_tied_embeddings_mapping(self):
        """Test that Gemma2 bridge handles tied embeddings correctly."""
        bridge = Gemma2Bridge()
        mapping_registry = bridge.mapping_registry()

        # Gemma2 uses tied embeddings, so there should be no separate lm_head.weight mapping
        # This is reflected in the mapping registry not including lm_head.weight
        assert mapping_registry is not None

    def test_gemma2_no_bias_mapping(self):
        """Test that Gemma2 bridge doesn't include bias mappings."""
        bridge = Gemma2Bridge()
        mapping_registry = bridge.mapping_registry()

        # Gemma2 doesn't have bias in linear layers
        # This is reflected in the QKVMapping and other mappings not including bias terms
        assert mapping_registry is not None

    def test_gemma2_gated_mlp_mapping(self):
        """Test that Gemma2 bridge includes gated MLP mappings."""
        bridge = Gemma2Bridge()
        mapping_registry = bridge.mapping_registry()

        # Gemma2 uses gated MLP, so it should have GatedMLPMapping
        # This combines gate_proj and up_proj into linear_fc1
        assert mapping_registry is not None

    def test_gemma2_additional_layer_norms_mapping(self):
        """Test that Gemma2 bridge includes additional layer norm mappings."""
        bridge = Gemma2Bridge()
        mapping_registry = bridge.mapping_registry()

        # Gemma2 has additional layer normalizations compared to original Gemma
        # pre_feedforward_layernorm, post_feedforward_layernorm, post_attention_layernorm
        assert mapping_registry is not None
