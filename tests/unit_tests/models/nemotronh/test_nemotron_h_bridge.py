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
from transformers import GenerationConfig
from transformers.configuration_utils import PretrainedConfig

from megatron.bridge.models import AutoBridge
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, QKVMapping
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.nemotronh.nemotron_h_bridge import NemotronHBridge
from megatron.bridge.models.nemotronh.nemotron_h_provider import NemotronHModelProvider


class TestNemotronHBridge:
    """Test cases for NemotronHBridge class."""

    @pytest.fixture
    def nemotronh_8b_config_dict(self):
        """Create a sample NemotronH configuration."""
        return {
            "architectures": ["NemotronHForCausalLM"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attention_head_dim": 128,
            "auto_map": {
                "AutoConfig": "configuration_nemotron_h.NemotronHConfig",
                "AutoModelForCausalLM": "modeling_nemotron_h.NemotronHForCausalLM",
            },
            "bos_token_id": 1,
            "chunk_size": 128,
            "conv_kernel": 4,
            "eos_token_id": 2,
            "expand": 2,
            "hidden_dropout": 0.0,
            "hidden_size": 4096,
            "hybrid_override_pattern": "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-",
            "initializer_range": 0.02,
            "intermediate_size": 21504,
            "layer_norm_epsilon": 1e-05,
            "mamba_head_dim": 64,
            "mamba_hidden_act": "silu",
            "mamba_num_heads": 128,
            "mamba_proj_bias": False,
            "max_position_embeddings": 8192,
            "mlp_bias": False,
            "mlp_hidden_act": "relu2",
            "model_type": "nemotron_h",
            "n_groups": 8,
            "num_attention_heads": 32,
            "num_hidden_layers": 52,
            "num_key_value_heads": 8,
            "num_logits_to_keep": 1,
            "pad_token_id": 0,
            "rescale_prenorm_residual": True,
            "residual_in_fp32": False,
            "rms_norm_eps": 1e-05,
            "ssm_state_size": 128,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.48.0.dev0",
            "use_bias": False,
            "use_cache": True,
            "use_conv_bias": True,
            "use_mamba_kernels": True,
            "vocab_size": 131072,
        }

    @pytest.fixture
    def mock_nemotronh_config(self, nemotronh_8b_config_dict):
        """Create mock config instance."""
        cfg = Mock()
        for k, v in nemotronh_8b_config_dict.items():
            setattr(cfg, k, v)
        return cfg

    @pytest.fixture
    def mock_pretrained_nemotronh(self, mock_nemotronh_config):
        """Create a mock PreTrainedCausalLM with NemotronH model."""

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mock_nemotronh_config
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)
        return mock_pretrained

    def test_bridge_registration(self):
        """Test that NemotronHBridge is properly registered."""
        assert issubclass(NemotronHBridge, MegatronModelBridge)

    def test_provider_bridge_basic(self, mock_pretrained_nemotronh, mock_nemotronh_config):
        """Test basic provider_bridge functionality."""
        bridge = NemotronHBridge()

        # Call provider_bridge
        result = bridge.provider_bridge(mock_pretrained_nemotronh)

        # Check that it returns a MambaModelProvider instance
        assert isinstance(result, NemotronHModelProvider)

        # Check basic configuration mapping
        assert result.num_layers == mock_nemotronh_config.num_hidden_layers
        assert result.hidden_size == mock_nemotronh_config.hidden_size
        assert result.add_bias_linear == mock_nemotronh_config.use_bias
        assert result.num_attention_heads == mock_nemotronh_config.num_attention_heads
        assert result.seq_length == mock_nemotronh_config.max_position_embeddings

    def test_provider_bridge_vocabulary(self, mock_pretrained_nemotronh, mock_nemotronh_config):
        """Test vocabulary size mapping."""
        bridge = NemotronHBridge()

        result = bridge.provider_bridge(mock_pretrained_nemotronh)

        # Check vocabulary configuration
        assert result.vocab_size == mock_nemotronh_config.vocab_size
        assert result.share_embeddings_and_output_weights == mock_nemotronh_config.tie_word_embeddings

    def test_provider_bridge_attention_config(self, mock_pretrained_nemotronh, mock_nemotronh_config):
        """Test attention configuration mapping."""
        bridge = NemotronHBridge()

        result = bridge.provider_bridge(mock_pretrained_nemotronh)

        # Check attention configuration
        assert result.num_attention_heads == mock_nemotronh_config.num_attention_heads
        assert result.num_query_groups == mock_nemotronh_config.num_key_value_heads

    def test_provider_bridge_mamba_config(self, mock_pretrained_nemotronh, mock_nemotronh_config):
        """Test Mamba-specific configuration mapping."""
        bridge = NemotronHBridge()

        result = bridge.provider_bridge(mock_pretrained_nemotronh)

        # Check Mamba-specific configuration
        assert result.mamba_state_dim == mock_nemotronh_config.ssm_state_size
        assert result.mamba_head_dim == mock_nemotronh_config.mamba_head_dim
        assert result.mamba_num_heads == mock_nemotronh_config.mamba_num_heads
        assert result.mamba_num_groups == mock_nemotronh_config.n_groups
        assert result.hybrid_override_pattern == mock_nemotronh_config.hybrid_override_pattern

    def test_provider_bridge_mlp_config(self, mock_pretrained_nemotronh, mock_nemotronh_config):
        """Test MLP configuration mapping."""
        bridge = NemotronHBridge()

        result = bridge.provider_bridge(mock_pretrained_nemotronh)

        # Check MLP configuration
        assert result.ffn_hidden_size == mock_nemotronh_config.intermediate_size
        assert result.gated_linear_unit == False  # Mamba doesn't use gated linear units

    def test_provider_bridge_normalization(self, mock_pretrained_nemotronh, mock_nemotronh_config):
        """Test normalization configuration."""
        bridge = NemotronHBridge()

        result = bridge.provider_bridge(mock_pretrained_nemotronh)

        # Check normalization settings
        assert result.layernorm_epsilon == mock_nemotronh_config.layer_norm_epsilon

    def test_provider_bridge_dtype_handling(self, mock_nemotronh_config):
        """Test dtype handling in provider_bridge."""
        # Create model with specific dtype
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_nemotronh_config.torch_dtype = "bfloat16"
        mock_pretrained.config = mock_nemotronh_config
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)

        bridge = NemotronHBridge()
        result = bridge.provider_bridge(mock_pretrained)

        # The provider should respect the model's dtype
        assert result.params_dtype == torch.bfloat16
        assert result.bf16 == True
        assert result.fp16 == False

    def test_mapping_registry_implementation(self, mock_pretrained_nemotronh):
        """Test that mapping_registry returns a proper MegatronMappingRegistry."""
        bridge = NemotronHBridge()

        # Get the mapping registry
        mapping_registry = bridge.mapping_registry()

        # Check it's not None
        assert mapping_registry is not None
        assert any([isinstance(m, AutoMapping) for m in mapping_registry.mappings])
        # assert any([isinstance(m, PrunedVocabMapping) for m in mapping_registry.mappings])
        assert any([isinstance(m, QKVMapping) for m in mapping_registry.mappings])

    def test_provider_bridge_fixed_settings(self, mock_pretrained_nemotronh):
        """Test fixed settings that should always be set regardless of config."""
        bridge = NemotronHBridge()

        result = bridge.provider_bridge(mock_pretrained_nemotronh)

        # These should always be set to these values for Mamba
        assert result.position_embedding_type == "none"  # Mamba doesn't use position embeddings
        assert result.rotary_percent == 1.0
        assert result.rotary_base == 10000


class TestAutoBridgeIntegration:
    """Integration tests for AutoBridge with NemotronH models."""

    @pytest.fixture
    def nemotronh_config_dict(self):
        """Create a sample NemotronH configuration."""
        return {
            "architectures": ["NemotronHForCausalLM"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attention_head_dim": 128,
            "auto_map": {
                "AutoConfig": "configuration_nemotron_h.NemotronHConfig",
                "AutoModelForCausalLM": "modeling_nemotron_h.NemotronHForCausalLM",
            },
            "bos_token_id": 1,
            "chunk_size": 128,
            "conv_kernel": 4,
            "eos_token_id": 2,
            "expand": 2,
            "hidden_dropout": 0.0,
            "hidden_size": 4096,
            "hybrid_override_pattern": "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-",
            "initializer_range": 0.02,
            "intermediate_size": 21504,
            "layer_norm_epsilon": 1e-05,
            "mamba_head_dim": 64,
            "mamba_hidden_act": "silu",
            "mamba_num_heads": 128,
            "mamba_proj_bias": False,
            "max_position_embeddings": 8192,
            "mlp_bias": False,
            "mlp_hidden_act": "relu2",
            "model_type": "nemotron_h",
            "n_groups": 8,
            "num_attention_heads": 32,
            "num_hidden_layers": 52,
            "num_key_value_heads": 8,
            "num_logits_to_keep": 1,
            "pad_token_id": 0,
            "rescale_prenorm_residual": True,
            "residual_in_fp32": False,
            "rms_norm_eps": 1e-05,
            "ssm_state_size": 128,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.48.0.dev0",
            "use_bias": False,
            "use_cache": True,
            "use_conv_bias": True,
            "use_mamba_kernels": True,
            "vocab_size": 131072,
        }

    @pytest.fixture
    def nemotronh_config(self, nemotronh_config_dict):
        """Create mock config instance."""
        cfg = Mock(spec=PretrainedConfig)
        for k, v in nemotronh_config_dict.items():
            setattr(cfg, k, v)
        return cfg

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
                "backbone.embeddings.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.0.mixer.in_proj.weight": "model-00001-of-00001.safetensors",
            },
        }
        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=2)

        # Create tokenizer files
        tokenizer_config = {
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
    @patch("transformers.dynamic_module_utils.get_class_from_dynamic_module")
    def test_from_pretrained_with_temp_dir(
        self, mock_dynamic_util, mock_autoconfig, mock_pretrained, nemotronh_config_dict, nemotronh_config
    ):
        """Test AutoBridge.from_hf_pretrained with temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dict = nemotronh_config_dict
            self.create_mock_model_files(config_dict, temp_dir)

            # Mock the config loading
            config = nemotronh_config
            mock_autoconfig.return_value = config

            # Mock the type from remote modeling file
            mock_nemtronh_type = Mock()
            mock_nemtronh_type.__name__ = "NemotronHForCausalLM"
            mock_dynamic_util.return_value = mock_nemtronh_type

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
            call_kwargs = mock_dynamic_util.call_args.kwargs
            assert call_kwargs["class_reference"] == config_dict["auto_map"]["AutoModelForCausalLM"]
            assert call_kwargs["pretrained_model_name_or_path"] == temp_dir
            mock_pretrained.assert_called_once_with(temp_dir)

    @patch("megatron.bridge.models.conversion.auto_bridge.PreTrainedCausalLM.from_pretrained")
    @patch("transformers.AutoConfig.from_pretrained")
    @patch("transformers.dynamic_module_utils.get_class_from_dynamic_module")
    def test_from_pretrained_with_kwargs(
        self, mock_dynamic_util, mock_autoconfig, mock_pretrained, nemotronh_config_dict, nemotronh_config
    ):
        """Test AutoBridge.from_hf_pretrained with various kwargs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dict = nemotronh_config_dict
            self.create_mock_model_files(config_dict, temp_dir)

            # Mock the config loading
            config = nemotronh_config
            mock_autoconfig.return_value = config

            # Mock the type from remote modeling file
            mock_nemtronh_type = Mock()
            mock_nemtronh_type.__name__ = "NemotronHForCausalLM"
            mock_dynamic_util.return_value = mock_nemtronh_type

            # Mock the pretrained model
            mock_model = Mock(spec=PreTrainedCausalLM)
            mock_model.config = config
            mock_pretrained.return_value = mock_model

            # Test with various kwargs
            kwargs = {
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
                "trust_remote_code": True,
            }

            _ = AutoBridge.from_hf_pretrained(temp_dir, **kwargs)

            # Verify kwargs were passed through
            mock_pretrained.assert_called_once_with(temp_dir, **kwargs)

    def test_supports_nemotronh_architectures(self, nemotronh_config):
        """Test that AutoBridge.supports correctly identifies NemotronH models."""
        config = nemotronh_config
        assert AutoBridge.supports(config) == True

        # Test non-causal LM architecture
        non_causal_config = Mock()
        non_causal_config.architectures = ["NemotronHModel"]  # Not ForCausalLM
        assert AutoBridge.supports(non_causal_config) == False
