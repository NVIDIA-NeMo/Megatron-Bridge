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
from transformers import GenerationConfig, MambaConfig, MambaForCausalLM

from megatron.bridge.models import AutoBridge
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, QKVMapping
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mamba.mamba_bridge import MambaBridge, PrunedVocabMapping
from megatron.bridge.models.mamba.mamba_provider import MambaProvider


class TestMegatronMambaBridge:
    """Test cases for MambaBridge class."""

    @pytest.fixture
    def mamba_130m_config_dict(self):
        """Create a sample Mamba configuration."""
        return {
            "architectures": ["MambaForCausalLM"],
            "bos_token_id": 0,
            "conv_kernel": 4,
            "d_inner": 1536,
            "d_model": 768,
            "eos_token_id": 0,
            "expand": 2,
            "fused_add_norm": True,
            "hidden_act": "silu",
            "hidden_size": 768,
            "initializer_range": 0.1,
            "intermediate_size": 1536,
            "layer_norm_epsilon": 1e-05,
            "model_type": "mamba",
            "n_layer": 24,
            "num_hidden_layers": 24,
            "pad_token_id": 0,
            "pad_vocab_size_multiple": 8,
            "rescale_prenorm_residual": False,
            "residual_in_fp32": True,
            "rms_norm": True,
            "ssm_cfg": {},
            "state_size": 16,
            "time_step_floor": 0.0001,
            "time_step_init_scheme": "random",
            "time_step_max": 0.1,
            "time_step_min": 0.001,
            "time_step_rank": 48,
            "time_step_scale": 1.0,
            "torch_dtype": "float32",
            "transformers_version": "4.39.0.dev0",
            "use_bias": False,
            "use_cache": True,
            "use_conv_bias": True,
            "vocab_size": 50280,
        }

    @pytest.fixture
    def mamba_config(self, mamba_130m_config_dict):
        """Create a MambaConfig instance."""
        return MambaConfig(**mamba_130m_config_dict)

    @pytest.fixture
    def mock_pretrained_mamba(self, mamba_config):
        """Create a mock PreTrainedCausalLM with Mamba model."""
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mamba_config
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)
        mock_pretrained.model = Mock(spec=MambaForCausalLM)
        mock_pretrained.model.dtype = torch.bfloat16
        return mock_pretrained

    def test_bridge_registration(self):
        """Test that MambaBridge is properly registered."""
        # The @MegatronModelBridge.register_bridge decorator should register the bridge
        # Check that the class exists and has the expected base class
        assert issubclass(MambaBridge, MegatronModelBridge)

    def test_provider_bridge_basic(self, mock_pretrained_mamba, mamba_config):
        """Test basic provider_bridge functionality."""
        bridge = MambaBridge()

        # Call provider_bridge
        result = bridge.provider_bridge(mock_pretrained_mamba)

        # Check that it returns a MambaProvider instance
        assert isinstance(result, MambaProvider)

        # Check basic configuration mapping
        assert result.num_layers == mamba_config.num_hidden_layers
        assert result.hidden_size == mamba_config.hidden_size
        assert result.add_bias_linear == mamba_config.use_bias

    def test_provider_bridge_vocabulary(self, mock_pretrained_mamba, mamba_config):
        """Test vocabulary size mapping."""
        bridge = MambaBridge()

        result = bridge.provider_bridge(mock_pretrained_mamba)

        # Check vocabulary configuration
        assert result.vocab_size == mamba_config.vocab_size
        assert result.make_vocab_size_divisible_by == mamba_config.pad_vocab_size_multiple

    def test_provider_bridge_mamba_config(self, mock_pretrained_mamba, mamba_config):
        """Test Mamba-specific configuration mapping."""
        bridge = MambaBridge()

        result = bridge.provider_bridge(mock_pretrained_mamba)

        # Check Mamba-specific configuration
        assert result.mamba_state_dim == mamba_config.state_size
        assert result.hybrid_override_pattern == "M" * mamba_config.num_hidden_layers

    def test_provider_bridge_mlp_config(self, mock_pretrained_mamba, mamba_config):
        """Test MLP configuration mapping."""
        bridge = MambaBridge()

        result = bridge.provider_bridge(mock_pretrained_mamba)

        # Check MLP configuration
        assert result.ffn_hidden_size == mamba_config.intermediate_size
        assert result.gated_linear_unit == False  # Mamba doesn't use gated linear units

    def test_provider_bridge_normalization(self, mock_pretrained_mamba, mamba_config):
        """Test normalization configuration."""
        bridge = MambaBridge()

        result = bridge.provider_bridge(mock_pretrained_mamba)

        # Check normalization settings
        assert result.layernorm_epsilon == mamba_config.layer_norm_epsilon

    def test_provider_bridge_dtype_handling(self, mamba_config):
        """Test dtype handling in provider_bridge."""
        # Create model with specific dtype
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mamba_config
        mock_pretrained.config["torch_dtype"] = "bfloat16"
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)

        bridge = MambaBridge()
        result = bridge.provider_bridge(mock_pretrained)

        # The provider should respect the model's dtype
        assert result.params_dtype == torch.bfloat16
        assert result.bf16 == True
        assert result.fp16 == False

    def test_mapping_registry_implementation(self, mock_pretrained_mamba):
        """Test that mapping_registry returns a proper MegatronMappingRegistry."""
        bridge = MambaBridge()

        # Get the mapping registry
        mapping_registry = bridge.mapping_registry()

        # Check it's not None
        assert mapping_registry is not None
        assert any([isinstance(m, AutoMapping) for m in mapping_registry.mappings])
        assert any([isinstance(m, PrunedVocabMapping) for m in mapping_registry.mappings])
        assert any([isinstance(m, QKVMapping) for m in mapping_registry.mappings])

    def test_provider_bridge_fixed_settings(self, mock_pretrained_mamba):
        """Test fixed settings that should always be set regardless of config."""
        bridge = MambaBridge()

        result = bridge.provider_bridge(mock_pretrained_mamba)

        # These should always be set to these values for Mamba
        assert result.position_embedding_type == "none"  # Mamba doesn't use position embeddings
        assert result.num_attention_heads == 1  # Mamba uses 1 attention head by default
        assert result.hybrid_attention_ratio == 0.0  # Pure Mamba by default
        assert result.hybrid_mlp_ratio == 0.0  # Pure Mamba by default


class TestAutoBridgeIntegration:
    """Integration tests for AutoBridge with Mamba models."""

    @pytest.fixture
    def mamba_config(self):
        """Create a sample Mamba configuration."""
        return {
            "architectures": ["MambaForCausalLM"],
            "bos_token_id": 0,
            "conv_kernel": 4,
            "d_inner": 1536,
            "d_model": 768,
            "eos_token_id": 0,
            "expand": 2,
            "fused_add_norm": True,
            "hidden_act": "silu",
            "hidden_size": 768,
            "initializer_range": 0.1,
            "intermediate_size": 1536,
            "layer_norm_epsilon": 1e-05,
            "model_type": "mamba",
            "n_layer": 24,
            "num_hidden_layers": 24,
            "pad_token_id": 0,
            "pad_vocab_size_multiple": 8,
            "rescale_prenorm_residual": False,
            "residual_in_fp32": True,
            "rms_norm": True,
            "ssm_cfg": {},
            "state_size": 16,
            "time_step_floor": 0.0001,
            "time_step_init_scheme": "random",
            "time_step_max": 0.1,
            "time_step_min": 0.001,
            "time_step_rank": 48,
            "time_step_scale": 1.0,
            "torch_dtype": "float32",
            "transformers_version": "4.39.0.dev0",
            "use_bias": False,
            "use_cache": True,
            "use_conv_bias": True,
            "vocab_size": 50280,
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
                "backbone.embeddings.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.0.mixer.dt_proj.weight": "model-00001-of-00001.safetensors",
            },
        }
        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=2)

        # Create tokenizer files
        tokenizer_config = {
            "model_max_length": 8192,
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
    @patch("megatron.bridge.models.conversion.auto_bridge.AutoConfig.from_pretrained")
    def test_from_pretrained_with_temp_dir(self, mock_autoconfig, mock_pretrained, mamba_config):
        """Test AutoBridge.from_hf_pretrained with temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dict = mamba_config
            self.create_mock_model_files(config_dict, temp_dir)

            # Mock the config loading
            config = MambaConfig(**config_dict)
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
    @patch("megatron.bridge.models.conversion.auto_bridge.AutoConfig.from_pretrained")
    def test_from_pretrained_with_kwargs(self, mock_autoconfig, mock_pretrained, mamba_config):
        """Test AutoBridge.from_hf_pretrained with various kwargs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dict = mamba_config
            self.create_mock_model_files(config_dict, temp_dir)

            # Mock the config loading
            config = MambaConfig(**config_dict)
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
            }

            _ = AutoBridge.from_hf_pretrained(temp_dir, **kwargs)

            # Verify kwargs were passed through
            mock_pretrained.assert_called_once_with(temp_dir, **kwargs)

    def test_supports_mamba_architectures(self, mamba_config):
        """Test that AutoBridge.supports correctly identifies Mamba models."""
        config = MambaConfig(**mamba_config)
        assert AutoBridge.supports(config) == True

        # Test non-causal LM architecture
        non_causal_config = Mock()
        non_causal_config.architectures = ["MambaModel"]  # Not ForCausalLM
        assert AutoBridge.supports(non_causal_config) == False
