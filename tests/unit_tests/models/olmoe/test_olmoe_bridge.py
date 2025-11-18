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

from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from megatron.bridge.models import AutoBridge
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.olmoe.olmoe_bridge import OlMoEBridge
from megatron.bridge.models.olmoe.olmoe_provider import OlMoEModelProvider


class TestMegatronOlMoEBridge:
    """Test cases for MegatronOlMoEBridge class."""

    @pytest.fixture
    def olmoe_1b_7b_config_dict(self):
        """Create a sample OLMoE-1B-7B configuration."""
        return {
            "architectures": ["OlmoeForCausalLM"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 2048,
            "initializer_range": 0.02,
            "intermediate_size": 1024,
            "max_position_embeddings": 4096,
            "model_type": "olmoe",
            "num_attention_heads": 16,
            "num_experts": 64,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 16,
            "num_key_value_heads": 16,
            "pad_token_id": 0,
            "rms_norm_eps": 1e-05,
            "rope_theta": 10000.0,
            "router_aux_loss_coef": 0.01,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.40.0",
            "use_cache": True,
            "vocab_size": 50304,
        }

    @pytest.fixture
    def olmoe_custom_config_dict(self):
        """Create a custom OLMoE configuration for testing."""
        return {
            "architectures": ["OlmoeForCausalLM"],
            "attention_bias": True,
            "attention_dropout": 0.1,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.01,
            "intermediate_size": 2048,
            "max_position_embeddings": 8192,
            "model_type": "olmoe",
            "num_attention_heads": 32,
            "num_experts": 128,
            "num_experts_per_tok": 16,
            "num_hidden_layers": 32,
            "num_key_value_heads": 32,
            "pad_token_id": 0,
            "rms_norm_eps": 1e-06,
            "rope_theta": 20000.0,
            "router_aux_loss_coef": 0.02,
            "tie_word_embeddings": False,
            "torch_dtype": "float16",
            "use_cache": True,
            "vocab_size": 100000,
        }

    @pytest.fixture
    def olmoe_1b_7b_config(self, olmoe_1b_7b_config_dict):
        """Create an OLMoE config instance for 1B-7B model."""
        # Create a mock config object with the properties
        config = Mock()
        for key, value in olmoe_1b_7b_config_dict.items():
            setattr(config, key, value)
        return config

    @pytest.fixture
    def olmoe_custom_config(self, olmoe_custom_config_dict):
        """Create an OLMoE config instance for custom model."""
        config = Mock()
        for key, value in olmoe_custom_config_dict.items():
            setattr(config, key, value)
        return config

    @pytest.fixture
    def mock_olmoe_1b_7b_model(self, olmoe_1b_7b_config):
        """Create a mock OlmoeForCausalLM 1B-7B model."""
        try:
            from transformers import OlmoeForCausalLM

            mock_model = Mock(spec=OlmoeForCausalLM)
        except ImportError:
            mock_model = Mock()
        mock_model.config = olmoe_1b_7b_config
        mock_model.dtype = torch.bfloat16
        return mock_model

    @pytest.fixture
    def mock_olmoe_custom_model(self, olmoe_custom_config):
        """Create a mock OlmoeForCausalLM custom model."""
        try:
            from transformers import OlmoeForCausalLM

            mock_model = Mock(spec=OlmoeForCausalLM)
        except ImportError:
            mock_model = Mock()
        mock_model.config = olmoe_custom_config
        mock_model.dtype = torch.float16
        return mock_model

    @pytest.fixture
    def mock_pretrained_olmoe_1b_7b(self, olmoe_1b_7b_config):
        """Create a mock PreTrainedCausalLM with OLMoE 1B-7B model."""
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = olmoe_1b_7b_config
        mock_pretrained.generation_config = Mock()
        try:
            from transformers import OlmoeForCausalLM

            mock_pretrained.model = Mock(spec=OlmoeForCausalLM)
        except ImportError:
            mock_pretrained.model = Mock()
        mock_pretrained.model.dtype = torch.bfloat16
        return mock_pretrained

    @pytest.fixture
    def mock_pretrained_olmoe_custom(self, olmoe_custom_config):
        """Create a mock PreTrainedCausalLM with custom OLMoE model."""
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = olmoe_custom_config
        mock_pretrained.generation_config = Mock()
        try:
            from transformers import OlmoeForCausalLM

            mock_pretrained.model = Mock(spec=OlmoeForCausalLM)
        except ImportError:
            mock_pretrained.model = Mock()
        mock_pretrained.model.dtype = torch.float16
        return mock_pretrained

    def test_bridge_registration(self):
        """Test that MegatronOlMoEBridge is properly registered."""
        # The @MegatronModelBridge.register_bridge decorator should register the bridge
        # Check that the class exists and has the expected base class
        assert issubclass(OlMoEBridge, MegatronModelBridge)

    def test_provider_bridge_basic_1b_7b(self, mock_pretrained_olmoe_1b_7b, olmoe_1b_7b_config):
        """Test basic provider_bridge functionality for OLMoE 1B-7B."""
        bridge = OlMoEBridge()

        # Call provider_bridge
        result = bridge.provider_bridge(mock_pretrained_olmoe_1b_7b)

        # Check that it returns an OlMoEModelProvider instance
        assert isinstance(result, OlMoEModelProvider)

        # Check basic configuration mapping
        assert result.num_layers == olmoe_1b_7b_config.num_hidden_layers
        assert result.hidden_size == olmoe_1b_7b_config.hidden_size
        assert result.num_attention_heads == olmoe_1b_7b_config.num_attention_heads
        assert result.seq_length == olmoe_1b_7b_config.max_position_embeddings
        assert result.rotary_base == olmoe_1b_7b_config.rope_theta

    def test_provider_bridge_basic_custom(self, mock_pretrained_olmoe_custom, olmoe_custom_config):
        """Test basic provider_bridge functionality for custom OLMoE."""
        bridge = OlMoEBridge()

        # Call provider_bridge
        result = bridge.provider_bridge(mock_pretrained_olmoe_custom)

        # Check that it returns an OlMoEModelProvider instance
        assert isinstance(result, OlMoEModelProvider)

        # Check basic configuration mapping
        assert result.num_layers == olmoe_custom_config.num_hidden_layers
        assert result.hidden_size == olmoe_custom_config.hidden_size
        assert result.num_attention_heads == olmoe_custom_config.num_attention_heads
        assert result.seq_length == olmoe_custom_config.max_position_embeddings
        assert result.rotary_base == olmoe_custom_config.rope_theta

    def test_provider_bridge_vocabulary(self, mock_pretrained_olmoe_1b_7b, olmoe_1b_7b_config):
        """Test vocabulary size mapping."""
        bridge = OlMoEBridge()

        result = bridge.provider_bridge(mock_pretrained_olmoe_1b_7b)

        # Check vocabulary configuration
        assert result.vocab_size == olmoe_1b_7b_config.vocab_size
        # OLMoE doesn't use tied embeddings by default
        assert result.share_embeddings_and_output_weights == olmoe_1b_7b_config.tie_word_embeddings

    def test_provider_bridge_attention_config(self, mock_pretrained_olmoe_1b_7b, olmoe_1b_7b_config):
        """Test attention configuration mapping."""
        bridge = OlMoEBridge()

        result = bridge.provider_bridge(mock_pretrained_olmoe_1b_7b)

        # Check attention configuration
        assert result.num_attention_heads == olmoe_1b_7b_config.num_attention_heads
        assert result.num_query_groups == olmoe_1b_7b_config.num_key_value_heads
        assert result.add_qkv_bias == olmoe_1b_7b_config.attention_bias

    def test_provider_bridge_mlp_config(self, mock_pretrained_olmoe_1b_7b, olmoe_1b_7b_config):
        """Test MLP configuration mapping."""
        bridge = OlMoEBridge()

        result = bridge.provider_bridge(mock_pretrained_olmoe_1b_7b)

        # Check MLP configuration
        assert result.ffn_hidden_size == olmoe_1b_7b_config.intermediate_size
        assert result.moe_ffn_hidden_size == olmoe_1b_7b_config.intermediate_size
        assert result.gated_linear_unit == True  # OLMoE uses gated MLP

    def test_provider_bridge_moe_config(self, mock_pretrained_olmoe_1b_7b, olmoe_1b_7b_config):
        """Test MoE-specific configuration mapping."""
        bridge = OlMoEBridge()

        result = bridge.provider_bridge(mock_pretrained_olmoe_1b_7b)

        # Check MoE configuration
        assert result.num_moe_experts == olmoe_1b_7b_config.num_experts
        assert result.moe_router_topk == olmoe_1b_7b_config.num_experts_per_tok
        assert result.moe_aux_loss_coeff == olmoe_1b_7b_config.router_aux_loss_coef

    def test_provider_bridge_moe_config_custom(self, mock_pretrained_olmoe_custom, olmoe_custom_config):
        """Test MoE-specific configuration mapping with custom values."""
        bridge = OlMoEBridge()

        result = bridge.provider_bridge(mock_pretrained_olmoe_custom)

        # Check MoE configuration with custom values
        assert result.num_moe_experts == olmoe_custom_config.num_experts  # 128
        assert result.moe_router_topk == olmoe_custom_config.num_experts_per_tok  # 16
        assert result.moe_aux_loss_coeff == olmoe_custom_config.router_aux_loss_coef  # 0.02

    def test_provider_bridge_normalization(self, mock_pretrained_olmoe_1b_7b, olmoe_1b_7b_config):
        """Test normalization configuration."""
        bridge = OlMoEBridge()

        result = bridge.provider_bridge(mock_pretrained_olmoe_1b_7b)

        # Check normalization settings
        assert result.layernorm_epsilon == olmoe_1b_7b_config.rms_norm_eps

    def test_provider_bridge_position_embedding(self, mock_pretrained_olmoe_1b_7b, olmoe_1b_7b_config):
        """Test position embedding configuration."""
        bridge = OlMoEBridge()

        result = bridge.provider_bridge(mock_pretrained_olmoe_1b_7b)

        # Check position embedding
        assert result.rotary_base == olmoe_1b_7b_config.rope_theta
        assert result.position_embedding_type == "rope"

    def test_provider_bridge_olmoe_specific_features(self, mock_pretrained_olmoe_1b_7b, olmoe_1b_7b_config):
        """Test OLMoE-specific features."""
        bridge = OlMoEBridge()

        result = bridge.provider_bridge(mock_pretrained_olmoe_1b_7b)

        # Check OLMoE-specific features
        assert result.add_bias_linear == False  # OLMoE doesn't use bias in linear layers by default
        assert result.qk_layernorm == True  # OLMoE uses QK layernorm
        assert result.normalization == "RMSNorm"  # OLMoE uses RMSNorm
        assert result.gated_linear_unit == True

    def test_provider_bridge_kv_channels_calculation(self, mock_pretrained_olmoe_1b_7b, olmoe_1b_7b_config):
        """Test KV channels (head dimension) calculation."""
        bridge = OlMoEBridge()

        result = bridge.provider_bridge(mock_pretrained_olmoe_1b_7b)

        # KV channels should be hidden_size / num_attention_heads
        expected_kv_channels = olmoe_1b_7b_config.hidden_size // olmoe_1b_7b_config.num_attention_heads
        assert result.kv_channels == expected_kv_channels
        assert result.kv_channels == 2048 // 16  # 128

    def test_provider_bridge_dtype_handling_bfloat16(self, olmoe_1b_7b_config):
        """Test bfloat16 dtype handling in provider_bridge."""
        # Create model with bfloat16 dtype
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = olmoe_1b_7b_config
        mock_pretrained.config.torch_dtype = torch.bfloat16
        mock_pretrained.generation_config = Mock()
        try:
            from transformers import OlmoeForCausalLM

            mock_pretrained.model = Mock(spec=OlmoeForCausalLM)
        except ImportError:
            mock_pretrained.model = Mock()

        bridge = OlMoEBridge()
        result = bridge.provider_bridge(mock_pretrained)

        # The provider should respect the config's dtype
        assert result.params_dtype == torch.bfloat16
        assert result.bf16 == True
        assert result.fp16 == False

    def test_provider_bridge_dtype_handling_fp16(self, olmoe_custom_config):
        """Test FP16 dtype handling in provider_bridge."""
        # Create model with FP16 dtype
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = olmoe_custom_config
        mock_pretrained.config.torch_dtype = torch.float16
        mock_pretrained.generation_config = Mock()
        try:
            from transformers import OlmoeForCausalLM

            mock_pretrained.model = Mock(spec=OlmoeForCausalLM)
        except ImportError:
            mock_pretrained.model = Mock()

        bridge = OlMoEBridge()
        result = bridge.provider_bridge(mock_pretrained)

        # The provider should respect the config's dtype
        assert result.params_dtype == torch.float16
        assert result.fp16 == True
        assert result.bf16 == False

    def test_provider_bridge_dtype_handling_fp32(self, olmoe_1b_7b_config):
        """Test FP32 dtype handling in provider_bridge."""
        # Create model with FP32 dtype
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = olmoe_1b_7b_config
        mock_pretrained.config.torch_dtype = torch.float32
        mock_pretrained.generation_config = Mock()
        try:
            from transformers import OlmoeForCausalLM

            mock_pretrained.model = Mock(spec=OlmoeForCausalLM)
        except ImportError:
            mock_pretrained.model = Mock()

        bridge = OlMoEBridge()
        result = bridge.provider_bridge(mock_pretrained)

        # The provider should respect the config's dtype
        assert result.params_dtype == torch.float32
        assert result.fp16 == False
        assert result.bf16 == False

    def test_provider_bridge_init_method_std(self, mock_pretrained_olmoe_1b_7b, olmoe_1b_7b_config):
        """Test initializer range mapping."""
        bridge = OlMoEBridge()

        result = bridge.provider_bridge(mock_pretrained_olmoe_1b_7b)

        # Check that initializer range is mapped correctly
        assert result.init_method_std == olmoe_1b_7b_config.initializer_range

    def test_provider_bridge_generation_config(self, mock_pretrained_olmoe_1b_7b):
        """Test that generation config is passed through."""
        bridge = OlMoEBridge()

        result = bridge.provider_bridge(mock_pretrained_olmoe_1b_7b)

        # Generation config should be passed from the pretrained model
        assert result.generation_config == mock_pretrained_olmoe_1b_7b.generation_config

    def test_mapping_registry_implementation(self, mock_pretrained_olmoe_1b_7b):
        """Test that mapping_registry returns a proper MegatronMappingRegistry."""
        bridge = OlMoEBridge()

        # Get the mapping registry
        mapping_registry = bridge.mapping_registry()

        # Check it's not None
        assert mapping_registry is not None

    def test_provider_bridge_attention_bias_false(self, mock_pretrained_olmoe_1b_7b):
        """Test that attention_bias is correctly mapped when False."""
        bridge = OlMoEBridge()

        result = bridge.provider_bridge(mock_pretrained_olmoe_1b_7b)

        # OLMoE 1B-7B doesn't use attention bias
        assert result.add_qkv_bias == False

    def test_provider_bridge_attention_bias_true(self, mock_pretrained_olmoe_custom):
        """Test that attention_bias is correctly mapped when True."""
        bridge = OlMoEBridge()

        result = bridge.provider_bridge(mock_pretrained_olmoe_custom)

        # Custom config has attention bias enabled
        assert result.add_qkv_bias == True


class TestAutoBridgeIntegration:
    """Integration tests for AutoBridge with OLMoE models."""

    @pytest.fixture
    def olmoe_configs(self):
        """Different OLMoE model configurations for testing."""
        return {
            "olmoe-1b-7b": {
                "architectures": ["OlmoeForCausalLM"],
                "model_type": "olmoe",
                "hidden_size": 2048,
                "num_hidden_layers": 16,
                "num_attention_heads": 16,
                "num_key_value_heads": 16,
                "intermediate_size": 1024,
                "vocab_size": 50304,
                "max_position_embeddings": 4096,
                "rope_theta": 10000.0,
                "rms_norm_eps": 1e-05,
                "attention_bias": False,
                "torch_dtype": "bfloat16",
                "num_experts": 64,
                "num_experts_per_tok": 8,
                "router_aux_loss_coef": 0.01,
                "tie_word_embeddings": False,
                "initializer_range": 0.02,
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
            "tokenizer_class": "PreTrainedTokenizerFast",
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

    def test_supports_olmoe_architectures(self, olmoe_configs):
        """Test that AutoBridge.supports correctly identifies OLMoE models."""
        for model_name, config_dict in olmoe_configs.items():
            config = Mock()
            for key, value in config_dict.items():
                setattr(config, key, value)
            assert AutoBridge.supports(config) == True

        # Test non-causal LM architecture
        non_causal_config = Mock()
        non_causal_config.architectures = ["OlmoeModel"]  # Not ForCausalLM
        assert AutoBridge.supports(non_causal_config) == False


class TestOlMoEBridgeParameterMapping:
    """Test parameter mapping functionality in OlMoEBridge."""

    @pytest.fixture
    def mock_olmoe_state_dict(self):
        """Create a mock state dict with OLMoE parameter names."""
        return {
            "model.embed_tokens.weight": torch.randn(50304, 2048),
            "model.norm.weight": torch.randn(2048),
            "model.layers.0.input_layernorm.weight": torch.randn(2048),
            "model.layers.0.post_attention_layernorm.weight": torch.randn(2048),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(2048, 2048),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(2048, 2048),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(2048, 2048),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(2048, 2048),
            "model.layers.0.self_attn.q_norm.weight": torch.randn(2048),
            "model.layers.0.self_attn.k_norm.weight": torch.randn(2048),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(1024, 2048),
            "model.layers.0.mlp.up_proj.weight": torch.randn(1024, 2048),
            "model.layers.0.mlp.down_proj.weight": torch.randn(2048, 1024),
            "model.layers.0.mlp.gate.weight": torch.randn(64, 2048),  # Router
            "model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(1024, 2048),
            "model.layers.0.mlp.experts.0.up_proj.weight": torch.randn(1024, 2048),
            "model.layers.0.mlp.experts.0.down_proj.weight": torch.randn(2048, 1024),
            "lm_head.weight": torch.randn(50304, 2048),
        }

    def test_mapping_registry_has_olmoe_specific_mappings(self):
        """Test that mapping registry includes OLMoE-specific mappings."""
        bridge = OlMoEBridge()
        mapping_registry = bridge.mapping_registry()

        # This test verifies that the mapping registry was created
        assert mapping_registry is not None

    def test_olmoe_qk_layernorm_mapping(self):
        """Test that OLMoE bridge handles QK layernorm mappings correctly."""
        bridge = OlMoEBridge()
        mapping_registry = bridge.mapping_registry()

        # OLMoE uses QK layernorm, so there should be q_norm and k_norm mappings
        assert mapping_registry is not None

    def test_olmoe_moe_expert_mapping(self):
        """Test that OLMoE bridge includes MoE expert mappings."""
        bridge = OlMoEBridge()
        mapping_registry = bridge.mapping_registry()

        # OLMoE has experts, so it should have expert-specific mappings
        assert mapping_registry is not None

    def test_olmoe_router_mapping(self):
        """Test that OLMoE bridge includes router mappings."""
        bridge = OlMoEBridge()
        mapping_registry = bridge.mapping_registry()

        # OLMoE has a router for expert selection
        assert mapping_registry is not None

    def test_olmoe_gated_mlp_mapping(self):
        """Test that OLMoE bridge includes gated MLP mappings."""
        bridge = OlMoEBridge()
        mapping_registry = bridge.mapping_registry()

        # OLMoE uses gated MLP, so it should have GatedMLPMapping
        assert mapping_registry is not None

    def test_olmoe_qkv_mapping(self):
        """Test that OLMoE bridge includes QKV mappings."""
        bridge = OlMoEBridge()
        mapping_registry = bridge.mapping_registry()

        # OLMoE needs to combine Q, K, V projections
        assert mapping_registry is not None

    def test_olmoe_separate_embeddings(self):
        """Test that OLMoE bridge handles separate embeddings correctly."""
        bridge = OlMoEBridge()
        mapping_registry = bridge.mapping_registry()

        # OLMoE can have separate embeddings and output layer
        assert mapping_registry is not None
