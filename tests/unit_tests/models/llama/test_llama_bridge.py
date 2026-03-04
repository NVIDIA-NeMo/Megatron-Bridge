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
import torch.nn.functional as F
from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM

from megatron.bridge.models import AutoBridge
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.transformers_compat import rope_theta_from_hf
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.llama.llama_bridge import LlamaBridge


# Note: CONFIG_MAPPING and ACTIVATION_MAPPING are inherited from MegatronModelBridge


class TestLlamaBridgeConfigConverter:
    """Test cases for LlamaBridge.provider_bridge method."""

    @pytest.fixture
    def llama_3_2_1b_config_dict(self):
        """Create a sample Llama 3.2 1B configuration with RoPE scaling."""
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
            "rope_parameters": {"rope_type": "llama3", "rope_theta": 500000.0},
            "tie_word_embeddings": True,
            "torch_dtype": "bfloat16",
            "transformers_version": "5.0.0",
            "use_cache": True,
            "vocab_size": 128256,
        }

    @pytest.fixture
    def llama_2_7b_config_dict(self):
        """Create a sample Llama 2 7B configuration without RoPE scaling."""
        return {
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "intermediate_size": 11008,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-05,
            "tie_word_embeddings": False,
            "model_type": "llama",
            "initializer_range": 0.02,
        }

    @pytest.fixture
    def llama_config(self, llama_3_2_1b_config_dict):
        """Create a LlamaConfig instance for Llama 3.2 1B."""
        return LlamaConfig(**llama_3_2_1b_config_dict)

    @pytest.fixture
    def llama_2_config(self, llama_2_7b_config_dict):
        """Create a LlamaConfig instance for Llama 2 7B."""
        return LlamaConfig(**llama_2_7b_config_dict)

    @pytest.fixture
    def mock_pretrained_llama(self, llama_config):
        """Create a mock PreTrainedCausalLM with Llama 3.2 config."""
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = llama_config
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)
        mock_pretrained.model = Mock(spec=LlamaForCausalLM)
        mock_pretrained.model.dtype = torch.bfloat16
        return mock_pretrained

    @pytest.fixture
    def mock_pretrained_llama_2(self, llama_2_config):
        """Create a mock PreTrainedCausalLM with Llama 2 config."""
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = llama_2_config
        mock_pretrained.generation_config = None
        mock_pretrained.model = Mock(spec=LlamaForCausalLM)
        mock_pretrained.model.dtype = torch.float32
        return mock_pretrained

    def test_bridge_registration(self):
        """Test that LlamaBridge is properly registered."""
        assert issubclass(LlamaBridge, MegatronModelBridge)

    def test_provider_bridge_returns_gpt_provider(self, mock_pretrained_llama_2):
        """Test that provider_bridge returns a GPTModelProvider for Llama 2."""
        bridge = LlamaBridge()
        result = bridge.provider_bridge(mock_pretrained_llama_2)

        # For Llama 2 (no RoPE scaling), should return base GPTModelProvider with rope_scaling=False
        assert isinstance(result, GPTModelProvider)
        assert result.rope_scaling is False

    def test_provider_bridge_stores_rope_scaling_for_llama31(self, mock_pretrained_llama):
        """Test that provider_bridge enables rope_scaling for Llama 3.1/3.2."""
        bridge = LlamaBridge()
        result = bridge.provider_bridge(mock_pretrained_llama)

        # For Llama 3.1/3.2 with RoPE scaling, provider should have rope_scaling=True
        assert isinstance(result, GPTModelProvider)
        assert result.rope_scaling is True
        assert result.rope_scaling_factor == 32.0

    def test_provider_bridge_architecture_mapping(self, mock_pretrained_llama, llama_config):
        """Test that architecture parameters are correctly mapped from HF config."""
        bridge = LlamaBridge()
        result = bridge.provider_bridge(mock_pretrained_llama)

        # Check architecture mapping
        assert result.num_layers == llama_config.num_hidden_layers
        assert result.hidden_size == llama_config.hidden_size
        assert result.ffn_hidden_size == llama_config.intermediate_size
        assert result.num_attention_heads == llama_config.num_attention_heads
        assert result.num_query_groups == llama_config.num_key_value_heads
        assert result.seq_length == llama_config.max_position_embeddings
        assert result.rotary_base == rope_theta_from_hf(llama_config)
        assert result.vocab_size == llama_config.vocab_size
        assert result.layernorm_epsilon == llama_config.rms_norm_eps
        assert result.init_method_std == llama_config.initializer_range
        # Check additional mappings from HF config
        assert result.attention_dropout == llama_config.attention_dropout
        assert result.add_qkv_bias == llama_config.attention_bias

    def test_provider_bridge_llama_defaults_applied(self, mock_pretrained_llama_2):
        """Test that Llama-specific defaults are correctly applied."""
        bridge = LlamaBridge()
        result = bridge.provider_bridge(mock_pretrained_llama_2)

        # Check Llama-specific defaults from MEGATRON_DEFAULTS
        assert result.normalization == "RMSNorm"
        assert result.gated_linear_unit is True
        assert result.position_embedding_type == "rope"
        assert result.hidden_dropout == 0.0
        assert result.bias_activation_fusion is True
        assert result.masked_softmax_fusion is True
        assert result.persist_layer_norm is True
        assert result.bias_dropout_fusion is True
        assert result.apply_rope_fusion is True
        assert result.rotary_percent == 1.0
        # Check activation function from hidden_act mapping
        assert result.activation_func == F.silu
        # Check values from CONFIG_MAPPING (defaults from HF config)
        assert result.add_bias_linear is False  # from mlp_bias
        assert result.attention_dropout == 0.0  # from attention_dropout

    def test_provider_bridge_dtype_handling_bfloat16(self, llama_config):
        """Test dtype handling for bfloat16."""
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = llama_config
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)

        bridge = LlamaBridge()
        result = bridge.provider_bridge(mock_pretrained)

        assert result.params_dtype == torch.bfloat16
        assert result.bf16 is True
        assert result.fp16 is False

    def test_provider_bridge_dtype_handling_float16(self, llama_2_7b_config_dict):
        """Test dtype handling for float16."""
        config_dict = llama_2_7b_config_dict.copy()
        config_dict["torch_dtype"] = "float16"
        config = LlamaConfig(**config_dict)

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = config
        mock_pretrained.generation_config = None

        bridge = LlamaBridge()
        result = bridge.provider_bridge(mock_pretrained)

        assert result.params_dtype == torch.float16
        assert result.fp16 is True
        assert result.bf16 is False

    def test_provider_bridge_rope_scaling_params(self, mock_pretrained_llama):
        """Test that RoPE scaling parameters are correctly captured."""
        bridge = LlamaBridge()
        result = bridge.provider_bridge(mock_pretrained_llama)

        assert isinstance(result, GPTModelProvider)
        # RoPE scaling is now handled via Megatron Core's built-in support
        assert result.rope_scaling is True
        assert result.rope_scaling_factor == 32.0
        # Check position embedding
        assert result.rotary_base == rope_theta_from_hf(mock_pretrained_llama.config)

    def test_provider_bridge_embedding_sharing(self, llama_config):
        """Test embedding sharing configuration."""
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = llama_config
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)

        bridge = LlamaBridge()
        result = bridge.provider_bridge(mock_pretrained)

        assert result.share_embeddings_and_output_weights == llama_config.tie_word_embeddings

    def test_provider_bridge_head_dim(self, llama_3_2_1b_config_dict):
        """Test head_dim (kv_channels) extraction."""
        config = LlamaConfig(**llama_3_2_1b_config_dict)

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = config
        mock_pretrained.generation_config = Mock(spec=GenerationConfig)

        bridge = LlamaBridge()
        result = bridge.provider_bridge(mock_pretrained)

        # head_dim should be captured as kv_channels
        assert result.kv_channels == 64

    def test_provider_bridge_backward_compatibility(self, mock_pretrained_llama):
        """Test that provider_bridge still works as an alias for provider_bridge."""
        bridge = LlamaBridge()

        # Both methods should return equivalent results
        result_config = bridge.provider_bridge(mock_pretrained_llama)
        result_provider = bridge.provider_bridge(mock_pretrained_llama)

        # They should have the same architecture
        assert result_config.num_layers == result_provider.num_layers
        assert result_config.hidden_size == result_provider.hidden_size
        assert result_config.normalization == result_provider.normalization

    def test_provider_bridge_is_the_hf_to_megatron_method(self, mock_pretrained_llama):
        """Test that provider_bridge is the HF -> Megatron conversion method."""
        bridge = LlamaBridge()

        # provider_bridge is the HF -> Megatron conversion (symmetric with megatron_to_hf_config)
        result = bridge.provider_bridge(mock_pretrained_llama)

        assert isinstance(result, GPTModelProvider)
        assert result.normalization == "RMSNorm"

    def test_config_mapping_inherited_from_base(self):
        """Test that CONFIG_MAPPING is inherited from MegatronModelBridge."""
        # CONFIG_MAPPING should be inherited from base class
        assert hasattr(LlamaBridge, "CONFIG_MAPPING")
        assert LlamaBridge.CONFIG_MAPPING is MegatronModelBridge.CONFIG_MAPPING
        assert isinstance(LlamaBridge.CONFIG_MAPPING, list)
        assert len(LlamaBridge.CONFIG_MAPPING) > 0

        # Each entry should be a tuple with 2 elements
        for mapping in LlamaBridge.CONFIG_MAPPING:
            assert isinstance(mapping, tuple)
            assert len(mapping) == 2

        # Check some known mappings
        mapping_dict = dict(LlamaBridge.CONFIG_MAPPING)
        assert mapping_dict["num_hidden_layers"] == "num_layers"
        assert mapping_dict["intermediate_size"] == "ffn_hidden_size"
        assert mapping_dict["num_key_value_heads"] == "num_query_groups"
        # Check additional mappings added from HF LlamaConfig
        assert mapping_dict["attention_dropout"] == "attention_dropout"
        assert mapping_dict["attention_bias"] == "add_qkv_bias"
        assert mapping_dict["mlp_bias"] == "add_bias_linear"

    def test_activation_mapping_inherited_from_base(self):
        """Test that ACTIVATION_MAPPING is inherited from MegatronModelBridge."""
        assert hasattr(LlamaBridge, "ACTIVATION_MAPPING")
        assert LlamaBridge.ACTIVATION_MAPPING is MegatronModelBridge.ACTIVATION_MAPPING
        mapping = LlamaBridge.ACTIVATION_MAPPING
        assert mapping["silu"] == F.silu
        assert mapping["gelu"] == F.gelu
        assert mapping["relu"] == F.relu

    def test_hf_to_megatron_activation_inherited(self):
        """Test HF to Megatron activation function conversion (inherited from base)."""
        assert LlamaBridge.hf_to_megatron_activation("silu") == F.silu
        assert LlamaBridge.hf_to_megatron_activation("gelu") == F.gelu
        assert LlamaBridge.hf_to_megatron_activation("relu") == F.relu

    def test_megatron_to_hf_activation_inherited(self):
        """Test Megatron to HF activation function conversion (inherited from base)."""
        assert LlamaBridge.megatron_to_hf_activation(F.silu) == "silu"
        assert LlamaBridge.megatron_to_hf_activation(F.gelu) == "gelu"
        assert LlamaBridge.megatron_to_hf_activation(F.relu) == "relu"

    def test_hf_to_megatron_activation_unsupported(self):
        """Test that unsupported activation raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported activation function"):
            LlamaBridge.hf_to_megatron_activation("unknown_activation")

    def test_llama_defaults_applied_via_provider_bridge(self, mock_pretrained_llama_2):
        """Test that Llama-specific defaults are applied via provider_bridge (not class attributes)."""
        # Per refactoring guide: MEGATRON_DEFAULTS removed, now use direct property assignment
        bridge = LlamaBridge()
        provider = bridge.provider_bridge(mock_pretrained_llama_2)

        # These values are set directly in provider_bridge, not via class attributes
        assert provider.normalization == "RMSNorm"
        assert provider.gated_linear_unit is True
        assert provider.position_embedding_type == "rope"

    def test_source_name_and_model_type_class_attributes(self):
        """Test that SOURCE_NAME and MODEL_TYPE are set via @register_bridge decorator."""
        # Per refactoring guide: HF_DEFAULTS removed, now use SOURCE_NAME and MODEL_TYPE
        assert LlamaBridge.SOURCE_NAME == "LlamaForCausalLM"
        assert LlamaBridge.MODEL_TYPE == "llama"


class TestBaseClassHelperMethods:
    """Test cases for base class helper methods used by LlamaBridge."""

    @pytest.fixture
    def mock_pretrained_llama_2(self):
        """Create a mock PreTrainedCausalLM with Llama 2 config."""
        llama_2_7b_config_dict = {
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "intermediate_size": 11008,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
            "rms_norm_eps": 1e-05,
            "tie_word_embeddings": False,
            "model_type": "llama",
            "initializer_range": 0.02,
        }
        config = LlamaConfig(**llama_2_7b_config_dict)
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = config
        mock_pretrained.generation_config = None
        mock_pretrained.model = Mock(spec=LlamaForCausalLM)
        mock_pretrained.model.dtype = torch.float32
        return mock_pretrained

    def test_provider_bridge_uses_base_class_and_direct_assignment(self, mock_pretrained_llama_2):
        """Test that LlamaBridge.provider_bridge uses base class + direct property assignment."""
        bridge = LlamaBridge()
        provider = bridge.provider_bridge(mock_pretrained_llama_2)

        # Verify Llama-specific values are applied via direct assignment in provider_bridge
        assert provider.normalization == "RMSNorm"
        assert provider.gated_linear_unit is True
        assert provider.position_embedding_type == "rope"
        assert provider.hidden_dropout == 0.0

    def testhf_config_to_provider_kwargs_from_base_class(self):
        """Test that hf_config_to_provider_kwargs is inherited from MegatronModelBridge."""
        assert hasattr(LlamaBridge, "hf_config_to_provider_kwargs")
        # It's an instance method, check it exists on the class
        assert hasattr(MegatronModelBridge, "hf_config_to_provider_kwargs")

    def test_megatron_to_hf_config_from_base_class(self):
        """Test that megatron_to_hf_config is inherited from MegatronModelBridge."""
        # megatron_to_hf_config is the Megatron -> HF conversion method
        assert hasattr(LlamaBridge, "megatron_to_hf_config")
        assert hasattr(MegatronModelBridge, "megatron_to_hf_config")

    def testhf_config_to_provider_kwargs_returns_correct_mappings(self):
        """Test that hf_config_to_provider_kwargs correctly maps HF config to provider kwargs."""
        # Create a mock HF config
        mock_hf_config = Mock()
        mock_hf_config.num_hidden_layers = 32
        mock_hf_config.hidden_size = 4096
        mock_hf_config.intermediate_size = 14336
        mock_hf_config.num_attention_heads = 32
        mock_hf_config.num_key_value_heads = 8
        mock_hf_config.vocab_size = 128256
        mock_hf_config.max_position_embeddings = 8192
        mock_hf_config.rope_theta = 500000.0
        mock_hf_config.rms_norm_eps = 1e-05
        mock_hf_config.initializer_range = 0.02
        mock_hf_config.hidden_act = "silu"
        mock_hf_config.torch_dtype = "bfloat16"
        mock_hf_config.attention_dropout = 0.0
        mock_hf_config.tie_word_embeddings = False
        mock_hf_config.attention_bias = False
        mock_hf_config.mlp_bias = False

        bridge = LlamaBridge()
        kwargs = bridge.hf_config_to_provider_kwargs(mock_hf_config)

        assert kwargs["num_layers"] == 32
        assert kwargs["hidden_size"] == 4096
        assert kwargs["ffn_hidden_size"] == 14336
        assert kwargs["num_attention_heads"] == 32
        assert kwargs["num_query_groups"] == 8
        assert kwargs["vocab_size"] == 128256
        assert kwargs["seq_length"] == 8192
        assert kwargs["rotary_base"] == 500000.0
        assert kwargs["activation_func"] == F.silu

    def test_megatron_to_hf_config_returns_correct_mappings(self):
        """Test that megatron_to_hf_config correctly maps provider to HF config."""
        provider = GPTModelProvider(
            num_layers=32,
            hidden_size=4096,
            ffn_hidden_size=14336,
            num_attention_heads=32,
            num_query_groups=8,
            vocab_size=128256,
            seq_length=8192,
            rotary_base=500000.0,
            layernorm_epsilon=1e-05,
            init_method_std=0.02,
            activation_func=F.silu,
            bf16=True,
        )

        # Use megatron_to_hf_config (the Megatron -> HF conversion method)
        hf_config = LlamaBridge.megatron_to_hf_config(provider)

        assert hf_config["num_hidden_layers"] == 32
        assert hf_config["hidden_size"] == 4096
        assert hf_config["intermediate_size"] == 14336
        assert hf_config["num_attention_heads"] == 32
        assert hf_config["num_key_value_heads"] == 8
        assert hf_config["vocab_size"] == 128256
        assert hf_config["max_position_embeddings"] == 8192
        assert hf_config["rope_theta"] == 500000.0
        assert hf_config["hidden_act"] == "silu"
        assert hf_config["torch_dtype"] == "bfloat16"


class TestLlamaBridgeBidirectionalConversion:
    """Test cases for bidirectional config conversion."""

    def test_roundtrip_hf_to_megatron_to_hf(self):
        """Test roundtrip conversion: HF -> Megatron -> HF preserves key values."""
        # Start with HF config
        hf_config_dict = {
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "vocab_size": 128256,
            "max_position_embeddings": 8192,
            "rope_parameters": {"rope_type": "default", "rope_theta": 500000.0},
            "rms_norm_eps": 1e-05,
            "tie_word_embeddings": False,
            "model_type": "llama",
            "initializer_range": 0.02,
            "torch_dtype": "bfloat16",
            "attention_dropout": 0.0,
            "attention_bias": False,
            "mlp_bias": False,
            "hidden_act": "silu",
        }
        config = LlamaConfig(**hf_config_dict)

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = config
        mock_pretrained.generation_config = None

        bridge = LlamaBridge()

        # HF -> Megatron (using provider_bridge, the correct method name)
        provider = bridge.provider_bridge(mock_pretrained)

        # Megatron -> HF
        result_hf_config = bridge.megatron_to_hf_config(provider)

        # Verify key values are preserved
        assert result_hf_config["num_hidden_layers"] == hf_config_dict["num_hidden_layers"]
        assert result_hf_config["hidden_size"] == hf_config_dict["hidden_size"]
        assert result_hf_config["intermediate_size"] == hf_config_dict["intermediate_size"]
        assert result_hf_config["num_attention_heads"] == hf_config_dict["num_attention_heads"]
        assert result_hf_config["num_key_value_heads"] == hf_config_dict["num_key_value_heads"]
        assert result_hf_config["vocab_size"] == hf_config_dict["vocab_size"]
        assert result_hf_config["max_position_embeddings"] == hf_config_dict["max_position_embeddings"]
        assert result_hf_config["rope_theta"] == rope_theta_from_hf(config)
        assert result_hf_config["rms_norm_eps"] == hf_config_dict["rms_norm_eps"]
        assert result_hf_config["tie_word_embeddings"] == hf_config_dict["tie_word_embeddings"]
        # Check new mappings are preserved
        assert result_hf_config["mlp_bias"] == hf_config_dict["mlp_bias"]
        assert result_hf_config["hidden_act"] == hf_config_dict["hidden_act"]

    def test_config_mapping_bidirectional(self):
        """Test that CONFIG_MAPPING works in both directions."""
        bridge = LlamaBridge()

        # Create lookup dicts for both directions
        hf_to_megatron = {hf: mg for hf, mg in bridge.CONFIG_MAPPING}
        megatron_to_hf = {mg: hf for hf, mg in bridge.CONFIG_MAPPING}

        # Verify bidirectional lookup
        assert hf_to_megatron["num_hidden_layers"] == "num_layers"
        assert megatron_to_hf["num_layers"] == "num_hidden_layers"

        assert hf_to_megatron["intermediate_size"] == "ffn_hidden_size"
        assert megatron_to_hf["ffn_hidden_size"] == "intermediate_size"


class TestLlamaBridgeMegatronToHFConfig:
    """Test cases for Megatron -> HF config conversion."""

    def test_megatron_to_hf_config_basic(self):
        """Test basic Megatron to HF config conversion."""
        provider = GPTModelProvider(
            num_layers=32,
            hidden_size=4096,
            ffn_hidden_size=11008,
            num_attention_heads=32,
            num_query_groups=32,
            init_method_std=0.02,
            layernorm_epsilon=1e-05,
            seq_length=4096,
            rotary_base=10000,
            vocab_size=32000,
            share_embeddings_and_output_weights=False,
            bf16=False,
            fp16=False,
        )

        hf_config = LlamaBridge.megatron_to_hf_config(provider)

        assert hf_config["architectures"] == ["LlamaForCausalLM"]
        assert hf_config["model_type"] == "llama"
        assert hf_config["num_hidden_layers"] == 32
        assert hf_config["hidden_size"] == 4096
        assert hf_config["intermediate_size"] == 11008
        assert hf_config["num_attention_heads"] == 32
        assert hf_config["num_key_value_heads"] == 32
        assert hf_config["rms_norm_eps"] == 1e-05
        assert hf_config["rope_theta"] == 10000
        assert hf_config["vocab_size"] == 32000
        assert hf_config["tie_word_embeddings"] is False
        assert hf_config["torch_dtype"] == "float32"

    def test_megatron_to_hf_config_with_rope_scaling(self):
        """Test Megatron to HF config conversion with RoPE scaling."""
        provider = GPTModelProvider(
            num_layers=16,
            hidden_size=2048,
            ffn_hidden_size=8192,
            num_attention_heads=32,
            num_query_groups=8,
            init_method_std=0.02,
            layernorm_epsilon=1e-05,
            seq_length=131072,
            rotary_base=500000,
            vocab_size=128256,
            share_embeddings_and_output_weights=True,
            bf16=True,
            # RoPE scaling is now handled via Megatron Core's built-in support
            rope_scaling=True,
            rope_scaling_factor=32.0,
        )

        hf_config = LlamaBridge.megatron_to_hf_config(provider)

        assert hf_config["torch_dtype"] == "bfloat16"
        assert hf_config["tie_word_embeddings"] is True
        assert "rope_scaling" in hf_config
        assert hf_config["rope_scaling"]["rope_type"] == "llama3"
        assert hf_config["rope_scaling"]["factor"] == 32.0
        # These use Megatron Core defaults
        assert hf_config["rope_scaling"]["low_freq_factor"] == 1.0
        assert hf_config["rope_scaling"]["high_freq_factor"] == 4.0
        assert hf_config["rope_scaling"]["original_max_position_embeddings"] == 8192


class TestLlamaBridgeMappingRegistry:
    """Test cases for LlamaBridge.mapping_registry method."""

    def test_mapping_registry_not_none(self):
        """Test that mapping_registry returns a valid registry."""
        bridge = LlamaBridge()
        registry = bridge.mapping_registry()

        assert registry is not None

    def test_mapping_registry_contains_embeddings(self):
        """Test that mapping registry contains embedding mappings."""
        bridge = LlamaBridge()
        registry = bridge.mapping_registry()

        # Check that we can look up embedding mapping
        mapping = registry.megatron_to_hf_lookup("embedding.word_embeddings.weight")
        assert mapping is not None
        assert mapping.hf_param == "model.embed_tokens.weight"

    def test_mapping_registry_contains_output_layer(self):
        """Test that mapping registry contains output layer mapping."""
        bridge = LlamaBridge()
        registry = bridge.mapping_registry()

        mapping = registry.megatron_to_hf_lookup("output_layer.weight")
        assert mapping is not None
        assert mapping.hf_param == "lm_head.weight"

    def test_mapping_registry_contains_final_layernorm(self):
        """Test that mapping registry contains final layernorm mapping."""
        bridge = LlamaBridge()
        registry = bridge.mapping_registry()

        mapping = registry.megatron_to_hf_lookup("decoder.final_layernorm.weight")
        assert mapping is not None
        assert mapping.hf_param == "model.norm.weight"


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
                "rope_parameters": {"rope_type": "llama3", "rope_theta": 500000.0},
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
                "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
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
                    mock_provider = Mock(spec=GPTModelProvider)
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
            assert AutoBridge.supports(config) is True

        # Test non-causal LM architecture
        non_causal_config = Mock()
        non_causal_config.architectures = ["LlamaModel"]  # Not ForCausalLM
        assert AutoBridge.supports(non_causal_config) is False

    def test_list_supported_models(self):
        """Test list_supported_models includes LlamaForCausalLM."""
        # This test requires the dispatch system to be set up
        # Since we're testing in isolation, we'll skip this test
        # In a real environment, this would work if the bridges are registered
        pass  # Skip for now as it requires full dispatch setup
