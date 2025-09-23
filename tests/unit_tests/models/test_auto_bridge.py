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

"""
Unit tests for AutoBridge automatic bridge selection and bridge functionality.
"""

from unittest.mock import Mock, PropertyMock, patch

import pytest
import torch
from transformers import LlamaConfig
from transformers.configuration_utils import PretrainedConfig

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


def create_mock_pretrained_causal_lm():
    """Helper function to create a mock PreTrainedCausalLM that passes isinstance checks."""

    class MockPreTrainedCausalLM(PreTrainedCausalLM):
        def __init__(self):
            pass  # Skip actual initialization

    return MockPreTrainedCausalLM()


class TestAutoBridge:
    """Test cases for AutoBridge automatic selection and full bridge functionality."""

    @pytest.fixture
    def llama_config(self):
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
    def llama_config_mock(self):
        """Create a mock Llama configuration."""
        config = Mock()
        config.architectures = ["LlamaForCausalLM"]
        config.model_type = "llama"
        config.vocab_size = 32000
        config.hidden_size = 2048
        config.num_hidden_layers = 16
        config.num_attention_heads = 32
        return config

    @pytest.fixture
    def bert_config(self):
        """Create a mock BERT configuration (unsupported)."""
        config = Mock()
        config.architectures = ["BertForMaskedLM"]
        config.model_type = "bert"
        return config

    @pytest.fixture
    def gpt2_config(self):
        """Create a mock GPT2 configuration."""
        config = Mock()
        config.architectures = ["GPT2ForCausalLM", "GPT2LMHeadModel"]
        config.model_type = "gpt2"
        return config

    def test_from_hf_pretrained_with_unsupported_model(self, bert_config):
        """Test AutoBridge raises ValueError for unsupported models."""
        with patch("megatron.bridge.models.conversion.auto_bridge.AutoConfig") as mock_auto_config:
            # Setup mocks
            mock_auto_config.from_pretrained.return_value = bert_config

            # Should raise ValueError
            with pytest.raises(ValueError) as exc_info:
                AutoBridge.from_hf_pretrained("bert-base-uncased")

            assert "Model architecture not supported by AutoBridge" in str(exc_info.value)
            assert "BertForMaskedLM" in str(exc_info.value)

    def test_from_pretrained_config_load_failure(self):
        """Test AutoBridge handles config loading failures gracefully."""
        with patch("megatron.bridge.models.conversion.auto_bridge.AutoConfig") as mock_auto_config:
            # Setup mock to raise exception
            mock_auto_config.from_pretrained.side_effect = Exception("Config not found")

            # Should raise ValueError with helpful message
            with pytest.raises(ValueError) as exc_info:
                AutoBridge.from_hf_pretrained("invalid/path")

            assert "Failed to load configuration" in str(exc_info.value)
            assert "Config not found" in str(exc_info.value)

    def test_can_handle_supported_model(self, llama_config_mock):
        """Test can_handle returns True for supported models."""
        with patch("megatron.bridge.models.conversion.auto_bridge.AutoConfig") as mock_auto_config:
            mock_auto_config.from_pretrained.return_value = llama_config_mock

            assert AutoBridge.can_handle("meta-llama/Llama-3-8B") is True
            mock_auto_config.from_pretrained.assert_called_with("meta-llama/Llama-3-8B", trust_remote_code=False)

    def test_can_handle_unsupported_model(self, bert_config):
        """Test can_handle returns False for unsupported models."""
        with patch("megatron.bridge.models.conversion.auto_bridge.AutoConfig") as mock_auto_config:
            mock_auto_config.from_pretrained.return_value = bert_config

            assert AutoBridge.can_handle("bert-base-uncased") is False

    def test_can_handle_invalid_path(self):
        """Test can_handle returns False for invalid paths."""
        with patch("megatron.bridge.models.conversion.auto_bridge.AutoConfig") as mock_auto_config:
            mock_auto_config.from_pretrained.side_effect = Exception("Not found")

            assert AutoBridge.can_handle("invalid/path") is False

    # Test core bridge functionality (from original AutoBridge tests)
    def test_from_hf_pretrained_with_model_id(self):
        """Test from_hf_pretrained with model ID string."""
        # This test checks that from_hf_pretrained creates correct bridge instance
        # We'll use a mock pretrained model
        mock_model = Mock(spec=PreTrainedCausalLM)
        mock_config = Mock(spec=PretrainedConfig)
        mock_config.architectures = ["GPT2LMHeadModel"]  # Use a real architecture
        mock_model.config = mock_config

        with patch(
            "megatron.bridge.models.conversion.auto_bridge.PreTrainedCausalLM.from_pretrained"
        ) as mock_from_pretrained:
            # Set up the from_pretrained class method properly
            mock_from_pretrained.return_value = mock_model

            with patch("megatron.bridge.models.conversion.auto_bridge.AutoConfig") as mock_autoconfig:
                mock_autoconfig.from_pretrained.return_value = mock_config

                # Skip architecture validation for this test
                with patch.object(AutoBridge, "_validate_config"):
                    # Call from_hf_pretrained
                    model_id = "gpt2"
                    result = AutoBridge.from_hf_pretrained(model_id, trust_remote_code=True)

                # Assertions
                assert isinstance(result, AutoBridge)
                assert result.hf_pretrained == mock_model
                mock_from_pretrained.assert_called_once_with(model_id, trust_remote_code=True)

    def test_from_pretrained_with_additional_kwargs(self):
        """Test from_pretrained with various kwargs."""
        # Setup mocks
        mock_model = Mock(spec=PreTrainedCausalLM)
        mock_config = Mock(spec=PretrainedConfig)
        mock_config.architectures = ["GPT2LMHeadModel"]
        mock_model.config = mock_config

        with patch(
            "megatron.bridge.models.conversion.auto_bridge.PreTrainedCausalLM.from_pretrained"
        ) as mock_from_pretrained:
            # Set up the from_pretrained class method properly
            mock_from_pretrained.return_value = mock_model

            with patch("megatron.bridge.models.conversion.auto_bridge.AutoConfig") as mock_autoconfig:
                mock_autoconfig.from_pretrained.return_value = mock_config

                # Skip architecture validation for this test
                with patch.object(AutoBridge, "_validate_config"):
                    # Call with multiple kwargs
                    result = AutoBridge.from_hf_pretrained(
                        "model-id",
                        torch_dtype=torch.bfloat16,
                        low_cpu_mem_usage=True,
                        attn_implementation="flash_attention_2",
                    )

                # Assertions
                assert isinstance(result, AutoBridge)
                assert result.hf_pretrained == mock_model
                mock_from_pretrained.assert_called_once_with(
                    "model-id",
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    attn_implementation="flash_attention_2",
                )

    def test_to_megatron_provider_basic(self, llama_config):
        """Test basic to_megatron_provider conversion."""
        # Setup mocks
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_hf_model.config = LlamaConfig(**llama_config)

        # Mock model bridge
        mock_model_bridge = Mock()
        mock_provider = Mock(spec=GPTModelProvider)
        mock_model_bridge.provider_bridge.return_value = mock_provider

        with patch.object(AutoBridge, "_model_bridge", mock_model_bridge):
            # Create bridge and convert
            bridge = AutoBridge(mock_hf_model)
            result = bridge.to_megatron_provider(load_weights=False)

            # Assertions
            assert result == mock_provider
            mock_model_bridge.provider_bridge.assert_called_once_with(mock_hf_model)

    def test_to_megatron_provider_with_different_model_types(self):
        """Test to_megatron_provider with different model architectures."""
        # Test with GPT2 model
        mock_gpt2_model = Mock(spec=PreTrainedCausalLM)
        mock_gpt2_model.config = Mock(model_type="gpt2")

        # Mock model bridge
        mock_model_bridge = Mock()
        mock_provider = Mock(spec=GPTModelProvider)
        mock_model_bridge.provider_bridge.return_value = mock_provider

        with patch.object(AutoBridge, "_model_bridge", mock_model_bridge):
            bridge = AutoBridge(mock_gpt2_model)
            result = bridge.to_megatron_provider(load_weights=False)

            assert result == mock_provider
            mock_model_bridge.provider_bridge.assert_called_once_with(mock_gpt2_model)

    def test_to_megatron_provider_with_custom_kwargs(self, llama_config):
        """Test to_megatron_provider with custom keyword arguments."""
        # Setup mocks
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_hf_model.config = LlamaConfig(**llama_config)

        # Mock model bridge
        mock_model_bridge = Mock()
        mock_provider = Mock(spec=GPTModelProvider)
        mock_provider.register_pre_wrap_hook = Mock()
        mock_model_bridge.provider_bridge.return_value = mock_provider
        mock_model_bridge.load_weights_hf_to_megatron = Mock()

        with patch.object(AutoBridge, "_model_bridge", mock_model_bridge):
            # Create bridge and convert with load_weights=True
            bridge = AutoBridge(mock_hf_model)
            result = bridge.to_megatron_provider(load_weights=True)

            # Assertions
            assert result == mock_provider
            mock_model_bridge.provider_bridge.assert_called_once_with(mock_hf_model)
            # Check that a pre-wrap hook was registered for loading weights
            mock_provider.register_pre_wrap_hook.assert_called_once()

    def test_to_megatron_provider_error_handling(self):
        """Test to_megatron_provider error handling."""
        # Setup mock to raise an exception
        mock_hf_model = Mock(spec=PreTrainedCausalLM)

        # Mock model bridge to raise error
        mock_model_bridge = Mock()
        mock_model_bridge.provider_bridge.side_effect = ValueError("Unsupported model type")

        with patch.object(AutoBridge, "_model_bridge", mock_model_bridge):
            bridge = AutoBridge(mock_hf_model)

            # Should propagate the exception
            with pytest.raises(ValueError, match="Unsupported model type"):
                bridge.to_megatron_provider()

    def test_bridge_instance_creation(self):
        """Test AutoBridge instance creation."""
        # Test with PreTrainedCausalLM
        mock_model = Mock(spec=PreTrainedCausalLM)

        bridge = AutoBridge(mock_model)

        # Should have the expected methods
        assert hasattr(bridge, "from_hf_pretrained")
        assert hasattr(bridge, "to_megatron_provider")
        assert hasattr(bridge, "load_hf_weights")
        assert hasattr(bridge, "export_hf_weights")
        assert hasattr(bridge, "save_hf_pretrained")
        assert bridge.hf_pretrained == mock_model

        # Test with PretrainedConfig
        mock_config = Mock(spec=PretrainedConfig)
        bridge_config = AutoBridge(mock_config)
        assert bridge_config.hf_pretrained == mock_config

        # Test with invalid type
        with pytest.raises(
            ValueError, match="hf_pretrained must be a PreTrainedCausalLM or PretrainedConfig instance"
        ):
            AutoBridge("invalid")

        # from_hf_pretrained should be a classmethod
        import inspect

        assert inspect.ismethod(AutoBridge.from_hf_pretrained)

    def test_from_hf_config(self):
        """Test creating bridge from config only."""
        # Create a mock config
        config = Mock(spec=PretrainedConfig)
        config.architectures = ["GPT2LMHeadModel"]

        # Skip architecture validation for this test
        with patch.object(AutoBridge, "_validate_config"):
            bridge = AutoBridge.from_hf_config(config)
            assert isinstance(bridge, AutoBridge)
            assert bridge.hf_pretrained == config

    def test_from_hf_config_invalid_architecture(self):
        """Test from_hf_config with unsupported architecture."""
        config = Mock(spec=PretrainedConfig)
        config.architectures = ["BertForMaskedLM"]  # Not a CausalLM

        with pytest.raises(ValueError, match="Model architecture not supported by AutoBridge"):
            AutoBridge.from_hf_config(config)

    def test_supports_method(self):
        """Test the supports class method."""
        # Supported config
        config = Mock()
        config.architectures = ["LlamaForCausalLM"]
        assert AutoBridge.supports(config) is True

        # Multiple architectures, one supported
        config.architectures = ["LlamaModel", "LlamaForCausalLM"]
        assert AutoBridge.supports(config) is True

        # No CausalLM architecture
        config.architectures = ["BertForMaskedLM"]
        assert AutoBridge.supports(config) is False

        # No architectures
        config.architectures = []
        assert AutoBridge.supports(config) is False

        # Missing architectures attribute
        config_no_arch = Mock(spec=[])
        assert AutoBridge.supports(config_no_arch) is False

    def test_list_supported_models(self):
        """Test listing supported models."""
        # Since this method looks at internal dispatch registry,
        # we'll just test that it returns a list
        with patch("megatron.bridge.models.conversion.auto_bridge.model_bridge") as mock_bridge:
            # Mock to avoid AttributeError
            mock_bridge.get_model_bridge = Mock()
            mock_bridge.get_model_bridge._exact_types = {}
            supported = AutoBridge.list_supported_models()
            assert isinstance(supported, list)
            # The list might be empty if no models are registered in test environment

    def test_load_hf_weights(self):
        """Test loading weights into a Megatron model."""
        # Setup mocks
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_config = Mock(spec=PretrainedConfig)
        mock_hf_model.config = mock_config

        mock_megatron_model = [Mock()]  # List of model instances

        mock_model_bridge = Mock()
        mock_model_bridge.load_weights_hf_to_megatron = Mock()

        with patch.object(AutoBridge, "_model_bridge", mock_model_bridge):
            bridge = AutoBridge(mock_hf_model)
            bridge.load_hf_weights(mock_megatron_model)

            mock_model_bridge.load_weights_hf_to_megatron.assert_called_once_with(mock_megatron_model, mock_hf_model)

    def test_load_hf_weights_from_path(self):
        """Test loading weights from a different path."""
        # Setup mocks
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_config = Mock(spec=PretrainedConfig)
        mock_hf_model.config = mock_config

        mock_megatron_model = [Mock()]

        mock_model_bridge = Mock()
        mock_model_bridge.load_weights_hf_to_megatron = Mock()

        # Create bridge first, then patch the from_pretrained method
        with patch.object(AutoBridge, "_model_bridge", mock_model_bridge):
            bridge = AutoBridge(mock_hf_model)

            # Now patch the from_pretrained method
            with patch(
                "megatron.bridge.models.conversion.auto_bridge.PreTrainedCausalLM.from_pretrained"
            ) as mock_from_pretrained:
                mock_loaded_model = Mock(spec=PreTrainedCausalLM)
                mock_from_pretrained.return_value = mock_loaded_model

                bridge.load_hf_weights(mock_megatron_model, "./custom_model")

                mock_from_pretrained.assert_called_once_with("./custom_model")
                mock_model_bridge.load_weights_hf_to_megatron.assert_called_once_with(
                    mock_megatron_model, mock_loaded_model
                )

    def test_load_hf_weights_no_path_config_only(self):
        """Test load_hf_weights fails when bridge has config only and no path provided."""
        mock_config = Mock(spec=PretrainedConfig)
        bridge = AutoBridge(mock_config)

        with pytest.raises(ValueError, match="hf_path is required when hf_pretrained is not a PreTrainedCausalLM"):
            bridge.load_hf_weights([Mock()])

    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.barrier")
    @patch("torch.distributed.is_available", return_value=True)
    @patch("torch.distributed.is_initialized", return_value=True)
    def test_save_hf_pretrained(self, mock_is_init, mock_is_avail, mock_barrier, mock_get_rank):
        """Test saving a model in HuggingFace format."""
        # Setup mocks
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_hf_model.save_artifacts = Mock()
        mock_hf_model.state = Mock()
        mock_hf_model.state.source = Mock(spec=["save_generator"])

        from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource

        mock_hf_model.state.source = Mock(spec=SafeTensorsStateSource)
        mock_hf_model.state.source.save_generator = Mock()

        mock_megatron_model = [Mock()]

        with patch.object(AutoBridge, "save_hf_weights") as mock_save_hf_weights:
            bridge = AutoBridge(mock_hf_model)
            bridge.save_hf_pretrained(mock_megatron_model, "./output_dir")

            # Check artifacts were saved on rank 0
            mock_hf_model.save_artifacts.assert_called_once_with("./output_dir")
            mock_save_hf_weights.assert_called_once_with(mock_megatron_model, "./output_dir", True)

    @patch("torch.distributed.get_rank", return_value=1)
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.is_available", return_value=True)
    @patch("torch.distributed.barrier")
    def test_save_hf_pretrained_non_zero_rank(
        self, mock_barrier, mock_is_available, mock_is_initialized, mock_get_rank
    ):
        """Test save_hf_pretrained on non-zero rank (should not save artifacts)."""
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_hf_model.save_artifacts = Mock()

        mock_megatron_model = [Mock()]

        with patch.object(AutoBridge, "save_hf_weights") as mock_save_hf_weights:
            bridge = AutoBridge(mock_hf_model)
            bridge.save_hf_pretrained(mock_megatron_model, "./output_dir")

            # Artifacts should NOT be saved on non-zero rank
            mock_hf_model.save_artifacts.assert_not_called()
            mock_save_hf_weights.assert_called_once_with(mock_megatron_model, "./output_dir", True)

    def test_export_hf_weights(self):
        """Test exporting weights from Megatron to HF format."""
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_hf_model.config = Mock()
        mock_hf_model.config.architectures = ["LlamaForCausalLM"]
        mock_hf_model.config.auto_map = None

        mock_megatron_model = [Mock()]
        mock_megatron_model[0].module = None  # No nested module

        # Mock the export process
        with patch(
            "megatron.bridge.models.conversion.auto_bridge.model_bridge.stream_weights_megatron_to_hf"
        ) as mock_bridge_state:
            mock_weight_iter = [("weight1", torch.randn(10, 10)), ("weight2", torch.randn(5, 5))]
            mock_bridge_state.return_value = iter(mock_weight_iter)

            with patch("megatron.bridge.models.conversion.auto_bridge.transformers") as mock_transformers:
                mock_arch_class = Mock()
                mock_transformers.LlamaForCausalLM = mock_arch_class

                bridge = AutoBridge(mock_hf_model)

                # Mock the cached property to avoid accessing transformers
                with patch.object(AutoBridge, "_causal_lm_architecture", new_callable=PropertyMock) as mock_prop:
                    mock_prop.return_value = mock_arch_class
                    weights = list(bridge.export_hf_weights(mock_megatron_model, cpu=True))

                    assert len(weights) == 2
                    assert weights[0][0] == "weight1"
                    assert weights[1][0] == "weight2"
                    assert isinstance(weights[0][1], torch.Tensor)
                    assert isinstance(weights[1][1], torch.Tensor)

    def test_get_causal_lm_architecture(self):
        """Test getting the CausalLM architecture class."""
        # Test with model that has architectures
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_hf_model.config = Mock()
        mock_hf_model.config.architectures = ["LlamaForCausalLM"]
        mock_hf_model.config.auto_map = None

        with patch("megatron.bridge.models.conversion.auto_bridge.transformers") as mock_transformers:
            mock_arch_class = Mock()
            mock_transformers.LlamaForCausalLM = mock_arch_class

            # Create bridge instance directly without isinstance validation
            bridge = AutoBridge.__new__(AutoBridge)
            bridge.hf_pretrained = mock_hf_model

            arch = bridge._causal_lm_architecture
            assert arch == mock_arch_class

    def test_get_causal_lm_architecture_no_architectures(self):
        """Test error when no architectures found."""
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_hf_model.config = Mock()
        mock_hf_model.config.architectures = []
        mock_hf_model.config.auto_map = None

        bridge = AutoBridge.__new__(AutoBridge)
        bridge.hf_pretrained = mock_hf_model
        with pytest.raises(ValueError, match="No architectures found in model config"):
            bridge._causal_lm_architecture

    def test_get_causal_lm_architecture_no_causal_lm(self):
        """Test error when no CausalLM architecture found."""
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_hf_model.config = Mock()
        mock_hf_model.config.architectures = ["BertForMaskedLM"]
        mock_hf_model.config.auto_map = None

        bridge = AutoBridge.__new__(AutoBridge)
        bridge.hf_pretrained = mock_hf_model
        with pytest.raises(ValueError, match="No CausalLM architecture found"):
            bridge._causal_lm_architecture

    def test_get_causal_lm_architecture_not_in_transformers(self):
        """Test error when architecture class not found in transformers."""
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_hf_model.config = Mock()
        mock_hf_model.config.architectures = ["CustomForCausalLM"]
        mock_hf_model.config.auto_map = None

        bridge = AutoBridge.__new__(AutoBridge)
        bridge.hf_pretrained = mock_hf_model

        # Mock transformers to not have the CustomForCausalLM attribute
        with patch("megatron.bridge.models.conversion.auto_bridge.transformers") as mock_transformers:
            # Configure mock to raise AttributeError when accessing CustomForCausalLM
            del mock_transformers.CustomForCausalLM

            with pytest.raises(ValueError, match="Architecture class 'CustomForCausalLM' not found in transformers"):
                bridge._causal_lm_architecture

    def test_repr(self):
        """Test string representation of AutoBridge."""
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_hf_model.__repr__ = Mock(return_value="PreTrainedCausalLM(\n  config=...\n)")

        mock_model_bridge = Mock()
        mock_model_bridge.__repr__ = Mock(return_value="ModelBridge(\n  mappings=...\n)")

        with patch.object(AutoBridge, "_model_bridge", mock_model_bridge):
            bridge = AutoBridge.__new__(AutoBridge)
            bridge.hf_pretrained = mock_hf_model
            repr_str = repr(bridge)

            assert "AutoBridge(" in repr_str
            assert "(hf_pretrained):" in repr_str
            assert "(model_bridge):" in repr_str
            assert "PreTrainedCausalLM" in repr_str
            assert "ModelBridge" in repr_str

    @patch("torch.cuda.current_device", return_value=0)
    @patch("torch.cuda.is_available", return_value=False)
    def test_cpu_compatibility(self, mock_cuda_avail, mock_cuda_device):
        """Test that bridge works on CPU-only systems."""
        # This test ensures the bridge doesn't require CUDA
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_hf_model.config = Mock()
        mock_hf_model.config.architectures = ["GPT2ForCausalLM"]
        mock_hf_model.config.auto_map = None

        # Create bridge - should work without CUDA
        bridge = AutoBridge.__new__(AutoBridge)
        bridge.hf_pretrained = mock_hf_model
        assert bridge.hf_pretrained == mock_hf_model

        # Test methods that might use device
        with patch("megatron.bridge.models.conversion.auto_bridge.transformers") as mock_transformers:
            mock_transformers.GPT2ForCausalLM = Mock()

            # These operations should work on CPU
            arch = bridge._causal_lm_architecture
            assert arch is not None

    def test_kwargs_passed_through(self, gpt2_config):
        """Test that all kwargs are properly passed to the underlying loader."""
        with patch("megatron.bridge.models.conversion.auto_bridge.AutoConfig") as mock_auto_config:
            with patch(
                "megatron.bridge.models.conversion.auto_bridge.PreTrainedCausalLM.from_pretrained"
            ) as mock_from_pretrained:
                with patch.object(AutoBridge, "_validate_config"):
                    mock_auto_config.from_pretrained.return_value = gpt2_config
                    mock_model = create_mock_pretrained_causal_lm()
                    mock_from_pretrained.return_value = mock_model

                    # Call with various kwargs
                    AutoBridge.from_hf_pretrained(
                        "gpt2",
                        trust_remote_code=True,
                        device_map="balanced",
                        torch_dtype="bfloat16",
                        custom_param="test",
                    )

                    # Verify all kwargs were passed
                    mock_from_pretrained.assert_called_once_with(
                        "gpt2",
                        trust_remote_code=True,
                        device_map="balanced",
                        torch_dtype="bfloat16",
                        custom_param="test",
                    )

    @patch.object(AutoBridge, "save_megatron_model")
    @patch.object(AutoBridge, "to_megatron_model")
    @patch.object(AutoBridge, "from_hf_pretrained")
    def test_import_ckpt_basic(self, mock_from_hf_pretrained, mock_to_megatron_model, mock_save_megatron_model):
        """Test basic import_ckpt functionality."""
        # Setup mocks
        mock_bridge = Mock(spec=AutoBridge)
        mock_from_hf_pretrained.return_value = mock_bridge

        mock_megatron_model = [Mock()]
        mock_bridge.to_megatron_model.return_value = mock_megatron_model
        mock_bridge.save_megatron_model = Mock()

        # Test import_ckpt
        AutoBridge.import_ckpt("meta-llama/Llama-3-8B", "./megatron_checkpoint")

        # Assertions
        mock_from_hf_pretrained.assert_called_once_with("meta-llama/Llama-3-8B")
        mock_bridge.to_megatron_model.assert_called_once_with(wrap_with_ddp=False, use_cpu_initialization=True)
        mock_bridge.save_megatron_model.assert_called_once_with(
            mock_megatron_model, "./megatron_checkpoint", hf_tokenizer_path="meta-llama/Llama-3-8B"
        )

    @patch.object(AutoBridge, "save_megatron_model")
    @patch.object(AutoBridge, "to_megatron_model")
    @patch.object(AutoBridge, "from_hf_pretrained")
    def test_import_ckpt_with_kwargs(self, mock_from_hf_pretrained, mock_to_megatron_model, mock_save_megatron_model):
        """Test import_ckpt with custom kwargs."""
        # Setup mocks
        mock_bridge = Mock(spec=AutoBridge)
        mock_from_hf_pretrained.return_value = mock_bridge

        mock_megatron_model = [Mock()]
        mock_bridge.to_megatron_model.return_value = mock_megatron_model
        mock_bridge.save_megatron_model = Mock()

        # Test import_ckpt with kwargs
        AutoBridge.import_ckpt("./local_model", "./megatron_checkpoint", torch_dtype=torch.float16, device_map="auto")

        # Assertions
        mock_from_hf_pretrained.assert_called_once_with("./local_model", torch_dtype=torch.float16, device_map="auto")
        mock_bridge.to_megatron_model.assert_called_once_with(wrap_with_ddp=False, use_cpu_initialization=True)
        mock_bridge.save_megatron_model.assert_called_once_with(
            mock_megatron_model, "./megatron_checkpoint", hf_tokenizer_path="./local_model"
        )

    def test_export_ckpt_basic(self):
        """Test basic export_ckpt functionality."""
        # Setup mocks
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_config = Mock(spec=PretrainedConfig)
        mock_config.architectures = ["LlamaForCausalLM"]
        mock_hf_model.config = mock_config

        mock_megatron_model = [Mock()]

        bridge = AutoBridge.__new__(AutoBridge)
        bridge.hf_pretrained = mock_hf_model

        with patch.object(bridge, "load_megatron_model") as mock_load_megatron_model:
            with patch.object(bridge, "save_hf_pretrained") as mock_save_hf_pretrained:
                mock_load_megatron_model.return_value = mock_megatron_model

                # Test export_ckpt
                bridge.export_ckpt("./megatron_checkpoint", "./hf_export")

                # Assertions
                mock_load_megatron_model.assert_called_once_with("./megatron_checkpoint", wrap_with_ddp=False)
                mock_save_hf_pretrained.assert_called_once_with(mock_megatron_model, "./hf_export", show_progress=True)

    def test_export_ckpt_with_kwargs(self):
        """Test export_ckpt with custom kwargs."""
        # Setup mocks
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_config = Mock(spec=PretrainedConfig)
        mock_config.architectures = ["LlamaForCausalLM"]
        mock_hf_model.config = mock_config

        mock_megatron_model = [Mock()]

        bridge = AutoBridge.__new__(AutoBridge)
        bridge.hf_pretrained = mock_hf_model

        with patch.object(bridge, "load_megatron_model") as mock_load_megatron_model:
            with patch.object(bridge, "save_hf_pretrained") as mock_save_hf_pretrained:
                mock_load_megatron_model.return_value = mock_megatron_model

                # Test export_ckpt with kwargs
                bridge.export_ckpt("./megatron_checkpoint", "./hf_export", show_progress=False)

                # Assertions
                mock_load_megatron_model.assert_called_once_with("./megatron_checkpoint", wrap_with_ddp=False)
                mock_save_hf_pretrained.assert_called_once_with(
                    mock_megatron_model, "./hf_export", show_progress=False
                )

    def test_save_megatron_model_basic(self):
        """Test save_megatron_model method."""
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_config = Mock(spec=PretrainedConfig)
        mock_config.architectures = ["LlamaForCausalLM"]
        mock_hf_model.config = mock_config

        mock_megatron_model = [Mock()]

        bridge = AutoBridge.__new__(AutoBridge)
        bridge.hf_pretrained = mock_hf_model

        with patch("megatron.bridge.training.model_load_save.save_megatron_model") as mock_save_megatron_model:
            bridge.save_megatron_model(mock_megatron_model, "./checkpoint_path")

            mock_save_megatron_model.assert_called_once_with(
                mock_megatron_model, "./checkpoint_path", hf_tokenizer_path=None
            )

    def test_save_megatron_model_with_tokenizer(self):
        """Test save_megatron_model method with tokenizer path."""
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_config = Mock(spec=PretrainedConfig)
        mock_config.architectures = ["LlamaForCausalLM"]
        mock_hf_model.config = mock_config

        mock_megatron_model = [Mock()]

        bridge = AutoBridge.__new__(AutoBridge)
        bridge.hf_pretrained = mock_hf_model

        with patch("megatron.bridge.training.model_load_save.save_megatron_model") as mock_save_megatron_model:
            bridge.save_megatron_model(
                mock_megatron_model, "./checkpoint_path", hf_tokenizer_path="meta-llama/Llama-3-8B"
            )

            mock_save_megatron_model.assert_called_once_with(
                mock_megatron_model, "./checkpoint_path", hf_tokenizer_path="meta-llama/Llama-3-8B"
            )

    def test_save_megatron_model_import_error(self):
        """Test save_megatron_model import error handling."""
        mock_hf_model = Mock(spec=PreTrainedCausalLM)

        bridge = AutoBridge.__new__(AutoBridge)
        bridge.hf_pretrained = mock_hf_model

        # Create a mock that raises ImportError when accessed
        def mock_import(*args, **kwargs):
            if "megatron.bridge.training.model_load_save" in args[0]:
                raise ImportError("No module named 'megatron.bridge.training.model_load_save'")
            return __import__(*args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="megatron.bridge.training is not available"):
                bridge.save_megatron_model([Mock()], "./path")

    def test_load_megatron_model_basic(self):
        """Test load_megatron_model method."""
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_config = Mock(spec=PretrainedConfig)
        mock_config.architectures = ["LlamaForCausalLM"]
        mock_hf_model.config = mock_config

        bridge = AutoBridge.__new__(AutoBridge)
        bridge.hf_pretrained = mock_hf_model

        with patch("megatron.bridge.training.model_load_save.load_megatron_model") as mock_load_megatron_model:
            from pathlib import Path

            with patch.object(Path, "iterdir") as mock_iterdir:
                # Setup mocks
                mock_model = Mock()
                mock_load_megatron_model.return_value = mock_model

                # Mock iterdir to return empty list (no iter_ folders)
                mock_iterdir.return_value = []

                result = bridge.load_megatron_model("./checkpoint_path")

                assert result == [mock_model]
                mock_load_megatron_model.assert_called_once()
                mock_iterdir.assert_called_once()

    def test_load_megatron_model_with_iter_folder(self):
        """Test load_megatron_model with iter_ folders."""
        mock_hf_model = Mock(spec=PreTrainedCausalLM)

        bridge = AutoBridge.__new__(AutoBridge)
        bridge.hf_pretrained = mock_hf_model

        with patch("megatron.bridge.training.model_load_save.load_megatron_model") as mock_load_megatron_model:
            from pathlib import Path

            # Create mock folder objects
            mock_iter_folder_1 = Mock()
            mock_iter_folder_1.is_dir.return_value = True
            mock_iter_folder_1.name = "iter_0000010"

            mock_iter_folder_2 = Mock()
            mock_iter_folder_2.is_dir.return_value = True
            mock_iter_folder_2.name = "iter_0000020"

            # Mock path.iterdir()
            with patch.object(Path, "iterdir") as mock_iterdir:
                # Setup mocks
                mock_model = Mock()
                mock_load_megatron_model.return_value = mock_model

                # Mock iterdir to return the iter folders
                mock_iterdir.return_value = [mock_iter_folder_1, mock_iter_folder_2]

                result = bridge.load_megatron_model("./checkpoint_path")

                assert result == [mock_model]
                mock_load_megatron_model.assert_called_once()
                mock_iterdir.assert_called_once()
                # Should use the latest iteration (iter_0000020)
