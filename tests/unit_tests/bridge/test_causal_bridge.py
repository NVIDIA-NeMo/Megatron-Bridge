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

from unittest.mock import Mock, patch

import pytest
import torch
from transformers import LlamaConfig
from transformers.configuration_utils import PretrainedConfig

from megatron.bridge.bridge.causal_bridge import CausalLMBridge
from megatron.bridge.bridge.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.gpt_provider import GPTModelProvider


class TestCausalLMBridge:
    """Test cases for CausalLMBridge class."""

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

    def test_from_hf_pretrained_with_model_id(self):
        """Test from_hf_pretrained with model ID string."""
        # This test checks that from_hf_pretrained creates correct bridge instance
        # We'll use a mock pretrained model
        mock_model = Mock(spec=PreTrainedCausalLM)
        mock_config = Mock(spec=PretrainedConfig)
        mock_config.architectures = ["GPT2LMHeadModel"]  # Use a real architecture
        mock_model.config = mock_config

        with patch("megatron.bridge.bridge.causal_bridge.PreTrainedCausalLM.from_pretrained") as mock_from_pretrained:
            # Set up the from_pretrained class method properly
            mock_from_pretrained.return_value = mock_model

            with patch("megatron.bridge.bridge.causal_bridge.AutoConfig") as mock_autoconfig:
                mock_autoconfig.from_pretrained.return_value = mock_config

                # Skip architecture validation for this test
                with patch.object(CausalLMBridge, "_validate_config"):
                    # Call from_hf_pretrained
                    model_id = "gpt2"
                    result = CausalLMBridge.from_hf_pretrained(model_id, trust_remote_code=True)

                # Assertions
                assert isinstance(result, CausalLMBridge)
                assert result.hf_pretrained == mock_model
                mock_from_pretrained.assert_called_once_with(model_id, trust_remote_code=True)

    def test_from_pretrained_with_path(self):
        """Test from_pretrained with Path object."""
        from pathlib import Path

        # Setup mocks
        mock_model = Mock(spec=PreTrainedCausalLM)
        mock_config = Mock(spec=PretrainedConfig)
        mock_config.architectures = ["GPT2LMHeadModel"]
        mock_model.config = mock_config

        with patch("megatron.bridge.bridge.causal_bridge.PreTrainedCausalLM.from_pretrained") as mock_from_pretrained:
            # Set up the from_pretrained class method properly
            mock_from_pretrained.return_value = mock_model

            with patch("megatron.bridge.bridge.causal_bridge.AutoConfig") as mock_autoconfig:
                mock_autoconfig.from_pretrained.return_value = mock_config

                # Skip architecture validation for this test
                with patch.object(CausalLMBridge, "_validate_config"):
                    # Call from_hf_pretrained with Path
                    model_path = Path("/path/to/model")
                    result = CausalLMBridge.from_hf_pretrained(model_path, device_map="auto")

                # Assertions
                assert isinstance(result, CausalLMBridge)
                assert result.hf_pretrained == mock_model
                mock_from_pretrained.assert_called_once_with(model_path, device_map="auto")

    def test_from_pretrained_with_additional_kwargs(self):
        """Test from_pretrained with various kwargs."""
        # Setup mocks
        mock_model = Mock(spec=PreTrainedCausalLM)
        mock_config = Mock(spec=PretrainedConfig)
        mock_config.architectures = ["GPT2LMHeadModel"]
        mock_model.config = mock_config

        with patch("megatron.bridge.bridge.causal_bridge.PreTrainedCausalLM.from_pretrained") as mock_from_pretrained:
            # Set up the from_pretrained class method properly
            mock_from_pretrained.return_value = mock_model

            with patch("megatron.bridge.bridge.causal_bridge.AutoConfig") as mock_autoconfig:
                mock_autoconfig.from_pretrained.return_value = mock_config

                # Skip architecture validation for this test
                with patch.object(CausalLMBridge, "_validate_config"):
                    # Call with multiple kwargs
                    result = CausalLMBridge.from_hf_pretrained(
                        "model-id",
                        torch_dtype=torch.bfloat16,
                        low_cpu_mem_usage=True,
                        attn_implementation="flash_attention_2",
                    )

                # Assertions
                assert isinstance(result, CausalLMBridge)
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

        with patch.object(CausalLMBridge, "_model_bridge", mock_model_bridge):
            # Create bridge and convert
            bridge = CausalLMBridge(mock_hf_model)
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

        with patch.object(CausalLMBridge, "_model_bridge", mock_model_bridge):
            bridge = CausalLMBridge(mock_gpt2_model)
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

        with patch.object(CausalLMBridge, "_model_bridge", mock_model_bridge):
            # Create bridge and convert with load_weights=True
            bridge = CausalLMBridge(mock_hf_model)
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

        with patch.object(CausalLMBridge, "_model_bridge", mock_model_bridge):
            bridge = CausalLMBridge(mock_hf_model)

            # Should propagate the exception
            with pytest.raises(ValueError, match="Unsupported model type"):
                bridge.to_megatron_provider()

    def test_bridge_instance_creation(self):
        """Test CausalLMBridge instance creation."""
        # Test with PreTrainedCausalLM
        mock_model = Mock(spec=PreTrainedCausalLM)

        bridge = CausalLMBridge(mock_model)

        # Should have the expected methods
        assert hasattr(bridge, "from_hf_pretrained")
        assert hasattr(bridge, "to_megatron_provider")
        assert hasattr(bridge, "load_hf_weights")
        assert hasattr(bridge, "export_hf_weights")
        assert hasattr(bridge, "save_hf_pretrained")
        assert bridge.hf_pretrained == mock_model

        # Test with PretrainedConfig
        mock_config = Mock(spec=PretrainedConfig)
        bridge_config = CausalLMBridge(mock_config)
        assert bridge_config.hf_pretrained == mock_config

        # Test with invalid type
        with pytest.raises(
            ValueError, match="hf_pretrained must be a PreTrainedCausalLM or PretrainedConfig instance"
        ):
            CausalLMBridge("invalid")

        # from_hf_pretrained should be a classmethod
        import inspect

        assert inspect.ismethod(CausalLMBridge.from_hf_pretrained)


class TestCausalLMBridgeEdgeCases:
    """Test edge cases and additional functionality for CausalLMBridge."""

    def test_from_hf_config(self):
        """Test creating bridge from config only."""
        # Create a mock config
        config = Mock(spec=PretrainedConfig)
        config.architectures = ["GPT2LMHeadModel"]

        # Skip architecture validation for this test
        with patch.object(CausalLMBridge, "_validate_config"):
            bridge = CausalLMBridge.from_hf_config(config)
            assert isinstance(bridge, CausalLMBridge)
            assert bridge.hf_pretrained == config

    def test_from_hf_config_invalid_architecture(self):
        """Test from_hf_config with unsupported architecture."""
        config = Mock(spec=PretrainedConfig)
        config.architectures = ["BertForMaskedLM"]  # Not a CausalLM

        with pytest.raises(ValueError, match="Model architecture not supported by CausalLMBridge"):
            CausalLMBridge.from_hf_config(config)

    def test_supports_method(self):
        """Test the supports class method."""
        # Supported config
        config = Mock()
        config.architectures = ["LlamaForCausalLM"]
        assert CausalLMBridge.supports(config) is True

        # Multiple architectures, one supported
        config.architectures = ["LlamaModel", "LlamaForCausalLM"]
        assert CausalLMBridge.supports(config) is True

        # No CausalLM architecture
        config.architectures = ["BertForMaskedLM"]
        assert CausalLMBridge.supports(config) is False

        # No architectures
        config.architectures = []
        assert CausalLMBridge.supports(config) is False

        # Missing architectures attribute
        config_no_arch = Mock(spec=[])
        assert CausalLMBridge.supports(config_no_arch) is False

    def test_list_supported_models(self):
        """Test listing supported models."""
        # Since this method looks at internal dispatch registry,
        # we'll just test that it returns a list
        with patch("megatron.bridge.bridge.causal_bridge.model_bridge") as mock_bridge:
            # Mock to avoid AttributeError
            mock_bridge.to_megatron = None
            supported = CausalLMBridge.list_supported_models()
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

        with patch.object(CausalLMBridge, "_model_bridge", mock_model_bridge):
            bridge = CausalLMBridge(mock_hf_model)
            result = bridge.load_hf_weights(mock_megatron_model)

            assert result == mock_megatron_model
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
        with patch.object(CausalLMBridge, "_model_bridge", mock_model_bridge):
            bridge = CausalLMBridge(mock_hf_model)

            # Now patch the from_pretrained method
            with patch(
                "megatron.bridge.bridge.causal_bridge.PreTrainedCausalLM.from_pretrained"
            ) as mock_from_pretrained:
                mock_loaded_model = Mock(spec=PreTrainedCausalLM)
                mock_from_pretrained.return_value = mock_loaded_model

                result = bridge.load_hf_weights(mock_megatron_model, "./custom_model")

                assert result == mock_megatron_model
                mock_from_pretrained.assert_called_once_with("./custom_model")
                mock_model_bridge.load_weights_hf_to_megatron.assert_called_once_with(
                    mock_megatron_model, mock_loaded_model
                )

    def test_load_hf_weights_no_path_config_only(self):
        """Test load_hf_weights fails when bridge has config only and no path provided."""
        mock_config = Mock(spec=PretrainedConfig)
        bridge = CausalLMBridge(mock_config)

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

        from megatron.bridge.common.state import SafeTensorsStateSource

        mock_hf_model.state.source = Mock(spec=SafeTensorsStateSource)
        mock_hf_model.state.source.save_generator = Mock()

        mock_megatron_model = [Mock()]

        with patch.object(CausalLMBridge, "save_hf_weights") as mock_save_hf_weights:
            bridge = CausalLMBridge(mock_hf_model)
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

        with patch.object(CausalLMBridge, "save_hf_weights") as mock_save_hf_weights:
            bridge = CausalLMBridge(mock_hf_model)
            bridge.save_hf_pretrained(mock_megatron_model, "./output_dir")

            # Artifacts should NOT be saved on non-zero rank
            mock_hf_model.save_artifacts.assert_not_called()
            mock_save_hf_weights.assert_called_once_with(mock_megatron_model, "./output_dir", True)

    def test_export_hf_weights(self):
        """Test exporting weights from Megatron to HF format."""
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_hf_model.config = Mock()
        mock_hf_model.config.architectures = ["LlamaForCausalLM"]

        mock_megatron_model = [Mock()]
        mock_megatron_model[0].module = None  # No nested module

        # Mock the export process
        with patch(
            "megatron.bridge.bridge.causal_bridge.model_bridge.stream_weights_megatron_to_hf"
        ) as mock_bridge_state:
            mock_weight_iter = [("weight1", torch.randn(10, 10)), ("weight2", torch.randn(5, 5))]
            mock_bridge_state.return_value = iter(mock_weight_iter)

            with patch("megatron.bridge.bridge.causal_bridge.transformers") as mock_transformers:
                mock_arch_class = Mock()
                mock_transformers.LlamaForCausalLM = mock_arch_class

                bridge = CausalLMBridge(mock_hf_model)

                # Mock _get_causal_lm_architecture to avoid accessing transformers
                with patch.object(bridge, "_get_causal_lm_architecture", return_value=mock_arch_class):
                    weights = list(bridge.export_hf_weights(mock_megatron_model, order="safetensors", cpu=True))

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

        with patch("megatron.bridge.bridge.causal_bridge.transformers") as mock_transformers:
            mock_arch_class = Mock()
            mock_transformers.LlamaForCausalLM = mock_arch_class

            bridge = CausalLMBridge(mock_hf_model)
            arch = bridge._get_causal_lm_architecture()
            assert arch == mock_arch_class

    def test_get_causal_lm_architecture_no_architectures(self):
        """Test error when no architectures found."""
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_hf_model.config = Mock()
        mock_hf_model.config.architectures = []

        bridge = CausalLMBridge(mock_hf_model)
        with pytest.raises(ValueError, match="No architectures found in model config"):
            bridge._get_causal_lm_architecture()

    def test_get_causal_lm_architecture_no_causal_lm(self):
        """Test error when no CausalLM architecture found."""
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_hf_model.config = Mock()
        mock_hf_model.config.architectures = ["BertForMaskedLM"]

        bridge = CausalLMBridge(mock_hf_model)
        with pytest.raises(ValueError, match="No CausalLM architecture found"):
            bridge._get_causal_lm_architecture()

    def test_get_causal_lm_architecture_not_in_transformers(self):
        """Test error when architecture class not found in transformers."""
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_hf_model.config = Mock()
        mock_hf_model.config.architectures = ["CustomForCausalLM"]

        bridge = CausalLMBridge(mock_hf_model)

        # Mock transformers to not have the CustomForCausalLM attribute
        with patch("megatron.bridge.bridge.causal_bridge.transformers") as mock_transformers:
            # Configure mock to raise AttributeError when accessing CustomForCausalLM
            del mock_transformers.CustomForCausalLM

            with pytest.raises(ValueError, match="Architecture class 'CustomForCausalLM' not found in transformers"):
                bridge._get_causal_lm_architecture()

    def test_repr(self):
        """Test string representation of CausalLMBridge."""
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_hf_model.__repr__ = Mock(return_value="PreTrainedCausalLM(\n  config=...\n)")

        with patch("megatron.bridge.bridge.causal_bridge.model_bridge") as mock_bridge:
            mock_bridge.to_megatron = Mock()
            mock_bridge.to_megatron.__repr__ = Mock(return_value="Dispatcher(\n  impls=...\n)")

            bridge = CausalLMBridge(mock_hf_model)
            repr_str = repr(bridge)

            assert "CausalLMBridge(" in repr_str
            assert "(hf_pretrained):" in repr_str
            assert "(to_megatron):" in repr_str
            assert "PreTrainedCausalLM" in repr_str
            assert "Dispatcher" in repr_str

    @patch("torch.cuda.current_device", return_value=0)
    @patch("torch.cuda.is_available", return_value=False)
    def test_cpu_compatibility(self, mock_cuda_avail, mock_cuda_device):
        """Test that bridge works on CPU-only systems."""
        # This test ensures the bridge doesn't require CUDA
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_hf_model.config = Mock()
        mock_hf_model.config.architectures = ["GPT2ForCausalLM"]

        # Create bridge - should work without CUDA
        bridge = CausalLMBridge(mock_hf_model)
        assert bridge.hf_pretrained == mock_hf_model

        # Test methods that might use device
        with patch("megatron.bridge.bridge.causal_bridge.transformers") as mock_transformers:
            mock_transformers.GPT2ForCausalLM = Mock()

            # These operations should work on CPU
            arch = bridge._get_causal_lm_architecture()
            assert arch is not None
