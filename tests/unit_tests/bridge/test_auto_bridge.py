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
Unit tests for AutoBridge automatic bridge selection.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from megatron.bridge.bridge.auto_bridge import _BRIDGES, AutoBridge
from megatron.bridge.bridge.causal_bridge import CausalLMBridge


class TestAutoBridge:
    """Test cases for AutoBridge automatic selection functionality."""

    @pytest.fixture
    def llama_config(self):
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

    def test_from_hf_pretrained_with_causal_lm_model(self, llama_config):
        """Test AutoBridge correctly selects CausalLMBridge for causal LM models."""
        with patch("megatron.bridge.bridge.auto_bridge.AutoConfig") as mock_auto_config:
            with patch.object(CausalLMBridge, "from_hf_pretrained") as mock_from_hf_pretrained:
                # Setup mocks
                mock_auto_config.from_pretrained.return_value = llama_config
                mock_bridge_instance = Mock(spec=CausalLMBridge)
                mock_from_hf_pretrained.return_value = mock_bridge_instance

                # Call AutoBridge
                result = AutoBridge.from_hf_pretrained("meta-llama/Llama-3-8B", trust_remote_code=True)

                # Verify
                mock_auto_config.from_pretrained.assert_called_once_with(
                    "meta-llama/Llama-3-8B", trust_remote_code=True
                )
                mock_from_hf_pretrained.assert_called_once_with("meta-llama/Llama-3-8B", trust_remote_code=True)
                assert result == mock_bridge_instance

    def test_from_hf_pretrained_with_unsupported_model(self, bert_config):
        """Test AutoBridge raises ValueError for unsupported models."""
        with patch("megatron.bridge.bridge.auto_bridge.AutoConfig") as mock_auto_config:
            # Setup mocks
            mock_auto_config.from_pretrained.return_value = bert_config

            # Should raise ValueError
            with pytest.raises(ValueError) as exc_info:
                AutoBridge.from_hf_pretrained("bert-base-uncased")

            assert "No bridge found for model" in str(exc_info.value)
            assert "bert" in str(exc_info.value).lower()
            assert "BertForMaskedLM" in str(exc_info.value)

    def test_from_pretrained_with_path_object(self, gpt2_config):
        """Test AutoBridge works with Path objects."""
        model_path = Path("/path/to/gpt2/model")

        with patch("megatron.bridge.bridge.auto_bridge.AutoConfig") as mock_auto_config:
            with patch.object(CausalLMBridge, "from_hf_pretrained") as mock_from_hf_pretrained:
                # Setup mocks
                mock_auto_config.from_pretrained.return_value = gpt2_config
                mock_bridge_instance = Mock(spec=CausalLMBridge)
                mock_from_hf_pretrained.return_value = mock_bridge_instance

                # Call with Path
                result = AutoBridge.from_hf_pretrained(model_path, device_map="auto")

                # Verify
                mock_auto_config.from_pretrained.assert_called_once_with(model_path, trust_remote_code=False)
                mock_from_hf_pretrained.assert_called_once_with(model_path, device_map="auto")
                assert result == mock_bridge_instance

    def test_from_pretrained_config_load_failure(self):
        """Test AutoBridge handles config loading failures gracefully."""
        with patch("megatron.bridge.bridge.auto_bridge.AutoConfig") as mock_auto_config:
            # Setup mock to raise exception
            mock_auto_config.from_pretrained.side_effect = Exception("Config not found")

            # Should raise ValueError with helpful message
            with pytest.raises(ValueError) as exc_info:
                AutoBridge.from_hf_pretrained("invalid/path")

            assert "Failed to load configuration" in str(exc_info.value)
            assert "Config not found" in str(exc_info.value)

    def test_from_pretrained_bridge_load_failure(self, gpt2_config):
        """Test AutoBridge raises error when selected bridge fails to load."""

        # Create a mock bridge that supports but fails to load
        class FailingBridge:
            @classmethod
            def supports(cls, config):
                return True

            @classmethod
            def from_pretrained(cls, path, **kwargs):
                raise RuntimeError("Loading failed")

            @classmethod
            def from_hf_pretrained(cls, path, **kwargs):
                raise RuntimeError("Loading failed")

            @classmethod
            def from_hf_config(cls, config):
                raise RuntimeError("Not implemented")

        # Temporarily modify registry
        original_bridges = _BRIDGES.copy()
        _BRIDGES.clear()
        _BRIDGES.extend([FailingBridge, CausalLMBridge])

        try:
            with patch("megatron.bridge.bridge.auto_bridge.AutoConfig") as mock_auto_config:
                # Setup mocks
                mock_auto_config.from_pretrained.return_value = gpt2_config

                # Call - should try FailingBridge and raise its error
                with pytest.raises(ValueError) as exc_info:
                    AutoBridge.from_hf_pretrained("gpt2")

                # Verify the error message includes both the bridge name and original error
                assert "Failed to load model with FailingBridge" in str(exc_info.value)
                assert "Loading failed" in str(exc_info.value)
        finally:
            # Restore original bridges
            _BRIDGES.clear()
            _BRIDGES.extend(original_bridges)

    def test_get_supported_bridges(self):
        """Test get_supported_bridges returns bridge names."""
        bridges = AutoBridge.get_supported_bridges()

        assert isinstance(bridges, list)
        assert "CausalLMBridge" in bridges
        assert len(bridges) == len(_BRIDGES)

    def test_can_handle_supported_model(self, llama_config):
        """Test can_handle returns True for supported models."""
        with patch("megatron.bridge.bridge.auto_bridge.AutoConfig") as mock_auto_config:
            mock_auto_config.from_pretrained.return_value = llama_config

            assert AutoBridge.can_handle("meta-llama/Llama-3-8B") is True
            mock_auto_config.from_pretrained.assert_called_with("meta-llama/Llama-3-8B", trust_remote_code=False)

    def test_can_handle_unsupported_model(self, bert_config):
        """Test can_handle returns False for unsupported models."""
        with patch("megatron.bridge.bridge.auto_bridge.AutoConfig") as mock_auto_config:
            mock_auto_config.from_pretrained.return_value = bert_config

            assert AutoBridge.can_handle("bert-base-uncased") is False

    def test_can_handle_invalid_path(self):
        """Test can_handle returns False for invalid paths."""
        with patch("megatron.bridge.bridge.auto_bridge.AutoConfig") as mock_auto_config:
            mock_auto_config.from_pretrained.side_effect = Exception("Not found")

            assert AutoBridge.can_handle("invalid/path") is False

    def test_causal_lm_bridge_supports_method(self):
        """Test CausalLMBridge.supports correctly identifies causal LM models."""
        # Test supported architectures
        config = Mock()
        config.architectures = ["LlamaForCausalLM"]
        assert CausalLMBridge.supports(config) is True

        config.architectures = ["GPT2ForCausalLM", "GPT2Model"]
        assert CausalLMBridge.supports(config) is True

        config.architectures = ["MistralForCausalLM"]
        assert CausalLMBridge.supports(config) is True

        # Test unsupported architectures
        config.architectures = ["BertForMaskedLM"]
        assert CausalLMBridge.supports(config) is False

        config.architectures = ["T5ForConditionalGeneration"]
        assert CausalLMBridge.supports(config) is False

        # Test edge cases
        config.architectures = []
        assert CausalLMBridge.supports(config) is False

        config = Mock(spec=[])  # No architectures attribute
        assert CausalLMBridge.supports(config) is False

    def test_multiple_architectures_priority(self):
        """Test that first supporting bridge is selected when multiple could work."""
        config = Mock()
        config.architectures = ["GPT2ForCausalLM"]
        config.model_type = "gpt2"

        with patch("megatron.bridge.bridge.auto_bridge.AutoConfig") as mock_auto_config:
            with patch.object(CausalLMBridge, "from_hf_pretrained") as mock_from_hf_pretrained:
                mock_auto_config.from_pretrained.return_value = config
                mock_bridge_instance = Mock(spec=CausalLMBridge)
                mock_from_hf_pretrained.return_value = mock_bridge_instance

                # Since CausalLMBridge is first in _BRIDGES, it should be selected
                result = AutoBridge.from_hf_pretrained("gpt2")

                assert isinstance(result, Mock)  # Our mocked instance
                mock_from_hf_pretrained.assert_called_once()

    def test_kwargs_passed_through(self, gpt2_config):
        """Test that all kwargs are properly passed to the selected bridge."""
        with patch("megatron.bridge.bridge.auto_bridge.AutoConfig") as mock_auto_config:
            with patch.object(CausalLMBridge, "from_hf_pretrained") as mock_from_hf_pretrained:
                mock_auto_config.from_pretrained.return_value = gpt2_config
                mock_bridge_instance = Mock(spec=CausalLMBridge)
                mock_from_hf_pretrained.return_value = mock_bridge_instance

                # Call with various kwargs
                AutoBridge.from_hf_pretrained(
                    "gpt2", trust_remote_code=True, device_map="balanced", torch_dtype="bfloat16", custom_param="test"
                )

                # Verify all kwargs were passed
                mock_from_hf_pretrained.assert_called_once_with(
                    "gpt2", trust_remote_code=True, device_map="balanced", torch_dtype="bfloat16", custom_param="test"
                )
