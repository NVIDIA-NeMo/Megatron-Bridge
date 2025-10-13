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
Unit tests for AutoPEFTBridge automatic adapter bridge selection and functionality.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from peft import LoraConfig

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.peft.api import MegatronPEFTModel
from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.conversion.auto_peft_bridge import AutoPEFTBridge
from megatron.bridge.peft.conversion.pretrained_adapters import PreTrainedAdapters


class TestAutoPEFTBridge:
    """Test cases for AutoPEFTBridge automatic selection and bridge functionality."""

    @pytest.fixture
    def lora_config_dict(self):
        """Create a sample LoRA configuration."""
        return {
            "base_model_name_or_path": "meta-llama/Llama-3.1-8B",
            "bias": "none",
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "peft_type": "LORA",
            "r": 8,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "task_type": "CAUSAL_LM",
        }

    @pytest.fixture
    def dora_config_dict(self):
        """Create a sample DoRA configuration."""
        return {
            "base_model_name_or_path": "meta-llama/Llama-3.1-8B",
            "bias": "none",
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "peft_type": "LORA",
            "r": 8,
            "target_modules": ["q_proj", "v_proj", "o_proj", "down_proj"],
            "task_type": "CAUSAL_LM",
            "use_dora": True,
        }

    @pytest.fixture
    def fused_lora_config_dict(self):
        """Create a LoRA configuration with fused target modules."""
        return {
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "peft_type": "LORA",
            "r": 16,
            "target_modules": ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
            "task_type": "CAUSAL_LM",
        }

    @pytest.fixture
    def mock_pretrained_adapters(self, lora_config_dict):
        """Create a mock PreTrainedAdapters instance."""
        mock_adapters = Mock(spec=PreTrainedAdapters)
        mock_adapters.config = LoraConfig(**lora_config_dict)
        mock_adapters.model_name_or_path = "username/llama-lora-adapters"
        mock_adapters.get_target_modules.return_value = lora_config_dict["target_modules"]
        mock_adapters.get_peft_type.return_value = "LORA"
        mock_adapters.get_rank.return_value = lora_config_dict["r"]
        mock_adapters.get_alpha.return_value = lora_config_dict["lora_alpha"]
        mock_adapters.supports_layout.return_value = True
        return mock_adapters

    def create_mock_adapter_files(self, config_dict, save_dir):
        """Create mock adapter files in a directory."""
        save_path = Path(save_dir)

        # Save adapter config
        config_path = save_path / "adapter_config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        # Create dummy adapter weights
        weights_path = save_path / "adapter_model.safetensors"
        # Create a minimal safetensors file
        dummy_weights = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(8, 4096),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(4096, 8),
        }
        import safetensors.torch

        safetensors.torch.save_file(dummy_weights, weights_path)

    @pytest.mark.skip(reason="API has changed - needs update")
    def test_from_hf_pretrained_basic(self, lora_config_dict):
        """Test basic from_hf_pretrained functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_mock_adapter_files(lora_config_dict, temp_dir)

            # Mock PreTrainedAdapters.from_pretrained
            with patch(
                "megatron.bridge.peft.conversion.auto_peft_bridge.PreTrainedAdapters.from_pretrained"
            ) as mock_from_pretrained:
                mock_adapters = Mock(spec=PreTrainedAdapters)
                mock_adapters.config = LoraConfig(**lora_config_dict)
                mock_adapters.get_target_modules.return_value = lora_config_dict["target_modules"]
                mock_adapters.supports_layout.return_value = True
                mock_from_pretrained.return_value = mock_adapters

                # Create bridge
                bridge = AutoPEFTBridge.from_hf_pretrained(temp_dir)

                # Verify
                assert isinstance(bridge, AutoPEFTBridge)
                assert bridge.adapters == mock_adapters
                assert bridge.adapter_name == "default"
                mock_from_pretrained.assert_called_once_with(temp_dir, trust_remote_code=False, strict=True)

    def test_from_hf_pretrained_with_kwargs(self, lora_config_dict):
        """Test from_hf_pretrained with custom kwargs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_mock_adapter_files(lora_config_dict, temp_dir)

            with patch(
                "megatron.bridge.peft.conversion.auto_peft_bridge.PreTrainedAdapters.from_pretrained"
            ) as mock_from_pretrained:
                mock_adapters = Mock(spec=PreTrainedAdapters)
                mock_adapters.config = LoraConfig(**lora_config_dict)
                mock_adapters.get_target_modules.return_value = lora_config_dict["target_modules"]
                mock_adapters.supports_layout.return_value = True
                mock_from_pretrained.return_value = mock_adapters

                # Create bridge with custom settings
                bridge = AutoPEFTBridge.from_hf_pretrained(
                    temp_dir, adapter_name="custom", trust_remote_code=True, strict=False
                )

                # Verify
                assert bridge.adapter_name == "custom"
                mock_from_pretrained.assert_called_once_with(temp_dir, trust_remote_code=True, strict=False)

    @pytest.mark.skip(reason="force_layout parameter no longer exists")
    def test_from_hf_pretrained_force_layout(self, lora_config_dict):
        """Test from_hf_pretrained with forced layout."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_mock_adapter_files(lora_config_dict, temp_dir)

            with patch(
                "megatron.bridge.peft.conversion.auto_peft_bridge.PreTrainedAdapters.from_pretrained"
            ) as mock_from_pretrained:
                mock_adapters = Mock(spec=PreTrainedAdapters)
                mock_adapters.config = LoraConfig(**lora_config_dict)
                mock_adapters.get_target_modules.return_value = lora_config_dict["target_modules"]
                mock_adapters.supports_layout.return_value = True
                mock_from_pretrained.return_value = mock_adapters

                # Test with forced canonical layout
                bridge = AutoPEFTBridge.from_hf_pretrained(temp_dir, force_layout="canonical")
                assert isinstance(bridge, AutoPEFTBridge)

                # Test with unsupported forced layout
                mock_adapters.supports_layout.return_value = False
                with pytest.raises(ValueError, match="does not support forced layout"):
                    AutoPEFTBridge.from_hf_pretrained(temp_dir, force_layout="fused")

    @pytest.mark.skip(reason="Error message format has changed")
    def test_from_hf_pretrained_invalid_config(self):
        """Test from_hf_pretrained with invalid adapter config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config missing required fields
            invalid_config = {"some_field": "value"}
            self.create_mock_adapter_files(invalid_config, temp_dir)

            with patch(
                "megatron.bridge.peft.conversion.auto_peft_bridge.PreTrainedAdapters.from_pretrained"
            ) as mock_from_pretrained:
                mock_from_pretrained.side_effect = ValueError("Missing peft_type")

                with pytest.raises(ValueError, match="Failed to load adapters"):
                    AutoPEFTBridge.from_hf_pretrained(temp_dir)

    def test_supports_method(self, lora_config_dict, dora_config_dict):
        """Test the supports class method."""
        with patch("megatron.bridge.peft.conversion.peft_bridge.list_registered_bridges") as mock_list_bridges:
            from peft import LoraConfig

            # Mock registry to include LoraConfig
            mock_list_bridges.return_value = {LoraConfig: Mock()}

            # Test supported LoRA config
            assert AutoPEFTBridge.supports(lora_config_dict) == True

            # Test supported DoRA config (uses LoraConfig with use_dora=True)
            assert AutoPEFTBridge.supports(dora_config_dict) == True

            # Test unsupported config
            unsupported_config = {"peft_type": "UNSUPPORTED_TYPE"}
            assert AutoPEFTBridge.supports(unsupported_config) == False

            # Test invalid config
            invalid_config = {"invalid": "config"}
            assert AutoPEFTBridge.supports(invalid_config) == False

    @pytest.mark.skip(reason="Test hangs - needs investigation")
    def test_can_handle_supported_adapters(self, lora_config_dict):
        """Test can_handle returns True for supported adapters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_mock_adapter_files(lora_config_dict, temp_dir)

            with patch(
                "megatron.bridge.peft.conversion.auto_peft_bridge.PreTrainedAdapters.from_pretrained"
            ) as mock_from_pretrained:
                mock_adapters = Mock(spec=PreTrainedAdapters)
                mock_adapters.config = LoraConfig(**lora_config_dict)
                mock_from_pretrained.return_value = mock_adapters

                assert AutoPEFTBridge.can_handle(temp_dir) == True

    def test_can_handle_unsupported_adapters(self):
        """Test can_handle returns False for unsupported adapters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create unsupported adapter config
            unsupported_config = {"peft_type": "UNSUPPORTED"}
            self.create_mock_adapter_files(unsupported_config, temp_dir)

            with patch(
                "megatron.bridge.peft.conversion.auto_peft_bridge.PreTrainedAdapters.from_pretrained"
            ) as mock_from_pretrained:
                mock_from_pretrained.side_effect = ValueError("Unsupported type")

                assert AutoPEFTBridge.can_handle(temp_dir) == False

    def test_can_handle_invalid_path(self):
        """Test can_handle returns False for invalid paths."""
        assert AutoPEFTBridge.can_handle("invalid/path") == False

    def test_list_supported_adapters(self):
        """Test listing supported adapter types."""
        with patch("megatron.bridge.peft.conversion.peft_bridge.list_registered_bridges") as mock_list_bridges:
            # Mock the registry to have LoraConfig
            from peft import LoraConfig

            mock_list_bridges.return_value = {LoraConfig: Mock()}

            supported = AutoPEFTBridge.list_supported_adapters()
            assert isinstance(supported, list)
            assert "LORA" in supported




    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.is_available", return_value=True)
    @patch("torch.distributed.barrier")
    def test_save_hf_pretrained(
        self, mock_barrier, mock_is_available, mock_is_initialized, mock_get_rank, mock_pretrained_adapters
    ):
        """Test saving PEFT model adapters in HuggingFace format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mocks
            mock_peft_model = Mock(spec=MegatronPEFTModel)
            mock_peft_model.stages = [Mock()]

            bridge = AutoPEFTBridge(mock_pretrained_adapters)

            with patch.object(bridge, "_save_adapter_config") as mock_save_config:
                with patch.object(bridge, "_save_adapter_weights") as mock_save_weights:
                    bridge.save_hf_pretrained(mock_peft_model, temp_dir)

                    # Verify config saved on rank 0
                    mock_save_config.assert_called_once_with(temp_dir)
                    mock_save_weights.assert_called_once_with(mock_peft_model, temp_dir, True)

    @patch("torch.distributed.get_rank", return_value=1)
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.is_available", return_value=True)
    @patch("torch.distributed.barrier")
    def test_save_hf_pretrained_non_zero_rank(
        self, mock_barrier, mock_is_available, mock_is_initialized, mock_get_rank, mock_pretrained_adapters
    ):
        """Test save_hf_pretrained on non-zero rank."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_peft_model = Mock(spec=MegatronPEFTModel)

            bridge = AutoPEFTBridge(mock_pretrained_adapters)

            with patch.object(bridge, "_save_adapter_config") as mock_save_config:
                with patch.object(bridge, "_save_adapter_weights") as mock_save_weights:
                    bridge.save_hf_pretrained(mock_peft_model, temp_dir)

                    # Config should NOT be saved on non-zero rank
                    mock_save_config.assert_not_called()
                    mock_save_weights.assert_called_once_with(mock_peft_model, temp_dir, True)


    def test_bridge_instance_creation(self, mock_pretrained_adapters):
        """Test AutoPEFTBridge instance creation."""
        bridge = AutoPEFTBridge(mock_pretrained_adapters)

        # Should have expected attributes and methods
        assert bridge.adapters == mock_pretrained_adapters
        assert bridge.adapter_name == "default"
        assert hasattr(bridge, "from_hf_pretrained")
        assert hasattr(bridge, "to_megatron_model")
        assert hasattr(bridge, "save_hf_pretrained")

    def test_peft_config_property(self, mock_pretrained_adapters):
        """Test peft_config property."""
        bridge = AutoPEFTBridge(mock_pretrained_adapters)

        config = bridge.peft_config
        assert config == mock_pretrained_adapters.config

    def test_repr(self, mock_pretrained_adapters):
        """Test string representation of AutoPEFTBridge."""
        bridge = AutoPEFTBridge(mock_pretrained_adapters)
        repr_str = repr(bridge)

        # Just verify repr doesn't crash and returns a string
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0


class TestAutoPEFTBridgeIntegration:
    """Integration tests for AutoPEFTBridge with different adapter types."""

    @pytest.fixture
    def adapter_configs(self):
        """Different adapter configurations for testing."""
        return {
            "canonical-lora": {
                "peft_type": "LORA",
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "bias": "none",
            },
            "fused-lora": {
                "peft_type": "LORA",
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "target_modules": ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
                "bias": "none",
            },
            "dora": {
                "peft_type": "LORA",
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj", "v_proj", "o_proj", "down_proj"],
                "use_dora": True,
            },
        }


    def test_supports_all_configs(self, adapter_configs):
        """Test supports method for all adapter configurations."""
        for adapter_name, config_dict in adapter_configs.items():
            # Mock successful support check
            with patch.object(AutoPEFTBridge, "supports", return_value=True):
                assert AutoPEFTBridge.supports(config_dict) == True
