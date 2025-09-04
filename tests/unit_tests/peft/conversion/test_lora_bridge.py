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
Unit tests for LoRABridge PEFT adapter bridge implementation.
"""

from unittest.mock import Mock

import pytest
import torch
from peft import LoraConfig

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.peft.conversion.peft_bridge import MegatronPEFTBridge
from megatron.bridge.peft.conversion.pretrained_adapters import PreTrainedAdapters
from megatron.bridge.peft.lora.lora import LoRA
from megatron.bridge.peft.lora.lora_bridge import LoRABridge


class TestLoRABridge:
    """Test cases for LoRABridge class."""

    @pytest.fixture
    def lora_config_dict(self):
        """Create a sample LoRA configuration."""
        return {
            "peft_type": "LORA",
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }

    @pytest.fixture
    def mock_pretrained_adapters(self, lora_config_dict):
        """Create a mock PreTrainedAdapters instance."""
        mock_adapters = Mock(spec=PreTrainedAdapters)
        mock_adapters.config = LoraConfig.from_dict(lora_config_dict)
        return mock_adapters

    @pytest.fixture
    def mock_base_bridge(self):
        """Create a mock base AutoBridge."""
        return Mock(spec=AutoBridge)

    def test_bridge_registration(self):
        """Test that LoRABridge is properly registered."""
        assert issubclass(LoRABridge, MegatronPEFTBridge)

    def test_peft_bridge_basic(self, mock_pretrained_adapters):
        """Test basic peft_bridge functionality."""
        bridge = LoRABridge()

        result = bridge.peft_bridge(mock_pretrained_adapters)

        # Check that it returns a LoRA instance
        assert isinstance(result, LoRA)

        # Check configuration mapping
        config = mock_pretrained_adapters.config
        assert result.dim == config.r
        assert result.alpha == config.lora_alpha
        assert result.dropout == config.lora_dropout

    def test_peft_bridge_target_module_conversion(self, mock_pretrained_adapters):
        """Test HF target module conversion to Megatron modules."""
        bridge = LoRABridge()

        result = bridge.peft_bridge(mock_pretrained_adapters)

        # The target modules should be converted from HF to Megatron names
        expected_megatron_modules = {"linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"}
        assert set(result.target_modules).issubset(expected_megatron_modules)

    def test_peft_bridge_dtype_parsing(self):
        """Test dtype parsing functionality."""
        bridge = LoRABridge()

        # Test various dtype strings
        assert bridge._parse_dtype("float16") == torch.float16
        assert bridge._parse_dtype("bfloat16") == torch.bfloat16
        assert bridge._parse_dtype("float32") == torch.float32
        assert bridge._parse_dtype("int8") == torch.int8
        assert bridge._parse_dtype(None) is None
        assert bridge._parse_dtype("unknown") is None

    def test_hf_to_megatron_target_modules(self):
        """Test HF to Megatron target module conversion."""
        bridge = LoRABridge()

        # Test individual projections -> fused modules
        hf_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        result = bridge._hf_to_megatron_target_modules(hf_targets)

        expected = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
        assert set(result) == set(expected)

        # Test partial targets
        partial_targets = ["q_proj", "o_proj", "down_proj"]
        result = bridge._hf_to_megatron_target_modules(partial_targets)
        expected_partial = ["linear_qkv", "linear_proj", "linear_fc2"]
        assert set(result) == set(expected_partial)

        # Test unknown targets (should pass through)
        unknown_targets = ["unknown_proj"]
        result = bridge._hf_to_megatron_target_modules(unknown_targets)
        assert result == ["unknown_proj"]

    def test_mapping_registry_implementation(self):
        """Test that mapping_registry returns proper mappings."""
        bridge = LoRABridge()

        registry = bridge.mapping_registry()

        assert isinstance(registry, MegatronMappingRegistry)

        # Check that it has mappings
        mappings = registry.get_all_mappings()
        assert len(mappings) > 0

        # Check for expected mapping types
        mapping_types = [type(mapping).__name__ for mapping in mappings]
        assert "AdapterAutoMapping" in mapping_types
        assert "AdapterQKVMapping" in mapping_types
        assert "AdapterGatedMLPMapping" in mapping_types

    def test_mapping_registry_parameter_patterns(self):
        """Test that mapping registry has correct parameter patterns."""
        bridge = LoRABridge()
        registry = bridge.mapping_registry()

        # Get all megatron parameter patterns
        megatron_params = []
        for mapping in registry.get_all_mappings():
            megatron_params.append(mapping.megatron_param)

        # Should include adapter parameters
        adapter_params = [p for p in megatron_params if ".adapter." in p]
        assert len(adapter_params) > 0

        # Should include QKV, projection, and MLP parameters
        qkv_params = [p for p in adapter_params if "linear_qkv.adapter" in p]
        proj_params = [p for p in adapter_params if "linear_proj.adapter" in p]
        mlp_params = [p for p in adapter_params if "linear_fc" in p and ".adapter" in p]

        assert len(qkv_params) > 0
        assert len(proj_params) > 0
        assert len(mlp_params) > 0


class TestLoRABridgeEdgeCases:
    """Test edge cases and error conditions for LoRABridge."""

    def test_peft_bridge_missing_config_fields(self):
        """Test peft_bridge with missing configuration fields."""
        bridge = LoRABridge()

        # Create config missing required fields
        incomplete_config_dict = {"peft_type": "LORA"}  # Missing 'r', 'lora_alpha'

        mock_adapters = Mock(spec=PreTrainedAdapters)

        with pytest.raises(KeyError):
            # Should fail when trying to access missing keys
            mock_adapters.config = LoraConfig.from_dict(incomplete_config_dict)
            bridge.peft_bridge(mock_adapters)

    def test_target_module_conversion_edge_cases(self):
        """Test target module conversion with edge cases."""
        bridge = LoRABridge()

        # Test empty list
        assert bridge._hf_to_megatron_target_modules([]) == []

        # Test already Megatron modules (should pass through)
        megatron_modules = ["linear_qkv", "linear_fc1"]
        result = bridge._hf_to_megatron_target_modules(megatron_modules)
        assert set(result) == set(megatron_modules)

        # Test mixed known and unknown
        mixed_modules = ["q_proj", "unknown_module"]
        result = bridge._hf_to_megatron_target_modules(mixed_modules)
        assert "linear_qkv" in result
        assert "unknown_module" in result

    def test_dtype_parsing_edge_cases(self):
        """Test dtype parsing with edge cases."""
        bridge = LoRABridge()

        # Test case variations
        assert bridge._parse_dtype("FLOAT16") == torch.float16
        assert bridge._parse_dtype("Float16") == torch.float16

        # Test empty string
        assert bridge._parse_dtype("") is None

        # Test whitespace
        assert bridge._parse_dtype("  float16  ") is None  # No strip implemented


class TestLoRABridgeIntegration:
    """Integration tests for LoRABridge with real configurations."""

    @pytest.fixture
    def real_lora_configs(self):
        """Real LoRA configurations from popular models."""
        return {
            "alpaca-lora": {
                "peft_type": "LORA",
                "r": 16,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj"],
                "bias": "none",
                "task_type": "CAUSAL_LM",
            },
            "full-lora": {
                "peft_type": "LORA",
                "r": 8,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "bias": "none",
                "task_type": "CAUSAL_LM",
            },
        }

    def test_peft_bridge_real_configs(self, real_lora_configs):
        """Test peft_bridge with real LoRA configurations."""
        bridge = LoRABridge()

        for config_name, config_dict in real_lora_configs.items():
            mock_adapters = Mock(spec=PreTrainedAdapters)
            mock_adapters.config = LoraConfig.from_dict(config_dict)

            result = bridge.peft_bridge(mock_adapters)

            # Verify LoRA configuration
            assert isinstance(result, LoRA)
            assert result.dim == config_dict["r"]
            assert result.alpha == config_dict["lora_alpha"]
            assert result.dropout == config_dict["lora_dropout"]

            # Verify target modules were converted
            assert len(result.target_modules) > 0
