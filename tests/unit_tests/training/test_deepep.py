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

from unittest.mock import MagicMock, patch

import pytest
from megatron.core.transformer import TransformerConfig

from megatron.bridge.training.deepep import apply_deepep, validate_deepep


class TestApplyDeepEP:
    """Test the apply_deepep function."""

    @patch("torch.cuda.get_device_properties")
    def test_apply_deepep_sets_configs_for_moe_model_on_ampere(self, mock_get_device_properties):
        """Test that apply_deepep sets DeepEP configs for MoE models on Ampere GPUs."""
        # Mock Ampere GPU (compute capability 8.x)
        mock_properties = MagicMock()
        mock_properties.major = 8
        mock_get_device_properties.return_value = mock_properties

        # Create a mock TransformerConfig with MoE enabled
        config = MagicMock(spec=TransformerConfig)
        config.num_moe_experts = 8  # MoE model

        # Apply DeepEP
        apply_deepep(config)

        # Verify the correct configs were set
        assert config.moe_token_dispatcher_type == "flex"
        assert config.moe_enable_deepep is True
        assert config.moe_shared_expert_overlap is False

    @patch("torch.cuda.get_device_properties")
    def test_apply_deepep_sets_configs_for_moe_model_on_hopper(self, mock_get_device_properties):
        """Test that apply_deepep sets DeepEP configs for MoE models on Hopper GPUs."""
        # Mock Hopper GPU (compute capability 9.x)
        mock_properties = MagicMock()
        mock_properties.major = 9
        mock_get_device_properties.return_value = mock_properties

        # Create a mock TransformerConfig with MoE enabled
        config = MagicMock(spec=TransformerConfig)
        config.num_moe_experts = 8  # MoE model

        # Apply DeepEP
        apply_deepep(config)

        # Verify the correct configs were set
        assert config.moe_token_dispatcher_type == "flex"
        assert config.moe_enable_deepep is True
        assert config.moe_shared_expert_overlap is False

    @patch("torch.cuda.get_device_properties")
    def test_apply_deepep_overrides_existing_configs_for_moe_model(self, mock_get_device_properties):
        """Test that apply_deepep overrides any existing config values for MoE models."""
        # Mock Ampere GPU (compute capability 8.x)
        mock_properties = MagicMock()
        mock_properties.major = 8
        mock_get_device_properties.return_value = mock_properties

        # Create a mock TransformerConfig with different initial values
        config = MagicMock(spec=TransformerConfig)
        config.num_moe_experts = 8  # MoE model
        config.moe_token_dispatcher_type = "legacy"
        config.moe_enable_deepep = False
        config.moe_shared_expert_overlap = True

        # Apply DeepEP
        apply_deepep(config)

        # Verify the configs were overridden
        assert config.moe_token_dispatcher_type == "flex"
        assert config.moe_enable_deepep is True
        assert config.moe_shared_expert_overlap is False

    @patch("megatron.bridge.training.deepep.logger")
    def test_apply_deepep_warns_for_non_moe_model_none_experts(self, mock_logger):
        """Test that apply_deepep logs warning and returns early when num_moe_experts is None."""
        # Create a mock TransformerConfig without MoE
        config = MagicMock(spec=TransformerConfig)
        config.num_moe_experts = None  # Explicitly set to None (default value)

        # Apply DeepEP
        apply_deepep(config)

        # Verify warning was logged
        mock_logger.warning.assert_called_once()
        assert "DeepEP is only applicable to MoE models" in mock_logger.warning.call_args[0][0]

        # Verify configs were NOT set
        assert config.moe_token_dispatcher_type != "flex"
        assert config.moe_enable_deepep != True
        assert config.moe_shared_expert_overlap != False

    @patch("megatron.bridge.training.deepep.logger")
    def test_apply_deepep_warns_for_non_moe_model_zero_experts(self, mock_logger):
        """Test that apply_deepep logs warning and returns early when num_moe_experts is 0."""
        # Create a mock TransformerConfig without MoE
        config = MagicMock(spec=TransformerConfig)
        config.num_moe_experts = 0  # Explicitly set to 0

        # Apply DeepEP
        apply_deepep(config)

        # Verify warning was logged
        mock_logger.warning.assert_called_once()
        assert "DeepEP is only applicable to MoE models" in mock_logger.warning.call_args[0][0]

        # Verify configs were NOT set
        assert config.moe_token_dispatcher_type != "flex"
        assert config.moe_enable_deepep != True
        assert config.moe_shared_expert_overlap != False

    @patch("torch.cuda.get_device_properties")
    @patch("megatron.bridge.training.deepep.logger")
    def test_apply_deepep_warns_for_unsupported_gpu_volta(self, mock_logger, mock_get_device_properties):
        """Test that apply_deepep logs warning and returns early on Volta GPUs."""
        # Mock Volta GPU (compute capability 7.x)
        mock_properties = MagicMock()
        mock_properties.major = 7
        mock_get_device_properties.return_value = mock_properties

        # Create a mock TransformerConfig with MoE enabled
        config = MagicMock(spec=TransformerConfig)
        config.num_moe_experts = 8  # MoE model

        # Apply DeepEP
        apply_deepep(config)

        # Verify warning was logged
        mock_logger.warning.assert_called_once()
        assert (
            "DeepEP is only applicable to Ampere (SM80) and Hopper (SM90) GPUs" in mock_logger.warning.call_args[0][0]
        )

        # Verify configs were NOT set
        assert config.moe_token_dispatcher_type != "flex"
        assert config.moe_enable_deepep != True
        assert config.moe_shared_expert_overlap != False

    @patch("torch.cuda.get_device_properties")
    @patch("megatron.bridge.training.deepep.logger")
    def test_apply_deepep_warns_for_unsupported_gpu_pascal(self, mock_logger, mock_get_device_properties):
        """Test that apply_deepep logs warning and returns early on Pascal GPUs."""
        # Mock Pascal GPU (compute capability 6.x)
        mock_properties = MagicMock()
        mock_properties.major = 6
        mock_get_device_properties.return_value = mock_properties

        # Create a mock TransformerConfig with MoE enabled
        config = MagicMock(spec=TransformerConfig)
        config.num_moe_experts = 8  # MoE model

        # Apply DeepEP
        apply_deepep(config)

        # Verify warning was logged
        mock_logger.warning.assert_called_once()
        assert (
            "DeepEP is only applicable to Ampere (SM80) and Hopper (SM90) GPUs" in mock_logger.warning.call_args[0][0]
        )

        # Verify configs were NOT set
        assert config.moe_token_dispatcher_type != "flex"
        assert config.moe_enable_deepep != True
        assert config.moe_shared_expert_overlap != False


class TestValidateDeepEP:
    """Test the validate_deepep function."""

    @patch("torch.cuda.get_device_properties")
    def test_validate_deepep_ampere_gpu_no_error(self, mock_get_device_properties):
        """Test that validate_deepep passes on Ampere GPUs when DeepEP is enabled."""
        # Mock Ampere GPU (compute capability 8.x)
        mock_properties = MagicMock()
        mock_properties.major = 8
        mock_get_device_properties.return_value = mock_properties

        # Create a mock TransformerConfig with DeepEP enabled
        config = MagicMock(spec=TransformerConfig)
        config.moe_enable_deepep = True

        # Should not raise any exception
        validate_deepep(config)

        # Verify get_device_properties was called
        mock_get_device_properties.assert_called_once_with(0)

    @patch("torch.cuda.get_device_properties")
    def test_validate_deepep_hopper_gpu_no_error(self, mock_get_device_properties):
        """Test that validate_deepep passes on Hopper GPUs when DeepEP is enabled."""
        # Mock Hopper GPU (compute capability 9.x)
        mock_properties = MagicMock()
        mock_properties.major = 9
        mock_get_device_properties.return_value = mock_properties

        # Create a mock TransformerConfig with DeepEP enabled
        config = MagicMock(spec=TransformerConfig)
        config.moe_enable_deepep = True

        # Should not raise any exception
        validate_deepep(config)

        # Verify get_device_properties was called
        mock_get_device_properties.assert_called_once_with(0)

    @patch("torch.cuda.get_device_properties")
    def test_validate_deepep_disabled_no_validation(self, mock_get_device_properties):
        """Test that validate_deepep skips validation when DeepEP is disabled."""
        # Mock unsupported GPU (compute capability 7.x)
        mock_properties = MagicMock()
        mock_properties.major = 7
        mock_get_device_properties.return_value = mock_properties

        # Create a mock TransformerConfig with DeepEP disabled
        config = MagicMock(spec=TransformerConfig)
        config.moe_enable_deepep = False

        # Should not raise any exception even on unsupported hardware
        validate_deepep(config)

        # Since DeepEP is disabled, get_device_properties should not be called
        mock_get_device_properties.assert_not_called()

    @patch("torch.cuda.get_device_properties")
    def test_validate_deepep_volta_gpu_raises_error(self, mock_get_device_properties):
        """Test that validate_deepep raises ValueError on Volta GPUs when DeepEP is enabled."""
        # Mock Volta GPU (compute capability 7.x)
        mock_properties = MagicMock()
        mock_properties.major = 7
        mock_get_device_properties.return_value = mock_properties

        # Create a mock TransformerConfig with DeepEP enabled
        config = MagicMock(spec=TransformerConfig)
        config.moe_enable_deepep = True

        # Should raise ValueError
        with pytest.raises(ValueError, match="DeepEP is supported for Ampere \\(SM80\\) and Hopper \\(SM90\\) GPUs"):
            validate_deepep(config)

        # Verify get_device_properties was called
        mock_get_device_properties.assert_called_once_with(0)

    @patch("torch.cuda.get_device_properties")
    def test_validate_deepep_future_gpu_raises_error(self, mock_get_device_properties):
        """Test that validate_deepep raises ValueError on future unsupported GPUs when DeepEP is enabled."""
        # Mock future GPU
        mock_properties = MagicMock()
        mock_properties.major = 200
        mock_get_device_properties.return_value = mock_properties

        # Create a mock TransformerConfig with DeepEP enabled
        config = MagicMock(spec=TransformerConfig)
        config.moe_enable_deepep = True

        # Should raise ValueError
        with pytest.raises(ValueError, match="DeepEP is supported for Ampere"):
            validate_deepep(config)

        # Verify get_device_properties was called
        mock_get_device_properties.assert_called_once_with(0)
