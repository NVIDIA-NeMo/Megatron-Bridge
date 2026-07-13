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

import argparse
from unittest.mock import MagicMock, call, patch

import pytest
from megatron.core.enums import ModelType
from megatron.core.transformer import TransformerConfig

from megatron.bridge.training.mlm_compat.model import _get_model


def create_mock_args() -> argparse.Namespace:
    """Create mock arguments for testing."""
    args = argparse.Namespace()
    args.init_model_with_meta_device = False
    args.fp16 = False
    args.bf16 = False
    return args


def create_mock_transformer_config() -> TransformerConfig:
    """Create a mock TransformerConfig for testing."""
    return TransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        use_cpu_initialization=True,
    )


class TestGetModel:
    """Test the _get_model function."""

    @pytest.fixture
    def mock_args(self):
        """Mock arguments."""
        return create_mock_args()

    @pytest.fixture
    def mock_config(self):
        """Mock transformer config."""
        return create_mock_transformer_config()

    @patch("megatron.bridge.training.mlm_compat.model.mpu")
    @patch("megatron.bridge.training.mlm_compat.model.tensor_parallel")
    @patch("megatron.bridge.training.mlm_compat.model.get_model_config")
    @patch("megatron.bridge.training.mlm_compat.model.Float16Module")
    @patch("megatron.bridge.training.mlm_compat.model.correct_amax_history_if_needed")
    @patch("torch.cuda")
    def test_get_model_basic_single_stage(
        self,
        mock_cuda,
        mock_correct_amax,
        mock_float16_module,
        mock_get_model_config,
        mock_tensor_parallel,
        mock_mpu,
        mock_args,
        mock_config,
    ):
        """Test basic model creation with single pipeline stage."""
        # Setup mocks
        mock_mpu.get_pipeline_model_parallel_world_size.return_value = 1
        mock_mpu.get_virtual_pipeline_model_parallel_world_size.return_value = None
        mock_mpu.is_pipeline_first_stage.return_value = True
        mock_mpu.is_pipeline_last_stage.return_value = True
        mock_mpu.get_data_parallel_rank.return_value = 0
        mock_mpu.get_context_parallel_rank.return_value = 0
        mock_mpu.get_tensor_model_parallel_rank.return_value = 0
        mock_mpu.get_pipeline_model_parallel_rank.return_value = 0

        # Mock CUDA
        mock_cuda.current_device.return_value = 0

        # Mock model parameters
        mock_param1 = MagicMock()
        mock_param1.nelement.return_value = 100
        mock_param2 = MagicMock()
        mock_param2.nelement.return_value = 200
        mock_model = MagicMock()
        mock_model.parameters.return_value = [mock_param1, mock_param2]
        mock_provider = MagicMock()
        mock_provider.return_value = mock_model

        # Call the function
        result = _get_model(mock_args, mock_provider, mock_config)

        # Assertions
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == mock_model
        assert mock_model.model_type == ModelType.encoder_or_decoder

        # Verify mpu calls
        mock_mpu.get_pipeline_model_parallel_world_size.assert_called_once()
        mock_mpu.get_virtual_pipeline_model_parallel_world_size.assert_not_called()
        mock_mpu.is_pipeline_first_stage.assert_called_once()
        mock_mpu.is_pipeline_last_stage.assert_called_once()

        # Verify tensor parallel setup
        assert mock_tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes.call_count == 2

        # Verify model provider was called correctly
        mock_provider.assert_called_once_with(mock_args, mock_config, pre_process=True, post_process=True)

        # Verify no fp16 conversion
        mock_float16_module.assert_not_called()

        # Verify amax correction was called
        mock_correct_amax.assert_called_once_with(result)

        # Verify model was moved to CUDA
        mock_model.cuda.assert_called_once_with(0)

    @patch("megatron.bridge.training.mlm_compat.model.mpu")
    @patch("megatron.bridge.training.mlm_compat.model.tensor_parallel")
    @patch("megatron.bridge.training.mlm_compat.model.get_model_config")
    @patch("megatron.bridge.training.mlm_compat.model.Float16Module")
    @patch("megatron.bridge.training.mlm_compat.model.correct_amax_history_if_needed")
    @patch("torch.cuda")
    def test_get_model_virtual_pipeline_parallel(
        self,
        mock_cuda,
        mock_correct_amax,
        mock_float16_module,
        mock_get_model_config,
        mock_tensor_parallel,
        mock_mpu,
        mock_args,
        mock_config,
    ):
        """Test model creation with virtual pipeline parallelism."""
        # Setup mocks for virtual pipeline parallelism
        mock_mpu.get_pipeline_model_parallel_world_size.return_value = 2
        mock_mpu.get_virtual_pipeline_model_parallel_world_size.return_value = 3
        mock_mpu.is_pipeline_first_stage.side_effect = lambda ignore_virtual=False, vp_stage=None: vp_stage == 0
        mock_mpu.is_pipeline_last_stage.side_effect = lambda ignore_virtual=False, vp_stage=None: vp_stage == 2
        mock_mpu.get_data_parallel_rank.return_value = 0
        mock_mpu.get_context_parallel_rank.return_value = 0
        mock_mpu.get_tensor_model_parallel_rank.return_value = 0
        mock_mpu.get_pipeline_model_parallel_rank.return_value = 0

        # Mock model parameters for each stage
        mock_models = []
        for i in range(3):
            mock_param1 = MagicMock()
            mock_param2 = MagicMock()
            mock_model = MagicMock()
            mock_model.parameters.return_value = [mock_param1, mock_param2]
            mock_models.append(mock_model)

        mock_provider = MagicMock()
        mock_provider.side_effect = mock_models

        # Call the function
        result = _get_model(mock_args, mock_provider, mock_config)

        # Assertions
        assert isinstance(result, list)
        assert len(result) == 3

        # Verify each model was created with correct parameters
        expected_calls = [
            call(mock_args, mock_config, pre_process=True, post_process=False, vp_stage=0),
            call(mock_args, mock_config, pre_process=False, post_process=False, vp_stage=1),
            call(mock_args, mock_config, pre_process=False, post_process=True, vp_stage=2),
        ]
        assert mock_provider.call_args_list == expected_calls

        # Verify each model has correct attributes
        for i, model in enumerate(result):
            assert model.model_type == ModelType.encoder_or_decoder
            assert model.vp_stage == i
            model.cuda.assert_called_once()

        # Verify tensor parallel setup for all models
        assert mock_tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes.call_count == 6

        # Verify amax correction was called
        mock_correct_amax.assert_called_once_with(result)

    @patch("megatron.bridge.training.mlm_compat.model.mpu")
    @patch("megatron.bridge.training.mlm_compat.model.tensor_parallel")
    @patch("megatron.bridge.training.mlm_compat.model.get_model_config")
    @patch("megatron.bridge.training.mlm_compat.model.Float16Module")
    @patch("megatron.bridge.training.mlm_compat.model.correct_amax_history_if_needed")
    @patch("torch.device")
    def test_get_model_meta_device(
        self,
        mock_torch_device,
        mock_correct_amax,
        mock_float16_module,
        mock_get_model_config,
        mock_tensor_parallel,
        mock_mpu,
        mock_args,
        mock_config,
    ):
        """Test model creation with meta device initialization."""
        # Setup mocks
        mock_args.init_model_with_meta_device = True
        mock_mpu.get_pipeline_model_parallel_world_size.return_value = 1
        mock_mpu.get_virtual_pipeline_model_parallel_world_size.return_value = None
        mock_mpu.is_pipeline_first_stage.return_value = True
        mock_mpu.is_pipeline_last_stage.return_value = True
        mock_mpu.get_data_parallel_rank.return_value = 0
        mock_mpu.get_context_parallel_rank.return_value = 0
        mock_mpu.get_tensor_model_parallel_rank.return_value = 0
        mock_mpu.get_pipeline_model_parallel_rank.return_value = 0

        # Mock model
        mock_model = MagicMock()
        mock_model.parameters.return_value = [MagicMock(), MagicMock()]
        mock_provider = MagicMock()
        mock_provider.return_value = mock_model

        # Mock torch.device context
        mock_device_context = MagicMock()
        mock_torch_device.return_value.__enter__ = MagicMock(return_value=mock_device_context)
        mock_torch_device.return_value.__exit__ = MagicMock(return_value=None)

        # Call the function
        result = _get_model(mock_args, mock_provider, mock_config)

        # Assertions
        assert isinstance(result, list)
        assert len(result) == 1

        # Verify meta device context was used
        mock_torch_device.assert_called_once_with("meta")

        # Verify model was not moved to CUDA
        mock_model.cuda.assert_not_called()

    @patch("megatron.bridge.training.mlm_compat.model.mpu")
    @patch("megatron.bridge.training.mlm_compat.model.tensor_parallel")
    @patch("megatron.bridge.training.mlm_compat.model.get_model_config")
    @patch("megatron.bridge.training.mlm_compat.model.Float16Module")
    @patch("megatron.bridge.training.mlm_compat.model.correct_amax_history_if_needed")
    def test_get_model_fp16_conversion(
        self,
        mock_correct_amax,
        mock_float16_module,
        mock_get_model_config,
        mock_tensor_parallel,
        mock_mpu,
        mock_args,
        mock_config,
    ):
        """Test model creation with FP16 conversion."""
        # Setup mocks
        mock_args.fp16 = True
        mock_mpu.get_pipeline_model_parallel_world_size.return_value = 1
        mock_mpu.get_virtual_pipeline_model_parallel_world_size.return_value = None
        mock_mpu.is_pipeline_first_stage.return_value = True
        mock_mpu.is_pipeline_last_stage.return_value = True
        mock_mpu.get_data_parallel_rank.return_value = 0
        mock_mpu.get_context_parallel_rank.return_value = 0
        mock_mpu.get_tensor_model_parallel_rank.return_value = 0
        mock_mpu.get_pipeline_model_parallel_rank.return_value = 0

        # Mock model config
        mock_get_model_config.return_value = mock_config

        # Mock model
        mock_model = MagicMock()
        mock_model.parameters.return_value = [MagicMock(), MagicMock()]
        mock_provider = MagicMock()
        mock_provider.return_value = mock_model

        # Mock Float16Module
        mock_fp16_model = MagicMock()
        mock_float16_module.return_value = mock_fp16_model

        # Call the function
        result = _get_model(mock_args, mock_provider, mock_config)

        # Assertions
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == mock_fp16_model

        # Verify Float16Module was created
        mock_get_model_config.assert_called_once_with(mock_model)
        mock_float16_module.assert_called_once_with(mock_config, mock_model)

    @patch("megatron.bridge.training.mlm_compat.model.mpu")
    @patch("megatron.bridge.training.mlm_compat.model.tensor_parallel")
    @patch("megatron.bridge.training.mlm_compat.model.get_model_config")
    @patch("megatron.bridge.training.mlm_compat.model.Float16Module")
    @patch("megatron.bridge.training.mlm_compat.model.correct_amax_history_if_needed")
    def test_get_model_bf16_conversion(
        self,
        mock_correct_amax,
        mock_float16_module,
        mock_get_model_config,
        mock_tensor_parallel,
        mock_mpu,
        mock_args,
        mock_config,
    ):
        """Test model creation with BF16 conversion."""
        # Setup mocks
        mock_args.bf16 = True
        mock_mpu.get_pipeline_model_parallel_world_size.return_value = 1
        mock_mpu.get_virtual_pipeline_model_parallel_world_size.return_value = None
        mock_mpu.is_pipeline_first_stage.return_value = True
        mock_mpu.is_pipeline_last_stage.return_value = True
        mock_mpu.get_data_parallel_rank.return_value = 0
        mock_mpu.get_context_parallel_rank.return_value = 0
        mock_mpu.get_tensor_model_parallel_rank.return_value = 0
        mock_mpu.get_pipeline_model_parallel_rank.return_value = 0

        # Mock model config
        mock_get_model_config.return_value = mock_config

        # Mock model
        mock_model = MagicMock()
        mock_model.parameters.return_value = [MagicMock(), MagicMock()]
        mock_provider = MagicMock()
        mock_provider.return_value = mock_model

        # Mock Float16Module
        mock_fp16_model = MagicMock()
        mock_float16_module.return_value = mock_fp16_model

        # Call the function
        result = _get_model(mock_args, mock_provider, mock_config)

        # Assertions
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == mock_fp16_model

        # Verify Float16Module was created
        mock_float16_module.assert_called_once_with(mock_config, mock_model)

    @patch("megatron.bridge.training.mlm_compat.model.mpu")
    @patch("megatron.bridge.training.mlm_compat.model.tensor_parallel")
    @patch("megatron.bridge.training.mlm_compat.model.get_model_config")
    @patch("megatron.bridge.training.mlm_compat.model.Float16Module")
    @patch("megatron.bridge.training.mlm_compat.model.correct_amax_history_if_needed")
    def test_get_model_empty_parameters(
        self,
        mock_correct_amax,
        mock_float16_module,
        mock_get_model_config,
        mock_tensor_parallel,
        mock_mpu,
        mock_args,
        mock_config,
    ):
        """Test model creation with empty parameters."""
        # Setup mocks
        mock_mpu.get_pipeline_model_parallel_world_size.return_value = 1
        mock_mpu.get_virtual_pipeline_model_parallel_world_size.return_value = None
        mock_mpu.is_pipeline_first_stage.return_value = True
        mock_mpu.is_pipeline_last_stage.return_value = True
        mock_mpu.get_data_parallel_rank.return_value = 0
        mock_mpu.get_context_parallel_rank.return_value = 0
        mock_mpu.get_tensor_model_parallel_rank.return_value = 0
        mock_mpu.get_pipeline_model_parallel_rank.return_value = 0

        # Mock model with no parameters
        def mock_provider(args, config, pre_process=True, post_process=True, vp_stage=None):
            mock_model = MagicMock()
            mock_model.parameters.return_value = []
            return mock_model

        # Call the function
        result = _get_model(mock_args, mock_provider, mock_config)

        # Assertions
        assert isinstance(result, list)
        assert len(result) == 1

        # Verify tensor parallel setup was called but with no parameters
        mock_tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes.assert_not_called()
