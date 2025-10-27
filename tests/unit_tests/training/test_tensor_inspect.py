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

"""Unit tests for tensor inspection integration."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from megatron.bridge.training.config import TensorInspectConfig
from megatron.bridge.training.tensor_inspect import (
    finalize_tensor_inspect_post_model_initialization,
    initialize_tensor_inspect_pre_model_initialization,
    tensor_inspect_end_if_enabled,
    tensor_inspect_step_if_enabled,
)


@pytest.fixture
def mock_nvinspect_api():
    """Provide a minimal fake nvinspect API for isolation."""
    mock_api = MagicMock()
    mock_api.initialize = Mock()
    mock_api.initialize_training_step = Mock()
    mock_api.infer_and_assign_layer_names = Mock()
    mock_api.set_tensor_reduction_group = Mock()
    mock_api.step = Mock()
    mock_api.end_debug = Mock()
    return mock_api


class TestTensorInspectInitialization:
    """Test pre-model initialization logic for tensor inspection."""

    def test_initialize_disabled_noop(self, mock_nvinspect_api):
        """Test that initialization is skipped when disabled."""
        cfg = TensorInspectConfig(enabled=False)

        with patch("megatron.bridge.training.tensor_inspect.HAVE_NVINSPECT", True), patch(
            "megatron.bridge.training.tensor_inspect.nvinspect_api", mock_nvinspect_api
        ):
            initialize_tensor_inspect_pre_model_initialization(cfg)

        mock_nvinspect_api.initialize.assert_not_called()

    def test_initialize_missing_dependency_raises(self):
        """Test that initialization raises ImportError when dependency is missing."""
        cfg = TensorInspectConfig(enabled=True)

        with patch("megatron.bridge.training.tensor_inspect.HAVE_NVINSPECT", False):
            with pytest.raises(ImportError, match="nvdlfw_inspect is not available"):
                initialize_tensor_inspect_pre_model_initialization(cfg)

    def test_initialize_success_invokes_nvinspect_api(self, mock_nvinspect_api):
        """Test that initialization calls nvinspect API with correct parameters."""
        cfg = TensorInspectConfig(
            enabled=True,
            features="/tmp/conf.yaml",
            feature_dirs=["/tmp/features"],
            log_dir="/tmp/logs",
            init_training_step=12,
        )

        with patch("megatron.bridge.training.tensor_inspect.HAVE_NVINSPECT", True), patch(
            "megatron.bridge.training.tensor_inspect.nvinspect_api", mock_nvinspect_api
        ):
            initialize_tensor_inspect_pre_model_initialization(cfg)

        mock_nvinspect_api.initialize.assert_called_once_with(
            config_file="/tmp/conf.yaml",
            feature_dirs=["/tmp/features"],
            log_dir="/tmp/logs",
            statistics_logger=None,
            init_training_step=12,
            default_logging_enabled=True,
        )


class TestTensorInspectFinalize:
    """Test post-model initialization logic for tensor inspection."""

    def test_finalize_success_sets_up_loggers_and_groups(self, mock_nvinspect_api):
        """Test that finalize attaches loggers and configures reduction groups."""
        cfg = TensorInspectConfig(enabled=True)
        model = [Mock()]

        with patch("megatron.bridge.training.tensor_inspect.HAVE_NVINSPECT", True), patch(
            "megatron.bridge.training.tensor_inspect._maybe_attach_metric_loggers"
        ) as mock_attach, patch(
            "megatron.bridge.training.tensor_inspect.nvinspect_api", mock_nvinspect_api
        ), patch(
            "megatron.core.parallel_state.get_tensor_and_data_parallel_group",
            return_value="mock-group",
        ):
            finalize_tensor_inspect_post_model_initialization(
                cfg, model, tensorboard_logger=Mock(), wandb_logger=Mock(), current_training_step=12
            )

        mock_attach.assert_called_once()
        
        mock_nvinspect_api.initialize_training_step.assert_called_once_with(12)
        mock_nvinspect_api.infer_and_assign_layer_names.assert_called_once_with(model)
        mock_nvinspect_api.set_tensor_reduction_group.assert_called_once_with("mock-group")

    def test_finalize_missing_dependency_raises(self):
        """Test that finalize raises ImportError when dependency is missing."""
        cfg = TensorInspectConfig(enabled=True)

        with patch("megatron.bridge.training.tensor_inspect.HAVE_NVINSPECT", False):
            with pytest.raises(ImportError, match="nvdlfw_inspect is not available"):
                finalize_tensor_inspect_post_model_initialization(
                    cfg, model=[Mock()], tensorboard_logger=None, wandb_logger=None, current_training_step=None
                )


class TestTensorInspectRuntime:
    """Test runtime stepping and shutdown APIs for tensor inspection."""

    def test_tensor_inspect_step_calls_api(self, mock_nvinspect_api):
        """Test that step advances nvinspect internal counter."""
        cfg = TensorInspectConfig(enabled=True)

        with patch("megatron.bridge.training.tensor_inspect.HAVE_NVINSPECT", True), patch(
            "megatron.bridge.training.tensor_inspect.nvinspect_api", mock_nvinspect_api
        ):
            tensor_inspect_step_if_enabled(cfg)
            
        mock_nvinspect_api.step.assert_called_once()

    def test_tensor_inspect_step_missing_dependency_raises(self):
        """Test that step raises ImportError when dependency is missing."""
        cfg = TensorInspectConfig(enabled=True)

        with patch("megatron.bridge.training.tensor_inspect.HAVE_NVINSPECT", False):
            with pytest.raises(ImportError, match="nvdlfw_inspect is not available"):
                tensor_inspect_step_if_enabled(cfg)

    def test_tensor_inspect_end_calls_end_debug(self, mock_nvinspect_api):
        """Test that shutdown invokes nvinspect end_debug."""
        cfg = TensorInspectConfig(enabled=True)

        with patch("megatron.bridge.training.tensor_inspect.HAVE_NVINSPECT", True), patch(
            "megatron.bridge.training.tensor_inspect.nvinspect_api", mock_nvinspect_api
        ):
            tensor_inspect_end_if_enabled(cfg)

        mock_nvinspect_api.end_debug.assert_called_once()

