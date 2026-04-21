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

"""Unit tests for NVRx straggler detection functionality."""

import logging
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest


# Test fixtures
@pytest.fixture(scope="session", autouse=True)
def mock_nvidia_resiliency_ext():
    """
    Mock the nvidia_resiliency_ext module for all tests in this session.

    This ensures tests can run without requiring the actual NVIDIA library,
    while properly cleaning up to maintain test isolation.
    """
    # Store original state
    original_modules = {}
    modules_to_mock = [
        "nvidia_resiliency_ext",
        "nvidia_resiliency_ext.attribution",
        "nvidia_resiliency_ext.attribution.straggler",
        "nvidia_resiliency_ext.straggler",
    ]

    for module in modules_to_mock:
        if module in sys.modules:
            original_modules[module] = sys.modules[module]

    # Mock the modules
    mock_module = MagicMock()
    mock_attribution = MagicMock()
    mock_straggler = MagicMock()

    # Set up the hierarchy: nvidia_resiliency_ext -> attribution -> straggler
    mock_attribution.straggler = mock_straggler
    mock_module.attribution = mock_attribution
    mock_module.straggler = mock_straggler

    sys.modules["nvidia_resiliency_ext"] = mock_module
    sys.modules["nvidia_resiliency_ext.attribution"] = mock_attribution
    sys.modules["nvidia_resiliency_ext.attribution.straggler"] = mock_straggler
    sys.modules["nvidia_resiliency_ext.straggler"] = mock_straggler

    yield mock_module

    # Cleanup - restore original state to maintain test isolation
    for module in modules_to_mock:
        if module in original_modules:
            sys.modules[module] = original_modules[module]
        else:
            sys.modules.pop(module, None)


# Import after mocking is set up
from megatron.bridge.training.config import NVRxStragglerDetectionConfig
from megatron.bridge.training.nvrx_straggler import (
    NVRxStragglerDetectionManager,
    check_nvrx_straggler_detection,
    safe_shutdown_nvrx_straggler_manager,
)


@pytest.fixture
def mock_torch_distributed():
    """Mock torch.distributed for testing distributed functionality."""
    with patch("torch.distributed") as mock_dist:
        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 0
        mock_dist.get_world_size.return_value = 1
        mock_dist.broadcast = MagicMock()
        yield mock_dist


@pytest.fixture
def mock_torch_cuda():
    """Mock torch.cuda for testing CUDA functionality."""
    with patch("torch.cuda") as mock_cuda:
        mock_cuda.current_device.return_value = 0
        yield mock_cuda


@pytest.fixture
def suppress_logging():
    """Suppress logging output during tests."""
    import logging

    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


class TestNVRxStragglerDetectionConfig:
    """Test the NVRxStragglerDetectionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = NVRxStragglerDetectionConfig()

        assert config.enabled is False
        assert config.report_time_interval == 300.0
        assert config.calc_relative_gpu_perf is True
        assert config.calc_individual_gpu_perf is True
        assert config.num_gpu_perf_scores_to_print == 5
        assert config.gpu_relative_perf_threshold == 0.7
        assert config.gpu_individual_perf_threshold == 0.7
        assert config.stop_if_detected is False
        assert config.enable_logging is True
        assert config.profiling_interval == 1
        assert config.logger_name == "megatron.bridge.NVRxStragglerDetection"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = NVRxStragglerDetectionConfig(
            enabled=False,
            report_time_interval=600.0,
            calc_relative_gpu_perf=False,
            stop_if_detected=True,
            logger_name="custom_logger",
        )

        assert config.enabled is False
        assert config.report_time_interval == 600.0
        assert config.calc_relative_gpu_perf is False
        assert config.stop_if_detected is True
        assert config.logger_name == "custom_logger"


class TestSafeShutdownFunction:
    """Test the safe_shutdown_nvrx_straggler_manager function."""

    def test_shutdown_with_none_manager(self):
        """Test shutdown with None manager should do nothing."""
        safe_shutdown_nvrx_straggler_manager(None)
        # Should not raise any exceptions

    def test_shutdown_with_valid_manager(self):
        """Test shutdown with valid manager."""
        mock_manager = Mock()
        mock_manager.shutdown = Mock()

        safe_shutdown_nvrx_straggler_manager(mock_manager)

        mock_manager.shutdown.assert_called_once()

    def test_shutdown_with_exception(self, caplog):
        """Test shutdown handles exceptions gracefully."""
        mock_manager = Mock()
        mock_manager.shutdown.side_effect = RuntimeError("Shutdown failed")

        with caplog.at_level(logging.ERROR):
            safe_shutdown_nvrx_straggler_manager(mock_manager, "test_logger")

        assert "Error shutting down NVRx straggler detection: Shutdown failed" in caplog.text


class TestNVRxStragglerDetectionManager:
    """Test the NVRxStragglerDetectionManager class."""

    @pytest.fixture
    def mock_straggler_module(self):
        """Mock the nvidia_resiliency_ext.straggler module."""
        with patch("megatron.bridge.training.nvrx_straggler.straggler") as mock_straggler:
            mock_straggler.Detector = Mock()
            mock_straggler.CallableId = Mock()
            yield mock_straggler

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return NVRxStragglerDetectionConfig(
            enabled=True,
            report_time_interval=100.0,
            calc_relative_gpu_perf=True,
            calc_individual_gpu_perf=True,
            num_gpu_perf_scores_to_print=3,
            stop_if_detected=False,
            logger_name="test_logger",
        )

    @pytest.fixture
    def manager(self, config, mock_straggler_module):
        """Create a manager instance for testing."""
        return NVRxStragglerDetectionManager(config)

    def test_init_success(self, config, mock_straggler_module):
        """Test successful initialization."""
        manager = NVRxStragglerDetectionManager(config)

        assert manager.config == config
        assert manager.initialized is False
        assert manager.wrapped_function is None
        assert "relative_perf_scores" in manager.scores_to_compute
        assert "individual_perf_scores" in manager.scores_to_compute

    def test_init_scores_to_compute_config(self, mock_straggler_module):
        """Test scores_to_compute based on configuration."""
        # Test with only relative performance
        config1 = NVRxStragglerDetectionConfig(calc_relative_gpu_perf=True, calc_individual_gpu_perf=False)
        manager1 = NVRxStragglerDetectionManager(config1)
        assert manager1.scores_to_compute == ["relative_perf_scores"]

        # Test with only individual performance
        config2 = NVRxStragglerDetectionConfig(calc_relative_gpu_perf=False, calc_individual_gpu_perf=True)
        manager2 = NVRxStragglerDetectionManager(config2)
        assert manager2.scores_to_compute == ["individual_perf_scores"]

        # Test with neither
        config3 = NVRxStragglerDetectionConfig(calc_relative_gpu_perf=False, calc_individual_gpu_perf=False)
        manager3 = NVRxStragglerDetectionManager(config3)
        assert manager3.scores_to_compute == []

    def test_initialize_disabled(self, manager, mock_straggler_module):
        """Test initialize when disabled."""
        manager.config.enabled = False

        manager.initialize()

        mock_straggler_module.Detector.initialize.assert_not_called()
        assert manager.initialized is False

    def test_initialize_already_initialized(self, manager, mock_straggler_module):
        """Test initialize when already initialized."""
        manager.initialized = True

        with pytest.raises(RuntimeError, match="already initialized"):
            manager.initialize()

    def test_initialize_success(self, manager, mock_straggler_module):
        """Test successful initialization."""
        manager.initialize()

        mock_straggler_module.Detector.initialize.assert_called_once_with(
            scores_to_compute=["relative_perf_scores", "individual_perf_scores"],
            gather_on_rank0=True,
            profiling_interval=1,
            report_time_interval=100.0,
        )
        assert manager.initialized is True

    def test_wrap_train_step_function_disabled(self, manager, mock_straggler_module):
        """Test wrapping when disabled."""
        manager.config.enabled = False

        def dummy_func():
            pass

        result = manager.wrap_train_step_function(dummy_func)

        assert result is dummy_func
        mock_straggler_module.Detector.wrap_callables.assert_not_called()

    def test_wrap_train_step_function_not_initialized(self, manager, mock_straggler_module):
        """Test wrapping when not initialized."""

        def dummy_func():
            pass

        result = manager.wrap_train_step_function(dummy_func)

        assert result is dummy_func
        mock_straggler_module.Detector.wrap_callables.assert_not_called()

    def test_wrap_train_step_function_already_wrapped(self, manager, mock_straggler_module):
        """Test wrapping when already wrapped."""
        manager.initialized = True
        manager.wrapped_function = Mock()

        def dummy_func():
            pass

        result = manager.wrap_train_step_function(dummy_func)

        assert result is dummy_func
        mock_straggler_module.Detector.wrap_callables.assert_not_called()

    def test_wrap_train_step_function_success(self, manager, mock_straggler_module):
        """Test successful wrapping."""
        manager.initialized = True

        def dummy_func():
            pass

        result = manager.wrap_train_step_function(dummy_func)

        assert result is dummy_func
        assert manager.wrapped_function is dummy_func

        # Verify CallableId was called with a TrainStepWrapper object, not the original function
        mock_straggler_module.CallableId.assert_called_once()
        call_args = mock_straggler_module.CallableId.call_args[0]
        assert len(call_args) == 2
        wrapper_obj, method_name = call_args
        assert method_name == "train_step"

        # Verify the wrapper object has the train_step method that wraps our dummy_func
        assert hasattr(wrapper_obj, "train_step")
        assert wrapper_obj.train_step is dummy_func

        mock_straggler_module.Detector.wrap_callables.assert_called_once()

    def test_check_stragglers_disabled(self, manager, mock_straggler_module):
        """Test check_stragglers when disabled."""
        manager.config.enabled = False

        result = manager.check_stragglers(0)

        assert result is False

    def test_check_stragglers_not_initialized(self, manager, mock_straggler_module):
        """Test check_stragglers when not initialized."""
        result = manager.check_stragglers(0)

        assert result is False

    @patch("megatron.bridge.training.nvrx_straggler.time.monotonic")
    def test_check_stragglers_no_report(self, mock_time, manager, mock_straggler_module):
        """Test check_stragglers when no report is generated."""
        mock_time.return_value = 1000.0
        manager.initialized = True
        mock_straggler_module.Detector.generate_report_if_interval_elapsed.return_value = None
        mock_straggler_module.Detector.is_interval_elapsed.return_value = False

        result = manager.check_stragglers(0)

        assert result is False

    @patch("megatron.bridge.training.nvrx_straggler.time.monotonic")
    def test_check_stragglers_with_report_no_stragglers(self, mock_time, manager, mock_straggler_module):
        """Test check_stragglers with report but no stragglers."""
        mock_time.side_effect = [1000.0, 1001.0]
        manager.initialized = True

        # Mock report
        mock_report = Mock()
        mock_report.identify_stragglers.return_value = {"straggler_gpus_relative": [], "straggler_gpus_individual": []}
        # Configure GPU performance scores as proper dictionaries
        mock_report.gpu_relative_perf_scores = {0: 0.95, 1: 0.92, 2: 0.88}
        mock_report.gpu_individual_perf_scores = {0: 0.93, 1: 0.90, 2: 0.85}
        mock_report.rank_to_node = {0: "node0", 1: "node1", 2: "node2"}

        mock_straggler_module.Detector.generate_report_if_interval_elapsed.return_value = mock_report
        mock_straggler_module.Detector.is_interval_elapsed.return_value = True

        result = manager.check_stragglers(0)

        assert result is False
        mock_report.identify_stragglers.assert_called_once_with(gpu_rel_threshold=0.7, gpu_indiv_threshold=0.7)

    @patch("megatron.bridge.training.nvrx_straggler.time.monotonic")
    def test_check_stragglers_with_stragglers_no_stop(self, mock_time, manager, mock_straggler_module):
        """Test check_stragglers with stragglers but stop_if_detected=False."""
        mock_time.side_effect = [1000.0, 1001.0]
        manager.initialized = True
        manager.config.stop_if_detected = False

        # Mock report with stragglers
        mock_report = Mock()
        mock_report.identify_stragglers.return_value = {
            "straggler_gpus_relative": [1, 2],
            "straggler_gpus_individual": [],
        }
        # Configure GPU performance scores as proper dictionaries
        mock_report.gpu_relative_perf_scores = {0: 0.95, 1: 0.65, 2: 0.60}  # 1,2 are stragglers
        mock_report.gpu_individual_perf_scores = {0: 0.93, 1: 0.90, 2: 0.85}
        mock_report.rank_to_node = {0: "node0", 1: "node1", 2: "node2"}

        mock_straggler_module.Detector.generate_report_if_interval_elapsed.return_value = mock_report
        mock_straggler_module.Detector.is_interval_elapsed.return_value = True

        result = manager.check_stragglers(0)

        assert result is False

    @patch("megatron.bridge.training.nvrx_straggler.time.monotonic")
    @patch("torch.distributed.broadcast")
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.cuda.current_device", return_value=0)
    def test_check_stragglers_with_stragglers_should_stop(
        self, mock_device, mock_dist_init, mock_broadcast, mock_time, manager, mock_straggler_module
    ):
        """Test check_stragglers with stragglers and stop_if_detected=True."""
        mock_time.side_effect = [1000.0, 1001.0]
        manager.initialized = True
        manager.config.stop_if_detected = True
        # Disable logging to avoid torch.tensor conflicts in _log_gpu_perf_scores
        manager.config.enable_logging = False

        # Mock report with stragglers
        mock_report = Mock()
        mock_report.identify_stragglers.return_value = {
            "straggler_gpus_relative": [1, 2],
            "straggler_gpus_individual": [],
        }
        # Configure GPU performance scores as proper dictionaries
        mock_report.gpu_relative_perf_scores = {0: 0.95, 1: 0.65, 2: 0.60}  # 1,2 are stragglers
        mock_report.gpu_individual_perf_scores = {0: 0.93, 1: 0.90, 2: 0.85}
        mock_report.rank_to_node = {0: "node0", 1: "node1", 2: "node2"}

        mock_straggler_module.Detector.generate_report_if_interval_elapsed.return_value = mock_report
        mock_straggler_module.Detector.is_interval_elapsed.return_value = True

        # Mock tensor for broadcast
        mock_tensor = Mock()
        mock_tensor.item.return_value = 1.0

        with patch("torch.tensor", return_value=mock_tensor):
            result = manager.check_stragglers(0)

        assert result is True
        mock_broadcast.assert_called_once()

    def test_format_gpu_scores_small_group(self):
        """Test _format_gpu_scores with small number of ranks."""
        rank_to_score = {0: 0.9, 1: 0.8, 2: 0.7}
        rank_to_node = {0: "node0", 1: "node1", 2: "node2"}

        result = NVRxStragglerDetectionManager._format_gpu_scores(rank_to_score, rank_to_node, num_best=2, num_worst=2)

        assert "Rank=0" in result
        assert "Rank=1" in result
        assert "Rank=2" in result
        assert "Score=0.90" in result
        assert "Score=0.80" in result
        assert "Score=0.70" in result

    def test_format_gpu_scores_large_group(self):
        """Test _format_gpu_scores with large number of ranks."""
        rank_to_score = {i: 1.0 - i * 0.1 for i in range(10)}
        rank_to_node = {i: f"node{i}" for i in range(10)}

        result = NVRxStragglerDetectionManager._format_gpu_scores(rank_to_score, rank_to_node, num_best=2, num_worst=2)

        assert "Worst performing 2/10 ranks:" in result
        assert "Best performing 2/10 ranks:" in result
        assert "Rank=0" in result  # Best performer

    def test_shutdown_not_initialized(self, manager, mock_straggler_module):
        """Test shutdown when not initialized."""
        manager.shutdown()

        mock_straggler_module.Detector.shutdown.assert_not_called()

    def test_shutdown_disabled(self, manager, mock_straggler_module):
        """Test shutdown when disabled."""
        manager.initialized = True
        manager.config.enabled = False

        manager.shutdown()

        mock_straggler_module.Detector.shutdown.assert_not_called()

    def test_shutdown_success(self, manager, mock_straggler_module):
        """Test successful shutdown."""
        manager.initialized = True

        manager.shutdown()

        mock_straggler_module.Detector.shutdown.assert_called_once()
        assert manager.initialized is False
        assert manager.wrapped_function is None


class TestCheckNVRxStragglerDetection:
    """Test the check_nvrx_straggler_detection function."""

    def test_with_none_manager(self):
        """Test function with None manager."""
        result = check_nvrx_straggler_detection(None)
        assert result is False

    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.is_initialized", return_value=True)
    def test_with_valid_manager_no_stragglers(self, mock_dist_init, mock_get_rank):
        """Test function with valid manager returning no stragglers."""
        mock_manager = Mock()
        mock_manager.check_stragglers.return_value = False

        result = check_nvrx_straggler_detection(mock_manager)

        assert result is False
        mock_manager.check_stragglers.assert_called_once_with(0)

    @patch("torch.distributed.get_rank", return_value=1)
    @patch("torch.distributed.is_initialized", return_value=True)
    def test_with_valid_manager_stragglers_detected(self, mock_dist_init, mock_get_rank):
        """Test function with valid manager detecting stragglers."""
        mock_manager = Mock()
        mock_manager.check_stragglers.return_value = True

        result = check_nvrx_straggler_detection(mock_manager)

        assert result is True
        mock_manager.check_stragglers.assert_called_once_with(1)

    @patch("torch.distributed.is_initialized", return_value=False)
    def test_with_distributed_not_initialized(self, mock_dist_init):
        """Test function when distributed is not initialized."""
        mock_manager = Mock()
        mock_manager.check_stragglers.return_value = False

        result = check_nvrx_straggler_detection(mock_manager)

        assert result is False
        mock_manager.check_stragglers.assert_called_once_with(0)


class TestNVRxNotAvailable:
    """Test behavior when HAVE_NVRX is False (nvidia-resiliency-ext not available)."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return NVRxStragglerDetectionConfig(
            enabled=True,
            report_time_interval=100.0,
            calc_relative_gpu_perf=True,
            calc_individual_gpu_perf=True,
            stop_if_detected=False,
            logger_name="test_logger",
        )

    def test_init_raises_import_error_without_nvrx(self, config):
        """Test that __init__ raises ImportError when HAVE_NVRX is False."""
        with patch("megatron.bridge.training.nvrx_straggler.HAVE_NVRX", False):
            # Should raise ImportError during instantiation with the expected message
            with pytest.raises(ImportError, match="nvidia-resiliency-ext is not available"):
                NVRxStragglerDetectionManager(config)


class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.fixture
    def full_config(self):
        """Create a full configuration for integration testing."""
        return NVRxStragglerDetectionConfig(
            enabled=True,
            report_time_interval=60.0,
            calc_relative_gpu_perf=True,
            calc_individual_gpu_perf=False,
            num_gpu_perf_scores_to_print=2,
            gpu_relative_perf_threshold=0.8,
            stop_if_detected=True,
            enable_logging=True,
            profiling_interval=2,
            logger_name="integration_test",
        )

    def test_full_workflow(self, full_config):
        """Test the complete workflow from initialization to shutdown."""
        # Use the same mocking pattern as other tests for consistency
        with patch("megatron.bridge.training.nvrx_straggler.straggler") as mock_straggler:
            # Setup mocks
            mock_straggler.Detector = Mock()
            mock_straggler.CallableId = Mock()

            # Configure the Detector's methods to return proper values
            mock_straggler.Detector.generate_report_if_interval_elapsed.return_value = None
            mock_straggler.Detector.is_interval_elapsed.return_value = False

            # Create manager and initialize
            manager = NVRxStragglerDetectionManager(full_config)
            manager.initialize()

            # Wrap a function
            def train_step():
                return "training"

            wrapped_func = manager.wrap_train_step_function(train_step)
            assert wrapped_func is train_step

            # Test straggler detection function
            with patch("torch.distributed.is_initialized", return_value=False):
                result = check_nvrx_straggler_detection(manager)
                # Should return False since we configured no report generation
                assert result is False

            # Test safe shutdown
            safe_shutdown_nvrx_straggler_manager(manager)

            # Verify calls
            mock_straggler.Detector.initialize.assert_called_once()
            mock_straggler.Detector.wrap_callables.assert_called_once()
            mock_straggler.Detector.shutdown.assert_called_once()
