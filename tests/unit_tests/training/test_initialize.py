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

import os
import tempfile
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.training.config import DistributedInitConfig
from megatron.bridge.training.initialize import _initialize_tp_communicators, _setup_flight_recorder_env


_FR_ENV_VARS = [
    "TORCH_FR_DUMP_TEMP_FILE",
    "TORCH_NCCL_DEBUG_INFO_TEMP_FILE",
    "TORCH_NCCL_TRACE_BUFFER_SIZE",
    "TORCH_NCCL_DUMP_ON_TIMEOUT",
    "TORCH_INCLUDE_STACK_TRACE",
    "TORCH_INCLUDE_ONLY_ACTIVE",
    "TORCH_NCCL_EXTRA_DUMP_ON_EXEC",
]


class TestInitializeTPCommunicators:
    """Test suite for _initialize_tp_communicators function."""

    @pytest.fixture
    def mock_gpt_config(self):
        """Create a mock GPT model configuration."""
        config = Mock(spec=GPTModelProvider)
        config.seq_length = 1024
        config.hidden_size = 768
        config.context_parallel_size = 1
        config.tensor_model_parallel_size = 2
        config.tp_comm_overlap_cfg = None
        config.fp8 = None
        config.first_last_layers_bf16 = False
        config.num_layers_at_start_in_bf16 = 0
        config.num_layers_at_end_in_bf16 = 0
        config.tp_comm_bootstrap_backend = "nccl"
        return config

    def test_import_error_transformer_engine_missing(self, mock_gpt_config):
        """Test ImportError when transformer_engine is not available."""
        with patch("builtins.__import__", side_effect=ImportError("No module named 'transformer_engine'")):
            with pytest.raises(
                RuntimeError,
                match="Tensor Parallel Communication/GEMM Overlap optimization needs 'yaml' and 'transformer_engine' packages",
            ):
                _initialize_tp_communicators(mock_gpt_config, micro_batch_size=4)

    def test_import_error_yaml_missing(self, mock_gpt_config):
        """Test ImportError when yaml is not available."""
        with patch("builtins.__import__", side_effect=ImportError("No module named 'yaml'")):
            with pytest.raises(
                RuntimeError,
                match="Tensor Parallel Communication/GEMM Overlap optimization needs 'yaml' and 'transformer_engine' packages",
            ):
                _initialize_tp_communicators(mock_gpt_config, micro_batch_size=4)

    @patch("transformer_engine.pytorch.module.base.initialize_ub")
    def test_config_loading_from_string_file(self, mock_init_ub, mock_gpt_config):
        """Test loading tp_comm_overlap_cfg from a string file path."""
        # Create a temporary YAML file
        config_data = {"buffer_size": 1024, "overlap_enabled": True}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file_path = f.name

        try:
            mock_gpt_config.tp_comm_overlap_cfg = config_file_path

            with patch("megatron.bridge.training.initialize.is_te_min_version", return_value=True):
                _initialize_tp_communicators(mock_gpt_config, micro_batch_size=4)

                # Verify that the config was loaded from the file
                mock_init_ub.assert_called_once()
                call_args = mock_init_ub.call_args
                assert call_args[1]["ub_cfgs"] == config_data
        finally:
            Path(config_file_path).unlink()

    @patch("transformer_engine.pytorch.module.base.initialize_ub")
    def test_config_loading_from_dict(self, mock_init_ub, mock_gpt_config):
        """Test loading tp_comm_overlap_cfg from a dictionary."""
        config_data = {"buffer_size": 2048, "overlap_enabled": False}
        mock_gpt_config.tp_comm_overlap_cfg = config_data

        with patch("megatron.bridge.training.initialize.is_te_min_version", return_value=True):
            _initialize_tp_communicators(mock_gpt_config, micro_batch_size=4)

            # Verify that the config dict was used directly
            mock_init_ub.assert_called_once()
            call_args = mock_init_ub.call_args
            assert call_args[1]["ub_cfgs"] == config_data

    @patch("transformer_engine.pytorch.module.base.initialize_ub")
    def test_config_loading_none(self, mock_init_ub, mock_gpt_config):
        """Test when tp_comm_overlap_cfg is None."""
        mock_gpt_config.tp_comm_overlap_cfg = None

        with patch("megatron.bridge.training.initialize.is_te_min_version", return_value=True):
            _initialize_tp_communicators(mock_gpt_config, micro_batch_size=4)

            # Verify that empty dict was used
            mock_init_ub.assert_called_once()
            call_args = mock_init_ub.call_args
            assert call_args[1]["ub_cfgs"] == {}

    @patch("transformer_engine.pytorch.module.base.initialize_ub")
    def test_input_shape_calculation_gpt(self, mock_init_ub, mock_gpt_config):
        """Test input_shape calculation for GPT model."""
        mock_gpt_config.seq_length = 1024
        mock_gpt_config.hidden_size = 768
        mock_gpt_config.context_parallel_size = 2
        micro_batch_size = 8

        expected_shape = [
            (1024 * 8) // 2,  # seq_length * micro_batch_size // context_parallel_size
            768,  # hidden_size
        ]

        with patch("megatron.bridge.training.initialize.is_te_min_version", return_value=True):
            _initialize_tp_communicators(mock_gpt_config, micro_batch_size)

            call_args = mock_init_ub.call_args
            assert call_args[1]["shape"] == expected_shape

    @patch("transformer_engine.pytorch.module.base.initialize_ub")
    @patch("transformer_engine.pytorch.module.base.UserBufferQuantizationMode")
    def test_te_version_2_7_0_fp8_disabled(self, mock_quant_mode, mock_init_ub, mock_gpt_config):
        """Test TE version 2.7.0+ path with FP8 disabled."""
        mock_gpt_config.fp8 = None

        with patch("megatron.bridge.training.initialize.is_te_min_version", side_effect=lambda v: v == "2.7.0"):
            mock_quant_mode.FP8 = "FP8"
            mock_quant_mode.NONE = "NONE"

            _initialize_tp_communicators(mock_gpt_config, micro_batch_size=4)

            mock_init_ub.assert_called_once()
            call_args = mock_init_ub.call_args
            assert call_args[1]["quantization_modes"] == ["NONE"]
            assert call_args[1]["tp_size"] == mock_gpt_config.tensor_model_parallel_size
            assert call_args[1]["bootstrap_backend"] == mock_gpt_config.tp_comm_bootstrap_backend

    @patch("transformer_engine.pytorch.module.base.initialize_ub")
    @patch("transformer_engine.pytorch.module.base.UserBufferQuantizationMode")
    def test_te_version_2_7_0_fp8_enabled(self, mock_quant_mode, mock_init_ub, mock_gpt_config):
        """Test TE version 2.7.0+ path with FP8 enabled."""
        mock_gpt_config.fp8 = "e4m3"

        with patch("megatron.bridge.training.initialize.is_te_min_version", side_effect=lambda v: v == "2.7.0"):
            mock_quant_mode.FP8 = "FP8"
            mock_quant_mode.NONE = "NONE"

            _initialize_tp_communicators(mock_gpt_config, micro_batch_size=4)

            mock_init_ub.assert_called_once()
            call_args = mock_init_ub.call_args
            assert call_args[1]["quantization_modes"] == ["FP8"]

    @patch("transformer_engine.pytorch.module.base.initialize_ub")
    @patch("transformer_engine.pytorch.module.base.UserBufferQuantizationMode")
    def test_te_version_2_7_0_fp8_with_bf16_layers(self, mock_quant_mode, mock_init_ub, mock_gpt_config):
        """Test TE version 2.7.0+ path with FP8 and BF16 first/last layers."""
        mock_gpt_config.fp8 = "e4m3"
        mock_gpt_config.first_last_layers_bf16 = True
        mock_gpt_config.num_layers_at_start_in_bf16 = 2
        mock_gpt_config.num_layers_at_end_in_bf16 = 1

        with patch("megatron.bridge.training.initialize.is_te_min_version", side_effect=lambda v: v == "2.7.0"):
            mock_quant_mode.FP8 = "FP8"
            mock_quant_mode.NONE = "NONE"

            _initialize_tp_communicators(mock_gpt_config, micro_batch_size=4)

            mock_init_ub.assert_called_once()
            call_args = mock_init_ub.call_args
            assert call_args[1]["quantization_modes"] == ["FP8", "NONE"]

    @patch("transformer_engine.pytorch.module.base.initialize_ub")
    @patch("transformer_engine.pytorch.module.base.UserBufferQuantizationMode")
    def test_te_version_2_7_0_fp8_with_bf16_layers_no_layers(self, mock_quant_mode, mock_init_ub, mock_gpt_config):
        """Test TE version 2.7.0+ path with FP8 and BF16 flag but no BF16 layers."""
        mock_gpt_config.fp8 = "e4m3"
        mock_gpt_config.first_last_layers_bf16 = True
        mock_gpt_config.num_layers_at_start_in_bf16 = 0
        mock_gpt_config.num_layers_at_end_in_bf16 = 0

        with patch("megatron.bridge.training.initialize.is_te_min_version", side_effect=lambda v: v == "2.7.0"):
            mock_quant_mode.FP8 = "FP8"
            mock_quant_mode.NONE = "NONE"

            _initialize_tp_communicators(mock_gpt_config, micro_batch_size=4)

            mock_init_ub.assert_called_once()
            call_args = mock_init_ub.call_args
            assert call_args[1]["quantization_modes"] == ["FP8"]

    @patch("transformer_engine.pytorch.module.base.initialize_ub")
    def test_te_version_1_9_0_fp8_disabled(self, mock_init_ub, mock_gpt_config):
        """Test TE version 1.9.0+ path with FP8 disabled."""
        mock_gpt_config.fp8 = None

        with patch("megatron.bridge.training.initialize.is_te_min_version", side_effect=lambda v: v == "1.9.0"):
            _initialize_tp_communicators(mock_gpt_config, micro_batch_size=4)

            mock_init_ub.assert_called_once()
            call_args = mock_init_ub.call_args
            assert call_args[1]["use_fp8"] is False
            assert call_args[1]["tp_size"] == mock_gpt_config.tensor_model_parallel_size
            assert call_args[1]["bootstrap_backend"] == mock_gpt_config.tp_comm_bootstrap_backend

    @patch("transformer_engine.pytorch.module.base.initialize_ub")
    def test_te_version_1_9_0_fp8_enabled(self, mock_init_ub, mock_gpt_config):
        """Test TE version 1.9.0+ path with FP8 enabled."""
        mock_gpt_config.fp8 = "e4m3"

        with patch("megatron.bridge.training.initialize.is_te_min_version", side_effect=lambda v: v == "1.9.0"):
            _initialize_tp_communicators(mock_gpt_config, micro_batch_size=4)

            mock_init_ub.assert_called_once()
            call_args = mock_init_ub.call_args
            assert call_args[1]["use_fp8"] is True

    @patch("transformer_engine.pytorch.module.base.initialize_ub")
    @patch("torch.distributed.new_group")
    def test_te_version_legacy_mpi_backend(self, mock_new_group, mock_init_ub, mock_gpt_config):
        """Test legacy TE version path with MPI backend."""
        mock_gpt_config.tp_comm_bootstrap_backend = "mpi"

        with (
            patch("megatron.bridge.training.initialize.is_te_min_version", return_value=False),
            patch("megatron.bridge.training.initialize.get_te_version", return_value="1.8.0"),
        ):
            _initialize_tp_communicators(mock_gpt_config, micro_batch_size=4)

            # Should not create new group for MPI backend
            mock_new_group.assert_called_once_with(backend="mpi")
            mock_init_ub.assert_called_once()
            call_args = mock_init_ub.call_args
            assert call_args[1]["use_fp8"] is False

    @patch("transformer_engine.pytorch.module.base.initialize_ub")
    @patch("torch.distributed.new_group")
    def test_te_version_legacy_non_mpi_backend_warning(self, mock_new_group, mock_init_ub, mock_gpt_config):
        """Test legacy TE version path with non-MPI backend shows warning."""
        mock_gpt_config.tp_comm_bootstrap_backend = "nccl"

        with (
            patch("megatron.bridge.training.initialize.is_te_min_version", return_value=False),
            patch("megatron.bridge.training.initialize.get_te_version", return_value="1.8.0"),
        ):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _initialize_tp_communicators(mock_gpt_config, micro_batch_size=4)

                # Check that warning was issued
                assert len(w) == 1
                assert "Transformer Engine v1.8.0 supports only MPI bootstrap backend" in str(w[0].message)

            # Should create MPI group for non-MPI backend
            mock_new_group.assert_called_once_with(backend="mpi")
            mock_init_ub.assert_called_once()

    @patch("transformer_engine.pytorch.module.base.initialize_ub")
    @patch("torch.distributed.new_group")
    def test_te_version_legacy_fp8_enabled(self, mock_new_group, mock_init_ub, mock_gpt_config):
        """Test legacy TE version path with FP8 enabled."""
        mock_gpt_config.fp8 = "e4m3"
        mock_gpt_config.tp_comm_bootstrap_backend = "mpi"

        with (
            patch("megatron.bridge.training.initialize.is_te_min_version", return_value=False),
            patch("megatron.bridge.training.initialize.get_te_version", return_value="1.8.0"),
        ):
            _initialize_tp_communicators(mock_gpt_config, micro_batch_size=4)

            mock_init_ub.assert_called_once()
            call_args = mock_init_ub.call_args
            assert call_args[1]["use_fp8"] is True

    @patch("transformer_engine.pytorch.module.base.initialize_ub")
    @patch("torch.distributed.new_group")
    def test_version_checking_logic(self, mock_new_group, mock_init_ub, mock_gpt_config):
        """Test that version checking logic works correctly."""
        # Test 2.7.0+ path
        with patch("megatron.bridge.training.initialize.is_te_min_version", side_effect=lambda v: v == "2.7.0"):
            _initialize_tp_communicators(mock_gpt_config, micro_batch_size=4)
            call_args = mock_init_ub.call_args
            assert "quantization_modes" in call_args[1]
            assert "use_fp8" not in call_args[1]

        # Test 1.9.0+ path
        with patch("megatron.bridge.training.initialize.is_te_min_version", side_effect=lambda v: v == "1.9.0"):
            _initialize_tp_communicators(mock_gpt_config, micro_batch_size=4)
            call_args = mock_init_ub.call_args
            assert "use_fp8" in call_args[1]
            assert "quantization_modes" not in call_args[1]

        # Test legacy path
        with patch("megatron.bridge.training.initialize.is_te_min_version", return_value=False):
            _initialize_tp_communicators(mock_gpt_config, micro_batch_size=4)
            call_args = mock_init_ub.call_args
            assert "use_fp8" in call_args[1]
            assert "quantization_modes" not in call_args[1]
            assert "bootstrap_backend" not in call_args[1]


class TestDistributedInitConfigFlightRecorder:
    """Test suite for DistributedInitConfig flight recorder fields."""

    def test_default_values(self):
        cfg = DistributedInitConfig()
        assert cfg.flight_recorder_dump_path is None
        assert cfg.flight_recorder_trace_buffer_size == 2000
        assert cfg.flight_recorder_dump_on_timeout is True
        assert cfg.flight_recorder_include_stack_trace is False
        assert cfg.flight_recorder_include_only_active is True
        assert cfg.flight_recorder_extra_dump_on_exec is True

    def test_custom_values(self):
        cfg = DistributedInitConfig(
            flight_recorder_dump_path="/tmp/fr_dump",
            flight_recorder_trace_buffer_size=5000,
            flight_recorder_dump_on_timeout=False,
            flight_recorder_include_stack_trace=True,
            flight_recorder_include_only_active=False,
            flight_recorder_extra_dump_on_exec=False,
        )
        assert cfg.flight_recorder_dump_path == "/tmp/fr_dump"
        assert cfg.flight_recorder_trace_buffer_size == 5000
        assert cfg.flight_recorder_dump_on_timeout is False
        assert cfg.flight_recorder_include_stack_trace is True
        assert cfg.flight_recorder_include_only_active is False
        assert cfg.flight_recorder_extra_dump_on_exec is False


class TestSetupFlightRecorderEnv:
    """Test suite for _setup_flight_recorder_env helper."""

    @pytest.fixture(autouse=True)
    def _clean_env(self):
        """Remove all flight-recorder env vars before and after each test."""
        saved = {}
        for var in _FR_ENV_VARS:
            if var in os.environ:
                saved[var] = os.environ.pop(var)
        yield
        for var in _FR_ENV_VARS:
            os.environ.pop(var, None)
        os.environ.update(saved)

    @patch("megatron.bridge.training.initialize.get_rank_safe", return_value=1)
    def test_no_path_no_env_is_noop(self, _mock_rank):
        """When no dump path is configured and no env vars are set, nothing happens."""
        cfg = DistributedInitConfig()
        _setup_flight_recorder_env(cfg)
        for var in _FR_ENV_VARS:
            assert var not in os.environ

    @patch("megatron.bridge.training.initialize.get_rank_safe", return_value=1)
    def test_config_path_sets_all_env_vars(self, _mock_rank):
        """When flight_recorder_dump_path is set in config, all env vars are populated."""
        cfg = DistributedInitConfig(flight_recorder_dump_path="/tmp/fr_test")
        _setup_flight_recorder_env(cfg)

        assert os.environ["TORCH_FR_DUMP_TEMP_FILE"] == "/tmp/fr_test"
        assert os.environ["TORCH_NCCL_DEBUG_INFO_TEMP_FILE"] == "/tmp/fr_test"
        assert os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] == "2000"
        assert os.environ["TORCH_NCCL_DUMP_ON_TIMEOUT"] == "1"
        assert os.environ["TORCH_INCLUDE_STACK_TRACE"] == "0"
        assert os.environ["TORCH_INCLUDE_ONLY_ACTIVE"] == "1"
        assert os.environ["TORCH_NCCL_EXTRA_DUMP_ON_EXEC"] == "1"

    @patch("megatron.bridge.training.initialize.get_rank_safe", return_value=1)
    def test_custom_config_values_reflected_in_env(self, _mock_rank):
        """Non-default config values are correctly converted to env var strings."""
        cfg = DistributedInitConfig(
            flight_recorder_dump_path="/data/traces",
            flight_recorder_trace_buffer_size=8000,
            flight_recorder_dump_on_timeout=False,
            flight_recorder_include_stack_trace=True,
            flight_recorder_include_only_active=False,
            flight_recorder_extra_dump_on_exec=False,
        )
        _setup_flight_recorder_env(cfg)

        assert os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] == "8000"
        assert os.environ["TORCH_NCCL_DUMP_ON_TIMEOUT"] == "0"
        assert os.environ["TORCH_INCLUDE_STACK_TRACE"] == "1"
        assert os.environ["TORCH_INCLUDE_ONLY_ACTIVE"] == "0"
        assert os.environ["TORCH_NCCL_EXTRA_DUMP_ON_EXEC"] == "0"

    @patch("megatron.bridge.training.initialize.get_rank_safe", return_value=1)
    def test_env_var_takes_precedence_over_config(self, _mock_rank):
        """Pre-existing env vars are preserved; config values are ignored with a warning."""
        os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "9999"
        cfg = DistributedInitConfig(flight_recorder_dump_path="/tmp/fr")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _setup_flight_recorder_env(cfg)

        assert os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] == "9999"
        warning_messages = [str(x.message) for x in w]
        assert any("TORCH_NCCL_TRACE_BUFFER_SIZE" in m and "9999" in m for m in warning_messages)

    @patch("megatron.bridge.training.initialize.get_rank_safe", return_value=1)
    def test_env_path_precedence_over_config_path(self, _mock_rank):
        """TORCH_FR_DUMP_TEMP_FILE env var takes precedence over config dump_path."""
        os.environ["TORCH_FR_DUMP_TEMP_FILE"] = "/env/path"
        cfg = DistributedInitConfig(flight_recorder_dump_path="/config/path")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _setup_flight_recorder_env(cfg)

        assert os.environ["TORCH_FR_DUMP_TEMP_FILE"] == "/env/path"
        assert os.environ["TORCH_NCCL_DEBUG_INFO_TEMP_FILE"] == "/env/path"

    @patch("megatron.bridge.training.initialize.get_rank_safe", return_value=1)
    def test_nccl_debug_info_env_triggers_setup(self, _mock_rank):
        """TORCH_NCCL_DEBUG_INFO_TEMP_FILE alone triggers flight recorder setup."""
        os.environ["TORCH_NCCL_DEBUG_INFO_TEMP_FILE"] = "/env/nccl_path"
        cfg = DistributedInitConfig()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _setup_flight_recorder_env(cfg)

        assert os.environ["TORCH_FR_DUMP_TEMP_FILE"] == "/env/nccl_path"
        assert os.environ["TORCH_NCCL_DEBUG_INFO_TEMP_FILE"] == "/env/nccl_path"
        assert "TORCH_NCCL_TRACE_BUFFER_SIZE" in os.environ

    @patch("megatron.bridge.training.initialize.get_rank_safe", return_value=1)
    def test_fr_dump_env_takes_precedence_over_nccl_debug_info_env(self, _mock_rank):
        """TORCH_FR_DUMP_TEMP_FILE has higher priority than TORCH_NCCL_DEBUG_INFO_TEMP_FILE."""
        os.environ["TORCH_FR_DUMP_TEMP_FILE"] = "/env/fr"
        os.environ["TORCH_NCCL_DEBUG_INFO_TEMP_FILE"] = "/env/nccl"
        cfg = DistributedInitConfig()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _setup_flight_recorder_env(cfg)

        assert os.environ["TORCH_FR_DUMP_TEMP_FILE"] == "/env/fr"
        assert os.environ["TORCH_NCCL_DEBUG_INFO_TEMP_FILE"] == "/env/nccl"

    @patch("megatron.bridge.training.initialize.get_rank_safe", return_value=1)
    def test_warning_for_every_preexisting_var(self, _mock_rank):
        """A warning is emitted for each pre-existing env var."""
        for var in _FR_ENV_VARS:
            os.environ[var] = "preset"
        cfg = DistributedInitConfig(flight_recorder_dump_path="/tmp/fr")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _setup_flight_recorder_env(cfg)

        warning_messages = [str(x.message) for x in w]
        for var in _FR_ENV_VARS:
            assert any(var in m for m in warning_messages), f"No warning for {var}"

    @patch("megatron.bridge.training.initialize.get_rank_safe", return_value=0)
    def test_rank_0_prints_env_vars(self, _mock_rank, capsys):
        """Rank 0 prints the flight recorder env var summary."""
        cfg = DistributedInitConfig(flight_recorder_dump_path="/tmp/fr")
        _setup_flight_recorder_env(cfg)
        captured = capsys.readouterr()
        assert "Flight recorder env vars:" in captured.out
        assert "TORCH_FR_DUMP_TEMP_FILE=/tmp/fr" in captured.out
        assert "TORCH_NCCL_TRACE_BUFFER_SIZE=2000" in captured.out

    @patch("megatron.bridge.training.initialize.get_rank_safe", return_value=1)
    def test_non_rank_0_does_not_print(self, _mock_rank, capsys):
        """Non-rank-0 processes do not print the flight recorder summary."""
        cfg = DistributedInitConfig(flight_recorder_dump_path="/tmp/fr")
        _setup_flight_recorder_env(cfg)
        captured = capsys.readouterr()
        assert "Flight recorder env vars:" not in captured.out
