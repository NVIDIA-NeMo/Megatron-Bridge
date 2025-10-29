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

import tempfile
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.training.initialize import _initialize_tp_communicators


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
