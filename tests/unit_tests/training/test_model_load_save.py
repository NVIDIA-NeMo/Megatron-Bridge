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
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.bridge.training.model_load_save import (
    dtype_from_hf,
    dtype_from_str,
    load_megatron_model,
    megatron_cpu_init_context,
    save_megatron_model,
    temporary_distributed_context,
    torch_dtype_from_mcore_config,
)


class TestTorchDtypeFromMcoreConfig:
    """Test torch_dtype_from_mcore_config function."""

    def test_torch_dtype_from_mcore_config_bf16(self):
        """Test bf16 configuration conversion."""
        config = Mock()
        config.bf16 = True
        config.fp16 = False

        result = torch_dtype_from_mcore_config(config)
        assert result == torch.bfloat16

    def test_torch_dtype_from_mcore_config_fp16(self):
        """Test fp16 configuration conversion."""
        config = Mock()
        config.bf16 = False
        config.fp16 = True

        result = torch_dtype_from_mcore_config(config)
        assert result == torch.float16

    def test_torch_dtype_from_mcore_config_fp32_default(self):
        """Test fp32 default configuration conversion."""
        config = Mock()
        config.bf16 = False
        config.fp16 = False

        result = torch_dtype_from_mcore_config(config)
        assert result == torch.float32

    def test_torch_dtype_from_mcore_config_no_attributes(self):
        """Test configuration without bf16/fp16 attributes defaults to fp32."""
        config = Mock(spec=[])  # Mock with no attributes

        result = torch_dtype_from_mcore_config(config)
        assert result == torch.float32

    def test_torch_dtype_from_mcore_config_bf16_priority(self):
        """Test that bf16 takes priority over fp16 when both are True."""
        config = Mock()
        config.bf16 = True
        config.fp16 = True

        result = torch_dtype_from_mcore_config(config)
        assert result == torch.bfloat16


class TestMegatronCpuInitContext:
    """Test megatron_cpu_init_context context manager."""

    def test_megatron_cpu_init_context_preserves_original_value(self):
        """Test that the context manager preserves original use_cpu_initialization value."""
        config = Mock()
        config.use_cpu_initialization = False

        with megatron_cpu_init_context(config):
            assert config.use_cpu_initialization is True

        assert config.use_cpu_initialization is False

    def test_megatron_cpu_init_context_with_already_true(self):
        """Test context manager when use_cpu_initialization is already True."""
        config = Mock()
        config.use_cpu_initialization = True

        with megatron_cpu_init_context(config):
            assert config.use_cpu_initialization is True

        assert config.use_cpu_initialization is True

    def test_megatron_cpu_init_context_exception_handling(self):
        """Test that the context manager restores value even when exception occurs."""
        config = Mock()
        config.use_cpu_initialization = False

        try:
            with megatron_cpu_init_context(config):
                assert config.use_cpu_initialization is True
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert config.use_cpu_initialization is False


class TestTemporaryDistributedContext:
    """Test temporary_distributed_context context manager."""

    @patch("megatron.bridge.training.model_load_save.dist")
    @patch("megatron.bridge.training.model_load_save.parallel_state")
    @patch("megatron.bridge.training.model_load_save.socket")
    @patch("megatron.bridge.training.model_load_save.os")
    def test_temporary_distributed_context_gloo(self, mock_os, mock_socket, mock_parallel_state, mock_dist):
        """Test temporary distributed context with gloo backend."""
        # Mock environment to not have MASTER_ADDR and MASTER_PORT
        mock_os.environ = {}

        # Mock socket for port selection
        mock_socket_instance = Mock()
        mock_socket_instance.getsockname.return_value = ("localhost", 12345)
        mock_socket.socket.return_value.__enter__.return_value = mock_socket_instance

        with temporary_distributed_context(backend="gloo"):
            pass

        mock_dist.init_process_group.assert_called_once_with(
            backend="gloo", init_method="tcp://localhost:12345", world_size=1, rank=0
        )
        mock_parallel_state.initialize_model_parallel.assert_called_once()
        mock_parallel_state.destroy_model_parallel.assert_called_once()
        mock_dist.destroy_process_group.assert_called_once()

    @patch("megatron.bridge.training.model_load_save.dist")
    @patch("megatron.bridge.training.model_load_save.parallel_state")
    @patch("megatron.bridge.training.model_load_save.os")
    def test_temporary_distributed_context_with_env_vars(self, mock_os, mock_parallel_state, mock_dist):
        """Test temporary distributed context when env vars are already set."""
        mock_os.environ = {"MASTER_ADDR": "localhost", "MASTER_PORT": "12345"}

        with temporary_distributed_context(backend="gloo"):
            pass

        mock_dist.init_process_group.assert_called_once_with(backend="gloo", init_method=None, world_size=1, rank=0)

    @patch("megatron.bridge.training.model_load_save.dist")
    @patch("megatron.bridge.training.model_load_save.parallel_state")
    @patch("megatron.bridge.training.model_load_save.socket")
    @patch("megatron.bridge.training.model_load_save.os")
    @patch("megatron.core.tensor_parallel.model_parallel_cuda_manual_seed")
    def test_temporary_distributed_context_nccl(self, mock_seed, mock_os, mock_socket, mock_parallel_state, mock_dist):
        """Test temporary distributed context with nccl backend."""
        # Mock environment to not have MASTER_ADDR and MASTER_PORT
        mock_os.environ = {}

        # Mock socket for port selection
        mock_socket_instance = Mock()
        mock_socket_instance.getsockname.return_value = ("localhost", 12345)
        mock_socket.socket.return_value.__enter__.return_value = mock_socket_instance

        with temporary_distributed_context(backend="nccl"):
            pass

        mock_dist.init_process_group.assert_called_once_with(
            backend="nccl", init_method="tcp://localhost:12345", world_size=1, rank=0
        )
        mock_seed.assert_called_once_with(0)
        mock_parallel_state.initialize_model_parallel.assert_called_once()
        mock_parallel_state.destroy_model_parallel.assert_called_once()
        mock_dist.destroy_process_group.assert_called_once()


class TestLoadMegatronModel:
    """Test load_megatron_model function."""

    @patch("megatron.bridge.training.model_load_save.temporary_distributed_context")
    @patch("megatron.bridge.training.model_load_save.TorchDistLoadShardedStrategy")
    @patch("megatron.bridge.training.model_load_save.megatron_cpu_init_context")
    @patch("megatron.bridge.training.model_load_save.dist")
    def test_load_megatron_model_return_state_dict(
        self, mock_dist, mock_cpu_context, mock_strategy_class, mock_temp_dist
    ):
        """Test loading model and returning state dict."""
        # Setup mocks
        mock_dist.is_available.return_value = False
        mock_dist.is_initialized.return_value = False

        mock_model = Mock()
        mock_model.sharded_state_dict.return_value = {"layer.weight": torch.randn(2, 2)}

        mock_config = Mock()
        mock_config.params_dtype = torch.float32
        mock_config.bf16 = True
        mock_config.fp16 = False
        mock_config.provide.return_value = mock_model
        mock_config.use_cpu_initialization = False

        mock_strategy = Mock()
        mock_strategy.load.return_value = {"layer.weight": torch.randn(2, 2)}
        mock_strategy_class.return_value = mock_strategy

        # Test
        with tempfile.TemporaryDirectory() as temp_dir:
            result = load_megatron_model(
                dist_ckpt_folder=Path(temp_dir), model_cfg=mock_config, return_state_dict=True, use_cpu_init=True
            )

        # Assertions
        assert isinstance(result, dict)
        mock_cpu_context.assert_called_once()
        mock_strategy.load.assert_called_once()

    @patch("megatron.bridge.training.model_load_save.temporary_distributed_context")
    @patch("megatron.bridge.training.model_load_save.TorchDistLoadShardedStrategy")
    @patch("megatron.bridge.training.model_load_save.megatron_cpu_init_context")
    @patch("megatron.bridge.training.model_load_save.dist")
    def test_load_megatron_model_return_model(self, mock_dist, mock_cpu_context, mock_strategy_class, mock_temp_dist):
        """Test loading model and returning model instance."""
        # Setup mocks
        mock_dist.is_available.return_value = False
        mock_dist.is_initialized.return_value = False

        mock_model = Mock()
        mock_model.sharded_state_dict.return_value = {"layer.weight": torch.randn(2, 2)}

        mock_config = Mock()
        mock_config.params_dtype = torch.float32
        mock_config.bf16 = False
        mock_config.fp16 = False
        mock_config.provide.return_value = mock_model
        mock_config.use_cpu_initialization = False

        mock_strategy = Mock()
        mock_strategy.load.return_value = {"layer.weight": torch.randn(2, 2)}
        mock_strategy_class.return_value = mock_strategy

        # Test
        with tempfile.TemporaryDirectory() as temp_dir:
            result = load_megatron_model(
                dist_ckpt_folder=Path(temp_dir), model_cfg=mock_config, return_state_dict=False, use_cpu_init=True
            )

        # Assertions
        assert result == mock_model
        mock_model.load_state_dict.assert_called_once()

    @patch("megatron.bridge.training.model_load_save.dist")
    def test_load_megatron_model_skip_temp_dist_context(self, mock_dist):
        """Test loading model when distributed is already initialized."""
        mock_dist.is_available.return_value = True
        mock_dist.is_initialized.return_value = True

        mock_model = Mock()
        mock_config = Mock()
        mock_config.params_dtype = torch.bfloat16
        mock_config.bf16 = True
        mock_config.fp16 = False
        mock_config.provide.return_value = mock_model
        mock_config.use_cpu_initialization = False

        with patch("megatron.bridge.training.model_load_save.TorchDistLoadShardedStrategy") as mock_strategy_class:
            mock_strategy = Mock()
            mock_strategy.load.return_value = {"layer.weight": torch.randn(2, 2)}
            mock_strategy_class.return_value = mock_strategy

            with patch("megatron.bridge.training.model_load_save.megatron_cpu_init_context"):
                with tempfile.TemporaryDirectory() as temp_dir:
                    result = load_megatron_model(
                        dist_ckpt_folder=Path(temp_dir), model_cfg=mock_config, use_cpu_init=True
                    )

        assert result == mock_model


class TestSaveMegatronModel:
    """Test save_megatron_model function."""

    @patch("megatron.bridge.training.model_load_save.save_checkpoint")
    @patch("megatron.bridge.training.model_load_save.get_model_config")
    @patch("megatron.bridge.training.model_load_save.GlobalState")
    @patch("megatron.bridge.training.model_load_save.ConfigContainer")
    @patch("megatron.bridge.training.model_load_save.OptimizerConfig")
    @patch("megatron.bridge.training.model_load_save.LoggerConfig")
    @patch("megatron.bridge.training.model_load_save.CheckpointConfig")
    def test_save_megatron_model(
        self,
        mock_ckpt_config,
        mock_logger_config,
        mock_opt_config,
        mock_config_container,
        mock_global_state,
        mock_get_model_config,
        mock_save_checkpoint,
    ):
        """Test saving megatron model."""
        # Setup mocks
        mock_model = Mock()

        class MockModelConfig(ModelProviderMixin, Mock):
            def provide(self, pre_process=None, post_process=None, vp_stage=None):
                return Mock()

        mock_model_config = MockModelConfig()
        mock_get_model_config.return_value = mock_model_config

        mock_state = Mock()
        mock_global_state.return_value = mock_state

        # Test
        with tempfile.TemporaryDirectory() as temp_dir:
            save_megatron_model([mock_model], temp_dir, ckpt_format="torch_dist")

        # Assertions
        mock_get_model_config.assert_called_once_with(mock_model)
        mock_global_state.assert_called_once()
        mock_save_checkpoint.assert_called_once_with(
            state=mock_state,
            model=[mock_model],
            optimizer=None,
            opt_param_scheduler=None,
            num_floating_point_operations_so_far=0,
        )


class TestDtypeFromStr:
    """Test dtype_from_str function."""

    @pytest.mark.parametrize(
        "dtype_str,expected",
        [
            ("float16", torch.float16),
            ("fp16", torch.float16),
            ("16", torch.float16),
            ("16-mixed", torch.float16),
            ("bfloat16", torch.bfloat16),
            ("bf16-mixed", torch.bfloat16),
            ("float32", torch.float32),
            ("unknown", torch.float32),
            ("", torch.float32),
        ],
    )
    def test_dtype_from_str_valid_inputs(self, dtype_str, expected):
        """Test dtype conversion from string."""
        result = dtype_from_str(dtype_str)
        assert result == expected

    def test_dtype_from_str_invalid_type(self):
        """Test dtype conversion with non-string input."""
        with pytest.raises(TypeError, match="Expected str, got"):
            dtype_from_str(123)

    def test_dtype_from_str_none_input(self):
        """Test dtype conversion with None input."""
        with pytest.raises(TypeError, match="Expected str, got"):
            dtype_from_str(None)


class TestDtypeFromHf:
    """Test dtype_from_hf function."""

    def test_dtype_from_hf_torch_dtype_attribute(self):
        """Test extracting torch.dtype from HF config with torch.dtype attribute."""
        config = Mock()
        config.torch_dtype = torch.bfloat16

        result = dtype_from_hf(config)
        assert result == torch.bfloat16

    def test_dtype_from_hf_string_attribute(self):
        """Test extracting torch.dtype from HF config with string attribute."""
        config = Mock()
        config.torch_dtype = "fp16"

        result = dtype_from_hf(config)
        assert result == torch.float16

    def test_dtype_from_hf_missing_attribute(self):
        """Test error when HF config missing torch_dtype attribute."""
        config = Mock(spec=[])  # Mock with no attributes

        with pytest.raises(AttributeError, match="Expected config to have attr `torch_dtype`"):
            dtype_from_hf(config)

    def test_dtype_from_hf_invalid_type(self):
        """Test error when torch_dtype is neither string nor torch.dtype."""
        config = Mock()
        config.torch_dtype = 123

        with pytest.raises(ValueError, match="torch_dtype is not of type str/torch.dtype"):
            dtype_from_hf(config)
