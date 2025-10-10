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
    load_tokenizer,
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
    @patch("megatron.bridge.training.checkpointing._load_model_weights_from_checkpoint")
    @patch("megatron.bridge.utils.instantiate_utils.instantiate")
    @patch("megatron.bridge.training.checkpointing.read_run_config")
    @patch("megatron.bridge.training.checkpointing.get_checkpoint_run_config_filename")
    @patch("megatron.bridge.training.model_load_save.megatron_cpu_init_context")
    @patch("megatron.bridge.training.model_load_save.dist")
    def test_load_mbridge_saved_model(
        self,
        mock_dist,
        mock_cpu_context,
        mock_run_config_fname,
        mock_run_config,
        mock_instantiate,
        mock_load_weights,
        mock_temp_dist,
    ):
        # Setup mocks
        mock_dist.is_available.return_value = False
        mock_dist.is_initialized.return_value = False

        mock_run_cfg_dict = {"model": {"tensor_model_parallel_size": 1}}
        mock_run_config.return_value = mock_run_cfg_dict

        mock_model = Mock()
        mock_model_cfg = Mock(spec=ModelProviderMixin)
        mock_model_cfg.params_dtype = torch.float32
        mock_model_cfg.bf16 = True
        mock_model_cfg.fp16 = False
        mock_model_cfg.provide_distributed_model.return_value = [mock_model]
        mock_model_cfg.use_cpu_initialization = False

        mock_instantiate.return_value = mock_model_cfg
        expected_result = {"layer.weight": torch.randn(2, 2)}
        mock_load_weights.return_value = expected_result

        with tempfile.TemporaryDirectory() as ckpt_path:
            config_file = Path(ckpt_path) / "run_config.yaml"
            config_file.touch()
            result = load_megatron_model(ckpt_path, return_state_dict=True, use_cpu_init=True)

        assert isinstance(result, dict)
        assert result == expected_result
        mock_run_config_fname.assert_called_once_with(ckpt_path)
        mock_run_config.assert_called_once()
        mock_instantiate.assert_called_once_with(mock_run_cfg_dict["model"])
        mock_cpu_context.assert_called_once()
        mock_model_cfg.provide_distributed_model.assert_called_once()
        mock_load_weights.assert_called_once_with(ckpt_path, [mock_model], return_state_dict=True)
        assert mock_model_cfg.params_dtype == torch.bfloat16

        result = load_megatron_model(ckpt_path, return_state_dict=False, use_cpu_init=True)
        assert result == [mock_model]
        mock_load_weights.assert_called_with(ckpt_path, [mock_model], return_state_dict=False)

    @pytest.mark.parametrize("model_type", ["gpt", "mamba", "resnet"])
    @patch("megatron.bridge.training.model_load_save.temporary_distributed_context")
    @patch("megatron.bridge.training.mlm_compat.model._mamba_provider")
    @patch("megatron.bridge.training.mlm_compat.model._gpt_provider")
    @patch("megatron.bridge.training.mlm_compat.model._get_model")
    @patch("megatron.bridge.training.checkpointing._load_model_weights_from_checkpoint")
    @patch("megatron.bridge.training.mlm_compat.arguments._transformer_config_from_args")
    @patch("megatron.bridge.training.mlm_compat.arguments._load_args_from_checkpoint")
    @patch("megatron.bridge.training.model_load_save.build_tokenizer")
    @patch("megatron.bridge.training.mlm_compat.arguments._tokenizer_config_from_args")
    @patch("megatron.bridge.training.model_load_save.megatron_cpu_init_context")
    @patch("megatron.bridge.training.model_load_save.dist")
    def test_load_mlm_saved_model(
        self,
        mock_dist,
        mock_cpu_context,
        mock_tokenizer_config_from_args,
        mock_build_tokenizer,
        mock_load_args,
        mock_transformer_cfg,
        mock_load_weights,
        mock_get_model,
        mock_gpt_provider,
        mock_mamba_provider,
        mock_temp_dist,
        model_type,
    ):
        # Setup mocks
        mock_dist.is_available.return_value = False
        mock_dist.is_initialized.return_value = False

        ckpt_path = "/path/to/mock/dist_checkpoint"
        mock_args = Mock()
        mock_args.vocab_size = 32000  # Add vocab_size for padded vocab calculation
        mock_args.make_vocab_size_divisible_by = 128  # Add for padded vocab calculation
        mock_args.tensor_model_parallel_size = 1  # Add for padded vocab calculation
        mock_load_args.return_value = mock_args

        # Setup tokenizer mocks for MLM compat path
        mock_tokenizer_cfg = Mock()
        mock_tokenizer_config_from_args.return_value = mock_tokenizer_cfg

        mock_tokenizer = Mock()
        mock_tokenizer.vocab_size = 32000  # Unpadded vocab size for calculate_padded_vocab_size
        mock_build_tokenizer.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model_cfg = Mock()
        mock_model_cfg.params_dtype = torch.float32
        mock_model_cfg.bf16 = True
        mock_model_cfg.fp16 = False
        mock_model_cfg.use_cpu_initialization = False
        mock_model_cfg.make_vocab_size_divisible_by = 128  # Add for padded vocab calculation
        mock_model_cfg.tensor_model_parallel_size = 1  # Add for padded vocab calculation
        mock_provider = None
        if model_type == "gpt":
            mock_provider = mock_gpt_provider
        elif model_type == "mamba":
            mock_provider = mock_mamba_provider
        mock_get_model.return_value = [mock_model]

        mock_transformer_cfg.return_value = mock_model_cfg
        expected_result = {"layer.weight": torch.randn(2, 2)}
        mock_load_weights.return_value = expected_result

        if model_type in ("gpt", "mamba"):
            result = load_megatron_model(ckpt_path, model_type=model_type, return_state_dict=True, use_cpu_init=True)

            assert isinstance(result, dict)
            assert result == expected_result
            mock_load_args.assert_called_once_with(ckpt_path)
            mock_transformer_cfg.assert_called_once_with(mock_args)
            mock_tokenizer_config_from_args.assert_called_once_with(mock_args)
            mock_build_tokenizer.assert_called_once_with(mock_tokenizer_cfg)
            # Verify padded vocab size was calculated and set
            assert mock_args.padded_vocab_size == 32000  # 32000 is already divisible by 128, so no padding
            mock_cpu_context.assert_called_once()
            mock_get_model.assert_called_once_with(mock_args, mock_provider, mock_model_cfg)
            mock_load_weights.assert_called_once_with(ckpt_path, [mock_model], return_state_dict=True)
            assert mock_model_cfg.params_dtype == torch.bfloat16

            result = load_megatron_model(ckpt_path, model_type=model_type, return_state_dict=False, use_cpu_init=True)
            assert result == [mock_model]
            mock_load_weights.assert_called_with(ckpt_path, [mock_model], return_state_dict=False)
        else:
            with pytest.raises(AssertionError, match=f"model type {model_type} not supported."):
                load_megatron_model(ckpt_path, model_type=model_type, return_state_dict=True, use_cpu_init=True)

    @patch("megatron.bridge.training.model_load_save.temporary_distributed_context")
    @patch("megatron.bridge.training.checkpointing._load_model_weights_from_checkpoint")
    @patch("megatron.bridge.utils.instantiate_utils.instantiate")
    @patch("megatron.bridge.training.checkpointing.read_run_config")
    @patch("megatron.bridge.training.checkpointing.get_checkpoint_run_config_filename")
    @patch("megatron.bridge.training.model_load_save.megatron_cpu_init_context")
    @patch("megatron.bridge.training.model_load_save.dist")
    def test_load_megatron_model_skip_temp_dist_context(
        self,
        mock_dist,
        mock_cpu_context,
        mock_run_config_fname,
        mock_run_config,
        mock_instantiate,
        mock_load_weights,
        mock_temp_dist,
    ):
        """Test loading model when distributed is already initialized."""

        # Setup mocks
        mock_dist.is_available.return_value = True
        mock_dist.is_initialized.return_value = True

        mock_run_cfg_dict = {"model": {"tensor_model_parallel_size": 1}}
        mock_run_config.return_value = mock_run_cfg_dict

        mock_model = Mock()
        mock_model_cfg = Mock(spec=ModelProviderMixin)
        mock_model_cfg.params_dtype = torch.bfloat16
        mock_model_cfg.bf16 = True
        mock_model_cfg.fp16 = False
        mock_model_cfg.provide_distributed_model.return_value = mock_model
        mock_model_cfg.use_cpu_initialization = False

        mock_instantiate.return_value = mock_model_cfg

        with tempfile.TemporaryDirectory() as ckpt_path:
            config_file = Path(ckpt_path) / "run_config.yaml"
            config_file.touch()
            result = load_megatron_model(ckpt_path, use_cpu_init=True)

        assert result == mock_model
        mock_temp_dist.assert_not_called()

    @patch("megatron.bridge.training.model_load_save.temporary_distributed_context")
    @patch("megatron.bridge.training.post_training.checkpointing.load_modelopt_state")
    @patch("megatron.bridge.training.post_training.checkpointing.has_modelopt_state")
    @patch("megatron.bridge.training.checkpointing._load_model_weights_from_checkpoint")
    @patch("megatron.bridge.utils.instantiate_utils.instantiate")
    @patch("megatron.bridge.training.checkpointing.read_run_config")
    @patch("megatron.bridge.training.checkpointing.get_checkpoint_run_config_filename")
    @patch("megatron.bridge.training.model_load_save.megatron_cpu_init_context")
    @patch("megatron.bridge.training.model_load_save.dist")
    def test_load_mbridge_saved_model_with_modelopt_state(
        self,
        mock_dist,
        mock_cpu_context,
        mock_run_config_fname,
        mock_run_config,
        mock_instantiate,
        mock_load_weights,
        mock_has_modelopt_state,
        mock_load_modelopt_state,
        mock_temp_dist,
    ):
        """Test loading model when modelopt state exists and model supports it."""
        # Setup mocks
        mock_dist.is_available.return_value = False
        mock_dist.is_initialized.return_value = False

        mock_run_cfg_dict = {"model": {"tensor_model_parallel_size": 1}}
        mock_run_config.return_value = mock_run_cfg_dict

        mock_model = Mock()
        mock_model_cfg = Mock(spec=ModelProviderMixin)
        mock_model_cfg.params_dtype = torch.float32
        mock_model_cfg.bf16 = True
        mock_model_cfg.fp16 = False
        mock_model_cfg.provide_distributed_model.return_value = [mock_model]
        mock_model_cfg.use_cpu_initialization = False
        mock_model_cfg.restore_modelopt_state = False  # Initially False

        mock_instantiate.return_value = mock_model_cfg
        expected_result = {"layer.weight": torch.randn(2, 2)}
        mock_load_weights.return_value = expected_result

        # Mock modelopt state exists
        mock_has_modelopt_state.return_value = True

        with tempfile.TemporaryDirectory() as ckpt_path:
            config_file = Path(ckpt_path) / "run_config.yaml"
            config_file.touch()
            result = load_megatron_model(ckpt_path, return_state_dict=True, use_cpu_init=True)

        # Verify modelopt state was detected and set
        mock_has_modelopt_state.assert_called_once_with(ckpt_path)
        assert mock_model_cfg.restore_modelopt_state is True

        # Verify modelopt state was loaded
        mock_load_modelopt_state.assert_called_once_with([mock_model], ckpt_path)

        assert isinstance(result, dict)
        assert result == expected_result

    @patch("megatron.bridge.training.model_load_save.temporary_distributed_context")
    @patch("megatron.bridge.training.post_training.checkpointing.load_modelopt_state")
    @patch("megatron.bridge.training.post_training.checkpointing.has_modelopt_state")
    @patch("megatron.bridge.training.checkpointing._load_model_weights_from_checkpoint")
    @patch("megatron.bridge.training.mlm_compat.model._get_model")
    @patch("megatron.bridge.training.model_load_save.build_tokenizer")
    @patch("megatron.bridge.training.mlm_compat.arguments._transformer_config_from_args")
    @patch("megatron.bridge.training.mlm_compat.arguments._load_args_from_checkpoint")
    @patch("megatron.bridge.training.model_load_save.file_exists")
    @patch("megatron.bridge.training.model_load_save.megatron_cpu_init_context")
    @patch("megatron.bridge.training.model_load_save.dist")
    def test_load_mlm_saved_model_without_modelopt_support(
        self,
        mock_dist,
        mock_cpu_context,
        mock_file_exists,
        mock_load_args,
        mock_transformer_config,
        mock_build_tokenizer,
        mock_get_model,
        mock_load_weights,
        mock_has_modelopt_state,
        mock_load_modelopt_state,
        mock_temp_dist,
    ):
        """Test loading MLM model when modelopt state exists but TransformerConfig doesn't support it."""
        # Setup mocks
        mock_dist.is_available.return_value = False
        mock_dist.is_initialized.return_value = False

        # Mock file_exists to return False for run_config (MLM checkpoint)
        mock_file_exists.return_value = False

        # Mock MLM args loading
        mock_args = Mock()
        mock_args.make_vocab_size_divisible_by = 128
        mock_load_args.return_value = mock_args

        # Create a TransformerConfig mock (doesn't have restore_modelopt_state)
        from megatron.bridge.models.transformer_config import TransformerConfig

        mock_model_cfg = Mock(spec=TransformerConfig)
        mock_model_cfg.params_dtype = torch.float32
        mock_model_cfg.bf16 = True
        mock_model_cfg.fp16 = False
        mock_model_cfg.use_cpu_initialization = False
        mock_model_cfg.tensor_model_parallel_size = 1

        mock_transformer_config.return_value = mock_model_cfg

        # Mock tokenizer creation
        mock_tokenizer = Mock()
        mock_tokenizer.vocab_size = 50000  # Set a realistic vocab size
        mock_build_tokenizer.return_value = mock_tokenizer

        # Mock model creation
        mock_model = Mock()
        mock_get_model.return_value = [mock_model]

        expected_result = {"layer.weight": torch.randn(2, 2)}
        mock_load_weights.return_value = expected_result

        # Mock modelopt state exists
        mock_has_modelopt_state.return_value = True

        with tempfile.TemporaryDirectory() as ckpt_path:
            result = load_megatron_model(ckpt_path, model_type="gpt", return_state_dict=True, use_cpu_init=True)

        # Verify modelopt state was detected but not set (no attribute on TransformerConfig)
        mock_has_modelopt_state.assert_called_once_with(ckpt_path)
        # TransformerConfig doesn't have restore_modelopt_state, so hasattr returns False
        assert not hasattr(mock_model_cfg, "restore_modelopt_state")

        # Verify modelopt state was NOT loaded (getattr returns False for missing attribute)
        mock_load_modelopt_state.assert_not_called()

        assert isinstance(result, dict)
        assert result == expected_result

    @patch("megatron.bridge.training.model_load_save.build_and_load_model")
    @patch("megatron.bridge.training.model_load_save.load_model_config")
    def test_load_megatron_model_resets_defaults(self, mock_load_model_config, mock_build_and_load):
        """Verify single-GPU default resets are applied before building the model."""
        # Prepare a config object with non-default values that should be reset
        cfg = Mock()
        cfg.tensor_model_parallel_size = 8
        cfg.pipeline_model_parallel_size = 4
        cfg.context_parallel_size = 2
        cfg.expert_model_parallel_size = 2
        cfg.expert_tensor_parallel_size = 2
        cfg.moe_extended_tp = True
        cfg.sequence_parallel = True
        cfg.virtual_pipeline_model_parallel_size = 2
        cfg.hierarchical_context_parallel_sizes = [2, 2]

        mock_load_model_config.return_value = (cfg, None)
        sentinel = object()
        mock_build_and_load.return_value = sentinel

        result = load_megatron_model("/ckpt", model_type=None, return_state_dict=False, use_cpu_init=True)

        # Ensure build_and_load_model was called and returned
        assert result is sentinel

        # After resets (no overrides), the following should hold
        assert cfg.tensor_model_parallel_size == 1
        assert cfg.pipeline_model_parallel_size == 1
        assert cfg.context_parallel_size == 1
        assert cfg.expert_model_parallel_size == 1
        assert cfg.expert_tensor_parallel_size == 1
        assert cfg.moe_extended_tp is False
        assert cfg.sequence_parallel is False
        assert cfg.virtual_pipeline_model_parallel_size is None
        assert cfg.hierarchical_context_parallel_sizes is None

    @patch("megatron.bridge.training.model_load_save.build_and_load_model")
    @patch("megatron.bridge.training.model_load_save.load_model_config")
    def test_load_megatron_model_applies_overrides(self, mock_load_model_config, mock_build_and_load):
        """Verify mp_overrides entries are applied to the config."""
        cfg = Mock()
        # Start with defaults to make verification straightforward
        cfg.tensor_model_parallel_size = 1
        cfg.pipeline_model_parallel_size = 1
        cfg.context_parallel_size = 1
        cfg.expert_model_parallel_size = 1
        cfg.expert_tensor_parallel_size = 1
        cfg.moe_extended_tp = False
        cfg.sequence_parallel = False
        cfg.virtual_pipeline_model_parallel_size = None
        cfg.hierarchical_context_parallel_sizes = None

        mock_load_model_config.return_value = (cfg, None)
        mock_build_and_load.return_value = Mock()

        overrides = {
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 3,
            "sequence_parallel": True,
            "virtual_pipeline_model_parallel_size": 4,
        }

        _ = load_megatron_model("/ckpt", mp_overrides=overrides)

        assert cfg.tensor_model_parallel_size == 2
        assert cfg.pipeline_model_parallel_size == 3
        assert cfg.sequence_parallel is True
        assert cfg.virtual_pipeline_model_parallel_size == 4


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

    @patch("megatron.bridge.training.model_load_save.save_checkpoint")
    @patch("megatron.bridge.training.model_load_save.get_model_config")
    @patch("megatron.bridge.training.model_load_save.GlobalState")
    @patch("megatron.bridge.training.model_load_save.ConfigContainer")
    @patch("megatron.bridge.training.model_load_save.OptimizerConfig")
    @patch("megatron.bridge.training.model_load_save.LoggerConfig")
    @patch("megatron.bridge.training.model_load_save.CheckpointConfig")
    def test_save_megatron_model_with_tokenizer(
        self,
        mock_ckpt_config,
        mock_logger_config,
        mock_opt_config,
        mock_config_container,
        mock_global_state,
        mock_get_model_config,
        mock_save_checkpoint,
    ):
        """Test saving megatron model with tokenizer configuration."""
        # Setup mocks
        mock_model = Mock()

        class MockModelConfig(ModelProviderMixin, Mock):
            def provide(self, pre_process=None, post_process=None, vp_stage=None):
                return Mock()

        mock_model_config = MockModelConfig()
        mock_get_model_config.return_value = mock_model_config

        mock_state = Mock()
        mock_global_state.return_value = mock_state

        # Mock the ConfigContainer to capture tokenizer config
        mock_container_instance = Mock()
        mock_config_container.return_value = mock_container_instance

        # Test with tokenizer path
        with tempfile.TemporaryDirectory() as temp_dir:
            save_megatron_model(
                [mock_model], temp_dir, ckpt_format="torch_dist", hf_tokenizer_path="meta-llama/Meta-Llama-3-8B"
            )

        # Assertions
        mock_get_model_config.assert_called_once_with(mock_model)
        mock_global_state.assert_called_once()

        # Check that ConfigContainer was called with a tokenizer config
        mock_config_container.assert_called_once()
        call_kwargs = mock_config_container.call_args[1]
        assert "tokenizer" in call_kwargs
        tokenizer_config = call_kwargs["tokenizer"]
        assert tokenizer_config.tokenizer_type == "HuggingFaceTokenizer"
        assert tokenizer_config.tokenizer_model == "meta-llama/Meta-Llama-3-8B"
        assert tokenizer_config.vocab_size is None

        mock_save_checkpoint.assert_called_once_with(
            state=mock_state,
            model=[mock_model],
            optimizer=None,
            opt_param_scheduler=None,
            num_floating_point_operations_so_far=0,
        )

    @patch("megatron.bridge.training.model_load_save.save_checkpoint")
    @patch("megatron.bridge.training.model_load_save.get_model_config")
    @patch("megatron.bridge.training.model_load_save.GlobalState")
    @patch("megatron.bridge.training.model_load_save.ConfigContainer")
    @patch("megatron.bridge.training.model_load_save.OptimizerConfig")
    @patch("megatron.bridge.training.model_load_save.LoggerConfig")
    @patch("megatron.bridge.training.model_load_save.CheckpointConfig")
    def test_save_megatron_model_without_tokenizer(
        self,
        mock_ckpt_config,
        mock_logger_config,
        mock_opt_config,
        mock_config_container,
        mock_global_state,
        mock_get_model_config,
        mock_save_checkpoint,
    ):
        """Test saving megatron model without tokenizer configuration."""
        # Setup mocks
        mock_model = Mock()

        class MockModelConfig(ModelProviderMixin, Mock):
            def provide(self, pre_process=None, post_process=None, vp_stage=None):
                return Mock()

        mock_model_config = MockModelConfig()
        mock_get_model_config.return_value = mock_model_config

        mock_state = Mock()
        mock_global_state.return_value = mock_state

        # Mock the ConfigContainer to capture tokenizer config
        mock_container_instance = Mock()
        mock_config_container.return_value = mock_container_instance

        # Test without tokenizer path (should be None)
        with tempfile.TemporaryDirectory() as temp_dir:
            save_megatron_model([mock_model], temp_dir, ckpt_format="torch_dist", hf_tokenizer_path=None)

        # Assertions
        mock_get_model_config.assert_called_once_with(mock_model)
        mock_global_state.assert_called_once()

        # Check that ConfigContainer was called with tokenizer=None
        mock_config_container.assert_called_once()
        call_kwargs = mock_config_container.call_args[1]
        assert "tokenizer" in call_kwargs
        assert call_kwargs["tokenizer"] is None

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


class TestLoadTokenizer:
    """Test load_tokenizer function."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Mock a tokenizer from build_tokenizer."""

        mock_tokenizer = Mock()
        mock_tokenizer.vocab_size = 32000
        mock_tokenizer.eod_id = 0
        mock_tokenizer.eos_id = 1

        return mock_tokenizer

    @patch("megatron.bridge.training.model_load_save.build_tokenizer")
    @patch("megatron.bridge.utils.instantiate_utils.instantiate")
    @patch("megatron.bridge.training.checkpointing.read_run_config")
    def test_load_mbridge_saved_tokenizer(self, mock_read_cfg, mock_instantiate, mock_build_tokenizer, mock_tokenizer):
        """Test loading tokenizer config from Megatron Bridge-saved checkpoint."""

        # Setup mocks
        mock_run_cfg_dict = {
            "model": {"tensor_model_parallel_size": 1, "make_vocab_size_divisible_by": 128},
            "tokenizer": {},
        }
        mock_read_cfg.return_value = mock_run_cfg_dict

        mock_tokenizer_cfg = Mock()
        mock_tokenizer_cfg.vocab_size = 32000
        mock_instantiate.return_value = mock_tokenizer_cfg

        mock_build_tokenizer.return_value = mock_tokenizer

        with tempfile.TemporaryDirectory() as ckpt_path:
            config_file = Path(ckpt_path) / "run_config.yaml"
            config_file.touch()
            result = load_tokenizer(ckpt_path)

        assert result == mock_tokenizer
        mock_read_cfg.assert_called_once()
        mock_instantiate.assert_called_once_with({})
        mock_build_tokenizer.assert_called_once_with(mock_tokenizer_cfg)

    @patch("megatron.bridge.training.model_load_save.build_tokenizer")
    @patch("megatron.bridge.training.mlm_compat.arguments._tokenizer_config_from_args")
    @patch("megatron.bridge.training.mlm_compat.arguments._load_args_from_checkpoint")
    def test_load_mlm_saved_tokenizer(self, mock_load_args, mock_cfg_from_args, mock_build_tokenizer, mock_tokenizer):
        """Test loading tokenizer config from MegatronLM-saved checkpoint."""

        # Setup mocks
        mock_args = Mock()
        mock_args.tensor_model_parallel_size = 2
        mock_args.make_vocab_size_divisible_by = 256
        mock_load_args.return_value = mock_args

        mock_tokenizer_cfg = Mock()
        mock_tokenizer_cfg.vocab_size = 32000
        mock_cfg_from_args.return_value = mock_tokenizer_cfg

        mock_build_tokenizer.return_value = mock_tokenizer

        ckpt_path = "/path/to/mock/dist_checkpoint"
        result = load_tokenizer(ckpt_path)

        assert result == mock_tokenizer
        mock_load_args.assert_called_once_with(ckpt_path)
        mock_cfg_from_args.assert_called_once_with(mock_args)
        mock_build_tokenizer.assert_called_once_with(mock_tokenizer_cfg)
