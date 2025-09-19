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
from unittest.mock import patch

import pytest
import torch

from megatron.bridge.models.qwen import Qwen25ModelProvider32B
from megatron.bridge.recipes.qwen.qwen25_32b import model_config, pretrain_config
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.config import ConfigContainer


@pytest.mark.unit
class TestModelConfig:
    """Test cases for the model_config function."""

    def test_model_config_default_parameters(self):
        """Test model_config with default parameters."""
        config = model_config()

        assert isinstance(config, Qwen25ModelProvider32B)
        assert config.tensor_model_parallel_size == 8  # Default for 32B model
        assert config.pipeline_model_parallel_size == 2  # Default for 32B model
        assert config.pipeline_dtype == torch.bfloat16  # Default for 32B model
        assert config.virtual_pipeline_model_parallel_size is None
        assert config.context_parallel_size == 1
        assert config.sequence_parallel is False


@pytest.mark.unit
class TestPretrainConfig:
    """Test cases for the pretrain_config function."""

    def test_pretrain_config_default_parameters(self):
        """Test pretrain_config with default parameters (mock mode)."""
        config = pretrain_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, Qwen25ModelProvider32B)

        # Check training configuration
        assert config.train.train_iters == 300000
        assert config.train.global_batch_size == 32
        assert config.train.micro_batch_size == 2
        assert config.train.eval_interval == 500
        assert config.train.eval_iters == 32

        # Check optimizer configuration
        assert config.optimizer.optimizer == "adam"
        assert config.optimizer.lr == 3e-4
        assert config.optimizer.min_lr == 3e-5
        assert config.optimizer.bf16 is True
        assert config.optimizer.fp16 is False

        # Check model configuration (32B specific defaults)
        assert config.model.tensor_model_parallel_size == 8  # Default for 32B model
        assert config.model.pipeline_model_parallel_size == 2  # Default for 32B model
        assert config.model.pipeline_dtype == torch.bfloat16  # Default for 32B model

        # Check dataset configuration (should be in mock mode)
        assert config.dataset.sequence_length == 4096
        assert config.dataset.split == "1,1,1"
        assert config.dataset.blend is None
        assert config.dataset.blend_per_split is None

    def test_pretrain_config_custom_training_parameters(self):
        """Test pretrain_config with custom training parameters."""
        config = pretrain_config(
            train_iters=10000,
            global_batch_size=256,
            micro_batch_size=4,
            seq_length=2048,
            lr=1e-4,
            min_lr=1e-5,
            lr_warmup_iters=1000,
        )

        assert config.train.train_iters == 10000
        assert config.train.global_batch_size == 256
        assert config.train.micro_batch_size == 4
        assert config.dataset.sequence_length == 2048
        assert config.optimizer.lr == 1e-4
        assert config.optimizer.min_lr == 1e-5
        assert config.scheduler.lr_warmup_iters == 1000
        assert config.scheduler.lr_decay_iters is None  # Will be set to train_iters during validation

    def test_pretrain_config_with_custom_directory(self):
        """Test pretrain_config with custom directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = pretrain_config(dir=temp_dir, name="test_run")

            expected_run_dir = os.path.join(temp_dir, "test_run")
            expected_checkpoint_dir = os.path.join(expected_run_dir, "checkpoints")
            expected_tensorboard_dir = os.path.join(expected_run_dir, "tb_logs")

            assert config.checkpoint.save == expected_checkpoint_dir
            assert config.checkpoint.load == expected_checkpoint_dir
            assert config.logger.tensorboard_dir == expected_tensorboard_dir

    def test_pretrain_config_explicit_mock_mode(self):
        """Test pretrain_config with explicit mock=True."""
        config = pretrain_config(mock=True)

        assert config.dataset.blend is None
        assert config.dataset.blend_per_split is None
        assert config.dataset.split == "1,1,1"

    def test_pretrain_config_with_data_paths(self):
        """Test pretrain_config with data paths provided."""

        data_paths = ["/path/to/data1", "/path/to/data2", "/path/to/data3"]
        config = pretrain_config(data_paths=data_paths)

        # Check that non-mock mode is configured
        assert config.dataset.split == "9999,8,2"
        assert config.dataset.blend is not None
        assert config.dataset.blend_per_split is None

    def test_pretrain_config_with_train_valid_test_paths(self):
        """Test pretrain_config with separate train/valid/test paths."""

        config = pretrain_config(
            train_data_path=["/path/to/train1", "/path/to/train2", "/path/to/train3"],
            valid_data_path=["/path/to/valid1", "/path/to/valid2", "/path/to/valid3"],
            test_data_path=["/path/to/test1", "/path/to/test2", "/path/to/test3"],
        )

        # When blend_per_split is used, split should be None
        assert config.dataset.split is None
        assert config.dataset.blend is None
        assert config.dataset.blend_per_split is not None

    def test_pretrain_config_prioritizes_blend(self):
        """Test that blend takes priority over blend_per_split when both are provided."""
        config = pretrain_config(
            train_data_path=["/path/to/train1", "/path/to/train2"],
            valid_data_path=["/path/to/valid1", "/path/to/valid2"],
            test_data_path=["/path/to/test1", "/path/to/test2"],
            data_paths=["/path/to/data1", "/path/to/data2"],
        )

        # Should prioritize blend over blend_per_split
        assert config.dataset.split == "9999,8,2"
        assert config.dataset.blend is not None
        assert config.dataset.blend_per_split is None

    @patch("megatron.bridge.recipes.utils.dataset_utils.get_blend_and_blend_per_split")
    def test_pretrain_config_fallback_to_mock_when_no_weights(self, mock_get_blend):
        """Test pretrain_config falls back to mock when no weights are returned."""
        # Mock function returns None for both weights
        mock_get_blend.return_value = (None, None)

        config = pretrain_config(data_paths=["/some/path"])

        # Should fall back to mock mode
        assert config.dataset.blend is None
        assert config.dataset.blend_per_split is None
        assert config.dataset.split == "1,1,1"

    def test_pretrain_config_checkpoint_configuration(self):
        """Test checkpoint configuration in pretrain_config."""
        config = pretrain_config()

        assert config.checkpoint.save_interval == 500
        assert config.checkpoint.ckpt_format == "torch_dist"
        assert config.checkpoint.fully_parallel_save is True

    def test_pretrain_config_ddp_configuration(self):
        """Test distributed data parallel configuration."""
        config = pretrain_config()

        assert config.ddp.use_distributed_optimizer is True
        assert config.ddp.check_for_nan_in_grad is True

    def test_pretrain_config_manual_gc(self):
        """Test manual garbage collection configuration."""
        config = pretrain_config()

        assert config.train.manual_gc is True
        assert config.train.manual_gc_interval == 100
        assert config.train.manual_gc_eval == 100

    def test_pretrain_config_scheduler_configuration(self):
        """Test scheduler configuration."""
        config = pretrain_config(train_iters=50000)

        assert config.scheduler.start_weight_decay == 0.033
        assert config.scheduler.end_weight_decay == 0.033
        assert config.scheduler.weight_decay_incr_style == "constant"
        assert config.scheduler.lr_decay_style == "cosine"
        assert config.scheduler.lr_warmup_iters == 500  # default for qwen25_32b
        assert config.scheduler.lr_warmup_init == 0.0
        assert config.scheduler.lr_decay_iters is None  # Will be set to train_iters during validation
        assert config.scheduler.override_opt_param_scheduler is True

    def test_pretrain_config_tokenizer_configuration(self):
        """Test tokenizer configuration."""
        config = pretrain_config()

        assert config.tokenizer.tokenizer_type == "NullTokenizer"
        assert config.tokenizer.vocab_size == DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    def test_pretrain_config_rng_configuration(self):
        """Test RNG configuration."""
        config = pretrain_config()

        assert config.rng.seed == 1234
        assert config.dataset.random_seed == 1234

    def test_pretrain_config_dataset_configuration(self):
        """Test dataset configuration details."""
        config = pretrain_config()

        assert config.dataset.reset_attention_mask is False
        assert config.dataset.reset_position_ids is False
        assert config.dataset.eod_mask_loss is False
        assert config.dataset.num_dataset_builder_threads == 1
        assert config.dataset.data_sharding is True
        assert config.dataset.dataloader_type == "single"

    def test_pretrain_config_logger_configuration(self):
        """Test logger configuration."""
        config = pretrain_config()

        assert config.logger.log_interval == 10
        assert config.logger.log_timers_to_tensorboard is True
        assert "tb_logs" in config.logger.tensorboard_dir

    @pytest.mark.parametrize("seq_length", [1024, 2048, 4096, 8192, 16384])
    def test_pretrain_config_sequence_lengths(self, seq_length):
        """Test various sequence lengths."""
        config = pretrain_config(seq_length=seq_length)

        assert config.dataset.sequence_length == seq_length
        assert config.model.seq_length == seq_length
