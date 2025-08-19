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

import pytest

from megatron.bridge.models.qwen import Qwen3ModelProvider4B
from megatron.bridge.recipes.qwen.qwen3_4b import model_config, pretrain_config
from megatron.bridge.training.config import ConfigContainer


@pytest.mark.unit
class TestModelConfig:
    """Test cases for the model_config function."""

    def test_model_config_default_parameters(self):
        """Test model_config with default parameters."""
        config = model_config()

        assert isinstance(config, Qwen3ModelProvider4B)
        assert config.tensor_model_parallel_size == 2  # Default for Qwen3 4B model
        assert config.pipeline_model_parallel_size == 1
        assert config.pipeline_dtype is None
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
        assert isinstance(config.model, Qwen3ModelProvider4B)

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

        # Check model configuration (Qwen3 4B specific defaults)
        assert config.model.tensor_model_parallel_size == 2  # Default for Qwen3 4B model
        assert config.model.pipeline_model_parallel_size == 1
        assert config.model.pipeline_dtype is None

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
        assert config.scheduler.lr_decay_iters == 10000  # Should match train_iters

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

    def test_pretrain_config_ddp_configuration(self):
        """Test distributed data parallel configuration."""
        config = pretrain_config()

        assert config.ddp.check_for_nan_in_grad is True
        assert config.ddp.grad_reduce_in_fp32 is True
        assert config.ddp.overlap_grad_reduce is True
        assert config.ddp.overlap_param_gather is True
        assert config.ddp.average_in_collective is True
        assert config.ddp.data_parallel_sharding_strategy == "optim_grads_params"
        assert config.ddp.use_distributed_optimizer is True

    def test_pretrain_config_tokenizer_configuration(self):
        """Test tokenizer configuration."""
        config = pretrain_config()

        assert config.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
        assert config.tokenizer.tokenizer_model == "Qwen/Qwen3-4B"

    @pytest.mark.parametrize("seq_length", [1024, 2048, 4096, 8192, 16384])
    def test_pretrain_config_sequence_lengths(self, seq_length):
        """Test various sequence lengths."""
        config = pretrain_config(seq_length=seq_length)

        assert config.dataset.sequence_length == seq_length
        assert config.model.seq_length == seq_length
