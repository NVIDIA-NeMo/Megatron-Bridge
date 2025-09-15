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
import torch

from megatron.bridge.models.llama import Llama3ModelProvider8B
from megatron.bridge.recipes.llama.llama3_8b_128k import SEQUENCE_LENGTH_128K, model_config, pretrain_config
from megatron.bridge.training.config import ConfigContainer


@pytest.mark.unit
class TestModelConfig:
    """Test cases for the model_config function."""

    def test_model_config_default_parameters_128k_optimized(self):
        """Test model_config with default parameters optimized for 128k sequences."""
        config = model_config()

        assert isinstance(config, Llama3ModelProvider8B)
        # Verify 128k-optimized defaults
        assert config.tensor_model_parallel_size == 4  # Same as 64k
        assert config.pipeline_model_parallel_size == 2  # Same as 64k
        assert config.pipeline_dtype == torch.bfloat16  # Specified for 128k
        assert config.virtual_pipeline_model_parallel_size is None
        assert config.context_parallel_size == 8  # Higher than 64k version (4)
        assert config.sequence_parallel is True  # Enabled for 128k
        # Verify model sequence length matches 128k
        assert config.seq_length == SEQUENCE_LENGTH_128K  # Model configured for 128k sequences

    def test_model_config_custom_parameters(self):
        """Test model_config with custom parameters."""
        config = model_config(
            tensor_parallelism=8,
            pipeline_parallelism=4,
            pipeline_parallelism_dtype=torch.float16,
            virtual_pipeline_parallelism=2,
            context_parallelism=16,
            sequence_parallelism=False,
        )

        assert config.tensor_model_parallel_size == 8
        assert config.pipeline_model_parallel_size == 4
        assert config.pipeline_dtype == torch.float16
        assert config.virtual_pipeline_model_parallel_size == 2
        assert config.context_parallel_size == 16
        assert config.sequence_parallel is False
        # Verify model sequence length is still 128k with custom parameters
        assert config.seq_length == SEQUENCE_LENGTH_128K


@pytest.mark.unit
class TestPretrainConfig:
    """Test cases for the pretrain_config function."""

    def test_pretrain_config_default_parameters_128k_optimized(self):
        """Test pretrain_config with default parameters optimized for 128k sequences."""
        config = pretrain_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, Llama3ModelProvider8B)

        # Check that sequence length is set to 128k in both model and dataset
        assert config.dataset.sequence_length == SEQUENCE_LENGTH_128K
        assert config.model.seq_length == SEQUENCE_LENGTH_128K

        # Check that model uses 128k-optimized defaults
        assert config.model.tensor_model_parallel_size == 4
        assert config.model.pipeline_model_parallel_size == 2
        assert config.model.pipeline_dtype == torch.bfloat16
        assert config.model.context_parallel_size == 8  # Higher than 64k (4)
        assert config.model.sequence_parallel is True

    def test_pretrain_config_custom_parameters(self):
        """Test pretrain_config with custom parameters."""
        config = pretrain_config(
            tensor_parallelism=8,
            pipeline_parallelism=8,
            context_parallelism=8,
            train_iters=10000,
            global_batch_size=256,
            micro_batch_size=2,
        )

        # Sequence length should be 128k from recipe
        assert config.dataset.sequence_length == SEQUENCE_LENGTH_128K
        assert config.model.seq_length == SEQUENCE_LENGTH_128K

        # Check custom model parameters
        assert config.model.tensor_model_parallel_size == 8
        assert config.model.pipeline_model_parallel_size == 8
        assert config.model.context_parallel_size == 8
        assert config.model.sequence_parallel is True

        # Check custom training parameters
        assert config.train.train_iters == 10000
        assert config.train.global_batch_size == 256
        assert config.train.micro_batch_size == 2

    def test_pretrain_config_128k_sequence_length_override(self):
        """Test that sequence length is hardcoded to 128k and cannot be overridden."""
        config = pretrain_config(
            tensor_parallelism=4,
            pipeline_parallelism=4,
            context_parallelism=8,
        )

        # Sequence length should always be 128k
        assert config.dataset.sequence_length == SEQUENCE_LENGTH_128K
        assert config.model.seq_length == SEQUENCE_LENGTH_128K

    def test_pretrain_config_model_dataset_sequence_length_match(self):
        """Test that model and dataset sequence lengths always match."""
        config = pretrain_config()
        assert config.model.seq_length == config.dataset.sequence_length, (
            "Model and dataset sequence lengths must match"
        )
        assert config.model.seq_length == SEQUENCE_LENGTH_128K, "Both should be 128k"

    def test_pretrain_config_with_custom_directory(self):
        """Test pretrain_config with custom directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = pretrain_config(dir=temp_dir, name="test_128k_run")

            expected_run_dir = os.path.join(temp_dir, "test_128k_run")
            expected_checkpoint_dir = os.path.join(expected_run_dir, "checkpoints")
            expected_tensorboard_dir = os.path.join(expected_run_dir, "tb_logs")

            assert config.checkpoint.save == expected_checkpoint_dir
            assert config.logger.tensorboard_dir == expected_tensorboard_dir
            assert config.dataset.sequence_length == SEQUENCE_LENGTH_128K

    def test_pretrain_config_with_data_paths(self):
        """Test pretrain_config with data paths provided."""
        data_paths = ["/path/to/data1", "/path/to/data2", "/path/to/data3"]
        config = pretrain_config(data_paths=data_paths)

        # Should still have 128k sequence length
        assert config.dataset.sequence_length == SEQUENCE_LENGTH_128K
        # Should use non-mock data configuration
        assert config.dataset.split == "9999,8,2"
        assert config.dataset.blend is not None

    @pytest.mark.parametrize(
        "tensor_parallelism,pipeline_parallelism,context_parallelism,sequence_parallelism",
        [
            (4, 2, 8, True),  # Default 128k-optimized
            (8, 2, 8, True),  # Higher tensor parallelism
            (4, 4, 16, True),  # Higher pipeline and context parallelism
            (2, 1, 4, False),  # Lower parallelism
        ],
    )
    def test_pretrain_config_128k_parallelism_combinations(
        self, tensor_parallelism, pipeline_parallelism, context_parallelism, sequence_parallelism
    ):
        """Test various parallelism combinations for 128k sequences."""
        config = pretrain_config(
            tensor_parallelism=tensor_parallelism,
            pipeline_parallelism=pipeline_parallelism,
            context_parallelism=context_parallelism,
            sequence_parallelism=sequence_parallelism,
        )

        assert config.model.tensor_model_parallel_size == tensor_parallelism
        assert config.model.pipeline_model_parallel_size == pipeline_parallelism
        assert config.model.context_parallel_size == context_parallelism
        assert config.model.sequence_parallel == sequence_parallelism
        assert config.dataset.sequence_length == SEQUENCE_LENGTH_128K  # Always 128k

    def test_pretrain_config_mock_mode_with_128k_sequence(self):
        """Test pretrain_config in mock mode with 128k sequence length."""
        config = pretrain_config(mock=True)

        assert config.dataset.sequence_length == SEQUENCE_LENGTH_128K  # Still 128k in mock mode
        assert config.dataset.split == "1,1,1"  # Mock mode split
        assert config.dataset.blend is None
        assert config.dataset.blend_per_split is None

    def test_pretrain_config_128k_optimized_parallelism(self):
        """Test 128k-optimized parallelism configuration."""
        # Test a realistic configuration for 128k sequences
        config = pretrain_config(
            tensor_parallelism=4,
            pipeline_parallelism=2,
            context_parallelism=8,  # Key difference from 64k (4) and 8k (2)
            sequence_parallelism=True,
        )

        assert config.model.tensor_model_parallel_size == 4
        assert config.model.pipeline_model_parallel_size == 2
        assert config.model.context_parallel_size == 8  # Optimized for 128k
        assert config.model.sequence_parallel is True
        assert config.dataset.sequence_length == SEQUENCE_LENGTH_128K

    @pytest.mark.parametrize("precision", ["fp16_mixed", "bf16_with_fp8_delayed_scaling_mixed"])
    def test_precision_recipes(self, precision):
        cfg = pretrain_config(precision_config=precision)
        assert cfg.mixed_precision == precision
