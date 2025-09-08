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

from megatron.bridge.models.llama import Llama3ModelProvider8B
from megatron.bridge.recipes.llama.llama3_8b_64k import SEQUENCE_LENGTH_64K, model_config, pretrain_config
from megatron.bridge.training.config import ConfigContainer


@pytest.mark.unit
class TestModelConfig:
    """Test cases for the model_config function."""

    def test_model_config_default_parameters_64k_optimized(self):
        """Test model_config with default parameters optimized for 64k sequences."""
        config = model_config()

        assert isinstance(config, Llama3ModelProvider8B)
        # Verify 64k-optimized defaults
        assert config.tensor_model_parallel_size == 4  # Higher than 8k version (1)
        assert config.pipeline_model_parallel_size == 2  # Higher than 8k version (1)
        assert config.pipeline_dtype == torch.bfloat16  # Specified for 64k
        assert config.virtual_pipeline_model_parallel_size is None
        assert config.context_parallel_size == 4  # Higher than 8k version (2)
        assert config.sequence_parallel is True  # Enabled for 64k (False for 8k)
        # Verify model sequence length matches 64k
        assert config.seq_length == SEQUENCE_LENGTH_64K  # Model configured for 64k sequences

    def test_model_config_custom_parameters(self):
        """Test model_config with custom parameters."""
        config = model_config(
            tensor_parallelism=8,
            pipeline_parallelism=4,
            pipeline_parallelism_dtype=torch.float16,
            virtual_pipeline_parallelism=2,
            context_parallelism=8,
            sequence_parallelism=False,
        )

        assert config.tensor_model_parallel_size == 8
        assert config.pipeline_model_parallel_size == 4
        assert config.pipeline_dtype == torch.float16
        assert config.virtual_pipeline_model_parallel_size == 2
        assert config.context_parallel_size == 8
        assert config.sequence_parallel is False
        # Verify model sequence length is still 64k with custom parameters
        assert config.seq_length == SEQUENCE_LENGTH_64K

    def test_model_config_inheritance_from_llama3_8b(self):
        """Test that model_config correctly delegates to llama3_8b.model_config."""
        with patch("megatron.bridge.recipes.llama.llama3_8b.model_config") as mock_base_config:
            mock_base_config.return_value = Llama3ModelProvider8B(
                tensor_model_parallel_size=4,
                pipeline_model_parallel_size=2,
                pipeline_dtype=torch.bfloat16,
                context_parallel_size=4,
                sequence_parallel=True,
            )

            config = model_config()

            # Verify the base function was called with correct parameters
            mock_base_config.assert_called_once_with(
                tensor_parallelism=4,
                pipeline_parallelism=2,
                pipeline_parallelism_dtype=torch.bfloat16,
                virtual_pipeline_parallelism=None,
                context_parallelism=4,
                sequence_parallelism=True,
            )
            assert isinstance(config, Llama3ModelProvider8B)


@pytest.mark.unit
class TestPretrainConfig:
    """Test cases for the pretrain_config function."""

    def test_pretrain_config_default_parameters_64k_optimized(self):
        """Test pretrain_config with default parameters optimized for 64k sequences."""
        config = pretrain_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, Llama3ModelProvider8B)

        # Check that sequence length is set to 64k in both model and dataset
        assert config.dataset.sequence_length == SEQUENCE_LENGTH_64K
        assert config.model.seq_length == SEQUENCE_LENGTH_64K

        # Check that model uses 64k-optimized defaults
        assert config.model.tensor_model_parallel_size == 4
        assert config.model.pipeline_model_parallel_size == 2
        assert config.model.pipeline_dtype == torch.bfloat16
        assert config.model.context_parallel_size == 4
        assert config.model.sequence_parallel is True

    def test_pretrain_config_custom_parameters(self):
        """Test pretrain_config with custom parameters."""
        config = pretrain_config(
            tensor_parallelism=8,
            pipeline_parallelism=4,
            context_parallelism=8,
            sequence_parallelism=False,
            train_iters=10000,
            global_batch_size=256,
            micro_batch_size=2,
        )

        # Check that sequence length is still 64k in both model and dataset
        assert config.dataset.sequence_length == SEQUENCE_LENGTH_64K
        assert config.model.seq_length == SEQUENCE_LENGTH_64K

        # Check custom model parameters
        assert config.model.tensor_model_parallel_size == 8
        assert config.model.pipeline_model_parallel_size == 4
        assert config.model.context_parallel_size == 8
        assert config.model.sequence_parallel is False

        # Check custom training parameters
        assert config.train.train_iters == 10000
        assert config.train.global_batch_size == 256
        assert config.train.micro_batch_size == 2

    def test_pretrain_config_64k_sequence_length_override(self):
        """Test that sequence length is always overridden to 64k."""
        # Test with various parameters, but sequence length should always be 64k
        configs = [
            pretrain_config(),
            pretrain_config(tensor_parallelism=8),
            pretrain_config(train_iters=100000),
            pretrain_config(global_batch_size=1024),
        ]

        for config in configs:
            assert config.dataset.sequence_length == SEQUENCE_LENGTH_64K, (
                "Dataset sequence length should always be 64k"
            )
            assert config.model.seq_length == SEQUENCE_LENGTH_64K, "Model sequence length should always be 64k"

    def test_pretrain_config_model_dataset_sequence_length_match(self):
        """Test that model and dataset sequence lengths always match."""
        config = pretrain_config()
        assert config.model.seq_length == config.dataset.sequence_length, (
            "Model and dataset sequence lengths must match"
        )
        assert config.model.seq_length == SEQUENCE_LENGTH_64K, "Both should be 64k"

    def test_pretrain_config_with_custom_directory(self):
        """Test pretrain_config with custom directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = pretrain_config(dir=temp_dir, name="test_64k_run")

            expected_run_dir = os.path.join(temp_dir, "test_64k_run")
            expected_checkpoint_dir = os.path.join(expected_run_dir, "checkpoints")
            expected_tensorboard_dir = os.path.join(expected_run_dir, "tb_logs")

            assert config.checkpoint.save == expected_checkpoint_dir
            assert config.logger.tensorboard_dir == expected_tensorboard_dir
            assert config.dataset.sequence_length == SEQUENCE_LENGTH_64K

    def test_pretrain_config_with_data_paths(self):
        """Test pretrain_config with data paths provided."""
        data_paths = ["/path/to/data1", "/path/to/data2", "/path/to/data3"]
        config = pretrain_config(data_paths=data_paths)

        # Should still have 64k sequence length
        assert config.dataset.sequence_length == SEQUENCE_LENGTH_64K
        # Should use non-mock data configuration
        assert config.dataset.split == "9999,8,2"
        assert config.dataset.blend is not None

    @pytest.mark.parametrize(
        "tensor_parallelism,pipeline_parallelism,context_parallelism,sequence_parallelism",
        [
            (4, 2, 4, True),  # Default 64k-optimized
            (8, 2, 4, True),  # Higher tensor parallelism
            (4, 4, 8, True),  # Higher pipeline and context parallelism
            (2, 1, 2, False),  # Lower parallelism
        ],
    )
    def test_pretrain_config_64k_parallelism_combinations(
        self, tensor_parallelism, pipeline_parallelism, context_parallelism, sequence_parallelism
    ):
        """Test various parallelism combinations for 64k sequences."""
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
        assert config.dataset.sequence_length == SEQUENCE_LENGTH_64K  # Always 64k

    @pytest.mark.parametrize("precision", ["fp16_mixed", "bf16_with_fp8_delayed_scaling_mixed"])
    def test_precision_recipes(self, precision):
        cfg = pretrain_config(precision_config=precision)
        assert cfg.mixed_precision == precision
