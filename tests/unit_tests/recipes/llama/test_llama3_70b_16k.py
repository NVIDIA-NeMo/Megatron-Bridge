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

from megatron.bridge.models.llama import Llama3ModelProvider70B
from megatron.bridge.recipes.llama.llama3_70b_16k import SEQUENCE_LENGTH_16K, model_config, pretrain_config
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.config import ConfigContainer


@pytest.mark.unit
class TestModelConfig:
    """Test cases for the model_config function."""

    def test_model_config_default_parameters_70b_16k_optimized(self):
        """Test model_config with default parameters optimized for 70B with 16k sequences."""
        config = model_config()

        assert isinstance(config, Llama3ModelProvider70B)
        # Verify 70B + 16k optimized defaults
        assert config.tensor_model_parallel_size == 8  # High for 70B model
        assert config.pipeline_model_parallel_size == 2  # Reasonable for 70B
        assert config.pipeline_dtype == torch.bfloat16  # Specified for efficiency
        assert config.virtual_pipeline_model_parallel_size is None
        assert config.context_parallel_size == 2  # Appropriate for 16k
        assert config.sequence_parallel is True  # Enabled for 70B + 16k
        # Verify model sequence length matches 16k
        assert config.seq_length == SEQUENCE_LENGTH_16K  # Model configured for 16k sequences

    def test_model_config_custom_parameters(self):
        """Test model_config with custom parameters."""
        config = model_config(
            tensor_parallelism=4,
            pipeline_parallelism=4,
            pipeline_parallelism_dtype=torch.float16,
            virtual_pipeline_parallelism=2,
            context_parallelism=4,
            sequence_parallelism=False,
        )

        assert config.tensor_model_parallel_size == 4
        assert config.pipeline_model_parallel_size == 4
        assert config.pipeline_dtype == torch.float16
        assert config.virtual_pipeline_model_parallel_size == 2
        assert config.context_parallel_size == 4
        assert config.sequence_parallel is False
        # Verify model sequence length is still 16k with custom parameters
        assert config.seq_length == SEQUENCE_LENGTH_16K

    def test_model_config_sequence_length_consistency(self):
        """Test that model_config always uses the 16k sequence length constant."""
        configs = [
            model_config(),
            model_config(tensor_parallelism=4),
            model_config(context_parallelism=4),
            model_config(sequence_parallelism=False),
        ]

        for config in configs:
            assert config.seq_length == SEQUENCE_LENGTH_16K, "Model sequence length should always be 16k"


@pytest.mark.unit
class TestPretrainConfig:
    """Test cases for the pretrain_config function."""

    def test_pretrain_config_default_parameters_70b_16k_optimized(self):
        """Test pretrain_config with default parameters optimized for 70B with 16k sequences."""
        config = pretrain_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, Llama3ModelProvider70B)

        # Check that sequence length is set to 16k in both model and dataset
        assert config.dataset.sequence_length == SEQUENCE_LENGTH_16K
        assert config.model.seq_length == SEQUENCE_LENGTH_16K

        # Check that model uses 70B + 16k optimized defaults
        assert config.model.tensor_model_parallel_size == 8
        assert config.model.pipeline_model_parallel_size == 2
        assert config.model.pipeline_dtype == torch.bfloat16
        assert config.model.context_parallel_size == 2
        assert config.model.sequence_parallel is True

    def test_pretrain_config_custom_parameters(self):
        """Test pretrain_config with custom parameters."""
        config = pretrain_config(
            dir="/custom/path",
            name="custom_run",
            tensor_parallelism=8,
            pipeline_parallelism=4,
            context_parallelism=2,
            train_iters=10000,
            global_batch_size=256,
            micro_batch_size=2,
        )

        assert config.model.tensor_model_parallel_size == 8
        assert config.model.pipeline_model_parallel_size == 4
        assert config.model.context_parallel_size == 2
        assert config.dataset.sequence_length == SEQUENCE_LENGTH_16K  # Should be 16k
        assert config.train.train_iters == 10000
        assert config.train.global_batch_size == 256
        assert config.train.micro_batch_size == 2

    def test_pretrain_config_16k_sequence_length_override(self):
        """Test that sequence length is always set to 16k."""
        # Test with various parameters, but sequence length should always be 16k
        configs = [
            pretrain_config(),
            pretrain_config(tensor_parallelism=4),
            pretrain_config(train_iters=100000),
            pretrain_config(global_batch_size=1024),
        ]

        for config in configs:
            assert config.dataset.sequence_length == SEQUENCE_LENGTH_16K, (
                "Dataset sequence length should always be 16k"
            )
            assert config.model.seq_length == SEQUENCE_LENGTH_16K, "Model sequence length should always be 16k"

    def test_pretrain_config_model_dataset_sequence_length_match(self):
        """Test that model and dataset sequence lengths always match."""
        config = pretrain_config()
        assert config.model.seq_length == config.dataset.sequence_length, (
            "Model and dataset sequence lengths must match"
        )
        assert config.model.seq_length == SEQUENCE_LENGTH_16K, "Both should be 16k"

    def test_pretrain_config_with_custom_directory(self):
        """Test pretrain_config with custom directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = pretrain_config(dir=temp_dir, name="test_70b_16k_run")

            expected_run_dir = os.path.join(temp_dir, "test_70b_16k_run")
            expected_checkpoint_dir = os.path.join(expected_run_dir, "checkpoints")
            expected_tensorboard_dir = os.path.join(expected_run_dir, "tb_logs")

            assert config.checkpoint.save == expected_checkpoint_dir
            assert config.logger.tensorboard_dir == expected_tensorboard_dir
            assert config.dataset.sequence_length == SEQUENCE_LENGTH_16K

    def test_pretrain_config_with_data_paths(self):
        """Test pretrain_config with data paths provided."""
        data_paths = ["/path/to/data1", "/path/to/data2", "/path/to/data3"]
        config = pretrain_config(data_paths=data_paths)

        # Should still have 16k sequence length
        assert config.dataset.sequence_length == SEQUENCE_LENGTH_16K
        # Should use non-mock data configuration
        assert config.dataset.split == "9999,8,2"
        assert config.dataset.blend is not None

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
        # Should still have 16k sequence length
        assert config.dataset.sequence_length == SEQUENCE_LENGTH_16K

    @pytest.mark.parametrize(
        "tensor_parallelism,pipeline_parallelism,context_parallelism,sequence_parallelism",
        [
            (8, 2, 2, True),  # Default 70B + 16k optimized
            (4, 4, 2, True),  # Different parallelism distribution
            (8, 1, 4, True),  # Higher context parallelism
            (4, 2, 1, False),  # Lower parallelism
        ],
    )
    def test_pretrain_config_70b_16k_parallelism_combinations(
        self, tensor_parallelism, pipeline_parallelism, context_parallelism, sequence_parallelism
    ):
        """Test various parallelism combinations for 70B model with 16k sequences."""
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
        assert config.dataset.sequence_length == SEQUENCE_LENGTH_16K  # Always 16k

    def test_pretrain_config_mock_mode_with_16k_sequence(self):
        """Test pretrain_config in mock mode with 16k sequence length."""
        config = pretrain_config(mock=True)

        assert config.dataset.sequence_length == SEQUENCE_LENGTH_16K  # Still 16k in mock mode
        assert config.dataset.split == "1,1,1"  # Mock mode split
        assert config.dataset.blend is None
        assert config.dataset.blend_per_split is None

    def test_pretrain_config_checkpoint_configuration(self):
        """Test checkpoint configuration in pretrain_config."""
        config = pretrain_config()

        assert config.checkpoint.save_interval == 2000
        assert config.checkpoint.ckpt_format == "torch_dist"
        assert config.checkpoint.fully_parallel_save is True
        assert config.checkpoint.async_save is False

    def test_pretrain_config_ddp_configuration(self):
        """Test distributed data parallel configuration."""
        config = pretrain_config()

        assert config.ddp.check_for_nan_in_grad is True
        assert config.ddp.grad_reduce_in_fp32 is True
        # Note: overlap_grad_reduce and overlap_param_gather are now controlled by CommOverlapConfig
        # and default to False when data_parallel_size is None or <= 1
        assert config.ddp.overlap_grad_reduce is True
        assert config.ddp.overlap_param_gather is True
        assert config.ddp.average_in_collective is True
        assert config.ddp.use_distributed_optimizer is True

    def test_pretrain_config_scheduler_configuration(self):
        """Test scheduler configuration."""
        config = pretrain_config(train_iters=50000)

        assert config.scheduler.start_weight_decay == 0.033
        assert config.scheduler.end_weight_decay == 0.033
        assert config.scheduler.weight_decay_incr_style == "constant"
        assert config.scheduler.lr_decay_style == "cosine"
        assert config.scheduler.lr_warmup_iters == 2000
        assert config.scheduler.lr_warmup_init == 0.0
        assert config.scheduler.lr_decay_iters == 50000  # Should match train_iters
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
        assert config.dataset.num_workers == 8

    def test_pretrain_config_logger_configuration(self):
        """Test logger configuration."""
        config = pretrain_config()

        assert config.logger.log_interval == 10
        assert "tb_logs" in config.logger.tensorboard_dir

    @pytest.mark.parametrize(
        "global_batch_size,micro_batch_size",
        [
            (256, 1),
            (512, 1),
            (1024, 2),
            (2048, 4),
        ],
    )
    def test_pretrain_config_batch_size_combinations(self, global_batch_size, micro_batch_size):
        """Test various batch size combinations for 70B model."""
        config = pretrain_config(global_batch_size=global_batch_size, micro_batch_size=micro_batch_size)

        assert config.train.global_batch_size == global_batch_size
        assert config.train.micro_batch_size == micro_batch_size
        # Sequence length should still be 16k regardless of batch size
        assert config.dataset.sequence_length == SEQUENCE_LENGTH_16K

    @pytest.mark.parametrize("virtual_pipeline_parallelism", [None, 1, 2, 4])
    def test_pretrain_config_virtual_pipeline_parallelism(self, virtual_pipeline_parallelism):
        """Test various virtual pipeline parallelism settings."""
        config = pretrain_config(virtual_pipeline_parallelism=virtual_pipeline_parallelism)

        assert config.model.virtual_pipeline_model_parallel_size == virtual_pipeline_parallelism
        # Sequence length should still be 16k
        assert config.dataset.sequence_length == SEQUENCE_LENGTH_16K

    @pytest.mark.parametrize("precision", ["fp16_mixed", "bf16_with_fp8_delayed_scaling_mixed"])
    def test_precision_recipes(self, precision):
        cfg = pretrain_config(precision_config=precision)
        assert cfg.mixed_precision == precision
