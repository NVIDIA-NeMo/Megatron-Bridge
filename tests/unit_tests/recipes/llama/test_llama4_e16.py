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

import pytest
import torch
from megatron.core.distributed import DistributedDataParallelConfig

from megatron.bridge.models.llama import Llama4Experts16ModelProvider
from megatron.bridge.recipes.llama.llama4_e16 import model_config, pretrain_config
from megatron.bridge.training.config import ConfigContainer, TrainingConfig
from megatron.bridge.training.mixed_precision import get_mixed_precision_config


@pytest.mark.unit
class TestModelConfig:
    """Test cases for the model_config function."""

    def test_model_config_default_parameters(self):
        """Test model_config with default parameters."""
        config = model_config()

        assert isinstance(config, Llama4Experts16ModelProvider)
        assert config.tensor_model_parallel_size == 4
        assert config.pipeline_model_parallel_size == 1
        assert config.pipeline_dtype is None
        assert config.virtual_pipeline_model_parallel_size is None
        assert config.context_parallel_size == 1
        assert config.sequence_parallel is True
        assert config.expert_tensor_parallel_size == 4
        assert config.expert_model_parallel_size == 16

    def test_model_config_custom_tensor_parallelism(self):
        """Test model_config with custom tensor parallelism."""
        config = model_config(tensor_parallelism=8)

        assert config.tensor_model_parallel_size == 8
        assert config.pipeline_model_parallel_size == 1  # default
        assert config.context_parallel_size == 1  # default
        assert config.expert_tensor_parallel_size == 4  # default
        assert config.expert_model_parallel_size == 16  # default

    def test_model_config_custom_pipeline_parallelism(self):
        """Test model_config with custom pipeline parallelism."""
        config = model_config(pipeline_parallelism=2, pipeline_parallelism_dtype=torch.float16)

        assert config.tensor_model_parallel_size == 4  # default
        assert config.pipeline_model_parallel_size == 2
        assert config.pipeline_dtype is torch.float16

    def test_model_config_with_pipeline_dtype(self):
        """Test model_config with pipeline dtype specified."""
        config = model_config(pipeline_parallelism=4, pipeline_parallelism_dtype=torch.bfloat16)

        assert config.pipeline_model_parallel_size == 4
        assert config.pipeline_dtype == torch.bfloat16

    def test_model_config_virtual_pipeline_parallelism(self):
        """Test model_config with virtual pipeline parallelism."""
        config = model_config(virtual_pipeline_parallelism=2)

        assert config.virtual_pipeline_model_parallel_size == 2

    def test_model_config_context_parallelism(self):
        """Test model_config with custom context parallelism."""
        config = model_config(context_parallelism=4)

        assert config.context_parallel_size == 4

    def test_model_config_expert_parallelism(self):
        """Test model_config with custom expert parallelism settings."""
        config = model_config(expert_tensor_parallelism=8, expert_model_parallelism=32)

        assert config.expert_tensor_parallel_size == 8
        assert config.expert_model_parallel_size == 32

    def test_model_config_all_custom_parameters(self):
        """Test model_config with all custom parameters."""
        config = model_config(
            tensor_parallelism=8,
            pipeline_parallelism=4,
            pipeline_parallelism_dtype=torch.float16,
            virtual_pipeline_parallelism=2,
            context_parallelism=2,
            expert_tensor_parallelism=8,
            expert_model_parallelism=32,
        )

        assert config.tensor_model_parallel_size == 8
        assert config.pipeline_model_parallel_size == 4
        assert config.pipeline_dtype == torch.float16
        assert config.virtual_pipeline_model_parallel_size == 2
        assert config.context_parallel_size == 2
        assert config.expert_tensor_parallel_size == 8
        assert config.expert_model_parallel_size == 32


@pytest.mark.unit
class TestPretrainConfig:
    """Test cases for the pretrain_config function."""

    def test_pretrain_config_default_parameters(self):
        """Test pretrain_config with default parameters."""
        config = pretrain_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, Llama4Experts16ModelProvider)
        assert isinstance(config.train, TrainingConfig)
        assert isinstance(config.ddp, DistributedDataParallelConfig)

        # Check default training settings
        assert config.train.train_iters == 1_168_251
        assert config.train.global_batch_size == 512
        assert config.train.micro_batch_size == 1
        assert config.train.eval_interval == 2000
        assert config.train.eval_iters == 32

        # Check default model settings
        assert config.model.tensor_model_parallel_size == 4
        assert config.model.pipeline_model_parallel_size == 1
        assert config.model.context_parallel_size == 1
        assert config.model.sequence_parallel is True
        assert config.model.expert_tensor_parallel_size == 4
        assert config.model.expert_model_parallel_size == 16

        # Check dataset settings
        assert config.dataset.sequence_length == 8192
        assert config.dataset.random_seed == 1234
        assert config.dataset.reset_attention_mask is False
        assert config.dataset.reset_position_ids is False
        assert config.dataset.eod_mask_loss is False

        # Check DDP settings
        assert config.ddp.check_for_nan_in_grad is True
        assert config.ddp.grad_reduce_in_fp32 is True
        assert config.ddp.overlap_grad_reduce is True
        assert config.ddp.overlap_param_gather is True
        assert config.ddp.average_in_collective is True
        assert config.ddp.use_distributed_optimizer is True

    def test_pretrain_config_custom_training_parameters(self):
        """Test pretrain_config with custom training parameters."""
        config = pretrain_config(
            train_iters=500_000,
            global_batch_size=1024,
            micro_batch_size=2,
            lr=1e-4,
            min_lr=1e-6,
            lr_warmup_iters=5000,
        )

        assert config.train.train_iters == 500_000
        assert config.train.global_batch_size == 1024
        assert config.train.micro_batch_size == 2

    def test_pretrain_config_custom_model_parameters(self):
        """Test pretrain_config with custom model parameters."""
        config = pretrain_config(
            tensor_parallelism=8,
            pipeline_parallelism=2,
            pipeline_parallelism_dtype=torch.bfloat16,
            virtual_pipeline_parallelism=2,
            context_parallelism=2,
            expert_tensor_parallelism=8,
            expert_model_parallelism=32,
        )

        assert config.model.tensor_model_parallel_size == 8
        assert config.model.pipeline_model_parallel_size == 2
        assert config.model.virtual_pipeline_model_parallel_size == 2
        assert config.model.context_parallel_size == 2
        assert config.model.expert_tensor_parallel_size == 8
        assert config.model.expert_model_parallel_size == 32

    def test_pretrain_config_with_fp16_precision_and_pipeline_dtype(self):
        """Test pretrain_config with fp16 precision and compatible pipeline dtype."""
        config = pretrain_config(
            pipeline_parallelism=2, pipeline_parallelism_dtype=torch.float16, precision_config="fp16_mixed"
        )

        assert config.model.pipeline_model_parallel_size == 2
        # With fp16_mixed precision, pipeline dtype should be compatible
        assert config.mixed_precision == "fp16_mixed"

    def test_pretrain_config_with_data_paths(self):
        """Test pretrain_config with custom data paths."""
        config = pretrain_config(
            data_paths=["/path/to/data1", "/path/to/data2"],
            train_data_path=["/path/to/train"],
            valid_data_path=["/path/to/valid"],
        )

        # Should have blend configuration from data paths
        assert config.dataset.blend is not None

    def test_pretrain_config_with_mock_data(self):
        """Test pretrain_config with mock data enabled."""
        config = pretrain_config(mock=True)

        # Should still create proper configuration
        assert isinstance(config, ConfigContainer)
        assert config.dataset.sequence_length == 8192

    def test_pretrain_config_with_custom_dir_and_name(self):
        """Test pretrain_config with custom directory and name."""
        config = pretrain_config(dir="/custom/path", name="test_run")

        # Should still create proper configuration
        assert isinstance(config, ConfigContainer)
        assert config.checkpoint.save.endswith("test_run/checkpoints")
        assert config.logger.tensorboard_dir.endswith("test_run/tb_logs")

    @pytest.mark.parametrize(
        "global_batch_size,micro_batch_size",
        [
            (256, 1),
            (512, 2),
            (1024, 4),
            (2048, 8),
        ],
    )
    def test_pretrain_config_batch_size_combinations(self, global_batch_size, micro_batch_size):
        """Test various batch size combinations."""
        config = pretrain_config(global_batch_size=global_batch_size, micro_batch_size=micro_batch_size)

        assert config.train.global_batch_size == global_batch_size
        assert config.train.micro_batch_size == micro_batch_size

    @pytest.mark.parametrize("train_iters", [50_000, 100_000, 500_000, 1_000_000])
    def test_pretrain_config_train_iters(self, train_iters):
        """Test various training iteration counts."""
        config = pretrain_config(train_iters=train_iters)

        assert config.train.train_iters == train_iters

    @pytest.mark.parametrize("precision", ["fp16_mixed", "bf16_mixed", "bf16_with_fp8_delayed_scaling_mixed"])
    def test_pretrain_config_precision_string(self, precision):
        """Test precision configuration with string values."""
        config = pretrain_config(precision_config=precision)

        assert isinstance(config, ConfigContainer)
        assert config.mixed_precision == precision

    @pytest.mark.parametrize("precision", ["fp16_mixed", "bf16_mixed", "bf16_with_fp8_delayed_scaling_mixed"])
    def test_pretrain_config_precision_object(self, precision):
        """Test precision configuration with MixedPrecisionConfig object."""
        precision_config = get_mixed_precision_config(precision)
        config = pretrain_config(precision_config=precision_config)

        assert isinstance(config, ConfigContainer)
        assert config.mixed_precision == precision_config

    def test_pretrain_config_llama4_e16_defaults(self):
        """Test that Llama4 16-Experts specific defaults are applied correctly."""
        config = pretrain_config()

        # Check model defaults for Llama4 16-Experts
        assert config.model.tensor_model_parallel_size == 4
        assert config.model.pipeline_model_parallel_size == 1
        assert config.model.context_parallel_size == 1
        assert config.model.sequence_parallel is True
        assert config.model.expert_tensor_parallel_size == 4
        assert config.model.expert_model_parallel_size == 16

        # Check dataset defaults
        assert config.dataset.sequence_length == 8192

    @pytest.mark.parametrize("expert_tensor_parallelism", [1, 2, 4, 8])
    def test_pretrain_config_expert_tensor_parallelism(self, expert_tensor_parallelism):
        """Test various expert tensor parallelism settings."""
        config = pretrain_config(expert_tensor_parallelism=expert_tensor_parallelism)

        assert config.model.expert_tensor_parallel_size == expert_tensor_parallelism

    @pytest.mark.parametrize("expert_model_parallelism", [8, 16, 32, 64])
    def test_pretrain_config_expert_model_parallelism(self, expert_model_parallelism):
        """Test various expert model parallelism settings."""
        config = pretrain_config(expert_model_parallelism=expert_model_parallelism)

        assert config.model.expert_model_parallel_size == expert_model_parallelism

    def test_pretrain_config_expert_parallelism_combination(self):
        """Test combination of expert parallelism settings."""
        config = pretrain_config(expert_tensor_parallelism=8, expert_model_parallelism=64)

        assert config.model.expert_tensor_parallel_size == 8
        assert config.model.expert_model_parallel_size == 64
