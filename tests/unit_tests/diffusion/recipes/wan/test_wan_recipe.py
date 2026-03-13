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

from megatron.bridge.diffusion.data.wan.wan_mock_datamodule import WanMockDataModuleConfig
from megatron.bridge.diffusion.models.wan.wan_provider import WanModelProvider
from megatron.bridge.diffusion.recipes.wan.wan import model_config, pretrain_config
from megatron.bridge.training.config import ConfigContainer


pytestmark = [pytest.mark.unit]


class TestModelConfig:
    """Tests for model_config function."""

    def test_model_config_returns_wan_provider_with_defaults(self):
        config = model_config()

        assert isinstance(config, WanModelProvider)

        assert config.tensor_model_parallel_size == 1
        assert config.pipeline_model_parallel_size == 1
        assert config.context_parallel_size == 1
        assert config.sequence_parallel is False
        assert config.seq_length == 1024

    def test_model_config_custom_parameters(self):
        config = model_config(
            tensor_parallelism=2,
            pipeline_parallelism=4,
            context_parallelism=2,
            sequence_parallelism=True,
            seq_length=2048,
        )

        assert config.tensor_model_parallel_size == 2
        assert config.pipeline_model_parallel_size == 4
        assert config.context_parallel_size == 2
        assert config.sequence_parallel is True
        assert config.seq_length == 2048

    def test_model_config_pipeline_dtype(self):
        config = model_config(pipeline_parallelism_dtype=torch.float16)
        assert config.pipeline_dtype == torch.float16

    def test_model_config_default_pipeline_dtype(self):
        config = model_config()
        assert config.pipeline_dtype == torch.bfloat16

    def test_model_config_wan_specific_defaults(self):
        config = model_config()
        assert config.num_layers == 30
        assert config.hidden_size == 1536
        assert config.num_attention_heads == 12
        assert config.ffn_hidden_size == 8960


class TestPretrainConfig:
    """Tests for pretrain_config function."""

    def test_pretrain_config_returns_complete_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = pretrain_config(dir=tmpdir, mock=True)

            assert isinstance(config, ConfigContainer)
            assert isinstance(config.model, WanModelProvider)
            assert isinstance(config.dataset, WanMockDataModuleConfig)

            assert hasattr(config, "train")
            assert hasattr(config, "optimizer")
            assert hasattr(config, "scheduler")
            assert hasattr(config, "ddp")
            assert hasattr(config, "logger")
            assert hasattr(config, "checkpoint")

    def test_pretrain_config_directory_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = pretrain_config(dir=tmpdir, name="test_run", mock=True)

            assert "test_run" in config.checkpoint.save
            assert "test_run" in config.logger.tensorboard_dir
            assert config.checkpoint.save.endswith("checkpoints")

    def test_pretrain_config_custom_training_parameters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = pretrain_config(
                dir=tmpdir,
                mock=True,
                train_iters=5000,
                global_batch_size=8,
                micro_batch_size=2,
                lr=5e-5,
            )

            assert config.train.train_iters == 5000
            assert config.train.global_batch_size == 8
            assert config.train.micro_batch_size == 2

    def test_pretrain_config_custom_model_parameters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = pretrain_config(
                dir=tmpdir,
                mock=True,
                tensor_parallelism=2,
                pipeline_parallelism=2,
                context_parallelism=2,
            )

            assert config.model.tensor_model_parallel_size == 2
            assert config.model.pipeline_model_parallel_size == 2
            assert config.model.context_parallel_size == 2

    def test_pretrain_config_mock_dataset_configuration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = pretrain_config(dir=tmpdir, mock=True, micro_batch_size=2, global_batch_size=16)

            assert isinstance(config.dataset, WanMockDataModuleConfig)
            assert config.dataset.micro_batch_size == 2
            assert config.dataset.global_batch_size == 16

    def test_pretrain_config_with_real_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = pretrain_config(dir=tmpdir, mock=False)

            from megatron.bridge.diffusion.data.wan.wan_energon_datamodule import WanDataModuleConfig

            assert isinstance(config.dataset, WanDataModuleConfig)

    def test_pretrain_config_default_dir(self):
        config = pretrain_config(mock=True)
        assert "nemo_experiments" in config.checkpoint.save

    def test_pretrain_config_checkpoint_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = pretrain_config(dir=tmpdir, mock=True)
            assert config.checkpoint.ckpt_format == "torch_dist"

    def test_pretrain_config_rng_seed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = pretrain_config(dir=tmpdir, mock=True)
            assert config.rng.seed == 1234

    def test_pretrain_config_precision_string(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = pretrain_config(dir=tmpdir, mock=True, precision_config="bf16_mixed")
            assert config.mixed_precision is not None
            assert config.mixed_precision.grad_reduce_in_fp32 is False

    def test_pretrain_config_ddp_settings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = pretrain_config(dir=tmpdir, mock=True)
            assert config.ddp.use_distributed_optimizer is True
            assert config.ddp.check_for_nan_in_grad is True

    def test_pretrain_config_fsdp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = pretrain_config(dir=tmpdir, mock=True, use_megatron_fsdp=True)
            assert config.ddp.use_megatron_fsdp is True
