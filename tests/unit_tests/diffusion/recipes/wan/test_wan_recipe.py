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

from megatron.bridge.diffusion.data.wan.wan_energon_datamodule import WanDatasetConfig
from megatron.bridge.diffusion.models.wan.wan_provider import WanModelProvider1_3B, WanModelProvider14B
from megatron.bridge.diffusion.recipes.wan.wan import (
    wan_1_3B_sft_config,
    wan_1_3B_pretrain_config,
    wan_14B_sft_config,
    wan_14B_pretrain_config,
)
from megatron.bridge.training.config import ConfigContainer


pytestmark = [pytest.mark.unit]


class TestWan1_3BPretrainConfig:
    """Tests for wan_1_3B_pretrain_config function (no-arg API)."""

    def test_pretrain_config_returns_complete_config(self):
        config = wan_1_3B_pretrain_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, WanModelProvider1_3B)
        assert isinstance(config.dataset, WanDatasetConfig)
        assert config.dataset.path is None  # default: mock/synthetic data

        assert hasattr(config, "train")
        assert hasattr(config, "optimizer")
        assert hasattr(config, "scheduler")
        assert hasattr(config, "ddp")
        assert hasattr(config, "logger")
        assert hasattr(config, "checkpoint")

    def test_pretrain_config_directory_structure(self):
        config = wan_1_3B_pretrain_config()

        assert "default" in config.checkpoint.save
        assert "default" in config.logger.tensorboard_dir
        assert config.checkpoint.save.endswith("checkpoints")

    def test_pretrain_config_default_training_parameters(self):
        config = wan_1_3B_pretrain_config()

        assert config.train.train_iters == 10000
        assert config.train.global_batch_size == 2
        assert config.train.micro_batch_size == 1

    def test_pretrain_config_default_model_parameters(self):
        config = wan_1_3B_pretrain_config()

        assert config.model.num_layers == 30
        assert config.model.hidden_size == 1536
        assert config.model.num_attention_heads == 12
        assert config.model.ffn_hidden_size == 8960
        assert config.model.tensor_model_parallel_size == 1
        assert config.model.context_parallel_size == 8

    def test_pretrain_config_default_dataset_configuration(self):
        config = wan_1_3B_pretrain_config()

        assert config.dataset.path is None
        assert config.dataset.F_latents == 24
        assert config.dataset.H_latents == 104
        assert config.dataset.W_latents == 60

    def test_pretrain_config_dataset_accepts_path(self):
        config = wan_1_3B_pretrain_config()
        assert config.dataset.path is None

        # WanDatasetConfig accepts a path to switch to real data
        config.dataset.path = "/some/data/path"
        assert config.dataset.path == "/some/data/path"

    def test_pretrain_config_checkpoint_format(self):
        config = wan_1_3B_pretrain_config()
        assert config.checkpoint.ckpt_format == "torch_dist"

    def test_pretrain_config_precision(self):
        config = wan_1_3B_pretrain_config()
        assert config.mixed_precision is not None
        assert config.mixed_precision.grad_reduce_in_fp32 is False

    def test_pretrain_config_ddp_settings(self):
        config = wan_1_3B_pretrain_config()
        assert config.ddp.use_distributed_optimizer is True
        assert config.ddp.check_for_nan_in_grad is True


class TestWan14BPretrainConfig:
    """Tests for wan_14B_pretrain_config function (no-arg API)."""

    def test_pretrain_config_returns_complete_config(self):
        config = wan_14B_pretrain_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, WanModelProvider14B)
        assert isinstance(config.dataset, WanDatasetConfig)
        assert config.dataset.path is None

    def test_pretrain_config_default_model_parameters(self):
        config = wan_14B_pretrain_config()

        assert config.model.num_layers == 40
        assert config.model.hidden_size == 5120
        assert config.model.num_attention_heads == 40
        assert config.model.ffn_hidden_size == 13824
        assert config.model.tensor_model_parallel_size == 2
        assert config.model.context_parallel_size == 4
        assert config.model.sequence_parallel is True


class TestWanFinetuneConfigs:
    """Tests for wan finetune config functions."""

    def test_1_3B_finetune_config_no_checkpoint(self):
        config = wan_1_3B_sft_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, WanModelProvider1_3B)
        assert config.checkpoint.pretrained_checkpoint is None

    def test_1_3B_finetune_config_with_checkpoint(self):
        config = wan_1_3B_sft_config(pretrained_checkpoint="/path/to/ckpt")

        assert config.checkpoint.pretrained_checkpoint == "/path/to/ckpt"

    def test_14B_finetune_config_no_checkpoint(self):
        config = wan_14B_sft_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, WanModelProvider14B)
        assert config.checkpoint.pretrained_checkpoint is None

    def test_14B_finetune_config_with_checkpoint(self):
        config = wan_14B_sft_config(pretrained_checkpoint="/path/to/ckpt")

        assert config.checkpoint.pretrained_checkpoint == "/path/to/ckpt"
