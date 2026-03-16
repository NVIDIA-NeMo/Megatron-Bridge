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

from megatron.bridge.diffusion.data.flux.flux_energon_datamodule import FluxDatasetConfig
from megatron.bridge.diffusion.models.flux.flux_provider import FluxProvider
from megatron.bridge.diffusion.recipes.flux.flux import flux_14b_pretrain_config
from megatron.bridge.training.config import ConfigContainer


pytestmark = [pytest.mark.unit]


class TestPretrainConfig:
    """Tests for pretrain_config function (flattened, no-arg API)."""

    def test_pretrain_config_returns_complete_config(self):
        """Test that pretrain_config returns a ConfigContainer with all required components."""
        config = flux_14b_pretrain_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, FluxProvider)
        assert isinstance(config.dataset, FluxDatasetConfig)
        assert config.dataset.path is None  # default: mock/synthetic data

        assert hasattr(config, "train")
        assert hasattr(config, "optimizer")
        assert hasattr(config, "scheduler")
        assert hasattr(config, "ddp")
        assert hasattr(config, "logger")
        assert hasattr(config, "checkpoint")

    def test_pretrain_config_directory_structure(self):
        """Test that pretrain_config uses default directory structure."""
        config = flux_14b_pretrain_config()

        assert "default" in config.checkpoint.save
        assert "default" in config.logger.tensorboard_dir
        assert config.checkpoint.save.endswith("checkpoints")

    def test_pretrain_config_default_training_parameters(self):
        """Test pretrain_config default training parameters."""
        config = flux_14b_pretrain_config()

        assert config.train.train_iters == 10000
        assert config.train.global_batch_size == 16
        assert config.train.micro_batch_size == 1

    def test_pretrain_config_default_model_parameters(self):
        """Test that default model parameters are set correctly."""
        config = flux_14b_pretrain_config()

        assert config.model.num_joint_layers == 19
        assert config.model.hidden_size == 3072
        assert config.model.guidance_embed is False
        assert config.model.tensor_model_parallel_size == 2

    def test_pretrain_config_default_dataset_configuration(self):
        """Test pretrain_config default dataset parameters."""
        config = flux_14b_pretrain_config()

        assert config.dataset.image_H == 1024
        assert config.dataset.image_W == 1024
        assert config.dataset.latent_channels == 16

    def test_pretrain_config_dataset_accepts_path_list(self):
        """Test that dataset config can be overridden to use real data paths."""
        config = flux_14b_pretrain_config()
        assert config.dataset.path is None

        # FluxDatasetConfig accepts path as list; recipe default is None
        config.dataset.path = ["/some/data/path"]
        assert config.dataset.path == ["/some/data/path"]
