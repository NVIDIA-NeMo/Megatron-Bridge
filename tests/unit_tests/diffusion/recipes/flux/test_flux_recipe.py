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

from megatron.bridge.diffusion.data.flux.flux_mock_datamodule import FluxMockDataModuleConfig
from megatron.bridge.diffusion.models.flux.flux_provider import FluxProvider
from megatron.bridge.diffusion.recipes.flux.flux import model_config, pretrain_config
from megatron.bridge.training.config import ConfigContainer


pytestmark = [pytest.mark.unit]


class TestModelConfig:
    """Tests for model_config function."""

    def test_model_config_returns_flux_provider_with_defaults(self):
        """Test that model_config returns a FluxProvider with correct defaults."""
        config = model_config()

        assert isinstance(config, FluxProvider)

        # Parallelism defaults
        assert config.tensor_model_parallel_size == 1
        assert config.pipeline_model_parallel_size == 1
        assert config.sequence_parallel is False

        # FLUX-specific defaults
        assert config.num_joint_layers == 19
        assert config.num_single_layers == 38
        assert config.hidden_size == 3072
        assert config.num_attention_heads == 24

    def test_model_config_custom_parameters(self):
        """Test model_config with custom parameters."""
        config = model_config(
            tensor_parallelism=2,
            pipeline_parallelism=4,
            num_joint_layers=10,
            num_single_layers=20,
            hidden_size=2048,
            guidance_embed=True,
        )

        assert config.tensor_model_parallel_size == 2
        assert config.pipeline_model_parallel_size == 4
        assert config.num_joint_layers == 10
        assert config.num_single_layers == 20
        assert config.hidden_size == 2048
        assert config.guidance_embed is True


class TestPretrainConfig:
    """Tests for pretrain_config function."""

    def test_pretrain_config_returns_complete_config(self):
        """Test that pretrain_config returns a ConfigContainer with all required components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = pretrain_config(dir=tmpdir, mock=True)

            assert isinstance(config, ConfigContainer)
            assert isinstance(config.model, FluxProvider)
            assert isinstance(config.dataset, FluxMockDataModuleConfig)

            # Check all required components exist
            assert hasattr(config, "train")
            assert hasattr(config, "optimizer")
            assert hasattr(config, "scheduler")
            assert hasattr(config, "ddp")
            assert hasattr(config, "logger")
            assert hasattr(config, "checkpoint")

    def test_pretrain_config_directory_structure(self):
        """Test that pretrain_config creates correct directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = pretrain_config(dir=tmpdir, name="test_run", mock=True)

            assert "test_run" in config.checkpoint.save
            assert "test_run" in config.logger.tensorboard_dir
            assert config.checkpoint.save.endswith("checkpoints")

    def test_pretrain_config_custom_training_parameters(self):
        """Test pretrain_config with custom training parameters."""
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
        """Test that model parameters propagate correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = pretrain_config(
                dir=tmpdir,
                mock=True,
                num_joint_layers=12,
                hidden_size=2048,
                guidance_embed=True,
                tensor_parallelism=2,
            )

            assert config.model.num_joint_layers == 12
            assert config.model.hidden_size == 2048
            assert config.model.guidance_embed is True
            assert config.model.tensor_model_parallel_size == 2

    def test_pretrain_config_mock_dataset_configuration(self):
        """Test pretrain_config with mock dataset parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = pretrain_config(
                dir=tmpdir,
                mock=True,
                image_H=512,
                image_W=512,
                vae_channels=16,
            )

            assert config.dataset.image_H == 512
            assert config.dataset.image_W == 512
            assert config.dataset.vae_channels == 16

    def test_pretrain_config_with_real_dataset(self):
        """Test pretrain_config with real dataset configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data")
            os.makedirs(data_path, exist_ok=True)

            config = pretrain_config(dir=tmpdir, mock=False, data_paths=[data_path])

            from megatron.bridge.diffusion.data.flux.flux_energon_datamodule import FluxDataModuleConfig

            assert isinstance(config.dataset, FluxDataModuleConfig)
            assert config.dataset.path == [data_path]
