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

import importlib
from types import SimpleNamespace

import pytest

from megatron.bridge.diffusion.data.flux.flux_energon_datamodule import FluxDatasetConfig
from megatron.bridge.diffusion.models.flux.flux_provider import FluxProvider
from megatron.bridge.diffusion.recipes.flux.flux import flux_12b_pretrain_config, flux_12b_sft_config
from megatron.bridge.training.config import ConfigContainer


pytestmark = [pytest.mark.unit]

# Recipe loads HF config via PreTrainedFlux; patch it so unit tests do not call the Hub
# (same idea as test_llama_recipes monkeypatching AutoBridge).
_flux_recipe_mod = importlib.import_module("megatron.bridge.diffusion.recipes.flux.flux")


def _fake_flux_diffusers_config() -> SimpleNamespace:
    """Shape expected by FluxBridge.provider_bridge; values match FLUX.1-dev / recipe defaults."""
    return SimpleNamespace(
        num_attention_heads=24,
        attention_head_dim=128,
        in_channels=64,
        patch_size=1,
        num_layers=19,
        num_single_layers=38,
        joint_attention_dim=4096,
        pooled_projection_dim=768,
        guidance_embeds=True,
        axes_dims_rope=[16, 56, 56],
        ffn_dim=12288,
    )


class _FakePreTrainedFlux:
    def __init__(self, model_name_or_path, **kwargs):
        self._model_name_or_path = str(model_name_or_path)
        self._config = _fake_flux_diffusers_config()

    @property
    def model_name_or_path(self) -> str:
        return self._model_name_or_path

    @property
    def config(self):
        return self._config


@pytest.fixture(autouse=True)
def _patch_pretrained_flux_no_hub(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_flux_recipe_mod, "PreTrainedFlux", _FakePreTrainedFlux)


class TestPretrainConfig:
    """Tests for pretrain_config function (flattened, no-arg API)."""

    def test_pretrain_config_returns_complete_config(self):
        """Test that pretrain_config returns a ConfigContainer with all required components."""
        config = flux_12b_pretrain_config()

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
        config = flux_12b_pretrain_config()

        assert "default" in config.checkpoint.save
        assert "default" in config.logger.tensorboard_dir
        assert config.checkpoint.save.endswith("checkpoints")

    def test_pretrain_config_default_training_parameters(self):
        """Test pretrain_config default training parameters."""
        config = flux_12b_pretrain_config()

        assert config.train.train_iters == 10000
        assert config.train.global_batch_size == 16
        assert config.train.micro_batch_size == 1

    def test_pretrain_config_default_model_parameters(self):
        """Test that default model parameters are set correctly."""
        config = flux_12b_pretrain_config()

        assert config.model.num_joint_layers == 19
        assert config.model.hidden_size == 3072
        assert config.model.guidance_embed is True
        assert config.model.tensor_model_parallel_size == 2

    def test_pretrain_config_default_dataset_configuration(self):
        """Test pretrain_config default dataset parameters."""
        config = flux_12b_pretrain_config()

        assert config.dataset.image_H == 1024
        assert config.dataset.image_W == 1024
        assert config.dataset.latent_channels == 16

    def test_pretrain_config_dataset_accepts_path_list(self):
        """Test that dataset config can be overridden to use real data paths."""
        config = flux_12b_pretrain_config()
        assert config.dataset.path is None

        # FluxDatasetConfig accepts path as str; recipe default is None
        config.dataset.path = "/some/data/path"
        assert config.dataset.path == "/some/data/path"


class TestSftConfig:
    """Tests for flux_12b_sft_config (SFT from pretrained checkpoint)."""

    def test_sft_config_matches_pretrain_except_checkpoint(self):
        pretrain = flux_12b_pretrain_config()
        sft = flux_12b_sft_config()

        assert sft.model.num_joint_layers == pretrain.model.num_joint_layers
        assert sft.train.train_iters == pretrain.train.train_iters
        assert sft.checkpoint.save_interval == 20
        assert sft.checkpoint.pretrained_checkpoint is None

    def test_sft_config_accepts_pretrained_checkpoint(self):
        ckpt = "/path/to/flux/iter_0000000"
        sft = flux_12b_sft_config(pretrained_checkpoint=ckpt)
        assert sft.checkpoint.pretrained_checkpoint == ckpt
