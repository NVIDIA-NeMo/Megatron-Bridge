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

from megatron.bridge.diffusion.data.flux import flux_energon_datamodule as flux_dm_mod
from megatron.bridge.diffusion.data.flux.flux_taskencoder import FluxTaskEncoder


class _FakeDiffusionDataModule:
    def __init__(
        self,
        *,
        path: str,
        seq_length: int,
        packing_buffer_size: int,
        task_encoder,
        micro_batch_size: int,
        global_batch_size: int,
        num_workers: int,
        use_train_split_for_val: bool = True,
    ):
        self.path = path
        self.seq_length = seq_length
        self.packing_buffer_size = packing_buffer_size
        self.task_encoder = task_encoder
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.num_workers = num_workers
        self.use_train_split_for_val = use_train_split_for_val

    # mimic API used by FluxDataModuleConfig.build_datasets
    def train_dataloader(self):
        return "train"


def test_flux_datamodule_config_initialization(monkeypatch):
    # Patch the symbol used inside flux_energon_datamodule module
    monkeypatch.setattr(flux_dm_mod, "DiffusionDataModule", _FakeDiffusionDataModule)

    cfg = flux_dm_mod.FluxDataModuleConfig(
        path="",
        seq_length=1024,
        packing_buffer_size=4,
        micro_batch_size=2,
        global_batch_size=8,
        num_workers=0,
        vae_scale_factor=8,
        latent_channels=16,
        task_encoder_seq_length=1024,
    )

    # __post_init__ should construct a dataset with FluxTaskEncoder and propagate seq_length
    assert isinstance(cfg.dataset, _FakeDiffusionDataModule)
    assert cfg.sequence_length == cfg.dataset.seq_length == 1024
    assert isinstance(cfg.dataset.task_encoder, FluxTaskEncoder)
    assert cfg.dataset.task_encoder.seq_length == 1024
    assert cfg.dataset.task_encoder.packing_buffer_size == 4
    assert cfg.dataset.task_encoder.vae_scale_factor == 8
    assert cfg.dataset.task_encoder.latent_channels == 16
    assert cfg.dataset.use_train_split_for_val is True

    # build_datasets should return train loader thrice
    train, val, test = cfg.build_datasets(context=None)
    assert train == "train" and val == "train" and test == "train"


def test_flux_datamodule_config_with_custom_parameters(monkeypatch):
    """Test FluxDataModuleConfig with custom VAE and latent parameters."""
    monkeypatch.setattr(flux_dm_mod, "DiffusionDataModule", _FakeDiffusionDataModule)

    cfg = flux_dm_mod.FluxDataModuleConfig(
        path="/path/to/dataset",
        seq_length=2048,
        packing_buffer_size=8,
        micro_batch_size=4,
        global_batch_size=16,
        num_workers=8,
        vae_scale_factor=16,
        latent_channels=32,
        task_encoder_seq_length=2048,
    )

    # Verify all parameters are correctly propagated
    assert cfg.dataset.path == "/path/to/dataset"
    assert cfg.dataset.seq_length == 2048
    assert cfg.dataset.packing_buffer_size == 8
    assert cfg.dataset.micro_batch_size == 4
    assert cfg.dataset.global_batch_size == 16
    assert cfg.dataset.num_workers == 8

    # Verify task encoder parameters
    assert cfg.dataset.task_encoder.vae_scale_factor == 16
    assert cfg.dataset.task_encoder.latent_channels == 32
    assert cfg.dataset.task_encoder.seq_length == 2048
    assert cfg.dataset.task_encoder.packing_buffer_size == 8
