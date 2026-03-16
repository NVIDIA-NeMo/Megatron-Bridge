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

from collections.abc import Iterator
from unittest.mock import patch

import pytest

from megatron.bridge.diffusion.data.flux import flux_energon_datamodule as flux_dm_mod
from megatron.bridge.diffusion.data.flux.flux_taskencoder import FluxTaskEncoder


pytestmark = [pytest.mark.unit]


# -----------------------------------------------------------------------------
# _is_empty_path
# -----------------------------------------------------------------------------


class TestIsEmptyPath:
    """Tests for _is_empty_path helper."""

    def test_none_is_empty(self):
        assert flux_dm_mod._is_empty_path(None) is True

    def test_empty_string_is_empty(self):
        assert flux_dm_mod._is_empty_path("") is True

    def test_whitespace_only_string_is_empty(self):
        assert flux_dm_mod._is_empty_path("   ") is True

    def test_non_empty_string_is_not_empty(self):
        assert flux_dm_mod._is_empty_path("/path/to/data") is False
        assert flux_dm_mod._is_empty_path("x") is False

    def test_empty_list_is_empty(self):
        assert flux_dm_mod._is_empty_path([]) is True

    def test_non_empty_list_is_not_empty(self):
        assert flux_dm_mod._is_empty_path(["/a"]) is False
        assert flux_dm_mod._is_empty_path(["/a", "/b"]) is False


# -----------------------------------------------------------------------------
# FluxDatasetConfig (unified config: mock vs real at runtime)
# -----------------------------------------------------------------------------


class TestFluxDatasetConfig:
    """Tests for FluxDatasetConfig and its build_datasets (mock vs real branch)."""

    def test_post_init_sets_sequence_length(self):
        cfg = flux_dm_mod.FluxDatasetConfig(path=None, seq_length=512)
        assert cfg.sequence_length == 512
        assert cfg.seq_length == 512

    def test_build_datasets_empty_path_uses_mock_branch(self):
        """When path is None/empty, build_datasets delegates to FluxMockDataModuleConfig."""
        fake_train, fake_val, fake_test = iter([1]), iter([2]), iter([3])

        class FakeMockConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def build_datasets(self, context):
                return (fake_train, fake_val, fake_test)

        with patch(
            "megatron.bridge.diffusion.data.flux.flux_mock_datamodule.FluxMockDataModuleConfig",
            FakeMockConfig,
        ):
            cfg = flux_dm_mod.FluxDatasetConfig(
                path=None,
                seq_length=1024,
                image_H=256,
                image_W=256,
                latent_channels=16,
            )
            train, val, test = cfg.build_datasets(context=None)
        assert list(train) == [1]
        assert list(val) == [2]
        assert list(test) == [3]

    def test_build_datasets_empty_string_path_uses_mock_branch(self):
        """When path is '' or whitespace, build_datasets uses mock branch (same as _is_empty_path)."""
        fake_train, fake_val, fake_test = iter(["a"]), iter(["b"]), iter(["c"])

        class FakeMockConfig:
            def build_datasets(self, context):
                return (fake_train, fake_val, fake_test)

        with patch(
            "megatron.bridge.diffusion.data.flux.flux_mock_datamodule.FluxMockDataModuleConfig",
            FakeMockConfig,
        ):
            cfg = flux_dm_mod.FluxDatasetConfig(path="", seq_length=512)
            train, val, test = cfg.build_datasets(context=None)
        assert list(train) == ["a"]
        assert list(val) == ["b"]
        assert list(test) == ["c"]

    def test_build_datasets_string_path_uses_real_branch(self, monkeypatch):
        """When path is a non-empty string, build_datasets uses FluxDataModuleConfig with path_str=path."""
        monkeypatch.setattr(flux_dm_mod, "DiffusionDataModule", _FakeDiffusionDataModule)
        cfg = flux_dm_mod.FluxDatasetConfig(
            path="/path/to/wds",
            seq_length=1024,
            packing_buffer_size=4,
            micro_batch_size=2,
            global_batch_size=8,
            num_workers=0,
        )
        train, val, test = cfg.build_datasets(context=None)
        assert isinstance(train, Iterator)
        assert list(train) == list("train")
        assert list(val) == list("val")
        assert list(test) == list("val")
        # Verify real config was built with the string path (via _FakeDiffusionDataModule)
        assert cfg.path == "/path/to/wds"

    def test_build_datasets_list_path_uses_first_element(self, monkeypatch):
        """When path is a list, build_datasets uses path[0] as path_str for FluxDataModuleConfig."""
        received_paths = []

        class CapturePathFake(_FakeDiffusionDataModule):
            def __init__(self, *, path, **kwargs):
                received_paths.append(path)
                super().__init__(path=path, **kwargs)

        monkeypatch.setattr(flux_dm_mod, "DiffusionDataModule", CapturePathFake)
        cfg = flux_dm_mod.FluxDatasetConfig(
            path=["/first/shard", "/second/shard"],
            seq_length=1024,
            packing_buffer_size=4,
            micro_batch_size=2,
            global_batch_size=8,
            num_workers=0,
        )
        train, val, test = cfg.build_datasets(context=None)
        assert list(train) == list("train")
        assert received_paths == ["/first/shard"]


# -----------------------------------------------------------------------------
# FluxDataModuleConfig (real data only)
# -----------------------------------------------------------------------------


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

    def val_dataloader(self):
        return "val"


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

    # build_datasets returns (iter(train_dataloader()), iter(val_dataloader()), iter(val_dataloader()))
    train, val, test = cfg.build_datasets(context=None)
    assert isinstance(train, Iterator) and isinstance(val, Iterator) and isinstance(test, Iterator)
    assert list(train) == list("train")
    assert list(val) == list("val")
    assert list(test) == list("val")


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
