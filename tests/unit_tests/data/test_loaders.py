# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.bridge.data.base import DatasetBuildContext, DatasetProvider
from megatron.bridge.data.loaders import build_train_valid_test_data_loaders
from megatron.bridge.data.utils import get_dataset_provider
from megatron.bridge.training.state import TrainState


@pytest.mark.unit
@mock.patch("torch.distributed.broadcast")
@mock.patch("torch.distributed.get_world_size", return_value=1)
@mock.patch("torch.distributed.get_rank", return_value=0)
def test_batch_loader_does_not_supervise_custom_dataset_padding(_mock_rank, _mock_world_size, _mock_broadcast):
    class OrdinaryDataset:
        def __init__(self, size):
            self.samples = [
                {"sample_id": torch.tensor(index), "loss_mask": torch.tensor(1.0)} for index in range(size)
            ]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            return self.samples[index]

    @dataclass
    class OrdinaryDatasetProvider(DatasetProvider):
        def build_datasets(self, context: DatasetBuildContext):
            return OrdinaryDataset(context.train_samples), None, None

    for dataset_size in (3, 4):
        provider = OrdinaryDatasetProvider(
            dataloader_type="batch",
            drop_last=False,
            num_workers=0,
            persistent_workers=False,
        )
        provider.finalize()
        cfg = SimpleNamespace(
            model=object(),
            dataset=provider,
            train=SimpleNamespace(
                train_samples=dataset_size,
                train_iters=1,
                global_batch_size=4,
                micro_batch_size=1,
                num_epochs=None,
                exit_signal=None,
                exit_signal_handler_for_dataloader=False,
            ),
            validation=SimpleNamespace(
                eval_interval=0,
                eval_iters=0,
                eval_global_batch_size=None,
                eval_micro_batch_size=None,
                skip_train=False,
            ),
        )
        real_torch_tensor = torch.tensor

        def tensor_on_cpu(*args, **kwargs):
            kwargs.pop("device", None)
            return real_torch_tensor(*args, **kwargs)

        try:
            with mock.patch(
                "megatron.bridge.data.loaders.torch.tensor",
                side_effect=tensor_on_cpu,
            ):
                train_dataloader, _, _ = build_train_valid_test_data_loaders(
                    cfg=cfg,
                    train_state=TrainState(),
                    build_train_valid_test_datasets_provider=get_dataset_provider(provider),
                    dp_group=object(),
                )
        except ValueError as error:
            assert dataset_size == 3
            assert "drop_last=False" in str(error)
            assert "padding" in str(error)
            continue

        batch = next(iter(train_dataloader))
        assert batch["loss_mask"].sum().item() == dataset_size, (
            "The padded batch must not supervise a duplicated real sample"
        )


@pytest.mark.unit
@mock.patch("torch.distributed.broadcast")
@mock.patch("torch.distributed.get_world_size", return_value=1)
@mock.patch("torch.distributed.get_rank", return_value=0)
def test_multiple_validation_sets_build_one_dataloader_per_set(_mock_rank, _mock_world_size, _mock_broadcast):
    class RangeDataset:
        def __init__(self, offset, size):
            self.samples = [{"sample_id": torch.tensor(offset + index)} for index in range(size)]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            return self.samples[index]

    @dataclass
    class MultiValDatasetProvider(DatasetProvider):
        def build_datasets(self, context: DatasetBuildContext):
            return (
                RangeDataset(0, context.train_samples),
                [RangeDataset(100, 4), RangeDataset(200, 4)],
                None,
            )

    def make_cfg(provider, multiple_validation_sets):
        return SimpleNamespace(
            model=object(),
            dataset=provider,
            train=SimpleNamespace(
                train_samples=4,
                train_iters=1,
                global_batch_size=2,
                micro_batch_size=1,
                num_epochs=None,
                exit_signal=None,
                exit_signal_handler_for_dataloader=False,
            ),
            validation=SimpleNamespace(
                eval_interval=1,
                eval_iters=1,
                eval_global_batch_size=None,
                eval_micro_batch_size=None,
                skip_train=False,
                multiple_validation_sets=multiple_validation_sets,
            ),
        )

    provider = MultiValDatasetProvider(
        dataloader_type="single",
        drop_last=True,
        num_workers=0,
        persistent_workers=False,
    )
    provider.finalize()

    real_torch_tensor = torch.tensor

    def tensor_on_cpu(*args, **kwargs):
        kwargs.pop("device", None)
        return real_torch_tensor(*args, **kwargs)

    with mock.patch("megatron.bridge.data.loaders.torch.tensor", side_effect=tensor_on_cpu):
        with pytest.raises(ValueError, match="multiple_validation_sets"):
            build_train_valid_test_data_loaders(
                cfg=make_cfg(provider, multiple_validation_sets=False),
                train_state=TrainState(),
                build_train_valid_test_datasets_provider=get_dataset_provider(provider),
                dp_group=object(),
            )

        _, valid_dataloader, test_dataloader = build_train_valid_test_data_loaders(
            cfg=make_cfg(provider, multiple_validation_sets=True),
            train_state=TrainState(),
            build_train_valid_test_datasets_provider=get_dataset_provider(provider),
            dp_group=object(),
        )

    assert isinstance(valid_dataloader, list) and len(valid_dataloader) == 2
    assert test_dataloader is None
    assert next(iter(valid_dataloader[0]))["sample_id"].item() == 100
    assert next(iter(valid_dataloader[1]))["sample_id"].item() == 200
