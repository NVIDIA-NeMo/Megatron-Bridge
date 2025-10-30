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

import types
from unittest import mock

import pytest
from torch.utils.data import DataLoader, Dataset

from megatron.bridge.data.loaders import dataloaders_to_iterators, set_flags_from_dataloaders


class _TinyDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return idx


def _make_cfg(dataloader_type="single", vpp=None, train_iters=10, eval_iters=2):
    cfg = types.SimpleNamespace()
    cfg.dataset = types.SimpleNamespace(
        dataloader_type=dataloader_type,
        num_workers=0,
        data_sharding=True,
        pin_memory=False,
        persistent_workers=False,
    )
    cfg.model = types.SimpleNamespace(virtual_pipeline_model_parallel_size=vpp)
    cfg.train = types.SimpleNamespace(
        train_iters=train_iters,
        eval_interval=5,
        eval_iters=eval_iters,
        global_batch_size=4,
        micro_batch_size=2,
    )
    return cfg


@pytest.mark.parametrize("dl_type", ["single", "cyclic", "external", "batch"])
def test_dataloaders_to_iterators_basic(dl_type):
    cfg = _make_cfg(dataloader_type=dl_type, vpp=None)
    ds = _TinyDataset()
    dl = DataLoader(ds, batch_size=2)

    train_it, valid_it, test_it = dataloaders_to_iterators(
        cfg=cfg, model_length=1, train_dataloader=dl, valid_dataloader=dl, test_dataloader=dl
    )

    # Always returns iterators (validation is forced cyclic)
    assert train_it is not None
    assert test_it is not None
    assert valid_it is not None


def test_dataloaders_to_iterators_vpp_replication():
    cfg = _make_cfg(dataloader_type="single", vpp=2)
    ds = _TinyDataset()
    dl = DataLoader(ds, batch_size=2)

    train_it, valid_it, test_it = dataloaders_to_iterators(
        cfg=cfg, model_length=2, train_dataloader=dl, valid_dataloader=dl, test_dataloader=dl
    )

    assert isinstance(train_it, list) and len(train_it) == 2
    assert isinstance(valid_it, list) and len(valid_it) == 2
    assert isinstance(test_it, list) and len(test_it) == 2


@mock.patch("megatron.bridge.data.loaders.torch.distributed.broadcast")
def test_set_flags_from_dataloaders_sets_flags(mock_broadcast, monkeypatch):
    # Force CPU tensors to avoid GPU requirement
    def _cpu_tensor(*args, **kwargs):
        # Ignore device kwarg
        if "device" in kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k != "device"}
        import torch

        return torch.tensor(*args, **kwargs)

    monkeypatch.setattr("megatron.bridge.data.loaders.torch.tensor", _cpu_tensor)

    cfg = _make_cfg(dataloader_type="single", vpp=None, train_iters=10, eval_iters=2)
    train_state = types.SimpleNamespace(do_train=False, do_valid=False, do_test=False)

    ds = _TinyDataset()
    dl = DataLoader(ds, batch_size=2)

    set_flags_from_dataloaders(
        cfg=cfg,
        train_state=train_state,
        train_dataloader=dl,
        valid_dataloader=dl,
        test_dataloader=dl,
    )

    assert train_state.do_train is True
    assert train_state.do_valid is True
    assert train_state.do_test is True
    mock_broadcast.assert_called_once()
