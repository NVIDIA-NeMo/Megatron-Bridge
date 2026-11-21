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

import pytest

import megatron.bridge.data.energon.base_energon_datamodule as mod
from megatron.bridge.data.energon.base_energon_datamodule import (
    EnergonDataloader,
    EnergonMultiModalDataModule,
)


pytestmark = pytest.mark.unit

_SENTINEL_POOL = object()


class TestCachePoolForwarding:
    def test_default_cache_pool_is_none(self):
        dm = EnergonMultiModalDataModule(path="dummy", tokenizer=None)
        assert dm.cache_pool is None

    def test_train_dataloader_forwards_cache_pool(self, monkeypatch):
        dm = EnergonMultiModalDataModule(path="dummy", tokenizer=None, cache_pool=_SENTINEL_POOL)
        calls = []

        def _fake_loader(dataset, *, worker_config, cache_pool):
            calls.append((dataset, cache_pool))
            return object()

        monkeypatch.setattr(mod, "get_savable_loader", _fake_loader)
        monkeypatch.setattr(dm, "datasets_provider", lambda worker_config, split: "dataset")

        loader = dm.train_dataloader()

        assert calls == [("dataset", _SENTINEL_POOL)]
        assert isinstance(loader, EnergonDataloader)
        dm.train_dataloader()
        assert len(calls) == 1

    def test_val_dataloader_forwards_cache_pool(self, monkeypatch):
        dm = EnergonMultiModalDataModule(path="dummy", tokenizer=None, cache_pool=_SENTINEL_POOL)
        calls = []

        def _fake_loader(dataset, *, worker_config, cache_pool):
            calls.append((dataset, cache_pool))
            return object()

        monkeypatch.setattr(mod, "get_savable_loader", _fake_loader)
        monkeypatch.setattr(dm, "datasets_provider", lambda worker_config, split: "dataset")

        loader = dm.val_dataloader()

        assert calls == [("dataset", _SENTINEL_POOL)]
        assert isinstance(loader, EnergonDataloader)
        dm.val_dataloader()
        assert len(calls) == 1
