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

from megatron.bridge.data.loaders import build_train_valid_test_datasets
from megatron.bridge.data.samplers import (
    RandomSeedDataset,
    build_pretraining_data_loader,
)
from megatron.bridge.data.utils import get_dataset_provider
from megatron.bridge.recipes.llama.llama3_8b import pretrain_config


class TestDataSamplers:
    def test_build_pretraining_data_loader(self):
        dataloader = build_pretraining_data_loader(
            dataset=None,
            consumed_samples=0,
            dataloader_type=None,
            micro_batch_size=1,
            num_workers=0,
            data_sharding=False,
        )

        assert dataloader == None

    def test_build_pretraining_data_loader_single(self):
        # Setup dataloader params
        cfg = pretrain_config()
        cfg.train.train_iters = 1000
        cfg.dataset.finalize()
        dataset_provider = get_dataset_provider(cfg.dataset)
        dataset = build_train_valid_test_datasets(cfg=cfg, build_train_valid_test_datasets_provider=dataset_provider)

        # Build dataloader with drop_last=True
        dataloader = build_pretraining_data_loader(
            dataset=dataset,
            consumed_samples=0,
            dataloader_type="single",
            micro_batch_size=1,
            num_workers=0,
            data_sharding=False,
            drop_last=True,
        )

        # Build dataloader with drop_last=False
        dataloader = build_pretraining_data_loader(
            dataset=dataset,
            consumed_samples=0,
            dataloader_type="single",
            micro_batch_size=1,
            num_workers=0,
            data_sharding=False,
            drop_last=False,
        )

        assert dataloader.num_workers == 0

    def test_build_pretraining_data_loader_cyclic(self):
        # Setup dataloader params
        cfg = pretrain_config()
        cfg.train.train_iters = 1000
        cfg.dataset.finalize()
        dataset_provider = get_dataset_provider(cfg.dataset)
        dataset = build_train_valid_test_datasets(cfg=cfg, build_train_valid_test_datasets_provider=dataset_provider)

        # Build dataloader with data_sharding=True
        dataloader = build_pretraining_data_loader(
            dataset=dataset,
            consumed_samples=1000,
            dataloader_type="cyclic",
            micro_batch_size=4,
            num_workers=2,
            data_sharding=True,
        )

        # Build dataloader with data_sharding=False
        dataloader = build_pretraining_data_loader(
            dataset=dataset,
            consumed_samples=0,
            dataloader_type="cyclic",
            micro_batch_size=1,
            num_workers=0,
            data_sharding=False,
        )

        # Build dataloader with RandomSeedDataset
        dataset = RandomSeedDataset(dataset=dataset, seed=1234)
        dataloader = build_pretraining_data_loader(
            dataset=dataset,
            consumed_samples=0,
            dataloader_type="cyclic",
            micro_batch_size=1,
            num_workers=0,
            data_sharding=False,
        )

        assert dataloader.num_workers == 0

    def test_build_pretraining_data_loader_external(self):
        cfg = pretrain_config()
        cfg.train.train_iters = 1000
        cfg.dataset.finalize()
        dataset_provider = get_dataset_provider(cfg.dataset)
        dataset = build_train_valid_test_datasets(cfg=cfg, build_train_valid_test_datasets_provider=dataset_provider)

        # Build dataloader with dataloader_type="external"
        dataloader = build_pretraining_data_loader(
            dataset=dataset,
            consumed_samples=0,
            dataloader_type="external",
            micro_batch_size=1,
            num_workers=2,
            data_sharding=cfg.dataset.data_sharding,
        )

        assert dataloader == dataset
