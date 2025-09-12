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

import json
import unittest.mock as mock

import torch

from megatron.bridge.data.loaders import (
    build_train_valid_test_data_loaders,
    get_blend_and_blend_per_split,
)
from megatron.bridge.data.utils import get_dataset_provider
from megatron.bridge.recipes.llama.llama3_8b import pretrain_config
from megatron.bridge.training.state import TrainState


class TestDataLoaders:
    def test_get_blend_and_blend_per_split_data_paths(self, ensure_test_data):
        data_path = f"{ensure_test_data}/datasets/train/test_text_document"
        blend, blend_per_split = get_blend_and_blend_per_split(data_paths=[1.0, data_path])

        assert blend == ([data_path], [1.0])
        assert blend_per_split == None

    def test_get_blend_and_blend_per_split_data_args_path(self, ensure_test_data):
        # Generate data args file
        data_path = ensure_test_data
        data_args_path = f"{ensure_test_data}/datasets/input/data_args.txt"
        data_path = f"{ensure_test_data}/datasets/train/test_text_document"
        with open(data_args_path, "w") as data_args_file:
            data_args_file.write(f"0.5 {data_path} 0.5 {data_path}")
        blend, blend_per_split = get_blend_and_blend_per_split(data_args_path=data_args_path)

        assert blend == ([data_path, data_path], [0.5, 0.5])
        assert blend_per_split == None

    def test_get_blend_and_blend_per_split_per_split_data_args_path(self, ensure_test_data):
        data_path = f"{ensure_test_data}/datasets/train/test_text_document"
        blend, blend_per_split = get_blend_and_blend_per_split(
            train_data_paths=[0.5, data_path, 0.5, data_path],
            valid_data_paths=[1.0, data_path],
            test_data_paths=[1.0, data_path],
        )

        assert blend == None
        assert blend_per_split == [
            ([data_path, data_path], [0.5, 0.5]),
            ([data_path], [1.0]),
            ([data_path], [1.0]),
        ]

        split_data = {
            "train": [data_path],
            "valid": [data_path],
            "test": [data_path],
        }
        split_data_path = f"{ensure_test_data}/datasets/input/split_data.json"
        with open(split_data_path, "w") as f:
            json.dump(split_data, f)

        blend, blend_per_split = get_blend_and_blend_per_split(per_split_data_args_path=split_data_path)

        assert blend == None
        assert blend_per_split == [
            ([data_path], None),
            ([data_path], None),
            ([data_path], None),
        ]

    @mock.patch("torch.distributed.broadcast")
    @mock.patch("megatron.core.mpu.get_data_parallel_rank")
    @mock.patch("megatron.core.mpu.get_data_parallel_world_size")
    def test_build_train_valid_test_data_loaders(
        self, mock_get_data_parallel_world_size, mock_get_data_parallel_rank, mock_broadcast
    ):
        mock_get_data_parallel_rank.return_value = 0
        mock_get_data_parallel_world_size.return_value = 1
        cfg = pretrain_config()
        cfg.train.train_iters = 1000
        dataset_provider = get_dataset_provider(cfg.dataset)
        train_dataloader, valid_dataloader, test_dataloader = build_train_valid_test_data_loaders(
            cfg=cfg, train_state=TrainState(), build_train_valid_test_datasets_provider=dataset_provider
        )

        mock_broadcast.assert_called_once_with(mock.ANY, 0)
        actual_flags = mock_broadcast.call_args[0][0]
        expected_flags = torch.tensor([1, 1, 1], dtype=torch.long, device="cuda")
        assert torch.equal(actual_flags, expected_flags)
        assert train_dataloader is not None
        assert valid_dataloader is not None
        assert test_dataloader is not None

    @mock.patch("torch.distributed.broadcast")
    @mock.patch("megatron.core.mpu.get_data_parallel_rank")
    @mock.patch("megatron.core.mpu.get_data_parallel_world_size")
    def test_build_train_valid_test_data_loaders_eval_iters_0(
        self, mock_get_data_parallel_world_size, mock_get_data_parallel_rank, mock_broadcast
    ):
        mock_get_data_parallel_rank.return_value = 0
        mock_get_data_parallel_world_size.return_value = 1
        cfg = pretrain_config()
        cfg.train.train_iters = 1000
        cfg.train.eval_iters = 0
        dataset_provider = get_dataset_provider(cfg.dataset)
        train_dataloader, valid_dataloader, test_dataloader = build_train_valid_test_data_loaders(
            cfg=cfg, train_state=TrainState(), build_train_valid_test_datasets_provider=dataset_provider
        )

        mock_broadcast.assert_called_once_with(mock.ANY, 0)
        actual_flags = mock_broadcast.call_args[0][0]
        expected_flags = torch.tensor([1, 0, 0], dtype=torch.long, device="cuda")
        assert torch.equal(actual_flags, expected_flags)
        assert train_dataloader is not None
        assert valid_dataloader is None
        assert test_dataloader is None
