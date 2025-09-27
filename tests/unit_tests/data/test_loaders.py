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
    get_train_valid_test_num_samples,
)
from megatron.bridge.data.utils import get_dataset_provider
from megatron.bridge.recipes.llama.llama3 import llama3_8b_pretrain_config as pretrain_config
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
        # Avoid HF download by mocking AutoBridge
        with mock.patch("megatron.bridge.recipes.llama.llama3.AutoBridge.from_hf_pretrained") as mock_from:

            class _DummyBridge:
                def to_megatron_provider(self, load_weights=False):
                    from megatron.bridge.models.llama.llama_provider import Llama3ModelProvider

                    return Llama3ModelProvider()

            mock_from.return_value = _DummyBridge()
            cfg = pretrain_config()
        cfg.train.train_iters = 1000
        cfg.dataset.finalize()
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
        # Avoid HF download by mocking AutoBridge
        with mock.patch("megatron.bridge.recipes.llama.llama3.AutoBridge.from_hf_pretrained") as mock_from:

            class _DummyBridge:
                def to_megatron_provider(self, load_weights=False):
                    from megatron.bridge.models.llama.llama_provider import Llama3ModelProvider

                    return Llama3ModelProvider()

            mock_from.return_value = _DummyBridge()
            cfg = pretrain_config()
        cfg.train.train_iters = 1000
        cfg.train.eval_iters = 0
        cfg.dataset.finalize()
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


class TestSampleBasedDataLoaders:
    """Tests for sample-based training data loader functionality."""

    def test_get_train_valid_test_num_samples_iteration_based(self):
        """Test sample calculation for iteration-based training."""
        cfg = pretrain_config()
        cfg.train.train_iters = 1000
        cfg.train.global_batch_size = 32
        cfg.train.eval_interval = 100
        cfg.train.eval_iters = 10

        train_samples, valid_samples, test_samples = get_train_valid_test_num_samples(cfg)

        expected_train_samples = cfg.train.train_iters * cfg.train.global_batch_size
        expected_eval_iters = (cfg.train.train_iters // cfg.train.eval_interval + 1) * cfg.train.eval_iters
        expected_valid_samples = expected_eval_iters * cfg.train.global_batch_size
        expected_test_samples = cfg.train.eval_iters * cfg.train.global_batch_size

        assert train_samples == expected_train_samples
        assert valid_samples == expected_valid_samples
        assert test_samples == expected_test_samples

    def test_get_train_valid_test_num_samples_sample_based(self):
        """Test sample calculation for sample-based training."""
        cfg = pretrain_config()
        cfg.train.train_samples = 50000  # Use sample-based training
        cfg.train.train_iters = None
        cfg.train.global_batch_size = 32
        cfg.train.eval_interval = 100
        cfg.train.eval_iters = 10

        # Need to calculate train_iters first for eval sample calculation
        cfg.train.train_iters = cfg.train.train_samples // cfg.train.global_batch_size

        train_samples, valid_samples, test_samples = get_train_valid_test_num_samples(cfg)

        expected_train_samples = cfg.train.train_samples  # Direct sample count
        expected_eval_iters = (cfg.train.train_iters // cfg.train.eval_interval + 1) * cfg.train.eval_iters
        expected_valid_samples = expected_eval_iters * cfg.train.global_batch_size
        expected_test_samples = cfg.train.eval_iters * cfg.train.global_batch_size

        assert train_samples == expected_train_samples
        assert valid_samples == expected_valid_samples
        assert test_samples == expected_test_samples

    @mock.patch("torch.distributed.broadcast")
    @mock.patch("megatron.core.mpu.get_data_parallel_rank")
    @mock.patch("megatron.core.mpu.get_data_parallel_world_size")
    def test_build_data_loaders_sample_based(
        self, mock_get_data_parallel_world_size, mock_get_data_parallel_rank, mock_broadcast
    ):
        """Test data loader building with sample-based training."""
        mock_get_data_parallel_rank.return_value = 0
        mock_get_data_parallel_world_size.return_value = 1

        cfg = pretrain_config()
        cfg.train.train_samples = 10000  # Sample-based training
        cfg.train.train_iters = None
        cfg.train.global_batch_size = 32
        cfg.model.context_parallel_size = 1  # Fix for world_size=1

        # Set sample-based scheduler config
        cfg.scheduler.lr_decay_samples = 8000
        cfg.scheduler.lr_decay_iters = None
        cfg.scheduler.lr_warmup_samples = 1000
        cfg.scheduler.lr_warmup_iters = 0

        # Need to validate config to calculate train_iters from train_samples
        with mock.patch("megatron.bridge.utils.common_utils.get_world_size_safe", return_value=1):
            cfg.validate()

        # Normal training state (no backward compatibility needed)
        train_state = TrainState()
        train_state.step = 0
        train_state.consumed_train_samples = 0
        train_state.consumed_valid_samples = 0

        dataset_provider = get_dataset_provider(cfg.dataset)

        # Should build data loaders successfully
        train_dataloader, valid_dataloader, test_dataloader = build_train_valid_test_data_loaders(
            cfg=cfg, train_state=train_state, build_train_valid_test_datasets_provider=dataset_provider
        )

        # Verify data loaders were created
        assert train_dataloader is not None
        assert valid_dataloader is not None
        assert test_dataloader is not None
