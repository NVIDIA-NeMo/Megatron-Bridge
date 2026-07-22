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
import os
import unittest.mock as mock
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.data.builders import GPTSFTDatasetConfig
from megatron.bridge.data.loaders import (
    build_train_valid_test_data_loaders,
    build_train_valid_test_datasets_for_num_epochs,
    get_blend_and_blend_per_split,
    get_train_valid_test_num_samples,
)
from megatron.bridge.data.utils import get_dataset_provider
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    DistributedInitConfig,
    LoggerConfig,
    MockGPTDatasetConfig,
    OptimizerConfig,
    RerunStateMachineConfig,
    RNGConfig,
    SchedulerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.bridge.training.state import TrainState
from megatron.bridge.training.tokenizers.config import TokenizerConfig


def _mock_tokenizer():
    """Create a lightweight mock tokenizer for MockGPTLowLevelDataset.

    MockGPTLowLevelDataset requires ``tokenizer.vocab_size`` and
    ``tokenizer.eod`` when building mock datasets.
    """
    return SimpleNamespace(
        vocab_size=1000,
        eod=0,
        unique_identifiers=OrderedDict({"class": "MockTokenizer"}),
    )


def create_simple_test_config():
    """Create a simple test configuration without HuggingFace dependencies."""
    return ConfigContainer(
        train=TrainingConfig(
            micro_batch_size=1,
            global_batch_size=32,
            train_iters=1000,
        ),
        validation=ValidationConfig(
            eval_interval=100,
            eval_iters=10,
        ),
        model=GPTModelProvider(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            seq_length=512,
            apply_rope_fusion=False,
            vocab_size=1000,
            make_vocab_size_divisible_by=1,
        ),
        optimizer=OptimizerConfig(
            lr=0.001,
            use_distributed_optimizer=False,
        ),
        scheduler=SchedulerConfig(),
        dataset=MockGPTDatasetConfig(
            random_seed=1234,
            seq_length=512,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            dataloader_type="single",
            num_workers=1,
        ),
        logger=LoggerConfig(),
        tokenizer=TokenizerConfig(
            tokenizer_type="NullTokenizer",
            vocab_size=1000,
        ),
        checkpoint=CheckpointConfig(),
        dist=DistributedInitConfig(),
        ddp=DistributedDataParallelConfig(),
        rng=RNGConfig(),
        rerun_state_machine=RerunStateMachineConfig(),
    )


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
        os.makedirs(os.path.dirname(data_args_path), exist_ok=True)
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
        os.makedirs(os.path.dirname(split_data_path), exist_ok=True)
        with open(split_data_path, "w") as f:
            json.dump(split_data, f)

        blend, blend_per_split = get_blend_and_blend_per_split(per_split_data_args_path=split_data_path)

        assert blend == None
        assert blend_per_split == [
            ([data_path], None),
            ([data_path], None),
            ([data_path], None),
        ]

    @mock.patch("torch.distributed.get_world_size")
    @mock.patch("torch.distributed.get_rank")
    @mock.patch("torch.distributed.broadcast")
    def test_build_train_valid_test_data_loaders(self, mock_broadcast, mock_get_rank, mock_get_world_size):
        mock_get_rank.return_value = 0
        mock_get_world_size.return_value = 1

        cfg = create_simple_test_config()
        cfg.dataset.tokenizer = _mock_tokenizer()
        cfg.dataset.finalize()
        dataset_provider = get_dataset_provider(cfg.dataset)
        dp_group = object()
        train_dataloader, valid_dataloader, test_dataloader = build_train_valid_test_data_loaders(
            cfg=cfg,
            train_state=TrainState(),
            build_train_valid_test_datasets_provider=dataset_provider,
            dp_group=dp_group,
        )

        mock_broadcast.assert_called_once_with(mock.ANY, 0)
        actual_flags = mock_broadcast.call_args[0][0]
        expected_flags = torch.tensor([1, 1, 1], dtype=torch.long, device="cuda")
        assert torch.equal(actual_flags, expected_flags)
        assert train_dataloader is not None
        assert valid_dataloader is not None
        assert test_dataloader is not None

    @mock.patch("torch.distributed.get_world_size")
    @mock.patch("torch.distributed.get_rank")
    @mock.patch("torch.distributed.broadcast")
    def test_build_train_valid_test_data_loaders_eval_iters_0(
        self, mock_broadcast, mock_get_rank, mock_get_world_size
    ):
        mock_get_rank.return_value = 0
        mock_get_world_size.return_value = 1

        cfg = create_simple_test_config()
        cfg.validation.eval_iters = 0
        cfg.dataset.tokenizer = _mock_tokenizer()
        cfg.dataset.finalize()
        dataset_provider = get_dataset_provider(cfg.dataset)
        dp_group = object()
        train_dataloader, valid_dataloader, test_dataloader = build_train_valid_test_data_loaders(
            cfg=cfg,
            train_state=TrainState(),
            build_train_valid_test_datasets_provider=dataset_provider,
            dp_group=dp_group,
        )

        mock_broadcast.assert_called_once_with(mock.ANY, 0)
        actual_flags = mock_broadcast.call_args[0][0]
        expected_flags = torch.tensor([1, 0, 0], dtype=torch.long, device="cuda")
        assert torch.equal(actual_flags, expected_flags)
        assert train_dataloader is not None
        assert valid_dataloader is None
        assert test_dataloader is None

    @mock.patch("torch.distributed.broadcast")
    @mock.patch("torch.distributed.get_world_size", return_value=1)
    @mock.patch("torch.distributed.get_rank", return_value=0)
    @mock.patch("megatron.bridge.data.loaders.build_pretraining_data_loader")
    @mock.patch("megatron.bridge.data.loaders.build_train_valid_test_datasets")
    def test_build_train_valid_test_data_loaders_specialized_dispatch_updates_flags(
        self, mock_build_datasets, mock_build_loader, mock_dp_rank, mock_dp_size, mock_broadcast
    ):
        cfg = create_simple_test_config()
        train_state = TrainState()
        dp_group = object()
        dataset_provider = mock.Mock()

        fake_train_ds = mock.MagicMock()
        fake_train_ds.__len__.return_value = cfg.train.global_batch_size
        fake_train_ds.collate_fn = None
        mock_build_datasets.return_value = (fake_train_ds, None, None)
        fake_train_loader = object()
        # Return a loader for train (non-None dataset), None for valid/test (None dataset)
        mock_build_loader.side_effect = lambda dataset, *args, **kwargs: (
            fake_train_loader if dataset is not None else None
        )

        train_dataloader, valid_dataloader, test_dataloader = build_train_valid_test_data_loaders(
            cfg=cfg,
            train_state=train_state,
            build_train_valid_test_datasets_provider=dataset_provider,
            dp_group=dp_group,
        )

        mock_broadcast.assert_called_once_with(mock.ANY, 0)
        actual_flags = mock_broadcast.call_args[0][0]
        expected_flags = torch.tensor([1, 0, 0], dtype=torch.long, device="cuda")
        assert torch.equal(actual_flags, expected_flags)
        assert train_state.do_train == 1
        assert train_state.do_valid == 0
        assert train_state.do_test == 0
        assert train_dataloader is not None
        assert valid_dataloader is None
        assert test_dataloader is None

    @mock.patch("torch.distributed.broadcast")
    @mock.patch("torch.distributed.get_world_size", return_value=1)
    @mock.patch("torch.distributed.get_rank", return_value=0)
    @mock.patch("megatron.bridge.data.loaders.build_pretraining_data_loader")
    @mock.patch("megatron.bridge.data.loaders.build_train_valid_test_datasets")
    def test_build_train_valid_test_data_loaders_allows_none_train_dataset(
        self, mock_build_datasets, mock_build_loader, mock_dp_rank, mock_dp_size, mock_broadcast
    ):
        cfg = create_simple_test_config()
        cfg.validation.eval_iters = 0
        train_state = TrainState()
        dp_group = object()
        dataset_provider = mock.Mock()

        mock_build_datasets.return_value = (None, None, None)
        mock_build_loader.return_value = None

        train_dataloader, valid_dataloader, test_dataloader = build_train_valid_test_data_loaders(
            cfg=cfg,
            train_state=train_state,
            build_train_valid_test_datasets_provider=dataset_provider,
            dp_group=dp_group,
        )

        mock_build_loader.assert_called_once()
        assert mock_build_loader.call_args.args[0] is None
        mock_broadcast.assert_called_once_with(mock.ANY, 0)
        actual_flags = mock_broadcast.call_args[0][0]
        expected_flags = torch.tensor([0, 0, 0], dtype=torch.long, device="cuda")
        assert torch.equal(actual_flags, expected_flags)
        assert train_state.do_train == 0
        assert train_state.do_valid == 0
        assert train_state.do_test == 0
        assert train_dataloader is None
        assert valid_dataloader is None
        assert test_dataloader is None

    @mock.patch("torch.distributed.broadcast")
    @mock.patch("torch.distributed.get_world_size")
    @mock.patch("torch.distributed.get_rank")
    @mock.patch("megatron.bridge.data.loaders.build_pretraining_data_loader", return_value=object())
    @mock.patch("megatron.bridge.data.loaders.build_train_valid_test_datasets")
    @pytest.mark.parametrize("dataset_kind", ["gpt", "sft"])
    def test_build_train_valid_test_data_loaders_uses_eval_dp_group(
        self,
        mock_build_datasets,
        mock_build_loader,
        mock_get_rank,
        mock_get_world_size,
        _mock_broadcast,
        dataset_kind,
    ):
        cfg = create_simple_test_config()
        if dataset_kind == "sft":
            cfg.dataset = GPTSFTDatasetConfig(
                dataset_root="/tmp/dataset",
                seq_length=512,
                seed=4321,
                num_workers=0,
                persistent_workers=False,
            )
            expected_seed = cfg.dataset.seed
        else:
            expected_seed = cfg.dataset.random_seed
        train_ds = mock.MagicMock()
        train_ds.__len__.return_value = cfg.train.global_batch_size
        valid_ds = mock.MagicMock()
        test_ds = mock.MagicMock()
        mock_build_datasets.return_value = (train_ds, valid_ds, test_ds)
        train_dp_group = object()
        eval_dp_group = object()

        mock_get_rank.side_effect = lambda *, group: 1 if group is train_dp_group else 0
        mock_get_world_size.side_effect = lambda *, group: 2 if group is train_dp_group else 1

        build_train_valid_test_data_loaders(
            cfg=cfg,
            train_state=TrainState(),
            build_train_valid_test_datasets_provider=mock.Mock(),
            dp_group=train_dp_group,
            eval_dp_group=eval_dp_group,
        )

        train_call, valid_call, test_call = mock_build_loader.call_args_list
        assert train_call.kwargs["data_parallel_rank"] == 1
        assert train_call.kwargs["data_parallel_size"] == 2
        assert valid_call.kwargs["data_parallel_rank"] == 0
        assert valid_call.kwargs["data_parallel_size"] == 1
        assert test_call.kwargs["data_parallel_rank"] == 0
        assert test_call.kwargs["data_parallel_size"] == 1
        assert train_call.kwargs["seed"] == expected_seed
        assert valid_call.kwargs["seed"] == expected_seed
        assert test_call.kwargs["seed"] == expected_seed

    @mock.patch("torch.distributed.broadcast")
    @mock.patch("torch.distributed.get_world_size", return_value=1)
    @mock.patch("torch.distributed.get_rank", return_value=0)
    def test_iteration_based_loader_respects_drop_last_false(self, _mock_rank, _mock_world_size, _mock_broadcast):
        class PaddingAwareDataset:
            def __len__(self):
                return 3

            def __getitem__(self, index):
                return index

        cfg = create_simple_test_config()
        cfg.train.global_batch_size = 4
        cfg.validation.eval_iters = 0
        cfg.dataset = GPTSFTDatasetConfig(
            dataset_root="/tmp/dataset",
            seq_length=512,
            drop_last=False,
            num_workers=0,
            persistent_workers=False,
        )
        real_torch_tensor = torch.tensor

        with mock.patch(
            "megatron.bridge.data.loaders.torch.tensor",
            side_effect=lambda data, *, dtype, device: real_torch_tensor(data, dtype=dtype),
        ):
            train_dataloader, _, _ = build_train_valid_test_data_loaders(
                cfg=cfg,
                train_state=TrainState(),
                build_train_valid_test_datasets_provider=mock.Mock(return_value=(PaddingAwareDataset(), None, None)),
                dp_group=object(),
            )

        batch = next(iter(train_dataloader)).tolist()
        assert sorted(batch[:-1]) == [0, 1, 2]
        assert batch[-1] == -1


class TestSampleBasedDataLoaders:
    """Tests for sample-based training data loader functionality."""

    def test_get_train_valid_test_num_samples_iteration_based(self):
        """Test sample calculation for iteration-based training."""
        cfg = create_simple_test_config()

        train_samples, valid_samples, test_samples = get_train_valid_test_num_samples(cfg)

        expected_train_samples = cfg.train.train_iters * cfg.train.global_batch_size
        expected_eval_iters = (cfg.train.train_iters // cfg.validation.eval_interval + 1) * cfg.validation.eval_iters
        expected_valid_samples = expected_eval_iters * cfg.train.global_batch_size
        expected_test_samples = cfg.validation.eval_iters * cfg.train.global_batch_size

        assert train_samples == expected_train_samples
        assert valid_samples == expected_valid_samples
        assert test_samples == expected_test_samples

    def test_get_train_valid_test_num_samples_sample_based(self):
        """Test sample calculation for sample-based training."""
        cfg = create_simple_test_config()
        cfg.train.train_samples = 50000  # Use sample-based training
        cfg.train.train_iters = None

        # Need to calculate train_iters first for eval sample calculation
        cfg.train.train_iters = cfg.train.train_samples // cfg.train.global_batch_size

        train_samples, valid_samples, test_samples = get_train_valid_test_num_samples(cfg)

        expected_train_samples = cfg.train.train_samples  # Direct sample count
        expected_eval_iters = (cfg.train.train_iters // cfg.validation.eval_interval + 1) * cfg.validation.eval_iters
        expected_valid_samples = expected_eval_iters * cfg.train.global_batch_size
        expected_test_samples = cfg.validation.eval_iters * cfg.train.global_batch_size

        assert train_samples == expected_train_samples
        assert valid_samples == expected_valid_samples
        assert test_samples == expected_test_samples

    @mock.patch("torch.distributed.get_world_size")
    @mock.patch("torch.distributed.get_rank")
    @mock.patch("torch.distributed.broadcast")
    def test_build_data_loaders_sample_based(self, mock_broadcast, mock_get_rank, mock_get_world_size):
        """Test data loader building with sample-based training."""
        mock_get_rank.return_value = 0
        mock_get_world_size.return_value = 1

        cfg = create_simple_test_config()
        cfg.train.train_samples = 10000  # Sample-based training
        cfg.train.train_iters = None

        # Set sample-based scheduler config
        cfg.scheduler.lr_decay_samples = 8000
        cfg.scheduler.lr_decay_iters = None
        cfg.scheduler.lr_warmup_samples = 1000
        cfg.scheduler.lr_warmup_iters = 0

        # Provide a mock tokenizer required by MockGPTLowLevelDataset
        cfg.dataset.tokenizer = _mock_tokenizer()

        # Need to validate config to calculate train_iters from train_samples
        with mock.patch("megatron.bridge.utils.common_utils.get_world_size_safe", return_value=1):
            cfg.validate()

        # Normal training state (no backward compatibility needed)
        train_state = TrainState()
        train_state.step = 0
        train_state.consumed_train_samples = 0
        train_state.consumed_valid_samples = 0

        dataset_provider = get_dataset_provider(cfg.dataset)

        dp_group = object()

        # Should build data loaders successfully
        train_dataloader, valid_dataloader, test_dataloader = build_train_valid_test_data_loaders(
            cfg=cfg,
            train_state=train_state,
            build_train_valid_test_datasets_provider=dataset_provider,
            dp_group=dp_group,
        )

        # Verify data loaders were created
        assert train_dataloader is not None
        assert valid_dataloader is not None
        assert test_dataloader is not None


class TestEpochBasedDataLoaders:
    """Tests for resolving epoch-based training from a finite dataset."""

    def test_build_datasets_resolves_num_epochs(self):
        cfg = create_simple_test_config()
        cfg.train.train_iters = None
        cfg.train.num_epochs = 0.5
        cfg.dataset = GPTSFTDatasetConfig(dataset_root="/tmp/dataset", seq_length=512)
        train_ds = list(range(100))
        dataset_provider = mock.Mock(return_value=(train_ds, None, None))

        datasets = build_train_valid_test_datasets_for_num_epochs(cfg, dataset_provider)

        dataset_provider.assert_called_once_with([0, 0, 0], cfg.dataset)
        assert datasets == (train_ds, None, None)
        assert cfg.train.train_iters == 2

    def test_build_datasets_rejects_non_batch_dataloader_for_num_epochs(self):
        cfg = create_simple_test_config()
        cfg.train.train_iters = None
        cfg.train.num_epochs = 1.0
        cfg.dataset = GPTSFTDatasetConfig(
            dataset_root="/tmp/dataset",
            seq_length=512,
            dataloader_type="single",
        )
        dataset_provider = mock.Mock()

        with pytest.raises(ValueError, match='dataloader_type="batch"'):
            build_train_valid_test_datasets_for_num_epochs(cfg, dataset_provider)

        dataset_provider.assert_not_called()

    def test_build_datasets_rejects_missing_train_dataset_for_num_epochs(self):
        cfg = create_simple_test_config()
        cfg.train.train_iters = None
        cfg.train.num_epochs = 1.0
        cfg.dataset = GPTSFTDatasetConfig(dataset_root="/tmp/dataset", seq_length=512)

        with pytest.raises(ValueError, match="num_epochs requires a training dataset"):
            build_train_valid_test_datasets_for_num_epochs(cfg, mock.Mock(return_value=(None, None, None)))

    def test_build_datasets_rejects_unsized_train_dataset_for_num_epochs(self):
        class UnsizedDataset:
            def __len__(self):
                raise NotImplementedError()

        cfg = create_simple_test_config()
        cfg.train.train_iters = None
        cfg.train.num_epochs = 1.0
        cfg.dataset = GPTSFTDatasetConfig(dataset_root="/tmp/dataset", seq_length=512)

        with pytest.raises(ValueError, match="finite length"):
            build_train_valid_test_datasets_for_num_epochs(cfg, mock.Mock(return_value=(UnsizedDataset(), None, None)))

    @mock.patch("torch.distributed.broadcast")
    @mock.patch("torch.distributed.get_world_size", return_value=1)
    @mock.patch("torch.distributed.get_rank", return_value=0)
    @mock.patch("megatron.bridge.data.loaders.build_pretraining_data_loader", return_value=object())
    def test_epoch_based_loader_allows_dataset_smaller_than_global_batch(
        self, mock_build_loader, _mock_rank, _mock_world_size, _mock_broadcast
    ):
        cfg = create_simple_test_config()
        cfg.train.num_epochs = 1.0
        cfg.validation.eval_iters = 0
        cfg.dataset = GPTSFTDatasetConfig(dataset_root="/tmp/dataset", seq_length=512)
        train_ds = mock.MagicMock()
        train_ds.__len__.return_value = 4

        build_train_valid_test_data_loaders(
            cfg=cfg,
            train_state=TrainState(),
            build_train_valid_test_datasets_provider=mock.Mock(return_value=(train_ds, None, None)),
            dp_group=object(),
        )

        assert mock_build_loader.call_args.kwargs["drop_last"] is False

    @mock.patch("torch.distributed.broadcast")
    @mock.patch("torch.distributed.get_world_size", return_value=1)
    @mock.patch("torch.distributed.get_rank", return_value=0)
    @mock.patch("megatron.bridge.data.loaders.build_pretraining_data_loader", return_value=object())
    def test_epoch_based_single_loader_is_rejected(
        self, mock_build_loader, _mock_rank, _mock_world_size, _mock_broadcast
    ):
        cfg = create_simple_test_config()
        cfg.train.num_epochs = 1.0
        cfg.validation.eval_iters = 0
        cfg.dataset = GPTSFTDatasetConfig(
            dataset_root="/tmp/dataset",
            seq_length=512,
            dataloader_type="single",
        )
        train_ds = mock.MagicMock()
        train_ds.__len__.return_value = 100

        with pytest.raises(ValueError, match='dataloader_type="batch"'):
            build_train_valid_test_data_loaders(
                cfg=cfg,
                train_state=TrainState(),
                build_train_valid_test_datasets_provider=mock.Mock(return_value=(train_ds, None, None)),
                dp_group=object(),
            )

        mock_build_loader.assert_not_called()
