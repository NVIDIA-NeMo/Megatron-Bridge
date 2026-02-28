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

import torch

from megatron.bridge.data.loaders import (
    build_train_valid_test_data_loaders,
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


class TestPhaseTransitionDataLoaders:
    """Tests for phase transition iterations in data loader functionality."""

    def test_get_train_valid_test_num_samples_phase_transition_first_phase(self):
        """Test sample calculation for first phase of multi-phase training."""
        cfg = create_simple_test_config()
        cfg.train.train_iters = 1000
        cfg.train.global_batch_size = 32
        cfg.train.phase_transition_iterations = [300, 600]  # Transitions at 300 and 600

        # At step 0, we're in phase 1 (0 to 300)
        train_samples, valid_samples, test_samples = get_train_valid_test_num_samples(cfg, current_step=0)

        # Phase 1: iterations 0-300, samples = 300 * 32 = 9600
        expected_train_samples = 300 * 32
        assert train_samples == expected_train_samples

    def test_get_train_valid_test_num_samples_phase_transition_second_phase(self):
        """Test sample calculation for second phase of multi-phase training."""
        cfg = create_simple_test_config()
        cfg.train.train_iters = 1000
        cfg.train.global_batch_size = 32
        cfg.train.phase_transition_iterations = [300, 600]

        # At step 300, we're in phase 2 (300 to 600)
        train_samples, valid_samples, test_samples = get_train_valid_test_num_samples(cfg, current_step=300)

        # Phase 2: iterations 300-600, samples = 300 * 32 = 9600
        expected_train_samples = 300 * 32
        assert train_samples == expected_train_samples

    def test_get_train_valid_test_num_samples_phase_transition_last_phase(self):
        """Test sample calculation for last phase of multi-phase training."""
        cfg = create_simple_test_config()
        cfg.train.train_iters = 1000
        cfg.train.global_batch_size = 32
        cfg.train.phase_transition_iterations = [300, 600]

        # At step 600, we're in phase 3 (600 to 1000)
        train_samples, valid_samples, test_samples = get_train_valid_test_num_samples(cfg, current_step=600)

        # Phase 3: iterations 600-1000, samples = 400 * 32 = 12800
        expected_train_samples = 400 * 32
        assert train_samples == expected_train_samples

    def test_get_train_valid_test_num_samples_phase_transition_mid_phase(self):
        """Test sample calculation when in the middle of a phase."""
        cfg = create_simple_test_config()
        cfg.train.train_iters = 1000
        cfg.train.global_batch_size = 32
        cfg.train.phase_transition_iterations = [300, 600]

        # At step 450, we're in phase 2 (300 to 600) - should still return full phase size
        train_samples, valid_samples, test_samples = get_train_valid_test_num_samples(cfg, current_step=450)

        # Phase 2: iterations 300-600, samples = 300 * 32 = 9600
        expected_train_samples = 300 * 32
        assert train_samples == expected_train_samples

    def test_get_train_valid_test_num_samples_no_phase_transition(self):
        """Test sample calculation without phase transitions returns total samples."""
        cfg = create_simple_test_config()
        cfg.train.train_iters = 1000
        cfg.train.global_batch_size = 32
        cfg.train.phase_transition_iterations = None

        train_samples, valid_samples, test_samples = get_train_valid_test_num_samples(cfg, current_step=500)

        # Without phase transitions, should return total samples
        expected_train_samples = 1000 * 32
        assert train_samples == expected_train_samples

    def test_get_train_valid_test_num_samples_empty_phase_transitions(self):
        """Test sample calculation with empty phase transitions list returns total samples."""
        cfg = create_simple_test_config()
        cfg.train.train_iters = 1000
        cfg.train.global_batch_size = 32
        cfg.train.phase_transition_iterations = []

        train_samples, valid_samples, test_samples = get_train_valid_test_num_samples(cfg, current_step=500)

        # With empty list, should return total samples
        expected_train_samples = 1000 * 32
        assert train_samples == expected_train_samples
