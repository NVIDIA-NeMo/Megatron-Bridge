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
from megatron.bridge.recipes.llama.llama3 import llama3_8b_pretrain_config as pretrain_config


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
        # Setup dataloader params (mock AutoBridge to avoid HF downloads)
        from unittest import mock as _mock

        with _mock.patch("megatron.bridge.recipes.llama.llama3.AutoBridge.from_hf_pretrained") as mock_from:

            class _DummyBridge:
                def to_megatron_provider(self, load_weights=False):
                    from megatron.bridge.models.llama.llama_provider import Llama3ModelProvider

                    return Llama3ModelProvider()

            mock_from.return_value = _DummyBridge()
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
        # Setup dataloader params (mock AutoBridge to avoid HF downloads)
        from unittest import mock as _mock

        with _mock.patch("megatron.bridge.recipes.llama.llama3.AutoBridge.from_hf_pretrained") as mock_from:

            class _DummyBridge:
                def to_megatron_provider(self, load_weights=False):
                    from megatron.bridge.models.llama.llama_provider import Llama3ModelProvider

                    return Llama3ModelProvider()

            mock_from.return_value = _DummyBridge()
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
        # Mock AutoBridge to avoid HF downloads
        from unittest import mock as _mock

        with _mock.patch("megatron.bridge.recipes.llama.llama3.AutoBridge.from_hf_pretrained") as mock_from:

            class _DummyBridge:
                def to_megatron_provider(self, load_weights=False):
                    from megatron.bridge.models.llama.llama_provider import Llama3ModelProvider

                    return Llama3ModelProvider()

            mock_from.return_value = _DummyBridge()
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


class TestMegatronPretrainingBatchSampler:
    """Test suite for MegatronPretrainingBatchSampler."""

    def test_batch_sampler_initialization(self):
        """Test basic initialization of batch sampler."""
        from megatron.bridge.data.samplers import MegatronPretrainingBatchSampler

        sampler = MegatronPretrainingBatchSampler(
            total_samples=100,
            consumed_samples=0,
            micro_batch_size=4,
            global_batch_size=16,
            data_parallel_rank=0,
            data_parallel_size=2,
            drop_last=True,
        )

        assert sampler.total_samples == 100
        assert sampler.consumed_samples == 0
        assert sampler.micro_batch_size == 4
        assert sampler._global_batch_size == 16
        assert sampler.data_parallel_size == 2

    def test_batch_sampler_length(self):
        """Test length calculation for batch sampler.

        After the fix, batch sampler yields microbatches one at a time,
        so length = num_global_batches × num_micro_batches.
        """
        from megatron.bridge.data.samplers import MegatronPretrainingBatchSampler

        # With drop_last=True
        # num_global_batches = 100 // 16 = 6
        # num_micro_batches = 16 // (4 * 2) = 2
        # total yields = 6 * 2 = 12
        sampler = MegatronPretrainingBatchSampler(
            total_samples=100,
            consumed_samples=0,
            micro_batch_size=4,
            global_batch_size=16,
            data_parallel_rank=0,
            data_parallel_size=2,
            drop_last=True,
        )
        assert len(sampler) == 12  # 6 global batches × 2 microbatches each

        # With drop_last=False
        # num_global_batches = ceil(100 / 16) = 7
        # total yields = 7 * 2 = 14
        sampler = MegatronPretrainingBatchSampler(
            total_samples=100,
            consumed_samples=0,
            micro_batch_size=4,
            global_batch_size=16,
            data_parallel_rank=0,
            data_parallel_size=2,
            drop_last=False,
        )
        assert len(sampler) == 14  # 7 global batches × 2 microbatches each

    def test_batch_sampler_interleaved_distribution(self):
        """Test that indices are distributed in interleaved fashion across ranks.

        With the fix, batch sampler yields microbatches one at a time. Since:
        - global_batch_size=8, micro_batch_size=4, dp_size=2
        - num_micro_batches = 8 // (4*2) = 1 per rank per global batch
        Each global batch yields 1 microbatch per rank.
        """
        from megatron.bridge.data.samplers import MegatronPretrainingBatchSampler

        # Simulate rank 0
        sampler_rank0 = MegatronPretrainingBatchSampler(
            total_samples=16,
            consumed_samples=0,
            micro_batch_size=4,
            global_batch_size=8,
            data_parallel_rank=0,
            data_parallel_size=2,
            drop_last=True,
        )

        # Simulate rank 1
        sampler_rank1 = MegatronPretrainingBatchSampler(
            total_samples=16,
            consumed_samples=0,
            micro_batch_size=4,
            global_batch_size=8,
            data_parallel_rank=1,
            data_parallel_size=2,
            drop_last=True,
        )

        # Get indices from both ranks
        rank0_batches = list(sampler_rank0)
        rank1_batches = list(sampler_rank1)

        # 2 global batches (16 / 8) × 1 microbatch each = 2 yields per rank
        assert len(rank0_batches) == 2
        assert len(rank1_batches) == 2

        # First microbatch: rank 0 gets [0, 2, 4, 6], rank 1 gets [1, 3, 5, 7]
        assert rank0_batches[0] == [0, 2, 4, 6]
        assert rank1_batches[0] == [1, 3, 5, 7]

        # Second microbatch: rank 0 gets [8, 10, 12, 14], rank 1 gets [9, 11, 13, 15]
        assert rank0_batches[1] == [8, 10, 12, 14]
        assert rank1_batches[1] == [9, 11, 13, 15]

    def test_batch_sampler_consumed_samples(self):
        """Test resumption from consumed_samples."""
        from megatron.bridge.data.samplers import MegatronPretrainingBatchSampler

        sampler = MegatronPretrainingBatchSampler(
            total_samples=32,
            consumed_samples=16,  # Start from sample 16
            micro_batch_size=4,
            global_batch_size=8,
            data_parallel_rank=0,
            data_parallel_size=2,
            drop_last=True,
        )

        batches = list(sampler)
        # Should start from index 16
        assert batches[0] == [16, 18, 20, 22]

    def test_batch_sampler_incomplete_batch_drop_last_true(self):
        """Test that incomplete batch is dropped when drop_last=True.

        With microbatch-by-microbatch yielding:
        - 1 global batch (16 samples) yields 2 microbatches of 4 samples each
        - Last 4 samples dropped
        """
        from megatron.bridge.data.samplers import MegatronPretrainingBatchSampler

        sampler = MegatronPretrainingBatchSampler(
            total_samples=20,  # Not divisible by global_batch_size=16
            consumed_samples=0,
            micro_batch_size=4,
            global_batch_size=16,
            data_parallel_rank=0,
            data_parallel_size=2,
            drop_last=True,
        )

        batches = list(sampler)
        # 1 global batch × 2 microbatches = 2 yields, each with 4 samples
        assert len(batches) == 2
        assert len(batches[0]) == 4
        assert len(batches[1]) == 4

    def test_batch_sampler_incomplete_batch_drop_last_false(self):
        """Test that incomplete batch is kept when drop_last=False.

        With microbatch-by-microbatch yielding:
        - First global batch (16 samples) → 2 microbatches of 4 samples each
        - Second global batch (4 samples) → 1 microbatch (partial)
        Total: 3 yields
        """
        from megatron.bridge.data.samplers import MegatronPretrainingBatchSampler

        sampler = MegatronPretrainingBatchSampler(
            total_samples=20,  # 16 + 4 remaining
            consumed_samples=0,
            micro_batch_size=4,
            global_batch_size=16,
            data_parallel_rank=0,
            data_parallel_size=2,
            drop_last=False,
            pad_samples_to_global_batch_size=False,
        )

        batches = list(sampler)
        # First global batch yields 2 microbatches, second yields 1 partial = 3 total
        assert len(batches) == 3
        assert batches[0] == [0, 2, 4, 6]  # First microbatch
        assert batches[1] == [8, 10, 12, 14]  # Second microbatch
        assert batches[2] == [16, 18]  # Partial microbatch from second global batch

    def test_batch_sampler_incomplete_batch_with_padding(self):
        """Test that incomplete batch is padded when pad_samples_to_global_batch_size=True.

        With padding and microbatch-by-microbatch yielding:
        - First global batch (16 samples) → 2 microbatches
        - Second global batch (4 samples, padded to 16) → 2 microbatches (with padding)
        Total: 4 yields
        """
        from megatron.bridge.data.samplers import MegatronPretrainingBatchSampler

        sampler = MegatronPretrainingBatchSampler(
            total_samples=20,  # 16 + 4 remaining
            consumed_samples=0,
            micro_batch_size=4,
            global_batch_size=16,
            data_parallel_rank=0,
            data_parallel_size=2,
            drop_last=False,
            pad_samples_to_global_batch_size=True,
        )

        batches = list(sampler)
        # First global batch: 2 microbatches, second global batch (padded): 2 microbatches = 4 total
        assert len(batches) == 4
        # First global batch microbatches
        assert batches[0] == [0, 2, 4, 6]
        assert batches[1] == [8, 10, 12, 14]
        # Second global batch microbatches (with padding)
        assert batches[2] == [16, 18, -1, -1]
        assert batches[3] == [-1, -1, -1, -1]

    def test_batch_sampler_global_batch_size_validation(self):
        """Test that invalid global_batch_size raises error."""
        import pytest

        from megatron.bridge.data.samplers import MegatronPretrainingBatchSampler

        with pytest.raises(RuntimeError, match="not divisible"):
            # global_batch_size=15 not divisible by micro_batch_size=4 * data_parallel_size=2 = 8
            MegatronPretrainingBatchSampler(
                total_samples=100,
                consumed_samples=0,
                micro_batch_size=4,
                global_batch_size=15,
                data_parallel_rank=0,
                data_parallel_size=2,
                drop_last=True,
            )

    def test_batch_sampler_multiple_data_parallel_ranks(self):
        """Test with multiple data parallel ranks.

        With microbatch-by-microbatch yielding, each rank gets multiple yields,
        but all indices should still appear exactly once across all ranks.
        """
        from megatron.bridge.data.samplers import MegatronPretrainingBatchSampler

        dp_size = 4
        global_batch_size = 16
        samplers = []

        # Create samplers for all ranks
        # num_micro_batches = 16 // (4*4) = 1
        # 2 global batches × 1 microbatch = 2 yields per rank
        for rank in range(dp_size):
            sampler = MegatronPretrainingBatchSampler(
                total_samples=32,
                consumed_samples=0,
                micro_batch_size=4,
                global_batch_size=global_batch_size,
                data_parallel_rank=rank,
                data_parallel_size=dp_size,
                drop_last=True,
            )
            samplers.append(sampler)

        # Collect all indices from all ranks
        all_indices = []
        for sampler in samplers:
            for batch in sampler:
                all_indices.extend(batch)

        # Verify all indices from 0-31 are present exactly once
        all_indices_sorted = sorted(all_indices)
        assert all_indices_sorted == list(range(32))


class TestBatchDataloaderIntegration:
    """Integration tests for batch dataloader type."""

    def test_build_batch_dataloader_basic(self):
        """Test building a dataloader with dataloader_type='batch'."""
        from unittest import mock as _mock

        with _mock.patch("megatron.bridge.recipes.llama.llama3.AutoBridge.from_hf_pretrained") as mock_from:

            class _DummyBridge:
                def to_megatron_provider(self, load_weights=False):
                    from megatron.bridge.models.llama.llama_provider import Llama3ModelProvider

                    return Llama3ModelProvider()

            mock_from.return_value = _DummyBridge()
            cfg = pretrain_config()
        cfg.train.train_iters = 1000
        cfg.train.global_batch_size = 16
        cfg.dataset.finalize()
        dataset_provider = get_dataset_provider(cfg.dataset)
        dataset = build_train_valid_test_datasets(cfg=cfg, build_train_valid_test_datasets_provider=dataset_provider)

        # Build dataloader with dataloader_type="batch"
        dataloader = build_pretraining_data_loader(
            dataset=dataset,
            consumed_samples=0,
            dataloader_type="batch",
            micro_batch_size=4,
            num_workers=0,
            data_sharding=False,
            global_batch_size=16,
            drop_last=True,
        )

        assert dataloader is not None
        assert dataloader.num_workers == 0

    def test_build_batch_dataloader_missing_global_batch_size(self):
        """Test that batch dataloader raises error without global_batch_size."""
        from unittest import mock as _mock

        import pytest

        with _mock.patch("megatron.bridge.recipes.llama.llama3.AutoBridge.from_hf_pretrained") as mock_from:

            class _DummyBridge:
                def to_megatron_provider(self, load_weights=False):
                    from megatron.bridge.models.llama.llama_provider import Llama3ModelProvider

                    return Llama3ModelProvider()

            mock_from.return_value = _DummyBridge()
            cfg = pretrain_config()
        cfg.train.train_iters = 1000
        cfg.dataset.finalize()
        dataset_provider = get_dataset_provider(cfg.dataset)
        dataset = build_train_valid_test_datasets(cfg=cfg, build_train_valid_test_datasets_provider=dataset_provider)

        with pytest.raises(RuntimeError, match="global_batch_size must be provided"):
            build_pretraining_data_loader(
                dataset=dataset,
                consumed_samples=0,
                dataloader_type="batch",
                micro_batch_size=4,
                num_workers=0,
                data_sharding=False,
                global_batch_size=None,
                drop_last=True,
            )
