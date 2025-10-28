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

        Batch sampler now yields full global batches, not individual microbatches.
        Length = number of global batches.
        """
        from megatron.bridge.data.samplers import MegatronPretrainingBatchSampler

        # With drop_last=True
        # num_global_batches = 100 // 16 = 6
        sampler = MegatronPretrainingBatchSampler(
            total_samples=100,
            consumed_samples=0,
            micro_batch_size=4,
            global_batch_size=16,
            data_parallel_rank=0,
            data_parallel_size=2,
            drop_last=True,
        )
        assert len(sampler) == 6  # 6 global batches

        # With drop_last=False
        # num_global_batches = ceil(100 / 16) = 7
        sampler = MegatronPretrainingBatchSampler(
            total_samples=100,
            consumed_samples=0,
            micro_batch_size=4,
            global_batch_size=16,
            data_parallel_rank=0,
            data_parallel_size=2,
            drop_last=False,
        )
        assert len(sampler) == 7  # 7 global batches

    def test_batch_sampler_interleaved_distribution(self):
        """Test that indices are distributed in interleaved fashion across ranks.

        Batch sampler now yields full global batches. With:
        - global_batch_size=8, micro_batch_size=4, dp_size=2
        - Each rank gets global_batch_size // dp_size = 4 samples per yield
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

        # 2 global batches (16 / 8), each yields full batch portion
        assert len(rank0_batches) == 2
        assert len(rank1_batches) == 2

        # First global batch: rank 0 gets [0, 2, 4, 6], rank 1 gets [1, 3, 5, 7]
        assert rank0_batches[0] == [0, 2, 4, 6]
        assert rank1_batches[0] == [1, 3, 5, 7]

        # Second global batch: rank 0 gets [8, 10, 12, 14], rank 1 gets [9, 11, 13, 15]
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

        Batch sampler yields full global batches:
        - 1 global batch (16 samples) yields once with 8 samples (for rank 0, dp_size=2)
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
        # 1 global batch, yields 8 samples for this rank
        assert len(batches) == 1
        assert len(batches[0]) == 8

    def test_batch_sampler_incomplete_batch_drop_last_false(self):
        """Test that incomplete batch is kept when drop_last=False.

        Batch sampler yields full global batches:
        - First global batch (16 samples) → yields 8 samples for this rank
        - Second global batch (4 samples) → yields 2 samples for this rank (partial)
        Total: 2 yields
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
        # 2 global batches
        assert len(batches) == 2
        assert batches[0] == [0, 2, 4, 6, 8, 10, 12, 14]  # Full global batch for this rank
        assert batches[1] == [16, 18]  # Partial batch for this rank

    def test_batch_sampler_incomplete_batch_with_padding(self):
        """Test that incomplete batch is padded when pad_samples_to_global_batch_size=True.

        Batch sampler yields full global batches:
        - First global batch (16 samples) → yields 8 samples for this rank
        - Second global batch (4 samples, padded to 8 for this rank) → yields 8 with padding
        Total: 2 yields
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
        # 2 global batches
        assert len(batches) == 2
        # First global batch: full 8 samples for this rank
        assert batches[0] == [0, 2, 4, 6, 8, 10, 12, 14]
        # Second global batch: 2 real + 6 padded samples
        assert batches[1] == [16, 18, -1, -1, -1, -1, -1, -1]

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

        Batch sampler yields full global batches.
        Each rank gets global_batch_size // dp_size samples per yield.
        All indices should appear exactly once across all ranks.
        """
        from megatron.bridge.data.samplers import MegatronPretrainingBatchSampler

        dp_size = 4
        global_batch_size = 16
        samplers = []

        # Create samplers for all ranks
        # Each rank gets 16 // 4 = 4 samples per global batch
        # 2 global batches (32 / 16) = 2 yields per rank
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


class TestFinetuningUtilities:
    """Tests for finetuning data handling utilities."""

    def test_split_batch_into_microbatches_basic(self):
        """Test basic batch splitting functionality."""
        import torch

        from megatron.bridge.data.finetuning import split_batch_into_microbatches

        batch = {
            "tokens": torch.randint(0, 1000, (16, 240)),
            "labels": torch.randint(0, 1000, (16, 240)),
            "loss_mask": torch.ones(16, 240),
        }

        microbatches = split_batch_into_microbatches(batch, num_microbatches=4)

        assert len(microbatches) == 4
        for i, mb in enumerate(microbatches):
            assert mb["tokens"].shape == (4, 240), f"MB{i} tokens shape mismatch"
            assert mb["labels"].shape == (4, 240), f"MB{i} labels shape mismatch"
            assert mb["loss_mask"].shape == (4, 240), f"MB{i} loss_mask shape mismatch"

    def test_split_batch_preserves_data(self):
        """Test that splitting preserves data integrity."""
        import torch

        from megatron.bridge.data.finetuning import split_batch_into_microbatches

        batch = {
            "tokens": torch.arange(16 * 10).reshape(16, 10),
            "labels": torch.arange(16 * 10, 32 * 10).reshape(16, 10),
        }

        microbatches = split_batch_into_microbatches(batch, num_microbatches=4)

        # Reconstruct original batch
        reconstructed_tokens = torch.cat([mb["tokens"] for mb in microbatches], dim=0)
        reconstructed_labels = torch.cat([mb["labels"] for mb in microbatches], dim=0)

        assert torch.equal(reconstructed_tokens, batch["tokens"])
        assert torch.equal(reconstructed_labels, batch["labels"])

    def test_split_batch_with_metadata(self):
        """Test splitting with mixed tensor and non-tensor items."""
        import torch

        from megatron.bridge.data.finetuning import split_batch_into_microbatches

        batch = {
            "tokens": torch.randint(0, 1000, (16, 240)),
            "labels": torch.randint(0, 1000, (16, 240)),
            "metadata": ["sample_" + str(i) for i in range(16)],
            "global_meta": {"dataset": "test"},
        }

        microbatches = split_batch_into_microbatches(batch, num_microbatches=4)

        # Check metadata splitting
        for i, mb in enumerate(microbatches):
            assert len(mb["metadata"]) == 4
            assert mb["metadata"][0] == f"sample_{i * 4}"
            assert mb["global_meta"] == {"dataset": "test"}  # Should be copied

    def test_prepare_finetuning_batch(self):
        """Test prepare_finetuning_batch function."""
        import torch

        from megatron.bridge.data.finetuning import prepare_finetuning_batch

        # Create mock dataloader that yields global batches
        def mock_dataloader():
            for seq_len in [240, 256, 224]:
                yield {
                    "tokens": torch.randint(0, 1000, (16, seq_len)),
                    "labels": torch.randint(0, 1000, (16, seq_len)),
                }

        iterator = iter(mock_dataloader())

        # First global batch (seq_len=240)
        microbatch_iter, seq_length = prepare_finetuning_batch(
            data_iterator=iterator,
            num_microbatches=4,
            default_seq_length=2048,
        )

        assert seq_length == 240, f"Expected seq_length=240, got {seq_length}"

        # Get all microbatches
        mb1 = next(microbatch_iter)
        assert mb1["tokens"].shape == (4, 240)

        mb2 = next(microbatch_iter)
        assert mb2["tokens"].shape == (4, 240)

        # All microbatches should have same seq_length
        for _ in range(2):  # Get remaining microbatches
            mb = next(microbatch_iter)
            assert mb["tokens"].shape[1] == 240

    def test_prepare_finetuning_batch_variable_lengths(self):
        """Test with different seq_lengths across global batches."""
        import torch

        from megatron.bridge.data.finetuning import prepare_finetuning_batch

        # Mock dataloader with variable seq_lengths
        def mock_dataloader():
            for seq_len in [240, 256, 224]:
                yield {
                    "tokens": torch.randint(0, 1000, (16, seq_len)),
                    "labels": torch.randint(0, 1000, (16, seq_len)),
                }

        # First global batch (seq_len=240)
        iterator = iter(mock_dataloader())
        microbatch_iter, seq_length = prepare_finetuning_batch(iterator, 4, 2048)
        assert seq_length == 240

        # Second global batch (seq_len=256)
        microbatch_iter, seq_length = prepare_finetuning_batch(iterator, 4, 2048)
        assert seq_length == 256

        # Third global batch (seq_len=224)
        microbatch_iter, seq_length = prepare_finetuning_batch(iterator, 4, 2048)
        assert seq_length == 224


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
