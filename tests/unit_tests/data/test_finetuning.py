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

"""Tests for finetuning data utilities."""

import torch

from megatron.bridge.data.finetuning import prepare_finetuning_batch


def test_prepare_finetuning_batch_basic():
    """Test basic microbatch splitting."""
    # Create a global batch: GBS=8, seq_len=128
    global_batch = {
        "tokens": torch.randint(0, 1000, (8, 128)),
        "labels": torch.randint(0, 1000, (8, 128)),
        "loss_mask": torch.ones(8, 128),
    }

    data_iter = iter([global_batch])

    microbatch_iter, seq_len = prepare_finetuning_batch(
        data_iterator=data_iter,
        num_microbatches=4,  # Split into 4 microbatches of size 2
        default_seq_length=2048,
        seq_key="tokens",
    )

    # Should extract seq_length from batch
    assert seq_len == 128

    # Collect all microbatches
    microbatches = list(microbatch_iter)
    assert len(microbatches) == 4

    # Each microbatch should have MBS=2
    for mb in microbatches:
        assert mb["tokens"].shape == (2, 128)
        assert mb["labels"].shape == (2, 128)
        assert mb["loss_mask"].shape == (2, 128)


def test_prepare_finetuning_batch_single_microbatch():
    """Test when num_microbatches=1 (no splitting)."""
    global_batch = {
        "tokens": torch.randint(0, 1000, (4, 256)),
        "labels": torch.randint(0, 1000, (4, 256)),
    }

    data_iter = iter([global_batch])

    microbatch_iter, seq_len = prepare_finetuning_batch(
        data_iterator=data_iter,
        num_microbatches=1,
        default_seq_length=2048,
        seq_key="tokens",
    )

    assert seq_len == 256

    microbatches = list(microbatch_iter)
    assert len(microbatches) == 1
    assert microbatches[0]["tokens"].shape == (4, 256)


def test_prepare_finetuning_batch_fallback_seq_length():
    """Test fallback to default_seq_length when batch has unexpected structure."""
    # Batch without the expected seq_key
    global_batch = {
        "data": torch.randn(8, 100),  # Different key, different shape
    }

    data_iter = iter([global_batch])

    microbatch_iter, seq_len = prepare_finetuning_batch(
        data_iterator=data_iter,
        num_microbatches=2,
        default_seq_length=2048,
        seq_key="tokens",  # This key doesn't exist
    )

    # Should fall back to default
    assert seq_len == 2048


def test_prepare_finetuning_batch_variable_seq_lengths():
    """Test that seq_length is extracted correctly for different batches."""
    batch1 = {"tokens": torch.randint(0, 1000, (4, 128))}
    batch2 = {"tokens": torch.randint(0, 1000, (4, 256))}
    batch3 = {"tokens": torch.randint(0, 1000, (4, 512))}

    data_iter1 = iter([batch1])
    _, seq_len1 = prepare_finetuning_batch(data_iter1, 2, 2048, "tokens")
    assert seq_len1 == 128

    data_iter2 = iter([batch2])
    _, seq_len2 = prepare_finetuning_batch(data_iter2, 2, 2048, "tokens")
    assert seq_len2 == 256

    data_iter3 = iter([batch3])
    _, seq_len3 = prepare_finetuning_batch(data_iter3, 2, 2048, "tokens")
    assert seq_len3 == 512


def test_prepare_finetuning_batch_preserves_all_keys():
    """Test that all keys in the batch are preserved in microbatches."""
    global_batch = {
        "tokens": torch.randint(0, 1000, (6, 100)),
        "labels": torch.randint(0, 1000, (6, 100)),
        "loss_mask": torch.ones(6, 100),
        "attention_mask": torch.ones(6, 100),
        "position_ids": torch.arange(100).unsqueeze(0).expand(6, -1),
    }

    data_iter = iter([global_batch])

    microbatch_iter, _ = prepare_finetuning_batch(
        data_iterator=data_iter,
        num_microbatches=3,
        default_seq_length=2048,
    )

    microbatches = list(microbatch_iter)

    # Each microbatch should have all keys
    for mb in microbatches:
        assert set(mb.keys()) == set(global_batch.keys())
        assert mb["tokens"].shape == (2, 100)
        assert mb["labels"].shape == (2, 100)
        assert mb["loss_mask"].shape == (2, 100)
        assert mb["attention_mask"].shape == (2, 100)
        assert mb["position_ids"].shape == (2, 100)
