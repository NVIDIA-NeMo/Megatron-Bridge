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

"""Synthetic variable-length SFT samples for global-batch packing."""

import numpy as np
import pytest
import torch

from megatron.bridge.data.base import DatasetBuildContext
from megatron.bridge.data.builders import MockGPTSFTDatasetBuilder, MockGPTSFTDatasetConfig
from megatron.bridge.data.builders.mock_gpt_sft import (
    MockGPTSFTDataset,
    length_bucket_bounds,
    length_bucket_counts,
    mock_gpt_sft_train_valid_test_datasets_provider,
    mock_sft_sequence_lengths,
)
from megatron.bridge.data.packing.global_batch import REQUIRED_SAMPLE_KEYS
from megatron.bridge.data.utils import get_dataset_provider


pytestmark = pytest.mark.unit


def _config(**overrides) -> MockGPTSFTDatasetConfig:
    return MockGPTSFTDatasetConfig(**{"seq_length": 1024, "vocab_size": 1000, **overrides})


def test_bucket_bounds_follow_cp_capacity():
    assert length_bucket_bounds(65536) == [(1, 16384), (16385, 32768), (32769, 65536)]


def test_bucket_counts_are_exact():
    assert length_bucket_counts(5120, (0.5, 0.3, 0.2)) == [2560, 1536, 1024]
    assert length_bucket_counts(256, (0.5, 0.3, 0.2)) == [128, 77, 51]
    assert sum(length_bucket_counts(7, (0.5, 0.3, 0.2))) == 7


def test_lengths_follow_the_bucket_split_and_are_deterministic():
    config = _config(seq_length=65536)
    lengths = mock_sft_sequence_lengths(config, 5120)

    counts = [int(((lengths >= low) & (lengths <= high)).sum()) for low, high in length_bucket_bounds(65536)]
    assert counts == [2560, 1536, 1024]
    shares = [
        lengths[(lengths >= low) & (lengths <= high)].sum() / lengths.sum()
        for low, high in length_bucket_bounds(65536)
    ]
    assert shares == pytest.approx([0.19, 0.35, 0.46], abs=0.02)
    assert np.array_equal(lengths, mock_sft_sequence_lengths(config, 5120))
    assert not np.array_equal(lengths, np.sort(lengths)), "buckets must be interleaved"
    assert not np.array_equal(lengths[:100], mock_sft_sequence_lengths(config, 5120, "validation")[:100])
    assert not np.array_equal(lengths, mock_sft_sequence_lengths(_config(seq_length=65536, random_seed=7), 5120))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"seq_length": 3}, "seq_length"),
        ({"vocab_size": 1}, "vocab_size"),
        ({"length_bucket_fractions": (0.5, 0.2, 0.2)}, "length_bucket_fractions"),
        ({"length_bucket_fractions": (1.2, -0.2, 0.0)}, "length_bucket_fractions"),
        ({"completion_fraction": 0.0}, "completion_fraction"),
        ({"enable_global_batch_packing": False}, "enable_global_batch_packing"),
        ({"global_batch_packing_pad_to_multiple_of": 0}, "global_batch_packing_pad_to_multiple_of"),
    ],
)
def test_invalid_settings_are_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        _config(**overrides).validate()


def test_samples_are_deterministic_random_tokens_of_the_drawn_length():
    dataset = MockGPTSFTDataset(_config(), 16, "train")

    for index in range(len(dataset)):
        ids = dataset[index]["input_ids"]
        assert ids.numel() == dataset.lengths[index] + 1
        assert int(ids.min()) >= 1 and int(ids.max()) < 1000
    assert torch.equal(dataset[3]["input_ids"], MockGPTSFTDataset(_config(), 16, "train")[3]["input_ids"])


def test_collate_shifts_once_supervises_the_completion_and_aligns_for_cp():
    config = _config(global_batch_packing_pad_to_multiple_of=8)
    dataset = MockGPTSFTDataset(config, 8, "train")
    batch = [dataset[i] for i in range(4)]

    samples = dataset.collate_fn(batch)

    assert len(samples) == 4
    for item, sample in zip(batch, samples):
        length = item["input_ids"].numel() - 1
        assert set(REQUIRED_SAMPLE_KEYS) <= set(sample)
        assert int(sample["original_seq_len"]) == length
        assert int(sample["padded_seq_len"]) % 8 == 0 and int(sample["padded_seq_len"]) >= length
        assert torch.equal(sample["tokens"][:length], item["input_ids"][:-1])
        assert torch.equal(sample["labels"][:length], item["input_ids"][1:])
        supervised = int(sample["loss_mask"].sum())
        assert supervised == max(1, round(length * 0.2))
        assert sample["loss_mask"][length - 1] == 1 and sample["loss_mask"][length:].sum() == 0


def test_fold_alignment_padding_reports_the_padded_length():
    config = _config(global_batch_packing_pad_to_multiple_of=8, fold_alignment_padding=True)
    dataset = MockGPTSFTDataset(config, 4, "train")

    sample = dataset.collate_fn([dataset[0]])[0]

    assert int(sample["original_seq_len"]) == int(sample["padded_seq_len"])


def test_builder_builds_requested_splits_and_dispatch_finds_the_provider():
    config = _config()

    train, valid, test = MockGPTSFTDatasetBuilder(config).build(
        DatasetBuildContext(train_samples=12, valid_samples=0, test_samples=3)
    )

    assert (len(train), valid, len(test)) == (12, None, 3)
    assert get_dataset_provider(config) is mock_gpt_sft_train_valid_test_datasets_provider
    assert config.yields_unpacked_samples is True
