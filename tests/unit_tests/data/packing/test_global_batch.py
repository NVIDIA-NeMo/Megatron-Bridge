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

"""Unit tests for the global-batch packing data contract."""

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.data.builders import GPTSFTDatasetConfig
from megatron.bridge.data.datasets.gpt_sft import GPTSFTDataset
from megatron.bridge.data.packing.global_batch import (
    REQUIRED_SAMPLE_KEYS,
    build_unpacked_sequence_sample,
    fold_alignment_padding,
    global_batch_packing_padding_multiple,
    identity_collate,
    make_unpacked_collate,
)


pytestmark = pytest.mark.unit


def test_identity_collate_keeps_samples_separate():
    samples = [{"tokens": torch.zeros(3)}, {"tokens": torch.zeros(5)}]
    collated = identity_collate(samples)
    assert collated == samples and collated is not samples


def test_build_unpacked_sequence_sample_pads_to_multiple_and_masks_padding():
    sample = build_unpacked_sequence_sample(
        [5, 6, 7, 8, 9], [6, 7, 8, 9, 0], [0, 1, 1, 1, 1], pad_to_multiple_of=4, pad_token_id=99
    )
    assert set(sample) == set(REQUIRED_SAMPLE_KEYS)
    assert sample["tokens"].tolist() == [5, 6, 7, 8, 9, 99, 99, 99]
    assert sample["labels"].tolist() == [6, 7, 8, 9, 0, 99, 99, 99]
    assert sample["loss_mask"].tolist() == [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert sample["position_ids"].tolist() == list(range(8))
    assert sample["original_seq_len"].dtype == torch.int32 and sample["original_seq_len"].tolist() == [5]
    assert sample["padded_seq_len"].tolist() == [8]
    assert sample["tokens"].dtype == torch.int64 and sample["loss_mask"].dtype == torch.float32


def test_build_unpacked_sequence_sample_rejects_bad_input():
    with pytest.raises(ValueError, match="empty"):
        build_unpacked_sequence_sample([], [], [], pad_to_multiple_of=1, pad_token_id=0)
    with pytest.raises(ValueError, match="same length"):
        build_unpacked_sequence_sample([1, 2], [1], [1, 1], pad_to_multiple_of=1, pad_token_id=0)


def test_fold_alignment_padding_reports_padded_length():
    sample = build_unpacked_sequence_sample([1, 2, 3], [2, 3, 0], [1, 1, 0], pad_to_multiple_of=4, pad_token_id=0)
    folded = fold_alignment_padding(sample)
    assert folded["original_seq_len"].tolist() == [4] == folded["padded_seq_len"].tolist()
    assert folded["loss_mask"].tolist() == [1.0, 1.0, 0.0, 0.0]


def test_make_unpacked_collate_optionally_folds():
    sample = build_unpacked_sequence_sample([1, 2, 3], [2, 3, 0], [1, 1, 0], pad_to_multiple_of=4, pad_token_id=0)
    assert make_unpacked_collate(fold_padding=False) is identity_collate
    [folded] = make_unpacked_collate(fold_padding=True)([sample])
    assert folded["original_seq_len"].tolist() == [4]


@pytest.mark.parametrize(
    ("dynamic_cp", "dp", "cp", "tp", "sp", "expected"),
    [
        (False, 2, 1, 1, False, 1),
        (False, 2, 4, 1, False, 8),
        (True, 2, 4, 1, False, 16),
        (True, 2, 4, 2, True, 32),
    ],
)
def test_padding_multiple_matches_megatron_divisor(dynamic_cp, dp, cp, tp, sp, expected):
    assert (
        global_batch_packing_padding_multiple(
            dynamic_cp=dynamic_cp, dp_size=dp, cp_size=cp, tp_size=tp, sequence_parallel=sp
        )
        == expected
    )


def _unpacked_dataset(*, fold: bool) -> GPTSFTDataset:
    dataset = GPTSFTDataset.__new__(GPTSFTDataset)
    dataset.max_seq_length = 16
    dataset.enable_global_batch_packing = True
    dataset.enable_in_batch_packing = False
    dataset.global_batch_packing_pad_to_multiple_of = 8
    dataset.fold_alignment_padding = fold
    dataset.tokenizer = SimpleNamespace(eos_id=2)
    dataset._build_loss_mask = lambda item: item["loss_mask"]
    return dataset


def test_gpt_sft_dataset_yields_unpacked_rows():
    dataset = _unpacked_dataset(fold=False)
    batch = [
        {"input_ids": [10, 11, 12, 13, 14, 15], "loss_mask": [0, 0, 0, 1, 1, 1]},
        {"input_ids": list(range(100, 120)), "loss_mask": [1] * 20},  # truncated to max_seq_length
    ]
    rows = dataset.collate_fn(batch)
    assert isinstance(rows, list) and len(rows) == 2
    first, second = rows
    assert first["tokens"].tolist() == [10, 11, 12, 13, 14, 2, 2, 2]
    assert first["labels"].tolist() == [11, 12, 13, 14, 15, 2, 2, 2]
    assert first["loss_mask"].tolist() == [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert first["original_seq_len"].tolist() == [5] and first["padded_seq_len"].tolist() == [8]
    assert second["original_seq_len"].tolist() == [16] and second["padded_seq_len"].tolist() == [16]
    assert set(first) == set(REQUIRED_SAMPLE_KEYS)


def test_gpt_sft_dataset_folds_alignment_padding_when_asked():
    dataset = _unpacked_dataset(fold=True)
    [row] = dataset.collate_fn([{"input_ids": [10, 11, 12, 13, 14, 15], "loss_mask": [0, 0, 0, 1, 1, 1]}])
    assert row["original_seq_len"].tolist() == [8] == row["padded_seq_len"].tolist()


def test_gpt_sft_dataset_config_declares_capability_and_exclusivity():
    config = GPTSFTDatasetConfig(
        seq_length=4096, dataset_root="/tmp/x", enable_global_batch_packing=True, dataloader_type="single"
    )
    assert config.yields_unpacked_samples is True
    config.validate()
    with pytest.raises(ValueError, match="mutually exclusive"):
        GPTSFTDatasetConfig(
            seq_length=4096,
            dataset_root="/tmp/x",
            enable_global_batch_packing=True,
            enable_in_batch_packing=True,
            dataloader_type="single",
        ).validate()
    with pytest.raises(ValueError, match="dataloader_type"):
        GPTSFTDatasetConfig(
            seq_length=4096, dataset_root="/tmp/x", enable_global_batch_packing=True, dataloader_type="batch"
        ).validate()
    assert GPTSFTDatasetConfig(seq_length=4096, dataset_root="/tmp/x").yields_unpacked_samples is False
