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

"""Unit tests for the synthetic variable-length dataset provider."""

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.data.base import DatasetBuildContext
from megatron.bridge.data.builders.synthetic_varlen import SyntheticVarlenDataset, SyntheticVarlenDatasetConfig
from megatron.bridge.data.collators.identity import identity_collate


pytestmark = pytest.mark.unit


def _dataset(**overrides) -> SyntheticVarlenDataset:
    kwargs = dict(
        num_samples=64,
        seq_length=1024,
        min_seq_length=8,
        median_seq_length=128,
        lognormal_sigma=1.0,
        length_distribution="lognormal",
        pad_to_multiple=16,
        vocab_size=1000,
        seed=7,
        fold_padding_into_sequence=False,
    )
    kwargs.update(overrides)
    return SyntheticVarlenDataset(**kwargs)


def test_sample_schema_dtypes_and_padding():
    ds = _dataset()
    for idx in range(len(ds)):
        sample = ds[idx]
        assert set(sample) == {"tokens", "labels", "loss_mask", "position_ids", "original_seq_len", "padded_seq_len"}
        length = int(sample["original_seq_len"][0])
        padded = int(sample["padded_seq_len"][0])
        assert sample["original_seq_len"].dtype == torch.int32 and sample["original_seq_len"].shape == (1,)
        assert sample["tokens"].dtype == torch.int64 and sample["tokens"].shape == (padded,)
        assert sample["labels"].dtype == torch.int64 and sample["position_ids"].dtype == torch.int64
        assert sample["loss_mask"].dtype == torch.float32
        assert 8 <= length <= padded <= 1024 and padded % 16 == 0 and padded - length < 16
        assert int(sample["tokens"].max()) < 1000 and int(sample["tokens"][:length].min()) >= 1
        assert torch.all(sample["tokens"][length:] == 0) and torch.all(sample["loss_mask"][length - 1 :] == 0)
        assert torch.equal(sample["labels"][: length - 1], sample["tokens"][1:length])
        assert torch.equal(sample["position_ids"], torch.arange(padded))


def test_samples_are_deterministic_and_length_varies():
    ds = _dataset()
    again = _dataset()
    lengths = [ds.sample_length(i) for i in range(64)]
    assert lengths == [again.sample_length(i) for i in range(64)]
    assert len(set(lengths)) > 8
    assert torch.equal(ds[3]["tokens"], again[3]["tokens"])


def test_fold_padding_reports_padded_length():
    ds = _dataset(fold_padding_into_sequence=True)
    sample = ds[1]
    assert torch.equal(sample["original_seq_len"], sample["padded_seq_len"])


def test_uniform_distribution_and_cap():
    ds = _dataset(length_distribution="uniform", seq_length=64, pad_to_multiple=8, median_seq_length=32)
    lengths = [ds.sample_length(i) for i in range(64)]
    assert min(lengths) >= 8 and max(lengths) <= 64
    assert all(int(ds[i]["padded_seq_len"][0]) <= 64 for i in range(64))


def test_rejects_seq_length_not_multiple_of_padding():
    with pytest.raises(ValueError, match="multiple"):
        _dataset(seq_length=1000, pad_to_multiple=16)


def test_collate_is_identity():
    ds = _dataset()
    assert ds.collate_fn is identity_collate
    batch = ds.collate_fn([ds[0], ds[1]])
    assert isinstance(batch, list) and len(batch) == 2


def test_config_builds_splits_from_context():
    config = SyntheticVarlenDatasetConfig(seq_length=2048, min_seq_length=16)
    config.finalize()
    assert config.median_seq_length == 256 and config.dataloader_type == "single"
    pg = SimpleNamespace(dp=SimpleNamespace(size=lambda: 2), cp=SimpleNamespace(size=lambda: 4))
    context = DatasetBuildContext(
        train_samples=32, valid_samples=8, test_samples=0, tokenizer=SimpleNamespace(vocab_size=512), pg_collection=pg
    )
    train, valid, test = config.build_datasets(context)
    assert len(train) == 32 and len(valid) == 8 and test is None
    assert train.pad_to_multiple == 16 and train.vocab_size == 512
    assert train.seed != valid.seed
    # Split seeds are mixed non-linearly: no train sample equals a validation sample at any offset.
    valid_tokens = {tuple(valid[i]["tokens"].tolist()) for i in range(len(valid))}
    assert all(tuple(train[i]["tokens"].tolist()) not in valid_tokens for i in range(len(train)))


def test_config_prefers_explicit_padding_multiple_and_rejects_batch_loader():
    config = SyntheticVarlenDatasetConfig(seq_length=2048, sequence_padding_multiple=32, vocab_size=100)
    config.finalize()
    train, _, _ = config.build_datasets(DatasetBuildContext(train_samples=4, valid_samples=0, test_samples=0))
    assert train.pad_to_multiple == 32
    bad = SyntheticVarlenDatasetConfig(seq_length=2048, dataloader_type="batch")
    with pytest.raises(ValueError, match="dataloader_type"):
        bad.finalize()
