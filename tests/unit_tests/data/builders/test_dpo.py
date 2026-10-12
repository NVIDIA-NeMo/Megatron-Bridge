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

"""Unit tests for DPODatasetConfig (the ConfigContainer.dataset entry for DPO)."""

import pytest

import megatron.bridge.data.builders.dpo as dpo_module
from megatron.bridge.data.base import DatasetBuildContext
from megatron.bridge.data.builders.dpo import (
    DPODatasetBuilder,
    DPODatasetConfig,
    dpo_train_valid_test_datasets_provider,
)
from megatron.bridge.data.sources.hf import HFDatasetSourceConfig
from tests.unit_tests.data.preference_fakes import ChatMLTokenizer


def chat_row(pair_id: int) -> dict:
    prompt = {"role": "user", "content": f"prompt {pair_id}"}
    return {
        "chosen": [prompt, {"role": "assistant", "content": "an answer"}],
        "rejected": [prompt, {"role": "assistant", "content": "worse"}],
    }


def ref_mapping(num_pairs: int) -> dict:
    return {
        pair_id: {
            "ref_chosen_logprob_sum": -1.0,
            "ref_chosen_num_tokens": 3,
            "ref_rejected_logprob_sum": -2.0,
            "ref_rejected_num_tokens": 3,
        }
        for pair_id in range(num_pairs)
    }


def test_provider_builds_each_split_from_its_own_scored_artifact(monkeypatch):
    """The registry resolves a DPODatasetConfig to the DPO provider, which loads each split's rows and
    reference artifact and yields datasets whose collate carries the DPO batch keys."""
    pytest.importorskip("megatron.core.datasets.blended_megatron_dataset_builder")
    from megatron.bridge.data.utils import get_dataset_provider

    config = DPODatasetConfig(
        seq_length=64,
        ref_artifact="/artifacts/train",
        source=HFDatasetSourceConfig(path_or_dataset="org/some-preference-set", split="train"),
        validation_ref_artifact="/artifacts/validation",
        validation_source=HFDatasetSourceConfig(path_or_dataset="org/some-preference-set", split="validation"),
    )
    rows = {"train": [chat_row(i) for i in range(6)], "validation": [chat_row(i) for i in range(4)]}
    loaded_artifacts = []

    def fake_load_ref_logprobs(path, expected_num_pairs):
        loaded_artifacts.append((path, expected_num_pairs))
        return ref_mapping(expected_num_pairs)

    monkeypatch.setattr(DPODatasetConfig, "load_source", lambda self, split="train": rows[split])
    monkeypatch.setattr(dpo_module, "load_ref_logprobs", fake_load_ref_logprobs)

    provider = get_dataset_provider(config)
    assert provider is dpo_train_valid_test_datasets_provider

    train_ds, valid_ds, test_ds = provider([0, 8, 0], config, tokenizer=ChatMLTokenizer())

    assert test_ds is None
    assert (len(train_ds), len(valid_ds)) == (6, 4)
    assert train_ds.require_ref_logprobs and valid_ds.require_ref_logprobs
    assert loaded_artifacts == [("/artifacts/train", 6), ("/artifacts/validation", 4)]

    batch = train_ds.collate_fn([train_ds[0], train_ds[1]])
    assert batch["tokens"].shape[0] == 4  # 2 pairs -> 4 interleaved rows
    for key in ("labels", "loss_mask", "pair_id", "loss_multiplier", "ref_logprob_sum", "ref_num_tokens"):
        assert key in batch, key


def test_builder_requires_the_run_tokenizer_to_carry_a_chat_template():
    """Pairs are rendered through the tokenizer's chat template, so a run without one cannot start."""
    config = DPODatasetConfig(
        seq_length=64,
        ref_artifact="/artifacts/train",
        source=HFDatasetSourceConfig(path_or_dataset="org/some-preference-set", split="train"),
    )
    context = DatasetBuildContext(train_samples=8, valid_samples=0, test_samples=0, tokenizer=None)

    with pytest.raises(ValueError, match="chat template"):
        DPODatasetBuilder(config).build(context)


def test_source_identity_names_file_backed_sources_by_their_files():
    """Two local files through the same loader must fingerprint differently; Hub sources keep their id."""
    local = DPODatasetConfig.resolve_source(
        {"path_or_dataset": "json", "load_kwargs": {"data_files": "/data/pairs.jsonl"}, "split": "train"},
        field="source",
    )
    assert isinstance(local, HFDatasetSourceConfig)
    assert DPODatasetConfig.source_identity(local) == ('json:"/data/pairs.jsonl"', "train")

    hub = DPODatasetConfig.resolve_source(
        HFDatasetSourceConfig(path_or_dataset="org/prefs", split="test"), field="source"
    )
    assert DPODatasetConfig.source_identity(hub) == ("org/prefs", "test")
