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

import json
import os
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock

import pyarrow.parquet as pq
import pytest

from megatron.bridge.data.builders import ChatSFTPreprocessingConfig, GPTSFTDatasetConfig
from megatron.bridge.data.builders import gpt_sft as builder_mod
from megatron.bridge.data.builders.gpt_sft import GPTSFTDatasetBuilder, _GPTSFTDataBlend, build_gpt_sft_split
from megatron.bridge.data.datasets.gpt_sft import GPTSFTBlendDataset
from megatron.bridge.data.packing import PackedSequenceSpecs
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer


@pytest.mark.parametrize("mkdir_error", [FileExistsError, FileNotFoundError])
def test_default_pack_path_ignores_shared_fs_mkdir_race(tmp_path, monkeypatch, mkdir_error):
    """Network filesystems can leak mkdir races even with exist_ok=True."""
    builder = GPTSFTDatasetBuilder(
        config=GPTSFTDatasetConfig(
            dataset_root=tmp_path,
            seq_length=2048,
            enable_offline_packing=True,
            offline_packing_specs=PackedSequenceSpecs(
                packed_sequence_size=128,
                tokenizer_model_name="mock-tokenizer",
                pad_seq_to_mult=8,
            ),
        ),
        tokenizer=MagicMock(),
    )
    expected_path = tmp_path / "packed" / f"mock-tokenizer_pad_seq_to_mult8_sft_{builder._packing_fingerprint}"

    monkeypatch.setattr(Path, "exists", lambda _: False)

    def raise_mkdir(self, parents=False, exist_ok=False):
        assert self == expected_path
        assert parents is True
        assert exist_ok is True
        raise mkdir_error("stale shared filesystem state")

    monkeypatch.setattr(Path, "mkdir", raise_mkdir)

    assert builder.default_pack_path == expected_path


def test_default_pack_path_fingerprints_preprocessing(tmp_path):
    specs = PackedSequenceSpecs(
        packed_sequence_size=128,
        tokenizer_model_name="mock-tokenizer",
        pad_seq_to_mult=8,
    )
    prompt_builder = GPTSFTDatasetBuilder(
        config=GPTSFTDatasetConfig(
            dataset_root=tmp_path,
            seq_length=2048,
            enable_offline_packing=True,
            offline_packing_specs=specs,
        ),
        tokenizer=MagicMock(),
    )
    chat_builder = GPTSFTDatasetBuilder(
        config=GPTSFTDatasetConfig(
            dataset_root=tmp_path,
            seq_length=2048,
            preprocessing=ChatSFTPreprocessingConfig(),
            enable_offline_packing=True,
            offline_packing_specs=specs,
        ),
        tokenizer=MagicMock(),
    )

    assert prompt_builder.default_pack_path != chat_builder.default_pack_path


def test_default_pack_path_fingerprints_max_single_sequence_length(tmp_path):
    """Different single-sequence caps must not reuse the same packed artifact."""

    def build(max_single_sequence_length: int) -> GPTSFTDatasetBuilder:
        return GPTSFTDatasetBuilder(
            config=GPTSFTDatasetConfig(
                dataset_root=tmp_path,
                seq_length=128,
                enable_offline_packing=True,
                offline_packing_specs=PackedSequenceSpecs(
                    packed_sequence_size=128,
                    max_single_sequence_length=max_single_sequence_length,
                    tokenizer_model_name="mock-tokenizer",
                ),
            ),
            tokenizer=MagicMock(),
        )

    assert build(120).default_pack_path != build(112).default_pack_path


def test_default_pack_path_is_stable_for_equivalent_non_hf_tokenizers(tmp_path):
    def build() -> GPTSFTDatasetBuilder:
        tokenizer = build_tokenizer(TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=128))
        return GPTSFTDatasetBuilder(
            config=GPTSFTDatasetConfig(
                dataset_root=tmp_path,
                seq_length=128,
                enable_offline_packing=True,
                offline_packing_specs=PackedSequenceSpecs(packed_sequence_size=128),
            ),
            tokenizer=tokenizer,
        )

    first = build()
    second = build()
    assert first.default_pack_path == second.default_pack_path


def test_default_pack_path_fingerprints_blend_paths_weights_and_source_sizes(tmp_path, monkeypatch):
    monkeypatch.setattr(builder_mod, "get_dataset_root", lambda name: tmp_path / "cache" / name)
    train_a = tmp_path / "train-a.jsonl"
    train_b = tmp_path / "train-b.jsonl"
    train_a.write_text('{"input": "a", "output": "a"}\n')
    train_b.write_text('{"input": "b", "output": "b"}\n')
    args_path = tmp_path / "per-split.json"

    def build(weights, *, first_source=train_a):
        args_path.write_text(
            json.dumps(
                {
                    "train": [str(weights[0]), str(first_source), str(weights[1]), str(train_b)],
                }
            )
        )
        return GPTSFTDatasetBuilder(
            config=GPTSFTDatasetConfig(
                seq_length=128,
                per_split_data_source_manifest_path=args_path,
                enable_offline_packing=True,
                offline_packing_specs=PackedSequenceSpecs(
                    packed_sequence_size=128,
                    tokenizer_model_name="mock-tokenizer",
                ),
                do_validation=False,
                do_test=False,
            ),
            tokenizer=MagicMock(),
        )

    equal_blend = build((1, 1))
    weighted_blend = build((3, 1))

    assert equal_blend.default_pack_path != weighted_blend.default_pack_path

    original_path = weighted_blend.default_pack_path
    renamed_train_a = tmp_path / "renamed-train-a.jsonl"
    renamed_train_a.write_text(train_a.read_text())
    path_changed_source = build((3, 1), first_source=renamed_train_a)
    assert original_path != path_changed_source.default_pack_path

    source_stat = train_a.stat()
    os.utime(train_a, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns + 1_000_000_000))
    assert train_a.stat().st_mtime_ns != source_stat.st_mtime_ns
    mtime_changed_source = build((3, 1))
    assert original_path == mtime_changed_source.default_pack_path

    train_a.write_text('{"input": "changed", "output": "a"}\n')
    changed_source = build((3, 1))
    assert original_path != changed_source.default_pack_path


def test_offline_packing_consumes_one_blended_raw_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(builder_mod, "get_dataset_root", lambda name: tmp_path / "cache" / name)
    train_a = tmp_path / "train-a.jsonl"
    train_b = tmp_path / "train-b.jsonl"
    for path in (train_a, train_b):
        path.write_text('{"input": "x", "output": "y"}\n')
    args_path = tmp_path / "per-split.json"
    args_path.write_text(json.dumps({"train": ["3", str(train_a), "1", str(train_b)]}))

    builder = GPTSFTDatasetBuilder(
        config=GPTSFTDatasetConfig(
            seq_length=128,
            per_split_data_source_manifest_path=args_path,
            enable_offline_packing=True,
            offline_packing_specs=PackedSequenceSpecs(
                packed_sequence_size=128,
                tokenizer_model_name="mock-tokenizer",
            ),
            do_validation=False,
            do_test=False,
        ),
        tokenizer=MagicMock(),
    )
    pack_calls = []
    monkeypatch.setattr(builder, "_packed_path_exists", lambda _: False)
    monkeypatch.setattr(
        "megatron.bridge.data.packing.offline.prepare_gpt_sft_packed_data",
        lambda **kwargs: pack_calls.append(kwargs),
    )

    builder.prepare_packed_data()

    assert len(pack_calls) == 1
    blend = pack_calls[0]["input_path"]
    assert blend.paths == (str(train_a), str(train_b))
    assert blend.weights == (3.0, 1.0)


def test_offline_packing_materializes_one_weighted_blend_parquet(tmp_path, monkeypatch):
    monkeypatch.setattr(builder_mod, "get_dataset_root", lambda name: tmp_path / "cache" / name)

    class PackingTokenizer:
        _tokenizer = object()
        eos_id = 127
        eod = 127
        bos_id = 126
        pad_id = 0
        space_sensitive = True
        legacy = True

        @staticmethod
        def text_to_ids(text):
            return [int(token) for token in text.split()]

    train_a = tmp_path / "train-a.jsonl"
    train_b = tmp_path / "train-b.jsonl"
    for path, source_token in ((train_a, 10), (train_b, 20)):
        path.write_text(
            "".join(json.dumps({"input": f"{source_token} 1", "output": f"2 {row}"}) + "\n" for row in range(4))
        )
    args_path = tmp_path / "per-split.json"
    args_path.write_text(json.dumps({"train": ["3", str(train_a), "1", str(train_b)]}))
    builder = GPTSFTDatasetBuilder(
        config=GPTSFTDatasetConfig(
            seq_length=32,
            per_split_data_source_manifest_path=args_path,
            enable_offline_packing=True,
            offline_packing_specs=PackedSequenceSpecs(
                packed_sequence_size=32,
                tokenizer_model_name="null-tokenizer",
                num_tokenizer_workers=1,
            ),
            do_validation=False,
            do_test=False,
        ),
        tokenizer=PackingTokenizer(),
    )

    builder.prepare_packed_data()

    packed_path = builder.train_path_packed
    table = pq.read_table(packed_path, columns=["input_ids", "seq_start_id"])
    source_tokens = Counter(
        input_ids[start]
        for input_ids, starts in zip(table["input_ids"].to_pylist(), table["seq_start_id"].to_pylist())
        for start in starts
    )
    assert source_tokens == {10: 6, 20: 2}
    assert list(packed_path.parent.glob("training_*.idx.parquet")) == [packed_path]


def test_build_gpt_sft_split_applies_max_samples_to_the_blend(tmp_path, monkeypatch):
    train_a = tmp_path / "train-a.jsonl"
    train_b = tmp_path / "train-b.jsonl"
    train_a.touch()
    train_b.touch()
    source_kwargs = []

    class TinyDataset:
        tokenizer = object()
        pad_seq_length_to_mult = 1

        def __init__(self, **kwargs):
            source_kwargs.append(kwargs)
            self.name = Path(kwargs["file_path"]).stem

        def __len__(self):
            return 4

        def __getitem__(self, index):
            return {"input_ids": [0, 1], "metadata": {"source": self.name, "index": index}}

        def collate_fn(self, batch):
            return batch

    monkeypatch.setattr(builder_mod, "GPTSFTDataset", TinyDataset)

    dataset = build_gpt_sft_split(
        _GPTSFTDataBlend(paths=(str(train_a), str(train_b)), weights=(3.0, 1.0)),
        tokenizer=object(),
        seq_length=128,
        memmap_workers=1,
        seed=17,
        packed_sequence_size=-1,
        dataset_kwargs={"max_num_samples": 12},
    )

    assert isinstance(dataset, GPTSFTBlendDataset)
    assert len(dataset) == 12
    assert all("max_num_samples" not in kwargs for kwargs in source_kwargs)
