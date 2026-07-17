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

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from megatron.bridge.data.builders import ChatSFTPreprocessingConfig, GPTSFTDatasetConfig
from megatron.bridge.data.builders.gpt_sft import GPTSFTDatasetBuilder
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
