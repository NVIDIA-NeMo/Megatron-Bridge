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
import torch

from megatron.bridge.data.builders.finetuning_dataset import FinetuningDatasetBuilder
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs


def _set_distributed_visibility(monkeypatch, visibility):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: len(visibility))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)

    def fake_all_gather_object(output, local_state):
        assert local_state["rank"] == 1
        output[:] = visibility[:-1] + [local_state]

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)


@pytest.mark.parametrize("mkdir_error", [FileExistsError, FileNotFoundError])
def test_default_pack_path_ignores_shared_fs_mkdir_race(tmp_path, monkeypatch, mkdir_error):
    """Network filesystems can leak mkdir races even with exist_ok=True."""
    builder = FinetuningDatasetBuilder(
        dataset_root=tmp_path,
        tokenizer=MagicMock(),
        enable_offline_packing=True,
        offline_packing_specs=PackedSequenceSpecs(
            packed_sequence_size=128,
            tokenizer_model_name="mock-tokenizer",
            pad_seq_to_mult=8,
        ),
    )
    expected_path = tmp_path / "packed" / "mock-tokenizer_pad_seq_to_mult8"

    monkeypatch.setattr(Path, "exists", lambda _: False)

    def raise_mkdir(self, parents=False, exist_ok=False):
        assert self == expected_path
        assert parents is True
        assert exist_ok is True
        raise mkdir_error("stale shared filesystem state")

    monkeypatch.setattr(Path, "mkdir", raise_mkdir)

    assert builder.default_pack_path == expected_path


def test_create_dataset_rejects_inconsistent_distributed_visibility(tmp_path, monkeypatch):
    builder = FinetuningDatasetBuilder(dataset_root=tmp_path, tokenizer=MagicMock())
    missing_path = tmp_path / "training.jsonl"
    monkeypatch.setenv("SLURM_LOCALID", "0")
    monkeypatch.setattr("socket.gethostname", lambda: "node-b")
    _set_distributed_visibility(
        monkeypatch,
        [
            {
                "rank": 0,
                "local_rank": "0",
                "hostname": "node-a",
                "path": str(missing_path),
                "exists": True,
            },
            None,
        ],
    )

    with pytest.raises(RuntimeError) as error:
        builder._create_dataset(missing_path)

    message = str(error.value)
    assert "Dataset path visibility is inconsistent" in message
    assert "rank 1 (local rank 0, host node-b" in message
    assert str(missing_path) in message
    assert "NEMO_HOME" in message


def test_create_dataset_preserves_none_when_path_is_missing_on_every_rank(tmp_path, monkeypatch):
    builder = FinetuningDatasetBuilder(dataset_root=tmp_path, tokenizer=MagicMock())
    missing_path = tmp_path / "training.jsonl"
    _set_distributed_visibility(
        monkeypatch,
        [
            {
                "rank": 0,
                "local_rank": "0",
                "hostname": "node-a",
                "path": str(missing_path),
                "exists": False,
            },
            None,
        ],
    )

    assert builder._create_dataset(missing_path) is None
