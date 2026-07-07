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

from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from megatron.bridge.data.builders.finetuning_dataset import FinetuningDatasetBuilder
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs


def _set_distributed_visibility(monkeypatch, mutate_remote=lambda _: None):
    calls = {"count": 0}
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)

    def fake_all_gather_object(output, local_state):
        calls["count"] += 1
        assert local_state["rank"] == 1
        remote_state = deepcopy(local_state)
        remote_state.update(rank=0, local_rank="0", hostname="node-a")
        mutate_remote(remote_state)
        output[:] = [remote_state, local_state]

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)
    return calls


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


def test_preflight_rejects_inconsistent_distributed_visibility(tmp_path, monkeypatch):
    builder = FinetuningDatasetBuilder(
        dataset_root=tmp_path,
        tokenizer=MagicMock(),
        do_validation=False,
        do_test=False,
    )
    missing_path = tmp_path / "training.jsonl"
    monkeypatch.setenv("SLURM_LOCALID", "0")
    monkeypatch.setattr("socket.gethostname", lambda: "node-b")
    monkeypatch.setattr(builder, "_path_exists", lambda _: False)

    def make_remote_path_visible(remote_state):
        remote_state["paths"]["training data"]["exists"] = True

    calls = _set_distributed_visibility(monkeypatch, make_remote_path_visible)

    with pytest.raises(RuntimeError) as error:
        builder._preflight_dataset_paths()

    message = str(error.value)
    assert "Dataset path visibility is inconsistent" in message
    assert "rank 1 (local rank 0, host node-b" in message
    assert str(missing_path) in message
    assert "NEMO_HOME" in message
    assert "NEMO_DATASETS_CACHE" in message
    assert calls["count"] == 1


def test_preflight_gathers_all_paths_once_and_preserves_all_missing(tmp_path, monkeypatch):
    builder = FinetuningDatasetBuilder(dataset_root=tmp_path, tokenizer=MagicMock())
    missing_path = tmp_path / "training.jsonl"
    calls = _set_distributed_visibility(monkeypatch)

    path_exists = builder._preflight_dataset_paths()

    assert path_exists == {
        "test data": False,
        "training data": False,
        "validation data": False,
    }
    assert calls["count"] == 1
    assert (
        builder._create_dataset(
            missing_path,
            path_exists=path_exists["training data"],
        )
        is None
    )


def test_preflight_gathers_probe_errors_before_raising(tmp_path, monkeypatch):
    builder = FinetuningDatasetBuilder(
        dataset_root=tmp_path,
        tokenizer=MagicMock(),
        do_validation=False,
        do_test=False,
    )
    monkeypatch.setattr(builder, "_path_exists", MagicMock(side_effect=OSError("storage unavailable")))

    def make_remote_path_visible(remote_state):
        remote_state["paths"]["training data"].update(exists=True, error=None)

    calls = _set_distributed_visibility(monkeypatch, make_remote_path_visible)

    with pytest.raises(RuntimeError, match="OSError: storage unavailable"):
        builder._preflight_dataset_paths()

    assert calls["count"] == 1


def test_preflight_validates_required_packed_metadata(tmp_path, monkeypatch):
    packed_path = tmp_path / "training.npy"
    packed_path.touch()
    metadata_path = tmp_path / "metadata.jsonl"
    builder = FinetuningDatasetBuilder(
        dataset_root=tmp_path,
        tokenizer=MagicMock(),
        enable_offline_packing=True,
        offline_packing_specs=PackedSequenceSpecs(
            packed_sequence_size=128,
            packed_train_data_path=packed_path,
            packed_metadata_path=metadata_path,
        ),
        do_validation=False,
        do_test=False,
    )

    def make_remote_metadata_visible(remote_state):
        remote_state["paths"]["training packed metadata"]["exists"] = True

    calls = _set_distributed_visibility(monkeypatch, make_remote_metadata_visible)

    with pytest.raises(RuntimeError) as error:
        builder._preflight_dataset_paths()

    assert "training packed metadata" in str(error.value)
    assert str(metadata_path) in str(error.value)
    assert calls["count"] == 1


def test_preflight_bounds_large_world_diagnostics(tmp_path, monkeypatch):
    builder = FinetuningDatasetBuilder(
        dataset_root=tmp_path,
        tokenizer=MagicMock(),
        do_validation=False,
        do_test=False,
    )
    monkeypatch.setattr(builder, "_path_exists", lambda _: False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 20)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 19)

    def fake_all_gather_object(output, local_state):
        states = []
        for rank in range(20):
            state = deepcopy(local_state)
            state.update(rank=rank, local_rank=str(rank % 8), hostname=f"node-{rank // 8}")
            state["paths"]["training data"]["exists"] = rank < 10
            states.append(state)
        output[:] = states

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)

    with pytest.raises(RuntimeError) as error:
        builder._preflight_dataset_paths()

    message = str(error.value)
    assert message.count("... and 2 more rank(s)") == 2
    assert "Current rank state: rank 19" in message
