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
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from megatron.bridge.models.hf_pretrained import state as state_module
from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource, _resolve_output_shard_path


pytestmark = pytest.mark.unit


def _write_safetensors_index(tmp_path, weight_map: dict[str, str], metadata: dict[str, object] | None = None) -> None:
    index_file = tmp_path / "model.safetensors.index.json"
    index: dict[str, object] = {"weight_map": weight_map}
    if metadata is not None:
        index["metadata"] = metadata
    index_file.write_text(json.dumps(index), encoding="utf-8")


def _run_destination_writer_rank(
    rank: int,
    world_size: int,
    init_method: str,
    output_path: str,
    save_every_n_ranks: int,
) -> None:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        tensors = [(f"model.weight_{index}", torch.full((4,), index, dtype=torch.float32)) for index in range(4)]
        state_module.save_generator_to_safetensors(
            iter(tensors),
            output_path,
            distributed_save=True,
            save_every_n_ranks=save_every_n_ranks,
            max_shard_size_bytes=20,
            expected_keys={name for name, _tensor in tensors},
        )
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize(
    "filename",
    [
        "../evil.safetensors",
        "nested/../../evil.safetensors",
        "/tmp/evil.safetensors",
        "C:/tmp/evil.safetensors",
        "nested\\evil.safetensors",
    ],
)
def test_safetensors_index_rejects_escaping_shard_filenames(tmp_path, filename: str) -> None:
    _write_safetensors_index(tmp_path, {"model.weight": filename})

    source = SafeTensorsStateSource(tmp_path)

    with pytest.raises(ValueError, match="relative path within the checkpoint directory"):
        _ = source.key_to_filename_map


def test_safetensors_index_rejects_non_safetensors_shard_filename(tmp_path) -> None:
    _write_safetensors_index(tmp_path, {"model.weight": "evil.pth"})

    source = SafeTensorsStateSource(tmp_path)

    with pytest.raises(ValueError, match="must end with '.safetensors'"):
        _ = source.key_to_filename_map


def test_safetensors_index_accepts_relative_safetensors_shard_filename(tmp_path) -> None:
    _write_safetensors_index(tmp_path, {"model.weight": "nested/model-00001-of-00002.safetensors"})

    source = SafeTensorsStateSource(tmp_path)

    assert source.key_to_filename_map == {"model.weight": "nested/model-00001-of-00002.safetensors"}


def test_resolve_output_shard_path_rejects_escaping_filename(tmp_path) -> None:
    with pytest.raises(ValueError, match="escapes output directory"):
        _resolve_output_shard_path(tmp_path, "../evil.safetensors")


def test_resolve_output_shard_path_accepts_nested_safetensors_filename(tmp_path) -> None:
    output_path = _resolve_output_shard_path(tmp_path, "nested/model-00001-of-00002.safetensors")

    assert output_path == tmp_path.resolve() / "nested/model-00001-of-00002.safetensors"


def test_save_generator_strict_false_writes_nested_partial_shard(tmp_path) -> None:
    shard_filename = "nested/model-00001-of-00001.safetensors"
    _write_safetensors_index(
        tmp_path,
        {
            "model.present": shard_filename,
            "model.missing": shard_filename,
        },
    )
    source = SafeTensorsStateSource(tmp_path)
    output_path = tmp_path / "output"

    source.save_generator(
        iter([("model.present", torch.ones(1))]),
        output_path,
        strict=False,
    )

    saved_shard = output_path / shard_filename
    assert saved_shard.exists()
    with safe_open(saved_shard, framework="pt", device="cpu") as shard:
        assert set(shard.keys()) == {"model.present"}
        torch.testing.assert_close(shard.get_tensor("model.present"), torch.ones(1))

    index_data = json.loads((output_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert index_data["weight_map"] == {"model.present": shard_filename}


@pytest.mark.parametrize("distributed_save", [False, True])
def test_save_generator_recomputes_index_total_size(tmp_path, monkeypatch, distributed_save: bool) -> None:
    first_shard = "model-00001-of-00002.safetensors"
    second_shard = "model-00002-of-00002.safetensors"
    _write_safetensors_index(
        tmp_path,
        {
            "model.embed_tokens.weight": first_shard,
            "lm_head.weight": second_shard,
        },
        metadata={"format": "pt", "total_size": 1},
    )
    source = SafeTensorsStateSource(tmp_path)
    output_path = tmp_path / "output"
    tensors = {
        "model.embed_tokens.weight": torch.ones((3, 2), dtype=torch.bfloat16),
        "lm_head.weight": torch.ones((2, 2), dtype=torch.float32),
    }
    if distributed_save:
        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
        monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

        def gather_rank_zero(output: list[object | None], value: object) -> None:
            output[0] = value

        monkeypatch.setattr(torch.distributed, "all_gather_object", gather_rank_zero)

    source.save_generator(iter(tensors.items()), output_path, distributed_save=distributed_save)

    index_data = json.loads((output_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    expected_total_size = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    assert index_data["metadata"] == {"format": "pt", "total_size": expected_total_size}


def test_save_generator_writes_shard_as_soon_as_its_remaining_keys_arrive(tmp_path, monkeypatch) -> None:
    class _SubsetForbiddenSet(set[str]):
        def issubset(self, _other: Iterable[object]) -> bool:
            raise AssertionError("save_generator must not scan shard subsets for each yielded tensor")

    def tracking_defaultdict(_default_factory: object) -> defaultdict[str, _SubsetForbiddenSet]:
        return defaultdict(_SubsetForbiddenSet)

    first_shard = "model-00001-of-00002.safetensors"
    second_shard = "model-00002-of-00002.safetensors"
    _write_safetensors_index(
        tmp_path,
        {
            "model.first.weight": first_shard,
            "model.first.bias": first_shard,
            "model.second.weight": second_shard,
            "model.second.bias": second_shard,
        },
    )
    source = SafeTensorsStateSource(tmp_path)
    saved_shards: list[tuple[str, set[str]]] = []

    def record_save(tensors: dict[str, torch.Tensor], output_file: str | Path) -> None:
        saved_shards.append((str(output_file), set(tensors)))

    monkeypatch.setattr(state_module, "defaultdict", tracking_defaultdict)
    monkeypatch.setattr("safetensors.torch.save_file", record_save)

    def tensors() -> Iterator[tuple[str, torch.Tensor]]:
        yield "model.first.weight", torch.ones(1)
        assert saved_shards == []

        yield "model.second.weight", torch.full((1,), 2.0)
        assert saved_shards == []

        yield "model.first.bias", torch.zeros(1)
        assert saved_shards == [(str(tmp_path / "output" / first_shard), {"model.first.weight", "model.first.bias"})]

        yield "model.second.bias", torch.full((1,), 3.0)

    source.save_generator(tensors(), tmp_path / "output")

    assert saved_shards == [
        (str(tmp_path / "output" / first_shard), {"model.first.weight", "model.first.bias"}),
        (str(tmp_path / "output" / second_shard), {"model.second.weight", "model.second.bias"}),
    ]


def test_destination_streaming_writer_saves_single_shard_without_index(tmp_path) -> None:
    tensors = {
        "model.embed_tokens.weight": torch.arange(6, dtype=torch.float32).reshape(3, 2),
        "lm_head.weight": torch.ones((2, 2), dtype=torch.bfloat16),
    }

    state_module.save_generator_to_safetensors(iter(tensors.items()), tmp_path)

    shard_path = tmp_path / "model.safetensors"
    assert shard_path.exists()
    assert not (tmp_path / "model.safetensors.index.json").exists()
    with safe_open(shard_path, framework="pt", device="cpu") as shard:
        assert set(shard.keys()) == set(tensors)
        for name, expected in tensors.items():
            torch.testing.assert_close(shard.get_tensor(name), expected)


def test_destination_streaming_writer_shards_and_indexes_oversized_tensor(tmp_path) -> None:
    tensors = {
        "model.small_a": torch.arange(4, dtype=torch.float32),
        "model.oversized": torch.arange(12, dtype=torch.float32),
        "model.small_b": torch.arange(2, dtype=torch.float32),
    }

    state_module.save_generator_to_safetensors(
        iter(tensors.items()),
        tmp_path,
        max_shard_size_bytes=20,
    )

    index = json.loads((tmp_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert index["metadata"]["total_size"] == sum(
        tensor.numel() * tensor.element_size() for tensor in tensors.values()
    )
    assert index["weight_map"] == {
        "model.oversized": "model-00002-of-00003.safetensors",
        "model.small_a": "model-00001-of-00003.safetensors",
        "model.small_b": "model-00003-of-00003.safetensors",
    }
    assert not list(tmp_path.glob("*.tmp"))
    for name, filename in index["weight_map"].items():
        with safe_open(tmp_path / filename, framework="pt", device="cpu") as shard:
            torch.testing.assert_close(shard.get_tensor(name), tensors[name])


def test_destination_streaming_writer_rejects_duplicate_names(tmp_path) -> None:
    tensors = iter(
        [
            ("model.weight", torch.ones(1)),
            ("model.weight", torch.zeros(1)),
        ]
    )

    with pytest.raises(ValueError, match="duplicate tensor name"):
        state_module.save_generator_to_safetensors(tensors, tmp_path)


def test_destination_streaming_writer_rejects_strict_key_mismatch(tmp_path) -> None:
    with pytest.raises(KeyError, match="missing=.*model.expected"):
        state_module.save_generator_to_safetensors(
            iter([("model.actual", torch.ones(1))]),
            tmp_path,
            expected_keys={"model.expected"},
        )
    assert not list(tmp_path.glob("*.tmp"))


def test_destination_streaming_writer_supports_world_size_one_distributed_save(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

    def gather_rank_zero(output: list[object | None], value: object) -> None:
        output[0] = value

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather_rank_zero)

    state_module.save_generator_to_safetensors(
        iter([("model.weight", torch.ones(4))]),
        tmp_path,
        distributed_save=True,
    )

    assert (tmp_path / "model.safetensors").exists()


@pytest.mark.skipif(
    not torch.distributed.is_gloo_available(), reason="Gloo is required for the multi-rank writer test."
)
@pytest.mark.parametrize("save_every_n_ranks", [1, 2])
def test_destination_streaming_writer_distributes_destination_shards(
    tmp_path,
    save_every_n_ranks: int,
) -> None:
    output_path = tmp_path / "output"
    init_method = f"file://{tmp_path / 'gloo_init'}"

    torch.multiprocessing.start_processes(
        _run_destination_writer_rank,
        args=(2, init_method, str(output_path), save_every_n_ranks),
        nprocs=2,
        join=True,
        start_method="fork",
    )

    index = json.loads((output_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert index["weight_map"] == {
        f"model.weight_{index}": f"model-{index + 1:05d}-of-00004.safetensors" for index in range(4)
    }
    assert not list(output_path.glob("*.tmp"))
    for index, filename in enumerate(index["weight_map"].values()):
        with safe_open(output_path / filename, framework="pt", device="cpu") as shard:
            torch.testing.assert_close(
                shard.get_tensor(f"model.weight_{index}"),
                torch.full((4,), index, dtype=torch.float32),
            )
