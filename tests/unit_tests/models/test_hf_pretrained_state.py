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

import pytest
import torch
from safetensors import safe_open

from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource, _resolve_output_shard_path


pytestmark = pytest.mark.unit


def _write_safetensors_index(tmp_path, weight_map: dict[str, str]) -> None:
    index_file = tmp_path / "model.safetensors.index.json"
    index_file.write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")


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
