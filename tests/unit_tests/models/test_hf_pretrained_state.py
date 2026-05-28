import json

import pytest

from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource


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
