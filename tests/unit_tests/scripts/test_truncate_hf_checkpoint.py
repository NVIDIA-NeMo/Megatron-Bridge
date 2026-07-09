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

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "examples" / "conversion" / "truncate_hf_checkpoint.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("truncate_hf_checkpoint_under_test", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_config(directory: Path, *, num_hidden_layers: int = 4) -> None:
    (directory / "config.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": num_hidden_layers,
                "max_window_layers": num_hidden_layers,
                "layer_types": ["full_attention"] * num_hidden_layers,
                "mlp_only_layers": [1, 3],
            }
        )
    )
    (directory / "tokenizer.json").write_text("{}")


def _checkpoint_tensors() -> dict[str, torch.Tensor]:
    return {
        "model.embed_tokens.weight": torch.arange(8, dtype=torch.float32).reshape(4, 2),
        "model.layers.0.self_attn.q_proj.weight": torch.full((2, 2), 10.0),
        "model.layers.1.self_attn.q_proj.weight": torch.full((2, 2), 11.0),
        "model.layers.2.self_attn.q_proj.weight": torch.full((2, 2), 12.0),
        "model.layers.3.self_attn.q_proj.weight": torch.full((2, 2), 13.0),
        "model.norm.weight": torch.ones(2),
    }


def _read_tensors(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {name: handle.get_tensor(name) for name in handle.keys()}


@pytest.mark.unit
def test_truncate_unsharded_checkpoint_preserves_retained_tensors(tmp_path: Path) -> None:
    module = _load_module()
    source = tmp_path / "source"
    source.mkdir()
    _write_config(source)
    source_tensors = _checkpoint_tensors()
    save_file(source_tensors, source / "model.safetensors", metadata={"format": "pt"})

    output = module.truncate_checkpoint(str(source), tmp_path / "output", num_hidden_layers=2)

    output_config = json.loads((output / "config.json").read_text())
    assert output_config["num_hidden_layers"] == 2
    assert output_config["max_window_layers"] == 2
    assert output_config["layer_types"] == ["full_attention", "full_attention"]
    assert output_config["mlp_only_layers"] == [1]
    assert (output / "tokenizer.json").read_text() == "{}"

    output_tensors = _read_tensors(output / "model.safetensors")
    assert set(output_tensors) == {
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
        "model.norm.weight",
    }
    for tensor_name, tensor in output_tensors.items():
        assert torch.equal(tensor, source_tensors[tensor_name])


@pytest.mark.unit
def test_truncate_sharded_checkpoint_rewrites_index_and_removes_empty_shards(tmp_path: Path) -> None:
    module = _load_module()
    source = tmp_path / "source"
    source.mkdir()
    _write_config(source)
    tensors = _checkpoint_tensors()
    first_shard = "model-00001-of-00003.safetensors"
    second_shard = "model-00002-of-00003.safetensors"
    third_shard = "model-00003-of-00003.safetensors"
    save_file(
        {name: tensor for name, tensor in tensors.items() if "layers.2" not in name and "layers.3" not in name},
        source / first_shard,
    )
    save_file(
        {"model.layers.2.self_attn.q_proj.weight": tensors["model.layers.2.self_attn.q_proj.weight"]},
        source / second_shard,
    )
    save_file(
        {"model.layers.3.self_attn.q_proj.weight": tensors["model.layers.3.self_attn.q_proj.weight"]},
        source / third_shard,
    )
    weight_map = {
        name: second_shard if "layers.2" in name else third_shard if "layers.3" in name else first_shard
        for name in tensors
    }
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 9999, "source": "unit-test"}, "weight_map": weight_map})
    )

    output = module.truncate_checkpoint(str(source), tmp_path / "output", num_hidden_layers=2)

    assert (output / first_shard).is_file()
    assert not (output / second_shard).exists()
    assert not (output / third_shard).exists()
    output_index = json.loads((output / "model.safetensors.index.json").read_text())
    assert set(output_index["weight_map"]) == set(_read_tensors(output / first_shard))
    assert set(output_index["weight_map"].values()) == {first_shard}
    assert output_index["metadata"]["total_size"] > 0
    assert output_index["metadata"]["total_size"] < 9999
    assert output_index["metadata"]["source"] == "unit-test"


@pytest.mark.unit
@pytest.mark.parametrize("num_hidden_layers", [0, 5])
def test_truncate_checkpoint_rejects_invalid_layer_count(tmp_path: Path, num_hidden_layers: int) -> None:
    module = _load_module()
    source = tmp_path / "source"
    source.mkdir()
    _write_config(source)

    with pytest.raises(ValueError, match="num_hidden_layers must be between"):
        module.truncate_checkpoint(
            str(source), tmp_path / f"output-{num_hidden_layers}", num_hidden_layers=num_hidden_layers
        )


@pytest.mark.unit
def test_truncate_checkpoint_rejects_unrecognized_layer_tensor_names(tmp_path: Path) -> None:
    module = _load_module()
    source = tmp_path / "source"
    source.mkdir()
    _write_config(source)
    save_file({"decoder.blocks.0.weight": torch.ones(2)}, source / "model.safetensors")

    with pytest.raises(ValueError, match="No layer tensors were removed"):
        module.truncate_checkpoint(str(source), tmp_path / "output", num_hidden_layers=2)
