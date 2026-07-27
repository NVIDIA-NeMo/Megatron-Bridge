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

import importlib.util
import json
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).parents[3] / "examples" / "conversion" / "create_hf_toy_model.py"
_SPEC = importlib.util.spec_from_file_location("create_hf_toy_model_under_test", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


@pytest.mark.parametrize("nested", [False, True])
def test_truncate_config_supports_flat_and_nested_text_configs(tmp_path: Path, nested: bool) -> None:
    """The toy-model helper truncates text-only and multimodal language configs."""
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    output_dir.mkdir()
    language_config = {
        "num_hidden_layers": 8,
        "max_window_layers": 6,
        "layer_types": ["full_attention", "linear_attention"] * 4,
        "linear_attn_config": {
            "full_attn_layers": [1, 4, 8],
            "kda_layers": [2, 3, 5, 6, 7],
        },
        "mlp_only_layers": [0, 3, 6],
    }
    config = {"model_type": "test", "text_config": language_config} if nested else language_config
    (source_dir / "config.json").write_text(json.dumps(config))

    original_num_hidden_layers = _MODULE._truncate_config(source_dir, output_dir, num_hidden_layers=4)

    output_config = json.loads((output_dir / "config.json").read_text())
    output_language_config = output_config["text_config"] if nested else output_config
    assert original_num_hidden_layers == 8
    assert output_language_config["num_hidden_layers"] == 4
    assert output_language_config["max_window_layers"] == 4
    assert output_language_config["layer_types"] == [
        "full_attention",
        "linear_attention",
        "full_attention",
        "linear_attention",
    ]
    assert output_language_config["mlp_only_layers"] == [0, 3]
    assert output_language_config["linear_attn_config"] == {
        "full_attn_layers": [1, 4],
        "kda_layers": [2, 3],
    }


def test_truncate_config_rejects_config_without_language_layers(tmp_path: Path) -> None:
    """A config without a transformer-layer count fails with an actionable error."""
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    output_dir.mkdir()
    (source_dir / "config.json").write_text(json.dumps({"model_type": "test"}))

    with pytest.raises(ValueError, match="top level or under text_config"):
        _MODULE._truncate_config(source_dir, output_dir, num_hidden_layers=4)


def test_select_hub_files_downloads_only_retained_layer_shards() -> None:
    """Hub sources avoid downloading shards that contain only removed layers."""
    repo_files = [
        "README.md",
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
        "nested/ignored.json",
    ]
    index = {
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00003.safetensors",
            "model.layers.3.self_attn.q_proj.weight": "model-00002-of-00003.safetensors",
            "model.layers.4.self_attn.q_proj.weight": "model-00003-of-00003.safetensors",
        }
    }

    selected = _MODULE._select_hub_files(repo_files, index, num_hidden_layers=4)

    assert selected == [
        "README.md",
        "config.json",
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model.safetensors.index.json",
    ]
