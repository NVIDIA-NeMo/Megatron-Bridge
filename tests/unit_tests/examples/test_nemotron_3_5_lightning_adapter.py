# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Unit tests for the Nemotron 3.5 Lightning adapter merge."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


_SCRIPT_DIR = Path(__file__).parents[3] / "examples" / "models" / "nemotron" / "nemotron_3_5_lightning"
_SCRIPT_PATH = _SCRIPT_DIR / "merge_adapter.py"


def _load_merge_adapter_module():
    sys.path.insert(0, str(_SCRIPT_DIR))
    try:
        spec = importlib.util.spec_from_file_location("nemotron_lightning_merge_under_test", _SCRIPT_PATH)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(_SCRIPT_DIR))


def _write_index(path: Path, weight_map: dict[str, str], total_size: int) -> None:
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map})
    )


def test_merge_training_only_mtp_weights_preserves_native_dtypes(tmp_path: Path) -> None:
    """Merged and unchanged MTP tensors retain their checkpoint dtypes."""
    module = _load_merge_adapter_module()
    base_path = tmp_path / "base"
    adapter_path = tmp_path / "adapter"
    output_path = tmp_path / "output"
    base_path.mkdir()
    adapter_path.mkdir()
    output_path.mkdir()

    base_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    base_bias = torch.tensor([0.25, -0.5], dtype=torch.float32)
    base_state = {
        "mtp.layers.0.linear.weight": base_weight,
        "mtp.layers.0.router_bias": base_bias,
    }
    save_file(base_state, base_path / "model.safetensors")
    _write_index(
        base_path,
        dict.fromkeys(base_state, "model.safetensors"),
        sum(tensor.numel() * tensor.element_size() for tensor in base_state.values()),
    )

    output_tensor = torch.ones(2, 2, dtype=torch.bfloat16)
    save_file({"model.linear.weight": output_tensor}, output_path / "model.safetensors")
    output_size = output_tensor.numel() * output_tensor.element_size()
    _write_index(output_path, {"model.linear.weight": "model.safetensors"}, output_size)

    adapter_state = {
        "base_model.model.mtp.layers.0.linear.lora_A.weight": torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
        "base_model.model.mtp.layers.0.linear.lora_B.weight": torch.tensor([[3.0], [4.0]], dtype=torch.bfloat16),
    }
    save_file(adapter_state, adapter_path / "adapter_model.safetensors")
    (adapter_path / "adapter_config.json").write_text(
        json.dumps({"r": 1, "lora_alpha": 1, "use_dora": False, "use_rslora": False})
    )

    merged_count, unchanged_count = module._merge_training_only_mtp_weights(
        hf_model=str(base_path),
        hf_revision="unused-for-local-model",
        adapter_path=adapter_path,
        output=output_path,
        device=torch.device("cpu"),
    )

    merged_state = load_file(output_path / "model-mtp.safetensors")
    expected_weight = base_weight + (
        adapter_state["base_model.model.mtp.layers.0.linear.lora_B.weight"].float()
        @ adapter_state["base_model.model.mtp.layers.0.linear.lora_A.weight"].float()
    ).to(torch.bfloat16)
    assert merged_count == 1
    assert unchanged_count == 1
    assert merged_state["mtp.layers.0.linear.weight"].dtype == torch.bfloat16
    assert torch.equal(merged_state["mtp.layers.0.linear.weight"], expected_weight)
    assert merged_state["mtp.layers.0.router_bias"].dtype == torch.float32
    assert torch.equal(merged_state["mtp.layers.0.router_bias"], base_bias)

    output_index = json.loads((output_path / "model.safetensors.index.json").read_text())
    mtp_size = sum(tensor.numel() * tensor.element_size() for tensor in merged_state.values())
    assert output_index["metadata"]["total_size"] == output_size + mtp_size
    assert set(output_index["weight_map"]) == {
        "model.linear.weight",
        "mtp.layers.0.linear.weight",
        "mtp.layers.0.router_bias",
    }
