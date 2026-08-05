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

"""Unit tests for adapter-export verification helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


_SCRIPT_PATH = Path(__file__).parents[3] / "examples" / "conversion" / "adapter" / "verify_adapter.py"


def _load_verify_adapter_module():
    spec = importlib.util.spec_from_file_location("verify_adapter_under_test", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def verify_adapter_module():
    """Load the adapter verification script as a module."""
    return _load_verify_adapter_module()


def test_load_megatron_export_accepts_only_training_mtp(verify_adapter_module) -> None:
    """Training-only MTP tensors may be absent from the HF inference class."""
    model = torch.nn.Linear(2, 2)
    state_dict = model.state_dict()
    state_dict["mtp.layers.0.weight"] = torch.ones(2, 2)

    omitted_keys = verify_adapter_module._load_megatron_export(model, state_dict)

    assert omitted_keys == ["mtp.layers.0.weight"]


def test_load_megatron_export_aligns_transformers_v5_dynamic_namespace(verify_adapter_module) -> None:
    """Native backbone tensors load into the Transformers v5 model namespace."""

    class DynamicWeightConversionModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Linear(2, 2)

    model = DynamicWeightConversionModel()
    expected_weight = torch.full_like(model.model.weight, 3.0)
    expected_bias = torch.full_like(model.model.bias, 4.0)
    state_dict = {
        "backbone.weight": expected_weight,
        "backbone.bias": expected_bias,
        "mtp.layers.0.weight": torch.ones(2, 2),
    }

    omitted_keys = verify_adapter_module._load_megatron_export(model, state_dict)

    assert omitted_keys == ["mtp.layers.0.weight"]
    assert torch.equal(model.model.weight, expected_weight)
    assert torch.equal(model.model.bias, expected_bias)


def test_load_megatron_export_rejects_dynamic_namespace_collision(verify_adapter_module) -> None:
    """Dynamic namespace alignment must not silently overwrite a tensor."""

    class DynamicWeightConversionModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Linear(2, 2)

    model = DynamicWeightConversionModel()
    state_dict = {
        "model.weight": torch.ones_like(model.model.weight),
        "backbone.weight": torch.zeros_like(model.model.weight),
        "model.bias": torch.ones_like(model.model.bias),
    }

    with pytest.raises(RuntimeError, match="duplicate tensors"):
        verify_adapter_module._load_megatron_export(model, state_dict)


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("missing", "missing="),
        ("unexpected", "unexpected="),
    ],
)
def test_load_megatron_export_rejects_other_mismatches(
    verify_adapter_module,
    mutation: str,
    expected_error: str,
) -> None:
    """Missing inference tensors and unrelated extras remain hard failures."""
    model = torch.nn.Linear(2, 2)
    state_dict = model.state_dict()
    if mutation == "missing":
        del state_dict["bias"]
    else:
        state_dict["other.weight"] = torch.ones(2, 2)

    with pytest.raises(RuntimeError, match=expected_error):
        verify_adapter_module._load_megatron_export(model, state_dict)
