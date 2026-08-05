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
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.conversion_mapping import (
    MergeModulelist,
    WeightConverter,
    WeightRenaming,
    register_checkpoint_conversion_mapping,
)


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


def test_load_converted_megatron_export_uses_transformers_mapping(verify_adapter_module, monkeypatch) -> None:
    """Native checkpoint tensors go through Transformers dynamic conversions."""
    expected_model = torch.nn.Linear(2, 2)
    state_dict = {
        "backbone.weight": torch.ones(2, 2),
        "backbone.experts.0.weight": torch.ones(2, 2),
        "mtp.layers.0.weight": torch.ones(2, 2),
    }

    expected_config = object()

    class ModelClass:
        @staticmethod
        def from_pretrained(model_path, **kwargs):
            assert model_path is None
            assert kwargs["config"] is expected_config
            assert kwargs["state_dict"] is state_dict
            assert kwargs["torch_dtype"] == torch.float32
            assert kwargs["output_loading_info"] is True
            return expected_model, {
                "missing_keys": [],
                "unexpected_keys": ["mtp.layers.0.weight"],
                "mismatched_keys": [],
                "error_msgs": [],
            }

    monkeypatch.setattr(
        verify_adapter_module.AutoConfig,
        "from_pretrained",
        lambda model_path, **kwargs: expected_config,
    )
    monkeypatch.setattr(verify_adapter_module, "_resolve_causal_lm_class", lambda *args, **kwargs: ModelClass)

    model, omitted_keys = verify_adapter_module._load_converted_megatron_export(
        "native-checkpoint",
        state_dict,
        trust_remote_code=True,
    )

    assert model is expected_model
    assert omitted_keys == ["mtp.layers.0.weight"]


def test_load_converted_megatron_export_packs_experts(verify_adapter_module, monkeypatch) -> None:
    """Transformers stacks native per-expert tensors into fused model parameters."""

    class ToyConfig(PretrainedConfig):
        model_type = "adapter_verifier_packed_experts_test"

    class ToyExperts(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up_proj = nn.Parameter(torch.empty(2, 2, 3))
            self.down_proj = nn.Parameter(torch.empty(2, 3, 2))

    class ToyBody(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = ToyExperts()

    class ToyModel(PreTrainedModel):
        config_class = ToyConfig

        def __init__(self, config: ToyConfig) -> None:
            super().__init__(config)
            self.model = ToyBody()
            self.post_init()

    register_checkpoint_conversion_mapping(
        ToyConfig.model_type,
        [
            WeightRenaming("backbone.", "model."),
            WeightConverter(
                "experts.*.up_proj.weight",
                "experts.up_proj",
                [MergeModulelist(dim=0)],
            ),
            WeightConverter(
                "experts.*.down_proj.weight",
                "experts.down_proj",
                [MergeModulelist(dim=0)],
            ),
        ],
        overwrite=True,
    )
    config = ToyConfig(experts_implementation="eager")
    state_dict = {
        "backbone.experts.0.up_proj.weight": torch.full((2, 3), 1.0),
        "backbone.experts.1.up_proj.weight": torch.full((2, 3), 2.0),
        "backbone.experts.0.down_proj.weight": torch.full((3, 2), 3.0),
        "backbone.experts.1.down_proj.weight": torch.full((3, 2), 4.0),
        "mtp.layers.0.weight": torch.ones(1),
    }

    monkeypatch.setattr(verify_adapter_module.AutoConfig, "from_pretrained", lambda *args, **kwargs: config)

    monkeypatch.setattr(verify_adapter_module, "_resolve_causal_lm_class", lambda *args, **kwargs: ToyModel)

    model, omitted_keys = verify_adapter_module._load_converted_megatron_export(
        "native-checkpoint",
        state_dict,
        trust_remote_code=False,
    )

    assert omitted_keys == ["mtp.layers.0.weight"]
    assert torch.equal(model.model.experts.up_proj[0], state_dict["backbone.experts.0.up_proj.weight"])
    assert torch.equal(model.model.experts.up_proj[1], state_dict["backbone.experts.1.up_proj.weight"])
    assert torch.equal(model.model.experts.down_proj[0], state_dict["backbone.experts.0.down_proj.weight"])
    assert torch.equal(model.model.experts.down_proj[1], state_dict["backbone.experts.1.down_proj.weight"])


def test_resolve_causal_lm_class_uses_remote_auto_map(verify_adapter_module, monkeypatch) -> None:
    """Trusted custom checkpoints resolve their concrete class from the original path."""

    class CustomConfig:
        auto_map = {"AutoModelForCausalLM": "modeling_custom.CustomForCausalLM"}

    expected_class = object()

    def get_class(class_reference, model_path):
        assert class_reference == "modeling_custom.CustomForCausalLM"
        assert model_path == "custom-checkpoint"
        return expected_class

    monkeypatch.setattr(verify_adapter_module, "get_class_from_dynamic_module", get_class)

    model_class = verify_adapter_module._resolve_causal_lm_class(
        CustomConfig(),
        "custom-checkpoint",
        trust_remote_code=True,
    )

    assert model_class is expected_class


@pytest.mark.parametrize("failure_key", ["mismatched_keys", "error_msgs"])
def test_load_converted_megatron_export_rejects_transformers_failures(
    verify_adapter_module,
    monkeypatch,
    failure_key: str,
) -> None:
    """Transformers shape mismatches and conversion errors remain hard failures."""
    loading_info = {
        "missing_keys": [],
        "unexpected_keys": [],
        "mismatched_keys": [],
        "error_msgs": [],
    }
    loading_info[failure_key] = ["conversion failed"]
    monkeypatch.setattr(
        verify_adapter_module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )

    class ModelClass:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return torch.nn.Linear(2, 2), loading_info

    monkeypatch.setattr(verify_adapter_module, "_resolve_causal_lm_class", lambda *args, **kwargs: ModelClass)

    with pytest.raises(RuntimeError, match="could not convert"):
        verify_adapter_module._load_converted_megatron_export(
            "native-checkpoint",
            {},
            trust_remote_code=False,
        )


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
