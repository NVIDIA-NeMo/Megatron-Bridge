# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


from unittest.mock import patch

import pytest
from transformers.configuration_utils import PretrainedConfig

from megatron.bridge.models.conversion.utils import (
    get_causal_lm_class_name_via_auto_map,
    is_modelopt_dynamic_module,
    moe_experts_stored_packed,
)


class DummyConfig(PretrainedConfig):
    def __init__(self, auto_map=None, name_or_path=None, architectures=None):
        super().__init__()
        if auto_map is not None:
            self.auto_map = auto_map
        if name_or_path is not None:
            setattr(self, "_name_or_path", name_or_path)
        if architectures is not None:
            self.architectures = architectures


def test_returns_none_when_auto_map_absent():
    config = DummyConfig(auto_map=None)
    result = get_causal_lm_class_name_via_auto_map(config=config)
    assert result is None


def test_returns_class_name_when_auto_map_present():
    config = DummyConfig(auto_map={"AutoModelForCausalLM": "some.module.Class"}, name_or_path=None)
    result = get_causal_lm_class_name_via_auto_map(config=config)
    assert result == "Class"


def test_splits_on_last_dot():
    config = DummyConfig(
        auto_map={"AutoModelForCausalLM": "pkg.subpkg.module.DeepClass"},
        name_or_path="repo/id",
    )
    result = get_causal_lm_class_name_via_auto_map(config)
    assert result == "DeepClass"


def test_returns_none_when_key_missing():
    config = DummyConfig(auto_map={"AutoModel": "some.module.Class"}, name_or_path="repo/id")
    result = get_causal_lm_class_name_via_auto_map(config)
    assert result is None


def test_is_modelopt_dynamic_module_returns_false_when_modelopt_not_installed():
    import builtins

    real_import = builtins.__import__

    def _block_modelopt(name, *args, **kwargs):
        if name == "modelopt.torch.opt.dynamic" or name.startswith("modelopt"):
            raise ImportError("No module named 'modelopt'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_block_modelopt):
        assert is_modelopt_dynamic_module(object()) is False


class _FakePretrained:
    """Minimal stand-in exposing state.source.get_all_keys() for moe_experts_stored_packed."""

    def __init__(self, keys):
        source = type("Source", (), {"get_all_keys": lambda self: keys})()
        self.state = type("State", (), {"source": source})()


def _expert_keys(layers_prefix, packed):
    if packed:
        return [f"{layers_prefix}0.mlp.experts.gate_up_proj", f"{layers_prefix}0.mlp.experts.down_proj"]
    return [
        f"{layers_prefix}0.mlp.experts.0.gate_proj.weight",
        f"{layers_prefix}0.mlp.experts.0.up_proj.weight",
        f"{layers_prefix}0.mlp.experts.0.down_proj.weight",
    ]


def test_moe_experts_stored_packed_detects_fused():
    hf = _FakePretrained(_expert_keys("model.layers.", packed=True))
    assert moe_experts_stored_packed(hf, "model.layers.") is True


def test_moe_experts_stored_packed_detects_per_expert():
    hf = _FakePretrained(_expert_keys("model.layers.", packed=False))
    assert moe_experts_stored_packed(hf, "model.layers.") is False


def test_moe_experts_stored_packed_vlm_prefix():
    prefix = "model.language_model.layers."
    assert moe_experts_stored_packed(_FakePretrained(_expert_keys(prefix, packed=True)), prefix) is True
    assert moe_experts_stored_packed(_FakePretrained(_expert_keys(prefix, packed=False)), prefix) is False


def test_moe_experts_stored_packed_returns_default_when_unavailable():
    # No routed-expert keys, or no usable source -> the provided default.
    assert moe_experts_stored_packed(_FakePretrained([]), "model.layers.", default=False) is False
    assert moe_experts_stored_packed(_FakePretrained([]), "model.layers.", default=True) is True
    assert moe_experts_stored_packed(None, "model.layers.", default=True) is True


def test_moe_experts_stored_packed_raises_on_mixed_layout():
    keys = _expert_keys("model.layers.", packed=True) + _expert_keys("model.layers.", packed=False)
    with pytest.raises(ValueError, match="mixes fused and per-expert"):
        moe_experts_stored_packed(_FakePretrained(keys), "model.layers.")
