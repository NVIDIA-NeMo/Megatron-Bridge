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

"""Focused tests for the single-process HF/Megatron round-trip example."""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit

_SCRIPT = Path(__file__).resolve().parents[4] / "examples" / "conversion" / "hf_megatron_roundtrip.py"


def _load_roundtrip_module(monkeypatch: pytest.MonkeyPatch):
    stubs = {
        "megatron.bridge": types.ModuleType("megatron.bridge"),
        "megatron.bridge.models.conversion": types.ModuleType("megatron.bridge.models.conversion"),
        "megatron.bridge.models.hf_pretrained.utils": types.ModuleType("megatron.bridge.models.hf_pretrained.utils"),
    }
    stubs["megatron.bridge"].AutoBridge = object
    stubs["megatron.bridge.models.conversion"].weights_verification_table = lambda bridge, model: "table"
    stubs["megatron.bridge.models.hf_pretrained.utils"].is_safe_repo = lambda **kwargs: False
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("test_hf_megatron_roundtrip_example", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeBridge:
    def __init__(self, use_model_config: bool):
        self._model_bridge = SimpleNamespace(USE_MODEL_CONFIG_FOR_CONVERSION=use_model_config)
        self.calls = []

    def get_model(self, **kwargs):
        self.calls.append(("get_model", kwargs))
        return ["builder_model"]

    def to_megatron_model(self, **kwargs):
        if self._model_bridge.USE_MODEL_CONFIG_FOR_CONVERSION:
            raise NotImplementedError("provider path is not supported")
        self.calls.append(("to_megatron_model", kwargs))
        return ["provider_model"]

    def save_hf_pretrained(self, model, path):
        self.calls.append(("save", (model, path)))


def _run_main(monkeypatch: pytest.MonkeyPatch, use_model_config: bool) -> _FakeBridge:
    module = _load_roundtrip_module(monkeypatch)
    fake_bridge = _FakeBridge(use_model_config)
    monkeypatch.setattr(module, "AutoBridge", SimpleNamespace(from_hf_pretrained=lambda *a, **k: fake_bridge))
    monkeypatch.setattr(module.console, "print", lambda *args, **kwargs: None)
    module.main(hf_model_id="example/model", output_dir="/out")
    return fake_bridge


def test_roundtrip_builds_builder_backed_model_with_get_model(monkeypatch: pytest.MonkeyPatch):
    fake_bridge = _run_main(monkeypatch, use_model_config=True)

    assert fake_bridge.calls == [
        ("get_model", {"wrap_with_ddp": False}),
        ("save", (["builder_model"], "/out/model")),
    ]


def test_roundtrip_keeps_provider_path_for_legacy_bridges(monkeypatch: pytest.MonkeyPatch):
    fake_bridge = _run_main(monkeypatch, use_model_config=False)

    assert fake_bridge.calls == [
        ("to_megatron_model", {"wrap_with_ddp": False}),
        ("save", (["provider_model"], "/out/model")),
    ]
