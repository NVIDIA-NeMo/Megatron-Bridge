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

"""Tests of the ERNIE 5.5 export of the shared MTP body copies that build no model (``ernie55_mappings``).

``conftest.py`` leaves ``test_ernie55_*.py`` out of the default collection, because building an ERNIE 5.5 model needs
Megatron-Core dev and CI runs Megatron-Core main. The cases here build no model: they read a config and call the
export hooks of the registered bridge on hand-built tensors and conversion tasks. They live outside that pattern so
that CI runs them, and can go back into ``test_ernie55_mappings.py`` once the exclusion is lifted. They cover:

* the names under which a checkpoint may store the shared MTP body a second time;
* the export hook, which adds the copies the source checkpoint holds to the tensors of a task;
* the note of whether the exported model has an MTP layer, taken from the tasks the bridge builds and from the
  tasks a caller supplies.

The cases that export a toy model are ``TestExportOfReleasedCheckpoints`` and ``TestExportOfAModelWithoutMtp`` in
``test_ernie55_mappings.py``.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.unit_tests.models.ernie.ernie55_test_utils import allow_conversion_imports_without_te


# Bridge's conversion modules import its LoRA layers, which need Transformer Engine at import time.
allow_conversion_imports_without_te()

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge  # noqa: E402
from megatron.bridge.models.ernie import Ernie55Bridge  # noqa: E402
from megatron.bridge.models.ernie import ernie55_mappings as mappings  # noqa: E402
from megatron.bridge.models.hf_pretrained.state import StateDict  # noqa: E402


pytestmark = pytest.mark.unit

_FIXTURES = Path(__file__).parent / "fixtures"
_PROXY_CONFIG = json.loads((_FIXTURES / "ernie55_proxy_config.json").read_text())
_PROXY_DTYPES = json.loads((_FIXTURES / "ernie55_proxy_checkpoint_dtypes.json").read_text())["tensors"]
# Megatron parameter names of a model with an MTP layer, and of one built without MTP.
_DECODER_PARAM = "decoder.layers.0.inner_layer.input_layernorm.weight"
_MTP_PARAM = "mtp.layers.0.enorm.weight"


def _bridge() -> Ernie55Bridge:
    """The registered ERNIE 5.5 bridge with the proxy config as its Hugging Face config; no model is built."""
    bridge = Ernie55Bridge()
    bridge.hf_config = _PROXY_CONFIG
    return bridge


def _tasks(*params: str) -> list:
    return [SimpleNamespace(global_param_name=name) for name in params]


class TestCopyNames:
    def test_mtp_body_copy_names(self):
        """Each tensor of the shared MTP body maps to its copies' names under the MTP layer, and to nothing else."""
        names = mappings.ernie55_mtp_body_copy_names(_PROXY_CONFIG)
        last, mtp = (
            f"model.layers.{_PROXY_CONFIG['num_hidden_layers'] - 1}.",
            f"model.layers.{_PROXY_CONFIG['num_hidden_layers']}.",
        )
        assert names and all(name.startswith(last) and name in _PROXY_DTYPES for name in names)
        copies = [copy for group in names.values() for copy in group]
        assert len(copies) == len(set(copies))
        assert set(copies) == mappings.ernie55_mtp_body_copy_keys(_PROXY_CONFIG)
        # never a tensor the registry writes: the MTP layer owns the router bias that has the body's name
        assert not set(copies) & set(_PROXY_DTYPES)
        router_bias = "block_sparse_moe.e_score_correction_bias"
        assert {last + router_bias, mtp + router_bias} <= set(_PROXY_DTYPES) and last + router_bias not in names
        for name, group in names.items():
            body = name[len(last) :]
            spellings = {body, f"transformer_layer.{body}"} if "_hyper_connection." in body else {body}
            assert {copy[len(mtp) :] for copy in group} == spellings


class TestExportHook:
    """``Ernie55Bridge.maybe_modify_converted_hf_weight`` on hand-built exports."""

    def test_adds_the_copies_the_source_holds(self):
        names = mappings.ernie55_mtp_body_copy_names(_PROXY_CONFIG)
        last = f"model.layers.{_PROXY_CONFIG['num_hidden_layers'] - 1}."
        experts = sorted(name for name in names if ".experts." in name)[:4]
        hyper = next(name for name in names if "_hyper_connection." in name)
        router_bias = f"{last}block_sparse_moe.e_score_correction_bias"
        other = "model.layers.0.input_layernorm.weight"
        # one task may export several tensors (the experts of every EP rank), in more than one dtype
        converted = {name: torch.randn(2, 3) for name in (*experts, router_bias, other)}
        converted[hyper] = torch.randn(4, dtype=torch.float64)
        # the source holds one spelling of the hyper-connection copy and the copies of two of the four experts
        held = {names[hyper][-1], *(names[name][0] for name in experts[:2])}
        source = StateDict({name: torch.zeros(1) for name in (*converted, *held)})
        result = _bridge().maybe_modify_converted_hf_weight(None, dict(converted), source)
        assert set(result) == set(converted) | held
        for body in (hyper, *experts[:2]):
            copy = (set(names[body]) & held).pop()
            assert torch.equal(result[copy], converted[body]) and result[copy].dtype == converted[body].dtype
            # two tensors: safetensors refuses to write two names that share one storage
            assert result[copy].data_ptr() != result[body].data_ptr()
        assert all(result[name] is converted[name] for name in converted)

    def test_adds_nothing_without_copies_to_write(self):
        names = mappings.ernie55_mtp_body_copy_names(_PROXY_CONFIG)
        body = sorted(names)[0]
        converted = {body: torch.randn(2, 3)}
        with_copy = StateDict({body: torch.zeros(1), names[body][0]: torch.zeros(1)})
        bridge = _bridge()
        assert bridge.maybe_modify_converted_hf_weight(None, {}, with_copy) == {}
        # a source checkpoint without the copy, and no source checkpoint at all (a config-only export)
        for source in (StateDict({body: torch.zeros(1)}), {}, None):
            assert bridge.maybe_modify_converted_hf_weight(None, converted, source) is converted
        assert set(bridge.maybe_modify_converted_hf_weight(None, converted, with_copy)) == {body, names[body][0]}
        # the copy names are kept per config: another config object is read again
        bridge.hf_config = dict(_PROXY_CONFIG)
        assert set(bridge.maybe_modify_converted_hf_weight(None, converted, with_copy)) == {body, names[body][0]}
        assert bridge._ernie55_mtp_body_copy_names[0] is bridge.hf_config
        # a model built without MTP exports no MTP layer, so no copies either
        bridge._ernie55_export_writes_mtp = False
        assert bridge.maybe_modify_converted_hf_weight(None, converted, with_copy) is converted


class TestWhetherTheModelHasMtp:
    """The copies belong to the MTP layer: they are written only when the export converts that layer's tensors."""

    def test_tasks_the_bridge_builds(self, monkeypatch):
        """``build_conversion_tasks`` notes whether the model has an MTP layer; every rank sees the same tasks."""
        bridge = _bridge()
        for tasks, has_mtp in ((_tasks(_DECODER_PARAM, _MTP_PARAM), True), (_tasks(_DECODER_PARAM), False)):
            seen = []
            monkeypatch.setattr(
                MegatronModelBridge,
                "build_conversion_tasks",
                lambda self, *a, _t=tasks, _seen=seen, **k: _seen.append((a, k)) or _t,
            )
            # the arguments reach the base method as they came
            assert bridge.build_conversion_tasks(None, [], torch.float32, exclude_param_patterns=("x",)) is tasks
            assert seen == [((None, [], torch.float32), {"exclude_param_patterns": ("x",)})]
            assert bridge._ernie55_export_writes_mtp is has_mtp

    def test_tasks_a_caller_supplies(self, monkeypatch):
        """An export with supplied tasks decides from those tasks, on a bridge that never built any."""
        base = inspect.signature(MegatronModelBridge._stream_weight_events_megatron_to_hf).parameters
        override = inspect.signature(Ernie55Bridge._stream_weight_events_megatron_to_hf).parameters
        # the override names every parameter of the base method: one added there must be added here too
        assert [(p.name, p.kind, p.default) for p in override.values()] == [
            (p.name, p.kind, p.default) for p in base.values()
        ]
        received = []

        def base_stream(self, *args, **kwargs):
            received.append((args, kwargs))
            yield from ()

        monkeypatch.setattr(MegatronModelBridge, "_stream_weight_events_megatron_to_hf", base_stream)
        bridge = _bridge()
        model, pretrained = [object()], object()
        options = {name: object() for name in list(base)[3:] if name != "conversion_tasks"}
        # a task list has ``None`` entries for the parameters a rank does not hold
        for tasks, has_mtp in (
            ([None, *_tasks(_DECODER_PARAM, _MTP_PARAM)], True),
            ([None, *_tasks(_DECODER_PARAM)], False),
            ([None, *_tasks(_DECODER_PARAM, _MTP_PARAM)], True),
        ):
            events = bridge._stream_weight_events_megatron_to_hf(model, pretrained, conversion_tasks=tasks, **options)
            assert list(events) == []
            # every argument reaches the base method as it came
            args, kwargs = received[-1]
            assert args == (model, pretrained) and kwargs.keys() == options.keys() | {"conversion_tasks"}
            assert kwargs["conversion_tasks"] is tasks and all(kwargs[name] is options[name] for name in options)
            assert bridge._ernie55_export_writes_mtp is has_mtp
        # without supplied tasks the export builds its own, and the note of build_conversion_tasks stands
        bridge._ernie55_export_writes_mtp = False
        assert list(bridge._stream_weight_events_megatron_to_hf(model, pretrained)) == []
        assert received[-1][1]["conversion_tasks"] is None
        assert bridge._ernie55_export_writes_mtp is False

    def test_the_note_decides_whether_the_hook_adds_copies(self, monkeypatch):
        names = mappings.ernie55_mtp_body_copy_names(_PROXY_CONFIG)
        body = sorted(names)[0]
        converted = {body: torch.randn(2, 3)}
        source = StateDict({body: torch.zeros(1), names[body][0]: torch.zeros(1)})
        bridge = _bridge()
        for tasks, added in ((_tasks(_DECODER_PARAM), set()), (_tasks(_DECODER_PARAM, _MTP_PARAM), {names[body][0]})):
            monkeypatch.setattr(MegatronModelBridge, "build_conversion_tasks", lambda self, *a, _t=tasks, **k: _t)
            bridge.build_conversion_tasks(None, [])
            assert set(bridge.maybe_modify_converted_hf_weight(None, converted, source)) == {body} | added
