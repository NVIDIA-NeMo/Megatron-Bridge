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

"""Regression tests for None-task handling in ``load_weights_hf_to_megatron`` (NVBug 6367442).

``build_conversion_tasks`` is typed ``List[None | WeightConversionTask]`` and intentionally
leaves a slot as ``None`` when no mapping matches a parameter. The consuming loop in
``load_weights_hf_to_megatron`` must skip such ``None`` slots; previously it accessed
``task.megatron_module`` directly, raising
``AttributeError: 'NoneType' object has no attribute 'megatron_module'`` during HF->Megatron
weight loading (observed for Gemma4 dense conversion and inference).
"""

from types import SimpleNamespace

import torch

from megatron.bridge.models.conversion import model_bridge as model_bridge_mod
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge


class _DummyBridge(MegatronModelBridge):
    def provider_bridge(self, hf_pretrained):  # pragma: no cover - not used in these tests
        return None

    def mapping_registry(self):  # pragma: no cover - not used in these tests
        return MegatronMappingRegistry()


def _bridge_loading(monkeypatch, tasks):
    """Build a bridge whose ``build_conversion_tasks`` yields ``tasks`` with trivial plumbing."""
    bridge = _DummyBridge()
    monkeypatch.setattr(bridge, "build_conversion_tasks", lambda *a, **k: tasks)
    monkeypatch.setattr(bridge, "_with_progress_tracking", lambda iterable, description: iterable)
    # ``unwrap_model`` is module-level in model_bridge; pass the model list through unchanged.
    monkeypatch.setattr(model_bridge_mod, "unwrap_model", lambda m: m)
    # After the task loop, load_weights broadcasts shared embeddings across ranks; stub it out so
    # the test needs no distributed/parallel-state init (we are only exercising the task loop).
    monkeypatch.setattr(bridge, "_broadcast_shared_embeddings", lambda models: None)
    return bridge


def test_load_weights_skips_none_task(monkeypatch):
    """A ``None`` task (no mapping matched) is skipped instead of dereferenced.

    Before the fix this raised ``AttributeError: 'NoneType' object has no attribute
    'megatron_module'``.
    """
    megatron_model = [torch.nn.Identity()]  # not FSDP -> FSDP branch skipped
    # First slot is ``None`` (no mapping); second is a task whose module is not on this rank.
    tasks = [None, SimpleNamespace(megatron_module=None)]
    bridge = _bridge_loading(monkeypatch, tasks)
    hf_pretrained = SimpleNamespace(model_name_or_path="dummy")  # no ``state`` -> empty state dict

    result = bridge.load_weights_hf_to_megatron(hf_pretrained, megatron_model)

    assert result is megatron_model  # completed without error; both slots skipped


def test_load_weights_processes_real_task_after_none(monkeypatch):
    """A real task following a ``None`` slot is still processed (the guard continues, not breaks)."""
    megatron_model = [torch.nn.Identity()]
    calls = []
    mapping = SimpleNamespace(hf_param="hf.weight", is_grouped_export=False)
    mapping.hf_to_megatron = lambda weights, module: calls.append("converted")
    real_task = SimpleNamespace(megatron_module=object(), mapping=mapping, param_weight=None, param_name="p")
    tasks = [None, real_task]
    bridge = _bridge_loading(monkeypatch, tasks)
    monkeypatch.setattr(
        bridge,
        "maybe_modify_loaded_hf_weight",
        lambda hf_param, state_dict: calls.append("fetched") or torch.zeros(1),
    )
    hf_pretrained = SimpleNamespace(model_name_or_path="dummy")

    bridge.load_weights_hf_to_megatron(hf_pretrained, megatron_model)

    # The None slot was skipped first, then the real task's mapping was exercised.
    assert calls == ["fetched", "converted"]
