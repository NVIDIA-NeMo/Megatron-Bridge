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

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask


class DummyBridge(MegatronModelBridge):
    def provider_bridge(self, hf_pretrained):  # pragma: no cover - not used in tests
        return None

    def mapping_registry(self):  # pragma: no cover - not used in tests
        return MegatronMappingRegistry()


def _make_lora_module(alpha=8, dim=4):
    linear_in = SimpleNamespace(weight=torch.eye(dim))
    linear_out = SimpleNamespace(weight=torch.eye(dim))
    adapter = SimpleNamespace(linear_in=linear_in, linear_out=linear_out, alpha=alpha, dim=dim)
    base_linear = torch.nn.Linear(dim, dim, bias=False)
    lora_module = SimpleNamespace(adapter=adapter, to_wrap=base_linear)
    return lora_module


def test_merge_lora_adapter_weights_merges(monkeypatch):
    bridge = DummyBridge()
    base_weight = torch.zeros(4, 4)
    converted = {"hf.weight": base_weight.clone()}
    task = WeightConversionTask(
        param_name="decoder.layers.0.mlp.linear_fc1.to_wrap.weight",
        mapping=Mock(),
        megatron_module=Mock(),
        vp_stage=0,
    )

    lora_module = _make_lora_module(alpha=4, dim=4)
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge.get_module_and_param_from_name",
        lambda *_, **__: (lora_module, None),
    )
    monkeypatch.setattr("megatron.bridge.models.conversion.model_bridge.print_rank_0", lambda *_, **__: None)
    monkeypatch.setattr("megatron.bridge.peft.utils.HAVE_TE", False)

    updated = bridge._merge_lora_adapter_weights(task, [Mock()], converted)
    expected = base_weight + torch.eye(4)
    torch.testing.assert_close(updated["hf.weight"], expected)


def test_merge_lora_adapter_weights_noop_without_adapter(monkeypatch):
    bridge = DummyBridge()
    converted = {"hf.weight": torch.ones(2, 2)}
    task = WeightConversionTask(
        param_name="decoder.layers.0.mlp.linear_fc1.weight",
        mapping=Mock(),
        megatron_module=Mock(),
    )

    updated = bridge._merge_lora_adapter_weights(task, [Mock()], converted)
    torch.testing.assert_close(updated["hf.weight"], converted["hf.weight"])


def test_global_param_names_skip_adapter(monkeypatch):
    bridge = DummyBridge()

    class DummyGroup:
        def size(self):
            return 1

    fake_param = torch.nn.Parameter(torch.zeros(1, 1))

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace()

        def named_parameters(self):
            return [
                ("decoder.layers.0.mlp.adapter.linear_in.weight", fake_param),
                ("decoder.layers.0.mlp.linear_fc1.to_wrap.weight", fake_param),
            ]

    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge.parallel_state.get_pipeline_model_parallel_group",
        lambda: DummyGroup(),
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge.persistent_buffers",
        lambda *_: [],
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge._megatron_local_name_to_global",
        lambda *_args, **_kwargs: _args[2],
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge.unwrap_model",
        lambda models: models if isinstance(models, list) else [models],
    )
    monkeypatch.setattr(
        "torch.distributed.all_gather_object",
        lambda output, obj, group=None: output.__setitem__(0, obj),
    )

    names = bridge._megatron_global_param_names_all_pp_ranks([FakeModel()])
    assert names == ["decoder.layers.0.mlp.linear_fc1.to_wrap.weight"]
