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
from megatron.bridge.models.conversion.model_bridge import (
    AdapterWeight,
    AdapterWeightConversionTask,
    MegatronModelBridge,
    MegatronWeightTuple,
    WeightConversionTask,
)
from megatron.bridge.models.conversion.param_mapping import ColumnParallelMapping, merge_qkv_weights


class DummyBridge(MegatronModelBridge):
    def provider_bridge(self, hf_pretrained):  # pragma: no cover - not used in tests
        return None

    def mapping_registry(self):  # pragma: no cover - not used in tests
        return MegatronMappingRegistry()


def test_merge_lora_adapter_weights_merges(monkeypatch):
    bridge = DummyBridge()
    base_weight = torch.zeros(4, 4)
    converted = {"hf.weight": base_weight.clone()}
    adapter_weight = AdapterWeight(
        global_base_prefix="decoder.layers.0.mlp.linear_fc1",
        adapter_key=None,
        alpha=4,
        dim=4,
        linear_in_weight=MegatronWeightTuple("in", torch.eye(4), vp_stage=0),
        linear_out_weight=MegatronWeightTuple("out", torch.eye(4), vp_stage=0),
    )

    updated = bridge._merge_lora_adapter_weights([Mock(config=SimpleNamespace())], converted, [adapter_weight])
    expected = base_weight + torch.eye(4)
    torch.testing.assert_close(updated["hf.weight"], expected)


def test_merge_single_adapter_weight_matches_loramerge():
    bridge = DummyBridge()
    base = torch.zeros(2, 2)
    linear_in = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    linear_out = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    merged = bridge._merge_single_adapter_weight(
        base, alpha=2, dim=2, linear_in_weight=linear_in, linear_out_weight=linear_out
    )
    expected = base + 2 / 2 * (linear_out @ linear_in)
    torch.testing.assert_close(merged, expected)


def test_merge_lora_adapter_weights_fused_fc1(monkeypatch):
    bridge = DummyBridge()
    base = torch.zeros(4, 4)
    converted = {
        "decoder.layers.0.mlp.gate_proj.weight": base.clone(),
        "decoder.layers.0.mlp.up_proj.weight": base.clone(),
    }

    linear_out = torch.cat([torch.eye(4), 2 * torch.eye(4)], dim=0)
    adapter_weight = AdapterWeight(
        global_base_prefix="decoder.layers.0.mlp.linear_fc1",
        adapter_key=None,
        alpha=1,
        dim=1,
        linear_in_weight=MegatronWeightTuple("in", torch.eye(4), vp_stage=0),
        linear_out_weight=MegatronWeightTuple("out", linear_out, vp_stage=0),
    )

    updated = bridge._merge_lora_adapter_weights([Mock(config=SimpleNamespace())], converted, [adapter_weight])
    torch.testing.assert_close(updated["decoder.layers.0.mlp.gate_proj.weight"], torch.eye(4))
    torch.testing.assert_close(updated["decoder.layers.0.mlp.up_proj.weight"], 2 * torch.eye(4))


def test_merge_lora_adapter_weights_qkv_split(monkeypatch):
    bridge = DummyBridge()
    config = SimpleNamespace(
        num_attention_heads=2,
        num_query_groups=1,
        kv_channels=None,
        hidden_size=4,
        attention_output_gate=False,
    )
    megatron_model = [SimpleNamespace(config=config)]
    converted = {
        "q_proj.weight": torch.zeros(4, 4),
        "k_proj.weight": torch.zeros(2, 4),
        "v_proj.weight": torch.zeros(2, 4),
    }

    q_weight = torch.eye(4)
    k_weight = torch.ones(2, 4)
    v_weight = torch.full((2, 4), 2.0)
    linear_out = merge_qkv_weights(config, q_weight, k_weight, v_weight)

    adapter_weight = AdapterWeight(
        global_base_prefix="decoder.layers.0.self_attention.linear_qkv",
        adapter_key=None,
        alpha=4,
        dim=4,
        linear_in_weight=MegatronWeightTuple("in", torch.eye(4), vp_stage=0),
        linear_out_weight=MegatronWeightTuple("out", linear_out, vp_stage=0),
    )

    updated = bridge._merge_lora_adapter_weights(megatron_model, converted, [adapter_weight])
    torch.testing.assert_close(updated["q_proj.weight"], q_weight)
    torch.testing.assert_close(updated["k_proj.weight"], k_weight)
    torch.testing.assert_close(updated["v_proj.weight"], v_weight)


def test_merge_canonical_adapter_from_weights(monkeypatch):
    bridge = DummyBridge()
    converted = {
        "decoder.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2),
        "decoder.layers.0.self_attn.k_proj.weight": torch.zeros(1, 2),
        "decoder.layers.0.self_attn.v_proj.weight": torch.zeros(1, 2),
    }

    adapter_q = AdapterWeight(
        global_base_prefix="decoder.layers.0.self_attn.linear_qkv",
        adapter_key="adapter_q",
        alpha=1,
        dim=1,
        linear_in_weight=MegatronWeightTuple("in_q", torch.eye(2), vp_stage=0),
        linear_out_weight=MegatronWeightTuple("out_q", torch.ones(2, 2), vp_stage=0),
    )
    adapter_k = AdapterWeight(
        global_base_prefix="decoder.layers.0.self_attn.linear_qkv",
        adapter_key="adapter_k",
        alpha=1,
        dim=1,
        linear_in_weight=MegatronWeightTuple("in_k", torch.eye(2), vp_stage=0),
        linear_out_weight=MegatronWeightTuple("out_k", 2 * torch.ones(1, 2), vp_stage=0),
    )
    adapter_v = AdapterWeight(
        global_base_prefix="decoder.layers.0.self_attn.linear_qkv",
        adapter_key="adapter_v",
        alpha=1,
        dim=1,
        linear_in_weight=MegatronWeightTuple("in_v", torch.eye(2), vp_stage=0),
        linear_out_weight=MegatronWeightTuple("out_v", 3 * torch.ones(1, 2), vp_stage=0),
    )

    updated = bridge._merge_canonical_adapter_from_weights(converted, [adapter_q, adapter_k, adapter_v])
    torch.testing.assert_close(updated["decoder.layers.0.self_attn.q_proj.weight"], torch.ones(2, 2))
    torch.testing.assert_close(updated["decoder.layers.0.self_attn.k_proj.weight"], 2 * torch.ones(1, 2))
    torch.testing.assert_close(updated["decoder.layers.0.self_attn.v_proj.weight"], 3 * torch.ones(1, 2))


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


def test_megatron_global_adapters_info_all_pp_ranks(monkeypatch):
    bridge = DummyBridge()

    class DummyGroup:
        def size(self):
            return 1

    class FakeAdapter:
        def __init__(self):
            self.linear_in = SimpleNamespace(weight=torch.ones(2, 2))
            self.linear_out = SimpleNamespace(weight=torch.ones(2, 2))
            self.alpha = 8
            self.dim = 2

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace()
            param = torch.nn.Parameter(torch.zeros(2, 2))
            self._params = [
                ("decoder.layers.0.mlp.linear_fc1.adapter.linear_in.weight", param),
                ("decoder.layers.0.mlp.linear_fc1.adapter.linear_out.weight", param),
            ]

        def named_parameters(self):
            return self._params

    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge.parallel_state.get_pipeline_model_parallel_group",
        lambda: DummyGroup(),
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge.parallel_state.get_pipeline_model_parallel_rank",
        lambda: 0,
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
        "megatron.bridge.models.conversion.model_bridge.get_adapter_attributes_from_linear",
        lambda *_args, **_kwargs: (True, None, None, None, None, False),
    )
    monkeypatch.setattr(
        "torch.distributed.all_gather_object",
        lambda output, obj, group=None: output.__setitem__(0, obj),
    )

    adapter = FakeAdapter()
    monkeypatch.setattr(bridge, "_get_adapter_wrap_module", lambda *_: (adapter, Mock()))

    info = bridge._megatron_global_adapters_info_all_pp_ranks([FakeModel()])
    assert len(info) == 1
    (
        global_base_name,
        local_base_prefix,
        input_is_parallel,
        base_linear_is_parallel,
        alpha,
        dim,
        pp_rank,
        vp_stage,
    ) = info[0]
    assert global_base_name == "decoder.layers.0.mlp.linear_fc1.adapter"
    assert local_base_prefix == "decoder.layers.0.mlp.linear_fc1"
    assert input_is_parallel is True and base_linear_is_parallel is False
    assert alpha == 8 and dim == 2 and pp_rank == 0 and vp_stage == 0


def test_construct_adapters_names():
    bridge = DummyBridge()
    linear_in, linear_out = bridge._construct_adapters_names("decoder.layers.0.mlp.linear_fc1", None)
    assert linear_in == "decoder.layers.0.mlp.linear_fc1.adapter.linear_in.weight"
    assert linear_out == "decoder.layers.0.mlp.linear_fc1.adapter.linear_out.weight"

    linear_in_k, linear_out_k = bridge._construct_adapters_names("decoder.layers.0.attn.q_proj", "adapter_q")
    assert linear_in_k.endswith("adapter_q.linear_in.weight")
    assert linear_out_k.endswith("adapter_q.linear_out.weight")


def test_build_adapter_conversion_tasks(monkeypatch):
    bridge = DummyBridge()

    adapters_info = [
        (
            "decoder.layers.0.mlp.linear_fc1.adapter",
            "decoder.layers.0.mlp.linear_fc1",
            False,
            False,
            4,
            8,
            0,
            0,
        )
    ]

    adapter = SimpleNamespace(
        linear_in=SimpleNamespace(weight=torch.ones(2, 2)),
        linear_out=SimpleNamespace(weight=torch.ones(2, 2)),
        alpha=4,
        dim=8,
    )

    monkeypatch.setattr(bridge, "_megatron_global_adapters_info_all_pp_ranks", lambda *_: adapters_info)
    monkeypatch.setattr(bridge, "_get_adapter_wrap_module", lambda *_: (adapter, Mock()))
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge.parallel_state.get_pipeline_model_parallel_rank",
        lambda: 0,
    )

    tasks_by_base = bridge.build_adapter_conversion_tasks([Mock()])
    assert "decoder.layers.0.mlp.linear_fc1" in tasks_by_base
    tasks = tasks_by_base["decoder.layers.0.mlp.linear_fc1"]
    assert len(tasks) == 1
    task = tasks[0]
    assert task.adapter_key is None
    assert task.linear_in_task.param_weight.shape == torch.Size([2, 2])
    assert task.linear_out_task.param_weight.shape == torch.Size([2, 2])


def test_materialize_adapter_weights(monkeypatch):
    bridge = DummyBridge()

    class DummyMapping:
        def __init__(self, payload):
            self.payload = payload

        def megatron_to_hf(self, weight, module):
            return {"hf": self.payload}

    adapter_tasks = [
        AdapterWeightConversionTask(
            global_base_prefix="decoder.layers.0.mlp.linear_fc1",
            adapter_key=None,
            alpha=2,
            dim=4,
            linear_in_task=WeightConversionTask(
                param_name="in_name",
                global_param_name="in_name",
                mapping=DummyMapping(torch.ones(2, 2)),
                megatron_module=None,
                param_weight=None,
            ),
            linear_out_task=WeightConversionTask(
                param_name="out_name",
                global_param_name="out_name",
                mapping=DummyMapping(2 * torch.ones(2, 2)),
                megatron_module=None,
                param_weight=None,
            ),
        )
    ]

    materials = bridge.materialize_adapter_weights(adapter_tasks)
    assert len(materials) == 1
    assert torch.all(materials[0].linear_in_weight.weight == torch.ones(2, 2))
    assert torch.all(materials[0].linear_out_weight.weight == 2 * torch.ones(2, 2))


def test_column_parallel_mapping_skips_ep_gather_for_adapters(monkeypatch):
    mapping = ColumnParallelMapping(
        "decoder.layers.0.mlp.experts.linear_fc1.adapter.linear_in.weight",
        "hf_param",
    )

    # Avoid distributed calls
    monkeypatch.setattr(ColumnParallelMapping, "broadcast_from_pp_rank", lambda self, tensor, cache_key=None: tensor)
    monkeypatch.setattr(ColumnParallelMapping, "gather_from_tp_ranks", lambda self, tensor: [tensor])
    monkeypatch.setattr(ColumnParallelMapping, "tp_size", property(lambda self: 1))

    def _raise(*args, **kwargs):
        raise AssertionError("gather_from_ep_ranks should not be called for adapters")

    monkeypatch.setattr(ColumnParallelMapping, "gather_from_ep_ranks", _raise)

    result = mapping.megatron_to_hf(torch.ones(2, 2), None)
    torch.testing.assert_close(result["hf_param"], torch.ones(2, 2))
