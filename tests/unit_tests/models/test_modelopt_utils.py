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

import copy
from types import SimpleNamespace

import pytest
import torch


mtq = pytest.importorskip("modelopt.torch.quantization")
quant_utils = pytest.importorskip("modelopt.torch.export.quant_utils")

from megatron.bridge.models.conversion import modelopt_utils
from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.conversion.model_bridge import HFWeightTuple, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import (
    ColumnParallelMapping,
    FusedExpertMapping,
    GatedMLPMapping,
    QKVMapping,
)


def _task(mapping, module, *, global_name="decoder.layers.0.projection.weight"):
    return WeightConversionTask(
        param_name="weight",
        global_param_name=global_name,
        mapping=mapping,
        megatron_module=module,
        param_weight=module.weight,
    )


def _fp8_linear(out_features=4, in_features=4):
    module = torch.nn.Linear(in_features, out_features, bias=False)
    with torch.no_grad():
        module.weight.copy_(
            torch.arange(out_features * in_features, dtype=torch.float32).reshape(out_features, in_features) / 8 - 1
        )
    return mtq.quantize(
        module,
        copy.deepcopy(mtq.FP8_DEFAULT_CFG),
        lambda candidate: candidate(torch.ones(2, in_features)),
    )


def _source(state, shape, parallelism, qkv_layout=None):
    return modelopt_utils._SourceState(state, shape, parallelism, qkv_layout)


def test_build_plan_delegates_fp8_packing_and_config_to_modelopt():
    module = _fp8_linear()
    hf_name = "model.layers.0.self_attn.o_proj.weight"
    task = _task(ColumnParallelMapping("projection.weight", hf_name), module)

    plan = modelopt_utils.build_modelopt_export_plan([task], model=[module])
    exported = dict(plan.conversion_tasks[0].export_hook(hf_name, module.weight))

    assert set(exported) == {
        hf_name,
        "model.layers.0.self_attn.o_proj.weight_scale",
        "model.layers.0.self_attn.o_proj.input_scale",
    }
    assert exported[hf_name].dtype == torch.float8_e4m3fn
    assert plan.quantization_config["quant_algo"] == "FP8"
    assert plan.quantization_config["config_groups"]["group_0"]["targets"] == ["Linear"]


def test_gated_state_is_split_before_tp_merge(monkeypatch):
    mapping = GatedMLPMapping(
        "decoder.layers.0.mlp.linear_fc1.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
    )
    task = SimpleNamespace(mapping=mapping, global_param_name="linear_fc1.weight")
    shards = [
        _source("rank0", (8, 4), "gated"),
        _source("rank1", (8, 4), "gated"),
    ]
    calls = []

    monkeypatch.setattr(modelopt_utils, "_all_gather_objects", lambda _value, _group: shards)

    def select(state, dim, indices):
        result = ("select", state, dim, tuple(indices.tolist()))
        calls.append(result)
        return result

    def merge(states, dim):
        result = ("merge", tuple(states), dim)
        calls.append(result)
        return result

    monkeypatch.setattr(quant_utils, "select_quantized_weight_export_state", select)
    monkeypatch.setattr(quant_utils, "merge_quantized_weight_export_states", merge)

    transformed = modelopt_utils._transform_source_state(task, shards[0])

    gate = transformed["model.layers.0.mlp.gate_proj.weight"]
    up = transformed["model.layers.0.mlp.up_proj.weight"]
    assert gate[0] == "merge" and up[0] == "merge"
    assert [entry[0] for entry in calls] == ["select", "select", "select", "select", "merge", "merge"]
    assert gate[1][0][3] == (0, 1, 2, 3)
    assert up[1][0][3] == (4, 5, 6, 7)


def test_qkv_state_uses_megatron_interleaving(monkeypatch):
    mapping = QKVMapping(
        "decoder.layers.0.self_attention.linear_qkv.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
    )
    task = SimpleNamespace(mapping=mapping, global_param_name="linear_qkv.weight")
    source = _source("qkv", (16, 8), "column", (4, 2, 2, 8, False))
    selections = []

    monkeypatch.setattr(modelopt_utils, "_all_gather_objects", lambda value, _group: [value])

    def select(state, dim, indices):
        selections.append((state, dim, tuple(indices.tolist())))
        return selections[-1]

    monkeypatch.setattr(quant_utils, "select_quantized_weight_export_state", select)

    transformed = modelopt_utils._transform_source_state(task, source)

    assert set(transformed) == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
    }
    assert [len(indices) for _, _, indices in selections] == [8, 4, 4]
    assert sorted(index for _, _, indices in selections for index in indices) == list(range(16))


def test_inconsistent_tp_quantization_is_rejected(monkeypatch):
    mapping = ColumnParallelMapping("projection.weight", "model.projection.weight")
    task = SimpleNamespace(mapping=mapping, global_param_name="projection.weight")
    quantized = _source("state", (4, 4), "column")
    unquantized = _source(None, (4, 4), "column")
    monkeypatch.setattr(
        modelopt_utils,
        "_all_gather_objects",
        lambda _value, _group: [quantized, unquantized],
    )

    with pytest.raises(RuntimeError, match="Inconsistent ModelOpt quantization across TP"):
        modelopt_utils._transform_source_state(task, quantized)


def test_quantized_expert_is_packed_before_ep_gather():
    module = _fp8_linear(out_features=8)
    mapping = GatedMLPMapping(
        "decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
    )
    task = _task(mapping, module, global_name=mapping.megatron_param)

    plan = modelopt_utils.build_modelopt_export_plan([task], model=[module])
    export_task = plan.conversion_tasks[0]
    mapped = export_task.mapping.megatron_to_hf(module.weight, module)
    exported = {
        name: value for hf_name, weight in mapped.items() for name, value in export_task.export_hook(hf_name, weight)
    }

    assert not export_task.mapping.is_expert
    assert export_task.mapping.tp_group is export_task.mapping._etp_group
    assert {
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight_scale",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight_scale",
    }.issubset(exported)


def test_grouped_hf_expert_mapping_is_rejected_before_export():
    module = _fp8_linear()
    mapping = FusedExpertMapping(
        "decoder.layers.0.mlp.experts.local_experts.0.linear_fc2.weight",
        "model.layers.0.mlp.experts.down_proj",
    )

    with pytest.raises(RuntimeError, match="grouped HF expert weights"):
        modelopt_utils.build_modelopt_export_plan(
            [_task(mapping, module, global_name=mapping.megatron_param)],
            model=[module],
        )


def test_custom_quantized_expert_mapping_is_rejected_before_ep_gather():
    class CustomExpertMapping(ColumnParallelMapping):
        pass

    module = _fp8_linear()
    mapping = CustomExpertMapping(
        "decoder.layers.0.mlp.experts.local_experts.0.linear_fc2.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight",
    )

    with pytest.raises(RuntimeError, match="cannot pack expert mapping"):
        modelopt_utils.build_modelopt_export_plan(
            [_task(mapping, module, global_name=mapping.megatron_param)],
            model=[module],
        )


def _distributed_topology_worker(rank, world_size, init_file):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        original_capture = modelopt_utils._capture_source_state
        original_get_pp_group = modelopt_utils.model_bridge_utils._get_pp_group
        modelopt_utils._capture_source_state = (
            lambda _task: (_ for _ in ()).throw(ValueError("rank-local failure"))
            if rank == 0
            else None
        )
        modelopt_utils.model_bridge_utils._get_pp_group = (
            lambda _model: torch.distributed.group.WORLD
        )
        try:
            modelopt_utils.build_modelopt_export_plan(
                [SimpleNamespace(global_param_name="weight")],
                model=[torch.nn.Linear(1, 1)],
            )
        except RuntimeError as error:
            assert "rank-local failure" in str(error)
        else:
            raise AssertionError("Every rank must observe the capture failure")
        finally:
            modelopt_utils._capture_source_state = original_capture
            modelopt_utils.model_bridge_utils._get_pp_group = original_get_pp_group

        mapping = ColumnParallelMapping("projection.weight", "model.projection.weight")
        mapping._tp_group = torch.distributed.group.WORLD
        task = SimpleNamespace(mapping=mapping, global_param_name="projection.weight")
        source = _source(f"rank{rank}", (2, 2), "column")
        gathered_sources = modelopt_utils._all_gather_objects(
            source, torch.distributed.group.WORLD
        )
        assert all(
            isinstance(item, modelopt_utils._SourceState)
            for item in gathered_sources
        ), repr(gathered_sources)
        original_merge = quant_utils.merge_quantized_weight_export_states
        quant_utils.merge_quantized_weight_export_states = (
            lambda states, dim: (tuple(states), dim)
        )
        try:
            transformed = modelopt_utils._transform_source_state(task, source)
            assert transformed["model.projection.weight"] == (
                ("rank0", "rank1"),
                0,
            )
        finally:
            quant_utils.merge_quantized_weight_export_states = original_merge

        gathered = list(
            modelopt_utils._gather_expert_output(
                f"expert.{rank}.weight",
                torch.tensor([rank], dtype=torch.int64),
                torch.distributed.group.WORLD,
            )
        )
        assert [name for name, _ in gathered] == [
            "expert.0.weight",
            "expert.1.weight",
        ]
        assert [tensor.item() for _, tensor in gathered] == [0, 1]
    finally:
        torch.distributed.destroy_process_group()


def test_two_rank_planning_and_expert_collectives(tmp_path):
    torch.multiprocessing.spawn(
        _distributed_topology_worker,
        args=(2, str(tmp_path / "modelopt-dist-init")),
        nprocs=2,
        join=True,
    )


def test_auto_bridge_can_reuse_a_prepared_plan():
    task = SimpleNamespace(name="task")
    plan = modelopt_utils.ModelOptExportPlan([task], {"quantized_layers": {}})

    class FakeBridge:
        def __init__(self):
            self.calls = []

        def build_hf_modelopt_export_plan(self, _model):
            raise AssertionError("the supplied plan must be reused")

        def export_hf_weights(self, model, **kwargs):
            self.calls.append((model, kwargs))
            yield HFWeightTuple("model.weight", torch.ones(1))

    bridge = FakeBridge()
    model = torch.nn.Linear(1, 1)

    exported = list(
        AutoBridge.export_hf_weights_modelopt(
            bridge,
            model,
            export_plan=plan,
            cpu=True,
            show_progress=False,
            merge_adapter_weights=False,
        )
    )

    assert [name for name, _ in exported] == ["model.weight"]
    assert bridge.calls == [
        (
            [model],
            {
                "cpu": True,
                "show_progress": False,
                "conversion_tasks": [task],
                "merge_adapter_weights": False,
            },
        )
    ]
