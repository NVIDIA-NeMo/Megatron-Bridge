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
from megatron.bridge.models.conversion.model_bridge import HFWeightTuple, MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    FusedExpertMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.conversion.quant_mapping import AmaxMapping


def _task(
    mapping,
    module,
    *,
    global_name="decoder.layers.0.projection.weight",
    weight_name="weight",
):
    return WeightConversionTask(
        param_name=weight_name,
        global_param_name=global_name,
        mapping=mapping,
        megatron_module=module,
        param_weight=getattr(module, weight_name),
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


class _GroupedWeights(torch.nn.Module):
    def __init__(self):
        super().__init__()
        source = _fp8_linear()
        self.weight0 = torch.nn.Parameter(source.weight.detach().clone())
        self.quantizers = torch.nn.ModuleList([source.weight_quantizer])
        self.input_quantizer = source.input_quantizer

    def iter_weights_for_calibration(self):
        yield self.weight0, self.quantizers[0]


def test_build_plan_delegates_fp8_packing_and_config_to_modelopt():
    module = _fp8_linear()
    hf_name = "model.layers.0.self_attn.o_proj.weight"
    task = _task(ColumnParallelMapping("projection.weight", hf_name), module)

    plan = modelopt_utils.build_modelopt_export_plan([task], model=[module])
    export_task = modelopt_utils.prepare_modelopt_export_tasks(plan)[0]
    exported = dict(export_task.export_hook(hf_name, module.weight))

    assert set(exported) == {
        hf_name,
        "model.layers.0.self_attn.o_proj.weight_scale",
        "model.layers.0.self_attn.o_proj.input_scale",
    }
    assert exported[hf_name].dtype == torch.float8_e4m3fn
    assert plan.quantization_config["quant_algo"] == "FP8"
    assert plan.quantization_config["config_groups"]["group_0"]["targets"] == ["Linear"]


def test_numbered_grouped_weight_uses_exact_modelopt_quantizer():
    module = _GroupedWeights()
    hf_name = "model.layers.0.mlp.experts.0.down_proj.weight"
    task = _task(
        ColumnParallelMapping("projection.weight0", hf_name),
        module,
        weight_name="weight0",
    )

    plan = modelopt_utils.build_modelopt_export_plan([task], model=[module])
    export_task = modelopt_utils.prepare_modelopt_export_tasks(plan)[0]
    exported = dict(export_task.export_hook(hf_name, module.weight0))

    assert exported[hf_name].dtype == torch.float8_e4m3fn
    assert f"{hf_name.removesuffix('.weight')}.weight_scale" in exported


def test_reused_plan_recaptures_mutable_quantizer_state():
    module = _fp8_linear()
    hf_name = "model.layers.0.self_attn.o_proj.weight"
    task = _task(ColumnParallelMapping("projection.weight", hf_name), module)
    plan = modelopt_utils.build_modelopt_export_plan([task], model=[module])

    first_task = modelopt_utils.prepare_modelopt_export_tasks(plan)[0]
    first = dict(first_task.export_hook(hf_name, module.weight))
    module.weight_quantizer._amax.mul_(2)
    second_task = modelopt_utils.prepare_modelopt_export_tasks(plan)[0]
    second = dict(second_task.export_hook(hf_name, module.weight))

    assert not torch.equal(
        first["model.layers.0.self_attn.o_proj.weight_scale"], second["model.layers.0.self_attn.o_proj.weight_scale"]
    )


def test_build_plan_rejects_quantized_adapter_weights():
    module = _fp8_linear()
    task = _task(
        ColumnParallelMapping("projection.to_wrap.weight", "model.projection.weight"),
        module,
        global_name="decoder.layers.0.projection.to_wrap.weight",
    )

    with pytest.raises(RuntimeError, match="folded into base weights before quantization calibration"):
        modelopt_utils.build_modelopt_export_plan([task], model=[module])


def test_build_plan_rejects_dimension_permutation():
    module = _fp8_linear()
    mapping = AutoMapping(
        "projection.weight",
        "model.projection.weight",
        permute_dims=(1, 0),
    )

    with pytest.raises(RuntimeError, match="dimension-permuting mappings"):
        modelopt_utils.build_modelopt_export_plan([_task(mapping, module)], model=[module])


def test_distributed_error_check_skips_object_gather_on_success(monkeypatch):
    group = object()
    monkeypatch.setattr(modelopt_utils, "get_pg_size", lambda _group: 2)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda _group: "gloo")
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        modelopt_utils,
        "_all_gather_objects",
        lambda *_args, **_kwargs: pytest.fail("success path used object gather"),
    )

    modelopt_utils._raise_distributed_errors(None, group, "capture")


def test_distributed_error_check_gathers_messages_only_on_failure(monkeypatch):
    group = object()
    monkeypatch.setattr(modelopt_utils, "get_pg_size", lambda _group: 2)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda _group: "gloo")

    def report_remote_failure(failed, **_kwargs):
        failed.fill_(1)

    monkeypatch.setattr(torch.distributed, "all_reduce", report_remote_failure)
    monkeypatch.setattr(
        modelopt_utils,
        "_all_gather_objects",
        lambda _error, _group: [None, "ValueError: rank-local failure"],
    )

    with pytest.raises(RuntimeError, match="rank-local failure"):
        modelopt_utils._raise_distributed_errors(None, group, "capture")


def test_weight_export_does_not_use_world_error_collective(monkeypatch):
    module = _fp8_linear()
    hf_name = "model.layers.0.self_attn.o_proj.weight"
    task = _task(ColumnParallelMapping("projection.weight", hf_name), module)
    plan = modelopt_utils.build_modelopt_export_plan([task], model=[module])
    export_task = modelopt_utils.prepare_modelopt_export_tasks(plan)[0]
    monkeypatch.setattr(
        modelopt_utils,
        "_raise_distributed_errors",
        lambda *_args, **_kwargs: pytest.fail("weight export used a WORLD error collective"),
    )

    exported = dict(export_task.export_hook(hf_name, module.weight))

    assert hf_name in exported


def test_build_plan_excludes_fake_quant_amax_mappings():
    module = _fp8_linear()
    hf_name = "model.layers.0.self_attn.o_proj.weight"
    weight_task = _task(ColumnParallelMapping("projection.weight", hf_name), module)
    amax_task = WeightConversionTask(
        param_name="weight_quantizer._amax",
        global_param_name="decoder.layers.0.projection.weight_quantizer._amax",
        mapping=AmaxMapping(
            "projection.weight_quantizer._amax",
            "model.layers.0.self_attn.o_proj.weight_quantizer._amax",
        ),
        megatron_module=module.weight_quantizer,
        param_weight=module.weight_quantizer._amax,
    )

    plan = modelopt_utils.build_modelopt_export_plan([weight_task, amax_task], model=[module])

    assert [task.global_param_name for task in plan.conversion_tasks] == [weight_task.global_param_name]


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

    monkeypatch.setattr(modelopt_utils, "_gather_source_states", lambda _source, _mapping, _error=None: shards)

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

    monkeypatch.setattr(modelopt_utils, "_gather_source_states", lambda value, _mapping, _error=None: [value])

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
        modelopt_utils._transform_source_spec(task, quantized)


def test_quantized_expert_is_packed_before_ep_gather():
    module = _fp8_linear(out_features=8)
    mapping = GatedMLPMapping(
        "decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
    )
    task = _task(mapping, module, global_name=mapping.megatron_param)

    plan = modelopt_utils.build_modelopt_export_plan([task], model=[module])
    export_task = modelopt_utils.prepare_modelopt_export_tasks(plan)[0]
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
        original_capture = modelopt_utils._capture_source_spec
        original_get_pp_group = modelopt_utils.model_bridge_utils._get_pp_group
        modelopt_utils._capture_source_spec = lambda _task: (
            (_ for _ in ()).throw(ValueError("rank-local failure")) if rank == 0 else None
        )
        modelopt_utils.model_bridge_utils._get_pp_group = lambda _model: torch.distributed.group.WORLD
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
            modelopt_utils._capture_source_spec = original_capture
            modelopt_utils.model_bridge_utils._get_pp_group = original_get_pp_group

        mapping = ColumnParallelMapping("projection.weight", "model.projection.weight")
        mapping._tp_group = torch.distributed.group.WORLD
        task = SimpleNamespace(mapping=mapping, global_param_name="projection.weight")
        module = _fp8_linear()
        source = _source(
            quant_utils.capture_quantized_weight_export_state(module),
            tuple(module.weight.shape),
            "column",
        )
        gathered_sources = modelopt_utils._all_gather_objects(source, torch.distributed.group.WORLD)
        assert all(isinstance(item, modelopt_utils._SourceState) for item in gathered_sources), repr(gathered_sources)
        transformed = modelopt_utils._transform_source_state(task, source)
        assert transformed["model.projection.weight"].weight_shape == (8, 4)

        gathered = list(
            modelopt_utils._gather_expert_outputs(
                [
                    HFWeightTuple(
                        f"expert.{rank}.weight",
                        torch.tensor([rank], dtype=torch.int64),
                    )
                ],
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


def _distributed_capture_failure_worker(rank, world_size, init_file):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        pp_groups = [
            torch.distributed.new_group([0, 2]),
            torch.distributed.new_group([1, 3]),
        ]
        tp_groups = [
            torch.distributed.new_group([0, 1]),
            torch.distributed.new_group([2, 3]),
        ]
        tp_rank = rank % 2
        pp_rank = rank // 2

        module = _fp8_linear()
        mapping = ColumnParallelMapping("projection.weight", "model.projection.weight")
        mapping.pp_group = pp_groups[tp_rank]
        mapping._tp_group = tp_groups[pp_rank]
        owns_weight = pp_rank == 0
        task = WeightConversionTask(
            param_name="weight",
            global_param_name="projection.weight",
            mapping=mapping,
            megatron_module=module if owns_weight else None,
            param_weight=module.weight if owns_weight else None,
        )

        original_capture = modelopt_utils._capture_source_state
        original_cuda_available = torch.cuda.is_available
        if rank == 0:

            def fail_capture(_task):
                raise ValueError("rank-local PP capture failure")

            modelopt_utils._capture_source_state = fail_capture
        torch.cuda.is_available = lambda: False
        try:
            source, capture_error = modelopt_utils._capture_current_source_state(task)
            modelopt_utils._transform_source_state(task, source, capture_error)
        except RuntimeError as error:
            assert "rank-local PP capture failure" in str(error)
        else:
            raise AssertionError("Every PP x TP rank must observe a capture failure")
        finally:
            modelopt_utils._capture_source_state = original_capture
            torch.cuda.is_available = original_cuda_available

        etp_group = tp_groups[pp_rank]
        ep_group = pp_groups[tp_rank]
        expert_mapping = ColumnParallelMapping("expert.weight", "model.expert.weight")
        expert_mapping._tp_group = etp_group
        source = _source(
            quant_utils.capture_quantized_weight_export_state(module),
            tuple(module.weight.shape),
            "column",
        )

        def expert_outputs():
            capture_error = "ValueError: rank-local ETP capture failure" if rank == 0 else None
            states = modelopt_utils._gather_source_states(
                None if rank == 0 else source,
                expert_mapping,
                capture_error,
            )
            assert states
            yield HFWeightTuple(
                f"expert.{rank}.weight",
                torch.tensor([rank], dtype=torch.int64),
            )

        try:
            list(modelopt_utils._gather_expert_outputs(expert_outputs(), ep_group))
        except RuntimeError as error:
            assert "rank-local ETP capture failure" in str(error)
        else:
            raise AssertionError("Every ETP x EP rank must observe an expert capture failure")

        mismatched_outputs = [
            HFWeightTuple(
                f"expert.{rank}.weight",
                torch.zeros(2 if pp_rank == 0 else 1),
            )
        ]
        try:
            list(modelopt_utils._gather_expert_outputs(mismatched_outputs, ep_group))
        except RuntimeError as error:
            assert "Inconsistent ModelOpt expert output tensors" in str(error)
        else:
            raise AssertionError("Every EP rank must observe an output layout mismatch")
    finally:
        torch.distributed.destroy_process_group()


def test_four_rank_capture_failure_topologies(tmp_path):
    torch.multiprocessing.spawn(
        _distributed_capture_failure_worker,
        args=(4, str(tmp_path / "modelopt-failure-init")),
        nprocs=4,
        join=True,
    )


def test_auto_bridge_can_reuse_a_prepared_plan():
    task = SimpleNamespace(name="task", global_param_name="model.weight")
    plan = modelopt_utils.ModelOptExportPlan([task], {"quantized_layers": {}}, frozenset())

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
            merge_adapter_weights=True,
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
                "merge_adapter_weights": True,
            },
        )
    ]


def test_auto_bridge_rejects_unmerged_adapters():
    bridge = object.__new__(AutoBridge)
    with pytest.raises(NotImplementedError, match="unmerged adapter"):
        list(
            bridge.export_hf_weights_modelopt(
                torch.nn.Linear(1, 1),
                export_plan=SimpleNamespace(),
                merge_adapter_weights=False,
            )
        )


def test_grouped_modelopt_export_rejects_custom_auto_bridge_export():
    class CustomAutoBridge(AutoBridge):
        def export_hf_weights(self, *args, **kwargs):
            yield from ()

    bridge = object.__new__(CustomAutoBridge)
    with pytest.raises(NotImplementedError, match="AutoBridge subclasses"):
        list(
            bridge.export_hf_weight_groups_modelopt(
                torch.nn.Linear(1, 1),
                export_plan=SimpleNamespace(),
            )
        )


def test_grouped_modelopt_export_rejects_custom_model_bridge_export(monkeypatch):
    class CustomModelBridge(MegatronModelBridge):
        def stream_weights_megatron_to_hf(self, *args, **kwargs):
            yield from ()

    monkeypatch.setattr(
        AutoBridge,
        "_model_bridge",
        property(lambda _self: CustomModelBridge()),
    )
    bridge = object.__new__(AutoBridge)
    with pytest.raises(NotImplementedError, match="model bridges"):
        list(
            bridge.export_hf_weight_groups_modelopt(
                torch.nn.Linear(1, 1),
                export_plan=SimpleNamespace(),
            )
        )
