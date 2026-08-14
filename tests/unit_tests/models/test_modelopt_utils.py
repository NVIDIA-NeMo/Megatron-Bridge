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

from copy import deepcopy
from types import SimpleNamespace

import modelopt.torch.quantization as mtq
import pytest
import torch
from megatron.core.transformer.moe.router import TopKRouter
from modelopt.torch.export.quantized_weight import (
    capture_quantized_weight_export_state,
    export_quantized_weight,
    quantized_weight_export_states_equal,
)
from modelopt.torch.quantization.nn import GroupedQuantizer

from megatron.bridge.models.conversion import modelopt_utils
from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.conversion.model_bridge import (
    HFWeightTuple,
    MegatronModelBridge,
    WeightConversionTask,
)
from megatron.bridge.models.conversion.modelopt_utils import (
    ModelOptExportPlan,
    build_hf_modelopt_quant_states,
    build_modelopt_export_plan,
    build_modelopt_quantization_config,
    collect_modelopt_config_weights,
    collect_modelopt_quant_states,
    find_modelopt_quantizers,
    sync_modelopt_quant_states,
)
from megatron.bridge.models.conversion.param_mapping import AutoMapping


def _state(
    *,
    w4a16=False,
    weight_amax=2.0,
    input_amax=3.0,
):
    module = torch.nn.Linear(16, 4, bias=False)
    mtq.quantize(
        module,
        deepcopy(mtq.W4A16_NVFP4_CFG if w4a16 else mtq.NVFP4_DEFAULT_CFG),
        lambda model: model(torch.ones(1, 16)),
    )
    module.weight_quantizer.amax = torch.tensor(weight_amax)
    if module.input_quantizer.is_enabled:
        module.input_quantizer.amax = torch.tensor(input_amax)
    return capture_quantized_weight_export_state(module)


def _assert_state_equal(actual, expected):
    assert quantized_weight_export_states_equal(actual, expected)


def _task(
    global_param_name,
    hf_param,
    *,
    megatron_module=None,
    param_weight=None,
    is_grouped_export=False,
    ep_size=1,
    transpose_on_export=False,
    export_hook=None,
):
    return WeightConversionTask(
        param_name=global_param_name,
        global_param_name=global_param_name,
        mapping=SimpleNamespace(
            hf_param=hf_param,
            is_grouped_export=is_grouped_export,
            ep_size=ep_size,
            transpose_on_export=transpose_on_export,
        ),
        megatron_module=megatron_module,
        param_weight=param_weight,
        export_hook=export_hook,
    )


def _model(*, num_moe_experts=None):
    model = torch.nn.Module()
    model.config = SimpleNamespace(num_moe_experts=num_moe_experts)
    return [model]


class _GroupedLinear(torch.nn.Module):
    def __init__(self, linears):
        super().__init__()
        self.weight0 = linears[0].weight
        self.weight1 = linears[1].weight
        self.weight_quantizer = GroupedQuantizer(
            linears[0].weight_quantizer,
            linears[1].weight_quantizer,
        )
        self.weight0_input_quantizer = linears[0].input_quantizer
        self.weight1_input_quantizer = linears[1].input_quantizer

    def iter_weights_for_calibration(self):
        yield self.weight0, self.weight_quantizer[0]
        yield self.weight1, self.weight_quantizer[1]


def _quantized_linears():
    linears = [torch.nn.Linear(16, 4, bias=False) for _ in range(2)]
    for linear, weight_amax, input_amax in zip(
        linears,
        (2.0, 5.0),
        (3.0, 7.0),
        strict=True,
    ):
        mtq.quantize(
            linear,
            deepcopy(mtq.NVFP4_DEFAULT_CFG),
            lambda module: module(torch.ones(1, 16)),
        )
        linear.weight_quantizer.amax = torch.tensor(weight_amax)
        linear.input_quantizer.amax = torch.tensor(input_amax)
    return linears


def test_find_modelopt_quantizers_returns_exact_grouped_quantizer():
    module = _GroupedLinear(_quantized_linears())

    found = find_modelopt_quantizers(module, module.weight1)

    assert found is not None
    owner, weight_name, weight_quantizer, input_quantizer = found
    assert owner is module
    assert weight_name == "weight1"
    assert weight_quantizer is module.weight_quantizer[1]
    assert input_quantizer is module.weight1_input_quantizer


def test_collect_modelopt_config_weights_includes_disabled_quantizers():
    quantized, plain = _quantized_linears()[0], torch.nn.Linear(16, 4, bias=False)
    quantized.weight_quantizer.disable()
    quantized_task = _task(
        "decoder.layers.0.mlp.linear_fc1.weight",
        "model.layers.0.mlp.up_proj.weight",
        megatron_module=quantized,
        param_weight=quantized.weight,
    )
    plain_task = _task(
        "decoder.layers.0.mlp.linear_fc2.weight",
        "model.layers.0.mlp.down_proj.weight",
        megatron_module=plain,
        param_weight=plain.weight,
    )

    assert collect_modelopt_config_weights([quantized_task, plain_task]) == {quantized_task.global_param_name}


def test_collect_modelopt_config_weights_includes_unquantized_moe_router():
    router = object.__new__(TopKRouter)
    torch.nn.Module.__init__(router)
    router.weight = torch.nn.Parameter(torch.ones(2, 16))
    router_task = _task(
        "decoder.layers.0.mlp.router.weight",
        "model.layers.0.mlp.gate.weight",
        megatron_module=router,
        param_weight=router.weight,
    )

    assert collect_modelopt_config_weights([router_task]) == {router_task.global_param_name}


def test_collect_modelopt_quant_states_preserves_independent_experts():
    module = _GroupedLinear(_quantized_linears())
    tasks = [
        _task(
            f"decoder.layers.0.mlp.experts.linear_fc1.weight{expert}",
            f"model.layers.0.mlp.experts.{expert}.gate_up_proj.weight",
            megatron_module=module,
            param_weight=getattr(module, f"weight{expert}"),
        )
        for expert in range(2)
    ]

    states = collect_modelopt_quant_states(tasks)

    _assert_state_equal(
        states[tasks[0].global_param_name],
        _state(weight_amax=2.0, input_amax=3.0),
    )
    _assert_state_equal(
        states[tasks[1].global_param_name],
        _state(weight_amax=5.0, input_amax=7.0),
    )


def test_collect_modelopt_quant_states_max_reduces_tp_scalars(monkeypatch):
    module = _GroupedLinear(_quantized_linears())
    expected = _state(weight_amax=11.0, input_amax=13.0)
    task = _task(
        "decoder.layers.0.mlp.experts.linear_fc1.weight0",
        "model.layers.0.mlp.experts.0.gate_up_proj.weight",
        megatron_module=module,
        param_weight=module.weight0,
    )
    reductions = iter((11.0, 13.0))

    monkeypatch.setattr(modelopt_utils, "_get_modelopt_tp_process_group", lambda _module: object())
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)

    def fake_all_reduce(value, **_kwargs):
        value.fill_(next(reductions))

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    state = collect_modelopt_quant_states([task])[task.global_param_name]

    _assert_state_equal(
        state,
        expected,
    )


def test_sync_modelopt_quant_states_merges_disjoint_ranks(monkeypatch):
    local = {"layer.0.weight": _state(weight_amax=1.0)}
    remote = {"layer.1.weight": _state(weight_amax=2.0)}
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)

    def fake_all_gather_object(gathered, _states, group=None):
        gathered[:] = [local.copy(), remote]

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)

    sync_modelopt_quant_states(local)

    assert set(local) == {"layer.0.weight", "layer.1.weight"}


def test_sync_modelopt_quant_states_rejects_conflicting_copies(monkeypatch):
    local = {"layer.weight": _state(weight_amax=1.0)}
    remote = {"layer.weight": _state(weight_amax=2.0)}
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)

    def fake_all_gather_object(gathered, _states, group=None):
        gathered[:] = [local.copy(), remote]

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)

    with pytest.raises(RuntimeError, match="Conflicting ModelOpt state"):
        sync_modelopt_quant_states(local)


def test_build_hf_states_copies_fused_projection_state():
    task = _task(
        "decoder.layers.0.self_attention.linear_qkv.weight",
        {
            "q": "model.layers.0.self_attn.q_proj.weight",
            "k": "model.layers.0.self_attn.k_proj.weight",
            "v": "model.layers.0.self_attn.v_proj.weight",
        },
    )
    state = _state()

    hf_states = build_hf_modelopt_quant_states(
        [task],
        {task.global_param_name: state},
    )

    assert set(hf_states) == set(task.mapping.hf_param.values())
    assert all(quantized_weight_export_states_equal(mapped_state, state) for mapped_state in hf_states.values())
    assert all(mapped_state is not state for mapped_state in hf_states.values())


def test_build_hf_states_orders_grouped_experts():
    hf_name = "model.layers.0.mlp.experts.gate_up_proj.weight"
    tasks = [
        _task(
            f"decoder.layers.0.mlp.experts.linear_fc1.weight{expert}",
            hf_name,
            is_grouped_export=True,
        )
        for expert in range(2)
    ]
    states = {
        tasks[0].global_param_name: _state(weight_amax=2.0),
        tasks[1].global_param_name: _state(weight_amax=5.0),
    }

    hf_states = build_hf_modelopt_quant_states(tasks, states, num_experts=2)

    grouped = hf_states[hf_name]
    assert isinstance(grouped, tuple)
    _assert_state_equal(grouped[0], states[tasks[0].global_param_name])
    _assert_state_equal(grouped[1], states[tasks[1].global_param_name])


def test_build_hf_states_rejects_missing_trailing_grouped_expert():
    task = _task(
        "decoder.layers.0.mlp.experts.linear_fc1.weight0",
        "model.layers.0.mlp.experts.gate_up_proj.weight",
        is_grouped_export=True,
    )

    with pytest.raises(RuntimeError, match=r"Missing ModelOpt state for experts \[1\]"):
        build_hf_modelopt_quant_states(
            [task],
            {task.global_param_name: _state()},
            num_experts=2,
        )


def test_build_modelopt_quantization_config_uses_canonical_hf_layers():
    linears = _quantized_linears()
    linears[1].input_quantizer.disable()
    router = object.__new__(TopKRouter)
    torch.nn.Module.__init__(router)
    router.weight = torch.nn.Parameter(torch.ones(2, 16))
    embedding = torch.nn.Embedding(8, 16)
    tasks = [
        _task(
            "decoder.layers.0.mlp.linear_fc1.weight",
            "model.layers.0.mlp.up_proj.weight",
            megatron_module=linears[0],
            param_weight=linears[0].weight,
        ),
        _task(
            "decoder.layers.1.mlp.linear_fc1.weight",
            "model.layers.1.mlp.up_proj.weight",
            megatron_module=linears[1],
            param_weight=linears[1].weight,
        ),
        _task(
            "decoder.layers.0.mlp.router.weight",
            "model.layers.0.mlp.gate.weight",
            megatron_module=router,
            param_weight=router.weight,
        ),
        _task(
            "embedding.word_embeddings.weight",
            "model.embed_tokens.weight",
            megatron_module=embedding,
            param_weight=embedding.weight,
        ),
    ]
    hf_states = {
        tasks[0].mapping.hf_param: _state(),
        tasks[1].mapping.hf_param: _state(w4a16=True),
    }

    config = build_modelopt_quantization_config(
        tasks,
        hf_states,
        config_weights=collect_modelopt_config_weights(tasks),
    )

    quantization = config
    assert quantization["quant_algo"] == "MIXED_PRECISION"
    assert quantization["quantized_layers"]["model.layers.0.mlp.up_proj"] == {
        "quant_algo": "NVFP4",
        "group_size": 16,
    }
    assert quantization["quantized_layers"]["model.layers.1.mlp.up_proj"] == {
        "quant_algo": "W4A16_NVFP4",
        "group_size": 16,
    }
    assert "model.layers.0.mlp.gate" in quantization["ignore"]
    assert "model.embed_tokens" not in quantization["ignore"]


def test_build_modelopt_export_plan_packs_lazily_and_preserves_source(monkeypatch):
    task = _task(
        "decoder.layers.0.mlp.linear_fc1.weight",
        "model.layers.0.mlp.up_proj.weight",
    )
    state = _state()
    monkeypatch.setattr(
        modelopt_utils,
        "collect_modelopt_quant_states",
        lambda _tasks: {task.global_param_name: state},
    )
    monkeypatch.setattr(
        modelopt_utils,
        "collect_modelopt_config_weights",
        lambda _tasks: {task.global_param_name},
    )
    export_calls = []
    real_export = export_quantized_weight

    def tracked_export(weight, export_state, *, dtype):
        export_calls.append(weight)
        return real_export(weight, export_state, dtype=dtype)

    import modelopt.torch.export.quantized_weight as quantized_weight

    monkeypatch.setattr(quantized_weight, "export_quantized_weight", tracked_export)
    source = torch.randn(4, 16)
    source_copy = source.clone()

    plan = build_modelopt_export_plan([None, task], model=_model())
    exported = plan.conversion_tasks[0].export_hook(task.mapping.hf_param, source)

    assert export_calls == []
    tensors = dict(exported)
    assert len(export_calls) == 1
    assert torch.equal(source, source_copy)
    assert list(tensors) == [
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.up_proj.weight_scale",
        "model.layers.0.mlp.up_proj.weight_scale_2",
        "model.layers.0.mlp.up_proj.input_scale",
    ]


def test_build_modelopt_export_plan_omits_internal_quantizer_state(monkeypatch):
    weight_task = _task(
        "decoder.layers.0.mlp.linear_fc1.weight",
        "model.layers.0.mlp.up_proj.weight",
    )
    quantizer_task = _task(
        "decoder.layers.0.mlp.linear_fc1.weight_quantizer._amax",
        "model.layers.0.mlp.up_proj.weight_quantizer._amax",
    )
    monkeypatch.setattr(
        modelopt_utils,
        "collect_modelopt_quant_states",
        lambda tasks: {tasks[0].global_param_name: _state(w4a16=True)},
    )
    monkeypatch.setattr(
        modelopt_utils,
        "collect_modelopt_config_weights",
        lambda tasks: {tasks[0].global_param_name},
    )

    plan = build_modelopt_export_plan(
        [weight_task, quantizer_task],
        model=_model(),
    )

    assert [task.global_param_name for task in plan.conversion_tasks] == [weight_task.global_param_name]


def test_build_modelopt_export_plan_orders_tasks_by_hf_name(monkeypatch):
    tasks = [
        _task(
            "decoder.layers.1.mlp.linear_fc1.weight",
            "model.layers.1.mlp.up_proj.weight",
        ),
        _task(
            "decoder.layers.0.mlp.linear_fc1.weight",
            "model.layers.0.mlp.up_proj.weight",
        ),
    ]
    states = {task.global_param_name: _state(w4a16=True) for task in tasks}
    monkeypatch.setattr(modelopt_utils, "collect_modelopt_quant_states", lambda _tasks: states)
    monkeypatch.setattr(
        modelopt_utils,
        "collect_modelopt_config_weights",
        lambda _tasks: set(states),
    )

    plan = build_modelopt_export_plan(tasks, model=_model())

    assert [_hf_task.mapping.hf_param for _hf_task in plan.conversion_tasks] == [
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.1.mlp.up_proj.weight",
    ]


def test_build_modelopt_export_plan_packs_grouped_experts_independently(monkeypatch):
    hf_name = "model.layers.0.mlp.experts.gate_up_proj.weight"
    tasks = [
        _task(
            f"decoder.layers.0.mlp.experts.linear_fc1.weight{expert}",
            hf_name,
            is_grouped_export=True,
        )
        for expert in range(2)
    ]
    states = {
        tasks[0].global_param_name: _state(weight_amax=2.0),
        tasks[1].global_param_name: _state(weight_amax=5.0),
    }
    monkeypatch.setattr(modelopt_utils, "collect_modelopt_quant_states", lambda _tasks: states)
    monkeypatch.setattr(
        modelopt_utils,
        "collect_modelopt_config_weights",
        lambda _tasks: set(states),
    )
    source = torch.randn(2, 4, 16)

    plan = build_modelopt_export_plan(tasks, model=_model(num_moe_experts=2))
    bridge = object.__new__(MegatronModelBridge)
    grouped_buffers = {}
    assert (
        bridge._accumulate_grouped_export(
            plan.conversion_tasks[0],
            {hf_name: source[0]},
            SimpleNamespace(num_moe_experts=2),
            grouped_buffers,
            {},
        )
        is None
    )
    tensors = bridge._accumulate_grouped_export(
        plan.conversion_tasks[1],
        {hf_name: source[1]},
        SimpleNamespace(num_moe_experts=2),
        grouped_buffers,
        {},
    )

    expected = [
        export_quantized_weight(
            source[index],
            states[task.global_param_name],
            dtype=source.dtype,
        ).named_tensors()
        for index, task in enumerate(tasks)
    ]
    assert tensors is not None
    assert torch.equal(tensors[hf_name], torch.stack([item["weight"] for item in expected]))
    assert torch.equal(
        tensors[hf_name.removesuffix("weight") + "weight_scale_2"],
        torch.stack([item["weight_scale_2"] for item in expected]),
    )


def test_pre_ep_export_packs_before_gather_and_preserves_hf_names(monkeypatch):
    hf_names = [f"model.layers.0.mlp.experts.{expert}.down_proj.weight" for expert in range(2)]
    tasks = [
        WeightConversionTask(
            param_name=f"decoder.layers.0.mlp.experts.linear_fc2.weight{expert}",
            global_param_name=f"decoder.layers.0.mlp.experts.linear_fc2.weight{expert}",
            mapping=AutoMapping(
                f"decoder.layers.0.mlp.experts.linear_fc2.weight{expert}",
                hf_name,
            ),
        )
        for expert, hf_name in enumerate(hf_names)
    ]
    states = {task.global_param_name: _state(weight_amax=2.0 + expert) for expert, task in enumerate(tasks)}
    monkeypatch.setattr(modelopt_utils, "collect_modelopt_quant_states", lambda _tasks: states)
    monkeypatch.setattr(
        modelopt_utils,
        "collect_modelopt_config_weights",
        lambda _tasks: set(states),
    )
    sources = [torch.randn(4, 16) for _ in tasks]
    packed_shapes = []
    real_export = export_quantized_weight

    def tracked_export(weight, state, *, dtype):
        packed_shapes.append(tuple(weight.shape))
        return real_export(weight, state, dtype=dtype)

    import modelopt.torch.export.quantized_weight as quantized_weight

    monkeypatch.setattr(quantized_weight, "export_quantized_weight", tracked_export)

    plan = build_modelopt_export_plan(tasks, model=_model(num_moe_experts=2))
    output = {}
    for task, source in zip(plan.conversion_tasks, sources, strict=True):
        assert getattr(task.mapping, "is_modelopt_pre_ep_export", False)
        output.update(dict(task.export_hook(task.mapping.hf_param, source)))

    assert packed_shapes == [(4, 16), (4, 16)]
    assert set(hf_names).issubset(output)
    expected = [
        export_quantized_weight(source, states[task.global_param_name], dtype=source.dtype).named_tensors()
        for task, source in zip(tasks, sources, strict=True)
    ]
    for expert, hf_name in enumerate(hf_names):
        assert torch.equal(output[hf_name], expected[expert]["weight"])


def test_grouped_export_packs_ep_gather_before_global_stack(monkeypatch):
    hf_name = "model.layers.0.mlp.experts.down_proj.weight"
    local_task = _task(
        "decoder.layers.0.mlp.experts.linear_fc2.weight0",
        hf_name,
        is_grouped_export=True,
        ep_size=2,
        transpose_on_export=True,
    )
    remote_name = "decoder.layers.0.mlp.experts.linear_fc2.weight1"
    states = {
        local_task.global_param_name: _state(weight_amax=2.0),
        remote_name: _state(weight_amax=5.0),
    }
    monkeypatch.setattr(modelopt_utils, "collect_modelopt_quant_states", lambda _tasks: states)
    monkeypatch.setattr(
        modelopt_utils,
        "collect_modelopt_config_weights",
        lambda _tasks: set(states),
    )
    gathered_experts = torch.randn(2, 16, 4)

    plan = build_modelopt_export_plan([local_task], model=_model(num_moe_experts=2))
    bridge = object.__new__(MegatronModelBridge)
    tensors = bridge._accumulate_grouped_export(
        plan.conversion_tasks[0],
        {hf_name: gathered_experts},
        SimpleNamespace(num_moe_experts=2),
        {},
        {hf_name: torch.empty(2, 4, 16)},
    )

    expected = [
        export_quantized_weight(
            gathered_experts[index].t().contiguous(),
            states[state_name],
            dtype=gathered_experts.dtype,
        ).named_tensors()
        for index, state_name in enumerate((local_task.global_param_name, remote_name))
    ]
    assert tensors is not None
    assert torch.equal(tensors[hf_name], torch.stack([item["weight"] for item in expected]))


def test_build_modelopt_export_plan_preserves_existing_export_hook(monkeypatch):
    def finalizer(name, tensor):
        yield HFWeightTuple(f"final.{name}", tensor)

    task = _task(
        "decoder.layers.0.mlp.linear_fc1.weight",
        "model.layers.0.mlp.up_proj.weight",
        export_hook=finalizer,
    )
    monkeypatch.setattr(
        modelopt_utils,
        "collect_modelopt_quant_states",
        lambda _tasks: {task.global_param_name: _state(w4a16=True)},
    )
    monkeypatch.setattr(
        modelopt_utils,
        "collect_modelopt_config_weights",
        lambda _tasks: {task.global_param_name},
    )

    plan = build_modelopt_export_plan([task], model=_model())
    names = [
        name
        for name, _ in plan.conversion_tasks[0].export_hook(
            task.mapping.hf_param,
            torch.randn(4, 16),
        )
    ]

    assert names == [
        "final.model.layers.0.mlp.up_proj.weight",
        "final.model.layers.0.mlp.up_proj.weight_scale",
        "final.model.layers.0.mlp.up_proj.weight_scale_2",
    ]


class _FakeAutoBridge:
    hf_pretrained = object()
    _build_modelopt_export_plan = AutoBridge._build_modelopt_export_plan

    def __init__(self, tasks, plan):
        self._model_bridge = SimpleNamespace(
            build_conversion_tasks=lambda *_args: tasks,
        )
        self.plan = plan
        self.export_kwargs = None

    def export_hf_weights(self, model, **kwargs):
        self.export_kwargs = kwargs
        return iter((HFWeightTuple("hf.weight", torch.ones(1)),))


def test_auto_bridge_exposes_config_and_uses_prepared_tasks(monkeypatch):
    task = _task("decoder.weight", "hf.weight")
    export_task = _task("decoder.weight", "hf.weight")
    plan = ModelOptExportPlan(
        conversion_tasks=[export_task],
        quantization_config={"quant_algo": "NVFP4"},
    )
    fake = _FakeAutoBridge([task], plan)
    monkeypatch.setattr(modelopt_utils, "build_modelopt_export_plan", lambda *_args, **_kwargs: plan)
    model = torch.nn.Module()

    config = AutoBridge.get_hf_modelopt_quantization_config(fake, model)
    weights = AutoBridge.export_hf_weights_modelopt(fake, model, cpu=True)

    assert fake.export_kwargs is not None
    weights = list(weights)

    assert config is plan.quantization_config
    assert [weight.param_name for weight in weights] == ["hf.weight"]
    assert torch.equal(weights[0].weight, torch.ones(1))
    assert fake.export_kwargs["conversion_tasks"] == [export_task]
    assert fake.export_kwargs["cpu"] is True
