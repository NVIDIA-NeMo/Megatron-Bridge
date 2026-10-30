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

"""Online KV companions share the weight task's ownership and stream lifecycle."""

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch


mtq = pytest.importorskip("modelopt.torch.quantization")
weight_export = pytest.importorskip("modelopt.torch.export.quantized_weight_export")
pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        not hasattr(weight_export, "capture_kv_cache_export_state"),
        reason="Requires ModelOpt's functional KV export API",
    ),
]

from modelopt.torch.quantization.config import QuantizerAttributeConfig

from megatron.bridge.models.conversion import modelopt_utils
from megatron.bridge.models.conversion._modelopt_kv import _KVExportHook
from megatron.bridge.models.conversion.model_bridge import WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import QKVMapping


@pytest.fixture
def identity_quant_backend() -> Iterator[str]:
    from modelopt.torch.quantization.nn.modules.tensor_quantizer import (
        register_quant_backend,
        unregister_quant_backend,
    )

    name = "bridge_kv_identity_test"
    register_quant_backend(name, lambda inputs, _quantizer: inputs)
    try:
        yield name
    finally:
        unregister_quant_backend(name)


def _model_and_task(kv_format="FP8", query=False):
    model = torch.nn.Module()
    model.self_attention = torch.nn.Module()
    attention = torch.nn.Module()
    model.self_attention.core_attention = attention
    for axis, value in (("q", 224.0), ("k", 112.0), ("v", 56.0)):
        config = {"num_bits": (4, 3), "axis": None}
        if kv_format == "NVFP4" and axis != "q":
            config.update(num_bits=(2, 1), block_sizes={-1: 16, "type": "dynamic", "scale_bits": (4, 3)})
        quantizer = mtq.nn.TensorQuantizer(QuantizerAttributeConfig(**config))
        quantizer.amax = value
        if axis == "q" and not query:
            quantizer.disable()
        setattr(attention, f"{axis}_bmm_quantizer", quantizer)
    linear = torch.nn.Linear(4, 8, bias=False)
    linear.tensor_model_parallel = True
    linear.partition_dim = 0
    linear.config = SimpleNamespace(num_attention_heads=2, num_query_groups=1, kv_channels=2, hidden_size=4)
    model.self_attention.linear_qkv = linear
    mapping = QKVMapping(
        "decoder.layers.0.self_attention.linear_qkv.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
    )
    task = WeightConversionTask(
        param_name="self_attention.linear_qkv.weight",
        global_param_name=mapping.megatron_param,
        mapping=mapping,
        megatron_module=linear,
        param_weight=linear.weight,
    )
    return model, attention, task


@pytest.mark.parametrize("kv_format", ["FP8", "NVFP4"])
@pytest.mark.parametrize("query", [False, True])
def test_kv_only_plan_captures_fresh_unclamped_scales_exactly_once(kv_format, query):
    model, attention, task = _model_and_task(kv_format, query)
    plan = modelopt_utils.build_modelopt_export_plan([task], model=[model])
    assert not plan.quantized_params
    assert plan.quantization_config["kv_cache_scheme"] is not None

    def stream():
        prepared = modelopt_utils.prepare_modelopt_export_tasks(plan)[0]
        return [
            item
            for hf_name in task.mapping.hf_param.values()
            for item in prepared.export_hook(hf_name, task.param_weight)
        ]

    first = stream()
    assert len(first) == 6
    assert len({name for name, _ in first}) == len(first)
    first = dict(first)
    prefix = "model.layers.0.self_attn"
    assert first[f"{prefix}.q_scale"].item() == (0.5 if query else 1.0)
    assert first[f"{prefix}.k_scale"].item() == 0.25
    assert first[f"{prefix}.v_scale"].item() == 0.125

    attention.k_bmm_quantizer.amax = 224.0
    second = dict(stream())
    assert second[f"{prefix}.k_scale"].item() == 0.5
    assert first[f"{prefix}.k_scale"].item() == 0.25


@pytest.mark.parametrize("kv_format", ["FP8", "NVFP4"])
@pytest.mark.parametrize("calibrated_kv", [False, True])
@pytest.mark.parametrize("keep_amax", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.run_only_on("gpu"))])
def test_kv_export_supports_constant_scales(kv_format, calibrated_kv, keep_amax, device):
    model, attention, task = _model_and_task(kv_format, query=True)
    constant_axes = ("q",) if calibrated_kv else ("q", "k", "v")
    for axis in constant_axes:
        quantizer = getattr(attention, f"{axis}_bmm_quantizer")
        quantizer.set_from_attribute_config({"use_constant_amax": True})
        if not keep_amax:
            del quantizer._amax
    model.to(device)
    task = replace(task, param_weight=model.self_attention.linear_qkv.weight)

    plan = modelopt_utils.build_modelopt_export_plan([task], model=[model])
    prepared = modelopt_utils.prepare_modelopt_export_tasks(plan)[0]
    exported = dict(prepared.export_hook(task.mapping.hf_param["q"], task.param_weight))
    for axis, expected in (("q", 1.0), ("k", 0.25 if calibrated_kv else 1.0), ("v", 0.125 if calibrated_kv else 1.0)):
        scale = exported[f"model.layers.0.self_attn.{axis}_scale"]
        assert scale.device == task.param_weight.device
        assert scale.item() == expected
    assert all(hasattr(getattr(attention, f"{axis}_bmm_quantizer"), "_amax") == keep_amax for axis in constant_axes)


def test_kv_plan_rejects_structural_change_before_streaming():
    model, attention, task = _model_and_task()
    plan = modelopt_utils.build_modelopt_export_plan([task], model=[model])
    attention.k_bmm_quantizer.disable()
    with pytest.raises(RuntimeError, match="export plan validation failed"):
        modelopt_utils.prepare_modelopt_export_tasks(plan)


def test_kv_plan_rejects_newly_enabled_quantizers():
    model, attention, task = _model_and_task()
    attention.k_bmm_quantizer.disable()
    attention.v_bmm_quantizer.disable()
    mtq.quantize(
        model.self_attention.linear_qkv,
        deepcopy(mtq.FP8_DEFAULT_CFG),
        lambda candidate: candidate(torch.ones(2, 4)),
    )
    plan = modelopt_utils.build_modelopt_export_plan([task], model=[model])
    attention.k_bmm_quantizer.enable()
    attention.v_bmm_quantizer.enable()
    with pytest.raises(RuntimeError, match="export plan validation failed"):
        modelopt_utils.prepare_modelopt_export_tasks(plan)


@pytest.mark.parametrize("kv_format", ["FP8", "NVFP4"])
@pytest.mark.parametrize("axis", ["q", "k", "v"])
@pytest.mark.parametrize("transform", ["rotation", "pre-quant scaling"])
@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("reuse_plan", [False, True])
def test_kv_export_rejects_unrepresented_transforms(kv_format, axis, transform, disabled, reuse_plan):
    model, attention, task = _model_and_task(kv_format, query=True)
    if disabled:
        for name in ("q", "k", "v"):
            getattr(attention, f"{name}_bmm_quantizer").disable()
        # Keep weight-only export eligible when all attention quantizers are disabled.
        mtq.quantize(
            model.self_attention.linear_qkv,
            deepcopy(mtq.FP8_DEFAULT_CFG),
            lambda candidate: candidate(torch.ones(2, 4)),
        )
    plan = modelopt_utils.build_modelopt_export_plan([task], model=[model]) if reuse_plan else None
    quantizer = getattr(attention, f"{axis}_bmm_quantizer")
    if transform == "rotation":
        quantizer.set_from_attribute_config({"rotate": True})
    else:
        quantizer.pre_quant_scale = torch.tensor(0.5)
    assert quantizer.is_enabled != disabled
    with pytest.raises(RuntimeError, match=f"{axis.upper()} export does not support {transform}"):
        if reuse_plan:
            modelopt_utils.prepare_modelopt_export_tasks(plan)
        else:
            modelopt_utils.build_modelopt_export_plan([task], model=[model])


@pytest.mark.parametrize("reuse_plan", [False, True])
def test_rotated_nvfp4_kv_recipe_is_rejected(reuse_plan):
    model, attention, task = _model_and_task("NVFP4")
    plan = modelopt_utils.build_modelopt_export_plan([task], model=[model]) if reuse_plan else None
    mtq.set_quantizer_by_cfg(model, deepcopy(mtq.NVFP4_KV_ROTATE_CFG["quant_cfg"]))
    assert not attention.q_bmm_quantizer.is_enabled
    assert attention.q_bmm_quantizer.rotate_is_enabled
    assert attention.k_bmm_quantizer.rotate_is_enabled
    with pytest.raises(RuntimeError, match="Q export does not support rotation"):
        if reuse_plan:
            modelopt_utils.prepare_modelopt_export_tasks(plan)
        else:
            modelopt_utils.build_modelopt_export_plan([task], model=[model])


@pytest.mark.parametrize("axis", ["k", "v"])
@pytest.mark.parametrize("reuse_plan", [False, True])
@pytest.mark.parametrize(
    ("kv_format", "attributes"),
    [
        ("FP8", {"axis": -1}),
        ("FP8", {"type": "dynamic"}),
        ("FP8", {"block_sizes": {-1: 16, "type": "dynamic"}}),
        ("NVFP4", {"type": "dynamic"}),
        ("NVFP4", {"block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (4, 3)}}),
        ("NVFP4", {"block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)}}),
        ("NVFP4", {"block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (8, 0)}}),
        ("NVFP4", {"block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3), "four_over_six": True}}),
    ],
)
def test_kv_export_rejects_unrepresented_layouts(kv_format, attributes, axis, reuse_plan):
    model, attention, task = _model_and_task(kv_format)
    plan = modelopt_utils.build_modelopt_export_plan([task], model=[model]) if reuse_plan else None
    getattr(attention, f"{axis}_bmm_quantizer").set_from_attribute_config(deepcopy(attributes))
    with pytest.raises(RuntimeError, match=f"{axis.upper()} export requires static symmetric"):
        if reuse_plan:
            modelopt_utils.prepare_modelopt_export_tasks(plan)
        else:
            modelopt_utils.build_modelopt_export_plan([task], model=[model])


@pytest.mark.parametrize("kv_format", ["FP8", "NVFP4"])
@pytest.mark.parametrize("axis", ["q", "k", "v"])
@pytest.mark.parametrize("reuse_plan", [False, True])
def test_kv_export_rejects_unrepresented_backend(kv_format, axis, reuse_plan, identity_quant_backend):
    model, attention, task = _model_and_task(kv_format, query=True)
    plan = modelopt_utils.build_modelopt_export_plan([task], model=[model]) if reuse_plan else None
    quantizer = getattr(attention, f"{axis}_bmm_quantizer")
    quantizer.set_from_attribute_config({"backend": identity_quant_backend})
    inputs = torch.tensor([1.1, 2.3, 3.7, 4.9])
    torch.testing.assert_close(quantizer(inputs), inputs, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match=f"{axis.upper()} export does not support custom quantization backends"):
        if reuse_plan:
            modelopt_utils.prepare_modelopt_export_tasks(plan)
        else:
            modelopt_utils.build_modelopt_export_plan([task], model=[model])


@pytest.mark.parametrize("kv_format", ["FP8", "NVFP4"])
def test_disabled_query_backend_remains_compatible(kv_format, identity_quant_backend):
    model, attention, task = _model_and_task(kv_format)
    attention.q_bmm_quantizer.set_from_attribute_config({"backend": identity_quant_backend})
    plan = modelopt_utils.build_modelopt_export_plan([task], model=[model])
    modelopt_utils.prepare_modelopt_export_tasks(plan)


@pytest.mark.parametrize("bad_scale", [0.0, float("nan"), float("inf")])
def test_live_invalid_kv_scales_fail_instead_of_using_native_defaults(monkeypatch, bad_scale):
    model, attention, task = _model_and_task()
    plan = modelopt_utils.build_modelopt_export_plan([task], model=[model])
    monkeypatch.setattr(
        weight_export,
        "capture_kv_cache_export_state",
        lambda *_args, **_kwargs: SimpleNamespace(k_scale=torch.tensor(bad_scale), v_scale=torch.tensor(1.0)),
    )
    hook = modelopt_utils.prepare_modelopt_export_tasks(plan)[0].export_hook
    with pytest.raises(RuntimeError, match="finite and positive"):
        list(hook(task.mapping.hf_param["q"], task.param_weight))


def test_kv_tp_disagreement_is_not_silently_merged(monkeypatch):
    model, _attention, task = _model_and_task()
    plan = modelopt_utils.build_modelopt_export_plan([task], model=[model])
    hook = modelopt_utils.prepare_modelopt_export_tasks(plan)[0].export_hook
    monkeypatch.setattr(task.mapping, "gather_from_tp_ranks", lambda value: [value, value * 2])
    with pytest.raises(ValueError, match="scales across TP"):
        list(hook(task.mapping.hf_param["q"], task.param_weight))


def test_remote_kv_capture_propagates_owner_failure(monkeypatch):
    _model, _attention, task = _model_and_task()
    task = replace(task, megatron_module=None, param_weight=None)
    hook = _KVExportHook(task, None, "model.layers.0.self_attn", ("FP8", False))
    monkeypatch.setattr(task.mapping, "broadcast_obj_from_pp_rank", lambda _value: ("owner capture failed",))
    with pytest.raises(RuntimeError, match="owner capture failed"):
        list(hook.scales())
