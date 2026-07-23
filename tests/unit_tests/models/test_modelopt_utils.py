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

from types import SimpleNamespace

import pytest
import torch


quant_utils = pytest.importorskip("modelopt.torch.export.quant_utils")
QUANTIZATION_NONE = quant_utils.QUANTIZATION_NONE
QUANTIZATION_NVFP4 = quant_utils.QUANTIZATION_NVFP4

from megatron.bridge.models.conversion import modelopt_utils
from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.conversion.model_bridge import WeightConversionTask
from megatron.bridge.models.conversion.modelopt_utils import (
    QuantMeta,
    _fuse_grouped_projection_names,
    _grouped_expert_projection_name,
    _modelopt_pre_ep_mapping,
    _stage_tensor_for_collective,
    build_hf_modelopt_quant_metadata,
    collect_modelopt_quant_metadata,
    compute_nvfp4_input_scale,
    find_modelopt_weight_quantizer_and_module,
    get_modelopt_quant_exporter,
    is_modelopt_quantizable_weight_name,
    matches_quant_ignore_pattern,
    quantize_nvfp4_weight,
    sync_modelopt_quant_metadata,
)
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping


def _task(
    global_param_name,
    hf_param,
    *,
    megatron_module=None,
    param_weight=None,
):
    return WeightConversionTask(
        param_name=global_param_name,
        global_param_name=global_param_name,
        mapping=SimpleNamespace(hf_param=hf_param),
        megatron_module=megatron_module,
        param_weight=param_weight,
    )


def _quant_meta(qformat=QUANTIZATION_NVFP4):
    return QuantMeta(
        qformat=qformat,
        block_size=16,
        weight_amax=torch.tensor([1.0]),
        weight_scale_2=torch.tensor([1.0 / (6.0 * 448.0)]),
        input_amax=torch.tensor([1.0]),
    )


def _bridge_for_export(conversion_tasks, exported_weights):
    class FakeBridge:
        hf_pretrained = object()
        _model_bridge = SimpleNamespace(
            build_conversion_tasks=lambda *_args, **_kwargs: conversion_tasks,
        )

        def __init__(self):
            self.export_calls = []

        def export_hf_weights(self, model, **kwargs):
            self.export_calls.append((model, kwargs))
            export_hf_weight = next(
                (task.export_hook for task in kwargs.get("conversion_tasks", ()) if task.export_hook is not None),
                None,
            )
            for hf_name, tensor in exported_weights:
                finalized_weights = (
                    export_hf_weight(hf_name, tensor.detach())
                    if export_hf_weight is not None
                    else ((hf_name, tensor.detach()),)
                )
                for finalized_name, finalized_tensor in finalized_weights:
                    finalized_tensor = finalized_tensor.detach()
                    if kwargs.get("cpu", False):
                        finalized_tensor = finalized_tensor.cpu()
                    yield finalized_name, finalized_tensor

    return FakeBridge()


def test_matches_quant_ignore_pattern_handles_model_prefix_and_scale_suffixes():
    ignore_patterns = [
        "lm_head",
        "*self_attn*",
        "*mlp.gate",
        "*router*",
    ]

    assert matches_quant_ignore_pattern(
        "model.layers.0.self_attn.o_proj.weight",
        ignore_patterns,
    )
    assert matches_quant_ignore_pattern(
        "layers.0.self_attn.o_proj.weight",
        ignore_patterns,
    )
    assert matches_quant_ignore_pattern("model.layers.0.mlp.gate.weight", ignore_patterns)
    assert matches_quant_ignore_pattern("model.layers.0.router.weight", ignore_patterns)
    assert matches_quant_ignore_pattern("lm_head.weight", ignore_patterns)
    assert matches_quant_ignore_pattern("model.layers.0.mlp.gate.weight_scale", ignore_patterns)
    assert not matches_quant_ignore_pattern(
        "model.layers.0.mlp.experts.0.w1.weight",
        ignore_patterns,
    )


def test_find_modelopt_weight_quantizer_uses_proxy_for_custom_weight():
    class FakeQuantModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.custom_weight = torch.nn.Parameter(torch.ones(1))
            self.weight_quantizer = SimpleNamespace(is_enabled=True)
            self.custom_quantizer = SimpleNamespace(is_enabled=True)
            self.input_quantizer = SimpleNamespace(is_enabled=True)

        def iter_weights_for_calibration(self):
            yield self.custom_weight, self.custom_quantizer

    module = FakeQuantModule()

    weight_quantizer, quant_module = find_modelopt_weight_quantizer_and_module(
        module,
        module.custom_weight,
    )

    assert weight_quantizer is module.custom_quantizer
    assert quant_module is not module
    assert quant_module.weight is module.custom_weight
    assert quant_module.weight_quantizer is module.custom_quantizer
    assert quant_module.input_quantizer is module.input_quantizer


def test_find_modelopt_weight_quantizer_returns_owner_for_weight_param():
    class FakeQuantModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.weight_quantizer = SimpleNamespace(is_enabled=True)

        def iter_weights_for_calibration(self):
            yield self.weight, self.weight_quantizer

    module = FakeQuantModule()

    weight_quantizer, quant_module = find_modelopt_weight_quantizer_and_module(
        module,
        module.weight,
    )

    assert weight_quantizer is module.weight_quantizer
    assert quant_module is module


def test_iter_modelopt_weight_quantizers_does_not_repeat_calibration_weight():
    class FakeQuantModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.weight_quantizer = SimpleNamespace(is_enabled=True)

        def iter_weights_for_calibration(self):
            yield self.weight, self.weight_quantizer

    module = FakeQuantModule()

    matches = list(modelopt_utils._iter_modelopt_weight_quantizers(module))

    assert len(matches) == 1
    weight, quantizer, can_use_module = matches[0]
    assert weight is module.weight
    assert quantizer is module.weight_quantizer
    assert can_use_module is True


def test_collect_modelopt_quant_metadata_skips_unquantized_tasks(monkeypatch):
    monkeypatch.delattr(quant_utils, "QUANTIZATION_W4A16_NVFP4", raising=False)
    quantizer_amax = torch.tensor([-2688.0])
    input_amax = torch.tensor([1344.0])
    quant_module = SimpleNamespace(
        weight_quantizer=SimpleNamespace(_amax=quantizer_amax),
        input_quantizer=SimpleNamespace(_amax=input_amax, is_enabled=True),
    )
    unquantized_module = SimpleNamespace(weight_quantizer=SimpleNamespace(_amax=torch.tensor([1.0])))
    blockless_module = SimpleNamespace(weight_quantizer=SimpleNamespace(_amax=torch.tensor([2.0])))

    qformat_by_module = {
        id(quant_module): QUANTIZATION_NVFP4,
        id(unquantized_module): QUANTIZATION_NONE,
        id(blockless_module): QUANTIZATION_NVFP4,
    }
    block_size_by_module = {
        id(quant_module): 16,
        id(unquantized_module): 16,
        id(blockless_module): 0,
    }

    monkeypatch.setattr(
        quant_utils,
        "get_quantization_format",
        lambda module: qformat_by_module[id(module)],
    )
    monkeypatch.setattr(
        quant_utils,
        "get_weight_block_size",
        lambda module: block_size_by_module[id(module)],
    )
    monkeypatch.setattr(
        modelopt_utils,
        "find_modelopt_weight_quantizer_and_module",
        lambda module, _weight: (module.weight_quantizer, module),
    )

    metadata = collect_modelopt_quant_metadata(
        [
            _task(
                "missing.module.weight",
                "hf.missing.weight",
                megatron_module=None,
                param_weight=torch.empty(1),
            ),
            _task(
                "missing.param.weight",
                "hf.missing_param.weight",
                megatron_module=quant_module,
            ),
            _task(
                "unquantized.weight",
                "hf.unquantized.weight",
                megatron_module=unquantized_module,
                param_weight=torch.empty(1),
            ),
            _task(
                "blockless.weight",
                "hf.blockless.weight",
                megatron_module=blockless_module,
                param_weight=torch.empty(1),
            ),
            _task(
                "quantized.weight",
                "hf.quantized.weight",
                megatron_module=quant_module,
                param_weight=torch.empty(1),
            ),
        ]
    )

    assert list(metadata) == ["quantized.weight"]
    assert metadata["quantized.weight"].qformat == QUANTIZATION_NVFP4
    assert metadata["quantized.weight"].block_size == 16
    torch.testing.assert_close(
        metadata["quantized.weight"].weight_amax,
        quantizer_amax.abs(),
    )
    torch.testing.assert_close(
        metadata["quantized.weight"].weight_scale_2,
        torch.tensor([1.0]),
    )
    torch.testing.assert_close(metadata["quantized.weight"].input_amax, input_amax)
    assert metadata["quantized.weight"].weight_amax.data_ptr() != quantizer_amax.data_ptr()
    assert metadata["quantized.weight"].input_amax.data_ptr() != input_amax.data_ptr()


def test_collect_modelopt_quant_metadata_collects_w4a16_scale_2_and_omits_input_amax(monkeypatch):
    w4a16_qformat = "modelopt_w4a16_nvfp4"
    quantizer_amax = torch.tensor([-2688.0])
    quantizer = SimpleNamespace(_amax=quantizer_amax)
    quant_module = SimpleNamespace(
        weight_quantizer=quantizer,
        input_quantizer=SimpleNamespace(_amax=torch.tensor([123.0]), is_enabled=True),
    )

    monkeypatch.setattr(
        quant_utils,
        "QUANTIZATION_W4A16_NVFP4",
        w4a16_qformat,
        raising=False,
    )
    monkeypatch.setattr(quant_utils, "get_quantization_format", lambda _module: w4a16_qformat)
    monkeypatch.setattr(quant_utils, "get_weight_block_size", lambda _module: 16)
    monkeypatch.setattr(
        modelopt_utils,
        "find_modelopt_weight_quantizer_and_module",
        lambda _module, _weight: (quantizer, quant_module),
    )
    metadata = collect_modelopt_quant_metadata(
        [
            _task(
                "quantized.weight",
                "hf.quantized.weight",
                megatron_module=quant_module,
                param_weight=torch.empty(1),
            )
        ]
    )

    meta = metadata["quantized.weight"]
    assert meta.qformat == w4a16_qformat
    assert meta.block_size == 16
    torch.testing.assert_close(meta.weight_amax, torch.tensor([2688.0]))
    torch.testing.assert_close(meta.weight_scale_2, torch.tensor([1.0]))
    assert meta.input_amax is None


def test_collect_modelopt_quant_metadata_rejects_static_nvfp4(monkeypatch):
    quantizer = SimpleNamespace(
        global_amax=torch.tensor([-2688.0]),
        _amax=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        block_sizes={"four_over_six": True},
    )
    quant_module = SimpleNamespace(weight_quantizer=quantizer)
    monkeypatch.setattr(quant_utils, "get_quantization_format", lambda _module: QUANTIZATION_NVFP4)
    monkeypatch.setattr(quant_utils, "get_weight_block_size", lambda _module: 16)
    monkeypatch.setattr(
        modelopt_utils,
        "find_modelopt_weight_quantizer_and_module",
        lambda _module, _weight: (quantizer, quant_module),
    )

    with pytest.raises(RuntimeError, match="Static NVFP4 weight quantizers.*dynamic NVFP4"):
        collect_modelopt_quant_metadata(
            [
                _task(
                    "quantized.weight",
                    "hf.quantized.weight",
                    megatron_module=quant_module,
                    param_weight=torch.empty(1),
                )
            ]
        )


def test_modelopt_weight_amax_prefers_live_then_restored_static_amax():
    public_amax = torch.tensor([2688.0])
    private_amax = torch.tensor([5376.0])
    block_amax = torch.tensor([1.0, 2.0])

    value, is_static = modelopt_utils._get_modelopt_weight_amax(
        SimpleNamespace(
            global_amax=public_amax,
            _global_amax=private_amax,
            _amax=block_amax,
        )
    )
    assert value is public_amax
    assert is_static

    value, is_static = modelopt_utils._get_modelopt_weight_amax(
        SimpleNamespace(
            _global_amax=private_amax,
            _amax=block_amax,
        )
    )
    assert value is private_amax
    assert is_static


def test_collect_modelopt_quant_metadata_max_reduces_scalars_per_tp_projection(monkeypatch):
    process_groups = [object(), object()]

    def tp_group(process_group):
        return SimpleNamespace(
            group=process_group,
            is_initialized=lambda: True,
            world_size=lambda: 2,
        )

    modules = [
        SimpleNamespace(
            weight_quantizer=SimpleNamespace(_amax=torch.tensor([2688.0])),
            input_quantizer=SimpleNamespace(_amax=torch.tensor([672.0]), is_enabled=True),
            parallel_state=SimpleNamespace(tensor_parallel_group=tp_group(process_groups[0])),
        ),
        SimpleNamespace(
            weight_quantizer=SimpleNamespace(_amax=torch.tensor([1344.0])),
            input_quantizer=SimpleNamespace(_amax=torch.tensor([336.0]), is_enabled=True),
            parallel_state=SimpleNamespace(tensor_parallel_group=tp_group(process_groups[1])),
        ),
    ]
    reduced_groups = []

    monkeypatch.setattr(quant_utils, "get_quantization_format", lambda _module: QUANTIZATION_NVFP4)
    monkeypatch.setattr(quant_utils, "get_weight_block_size", lambda _module: 16)
    monkeypatch.setattr(
        modelopt_utils,
        "find_modelopt_weight_quantizer_and_module",
        lambda module, _weight: (module.weight_quantizer, module),
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def fake_all_reduce(value, op, group):
        assert op == torch.distributed.ReduceOp.MAX
        reduced_groups.append(group)
        value.mul_(2)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    metadata = collect_modelopt_quant_metadata(
        [
            _task(
                f"projection_{index}.weight",
                f"hf.projection_{index}.weight",
                megatron_module=module,
                param_weight=torch.empty(1),
            )
            for index, module in enumerate(modules)
        ]
    )

    torch.testing.assert_close(metadata["projection_0.weight"].weight_amax, torch.tensor([5376.0]))
    torch.testing.assert_close(metadata["projection_0.weight"].weight_scale_2, torch.tensor([2.0]))
    torch.testing.assert_close(metadata["projection_0.weight"].input_amax, torch.tensor([1344.0]))
    torch.testing.assert_close(metadata["projection_1.weight"].weight_amax, torch.tensor([2688.0]))
    torch.testing.assert_close(metadata["projection_1.weight"].weight_scale_2, torch.tensor([1.0]))
    torch.testing.assert_close(metadata["projection_1.weight"].input_amax, torch.tensor([672.0]))
    assert reduced_groups == [
        process_groups[0],
        process_groups[0],
        process_groups[1],
        process_groups[1],
    ]


def test_collect_modelopt_quant_metadata_reuses_only_compatible_shared_quantizer_metadata(monkeypatch):
    quantizer = SimpleNamespace(_amax=torch.tensor([2688.0]))
    quant_module = SimpleNamespace(weight_quantizer=quantizer, block_size=16)
    different_block_module = SimpleNamespace(weight_quantizer=quantizer, block_size=32)
    tasks = [
        _task(
            f"decoder.layers.0.mlp.experts.linear_fc1.weight{expert}",
            f"model.layers.0.mlp.experts.{expert}.up_proj.weight",
            megatron_module=module,
            param_weight=torch.ones(1),
        )
        for expert, module in enumerate((quant_module, quant_module, different_block_module))
    ]
    clone_calls = []

    monkeypatch.setattr(
        modelopt_utils,
        "find_modelopt_weight_quantizer_and_module",
        lambda module, _weight: (quantizer, module),
    )
    monkeypatch.setattr(
        quant_utils,
        "get_quantization_format",
        lambda _module: QUANTIZATION_NVFP4,
    )
    monkeypatch.setattr(quant_utils, "get_weight_block_size", lambda module: module.block_size)
    monkeypatch.setattr(
        modelopt_utils,
        "_clone_positive_cpu",
        lambda value: clone_calls.append(value) or value.clone(),
    )

    metadata = collect_modelopt_quant_metadata(tasks)

    assert list(metadata) == [task.global_param_name for task in tasks]
    assert metadata[tasks[0].global_param_name] is metadata[tasks[1].global_param_name]
    assert metadata[tasks[2].global_param_name] is not metadata[tasks[0].global_param_name]
    assert metadata[tasks[2].global_param_name].block_size == 32
    assert len(clone_calls) == 2


def test_collect_modelopt_quant_metadata_does_not_reuse_different_input_amax(monkeypatch):
    weight_quantizer = SimpleNamespace(_amax=torch.tensor([2688.0]))
    modules = [
        SimpleNamespace(
            weight_quantizer=weight_quantizer,
            input_quantizer=SimpleNamespace(_amax=torch.tensor([value]), is_enabled=True),
        )
        for value in (672.0, 336.0)
    ]
    monkeypatch.setattr(quant_utils, "get_quantization_format", lambda _module: QUANTIZATION_NVFP4)
    monkeypatch.setattr(quant_utils, "get_weight_block_size", lambda _module: 16)
    monkeypatch.setattr(
        modelopt_utils,
        "find_modelopt_weight_quantizer_and_module",
        lambda module, _weight: (weight_quantizer, module),
    )
    metadata = collect_modelopt_quant_metadata(
        [
            _task(
                f"projection_{index}.weight",
                f"hf.projection_{index}.weight",
                megatron_module=module,
                param_weight=torch.empty(1),
            )
            for index, module in enumerate(modules)
        ]
    )

    assert metadata["projection_0.weight"] is not metadata["projection_1.weight"]
    torch.testing.assert_close(metadata["projection_0.weight"].input_amax, torch.tensor([672.0]))
    torch.testing.assert_close(metadata["projection_1.weight"].input_amax, torch.tensor([336.0]))


def test_sync_modelopt_quant_metadata_merges_gathered_rank_metadata(monkeypatch):
    rank1_meta = QuantMeta(
        qformat=QUANTIZATION_NVFP4,
        block_size=16,
        weight_amax=torch.tensor([2.0]),
    )
    metadata = {
        "rank0.weight": QuantMeta(
            qformat=QUANTIZATION_NVFP4,
            block_size=16,
            weight_amax=torch.tensor([1.0]),
        )
    }

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)

    def fake_all_gather_object(gathered, local_metadata, group=None):
        gathered[0] = dict(local_metadata)
        gathered[1] = {"rank1.weight": rank1_meta}

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)

    sync_modelopt_quant_metadata(metadata, group=object())

    assert set(metadata) == {"rank0.weight", "rank1.weight"}
    assert metadata["rank1.weight"] is rank1_meta


def test_build_hf_modelopt_quant_metadata_stacks_synced_grouped_experts():
    hf_name = "model.layers.0.mlp.experts.gate_up_proj.weight"
    task = SimpleNamespace(
        global_param_name="decoder.layers.0.mlp.experts.linear_fc1.weight0",
        mapping=SimpleNamespace(hf_param=hf_name, is_grouped_export=True),
    )
    metadata = {
        f"decoder.layers.0.mlp.experts.linear_fc1.weight{expert_idx}": QuantMeta(
            qformat=QUANTIZATION_NVFP4,
            block_size=16,
            weight_amax=torch.tensor([float(expert_idx)]),
            weight_scale_2=torch.tensor([float(expert_idx + 10)]),
            input_amax=torch.tensor([float(expert_idx + 20)]),
        )
        for expert_idx in range(4)
    }

    hf_metadata = build_hf_modelopt_quant_metadata([task], metadata)

    meta = hf_metadata[hf_name]
    assert meta.qformat == QUANTIZATION_NVFP4
    assert meta.block_size == 16
    torch.testing.assert_close(
        meta.weight_amax,
        torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
    )
    torch.testing.assert_close(
        meta.weight_scale_2,
        torch.tensor([[10.0], [11.0], [12.0], [13.0]]),
    )
    torch.testing.assert_close(
        meta.input_amax,
        torch.tensor([[20.0], [21.0], [22.0], [23.0]]),
    )


def test_build_hf_modelopt_quant_metadata_slices_non_grouped_gated_amax():
    gate_name = "model.layers.0.mlp.gate_proj.weight"
    up_name = "model.layers.0.mlp.up_proj.weight"
    task = SimpleNamespace(
        global_param_name="decoder.layers.0.mlp.linear_fc1.weight",
        mapping=SimpleNamespace(
            hf_param={"gate": gate_name, "up": up_name},
            is_grouped_export=False,
        ),
    )
    shared_scale_2 = torch.tensor(0.5)
    shared_input_amax = torch.tensor([1344.0])
    metadata = {
        task.global_param_name: QuantMeta(
            qformat=QUANTIZATION_NVFP4,
            block_size=16,
            weight_amax=torch.tensor(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0],
                    [7.0, 8.0],
                ]
            ),
            weight_scale_2=shared_scale_2,
            input_amax=shared_input_amax,
        )
    }

    hf_metadata = build_hf_modelopt_quant_metadata([task], metadata)

    torch.testing.assert_close(
        hf_metadata[gate_name].weight_amax,
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    torch.testing.assert_close(
        hf_metadata[up_name].weight_amax,
        torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
    )
    torch.testing.assert_close(hf_metadata[gate_name].weight_scale_2, shared_scale_2)
    torch.testing.assert_close(hf_metadata[up_name].weight_scale_2, shared_scale_2)
    torch.testing.assert_close(hf_metadata[gate_name].input_amax, shared_input_amax)
    torch.testing.assert_close(hf_metadata[up_name].input_amax, shared_input_amax)


def test_build_hf_modelopt_quant_metadata_shares_grouped_gated_scale_2():
    gate_name = "model.layers.0.mlp.experts.gate_proj.weight"
    up_name = "model.layers.0.mlp.experts.up_proj.weight"
    task = SimpleNamespace(
        global_param_name="decoder.layers.0.mlp.experts.linear_fc1.weight0",
        mapping=SimpleNamespace(
            hf_param={"gate": gate_name, "up": up_name},
            is_grouped_export=True,
        ),
    )
    metadata = {
        f"decoder.layers.0.mlp.experts.linear_fc1.weight{expert_idx}": QuantMeta(
            qformat=QUANTIZATION_NVFP4,
            block_size=16,
            weight_amax=torch.tensor(
                [
                    float(expert_idx * 4 + 1),
                    float(expert_idx * 4 + 2),
                    float(expert_idx * 4 + 3),
                    float(expert_idx * 4 + 4),
                ]
            ),
            weight_scale_2=torch.tensor(float(expert_idx + 10)),
            input_amax=torch.tensor([float(expert_idx + 20)]),
        )
        for expert_idx in range(2)
    }

    hf_metadata = build_hf_modelopt_quant_metadata([task], metadata)

    torch.testing.assert_close(
        hf_metadata[gate_name].weight_amax,
        torch.tensor([[1.0, 2.0], [5.0, 6.0]]),
    )
    torch.testing.assert_close(
        hf_metadata[up_name].weight_amax,
        torch.tensor([[3.0, 4.0], [7.0, 8.0]]),
    )
    expected_scale_2 = torch.tensor([10.0, 11.0])
    torch.testing.assert_close(hf_metadata[gate_name].weight_scale_2, expected_scale_2)
    torch.testing.assert_close(hf_metadata[up_name].weight_scale_2, expected_scale_2)
    expected_input_amax = torch.tensor([[20.0], [21.0]])
    torch.testing.assert_close(hf_metadata[gate_name].input_amax, expected_input_amax)
    torch.testing.assert_close(hf_metadata[up_name].input_amax, expected_input_amax)


def test_build_hf_modelopt_pre_ep_quant_metadata_rebases_local_expert_ids():
    hf_name = "backbone.layers.0.mixer.experts.up_proj"
    pre_ep_mapping = SimpleNamespace(hf_param=hf_name, is_modelopt_pre_ep_export=True)
    tasks = [
        SimpleNamespace(
            global_param_name=f"decoder.layers.0.mlp.experts.linear_fc1.weight{expert_idx}",
            mapping=pre_ep_mapping,
        )
        for expert_idx in (5, 4)
    ]
    metadata = {
        f"decoder.layers.0.mlp.experts.linear_fc1.weight{expert_idx}": QuantMeta(
            qformat=QUANTIZATION_NVFP4,
            block_size=16,
            weight_amax=torch.tensor(float(expert_idx)),
            weight_scale_2=torch.tensor(float(expert_idx + 10)),
            input_amax=torch.tensor(float(expert_idx + 20)),
        )
        for expert_idx in (4, 5)
    }

    hf_metadata = modelopt_utils._build_hf_modelopt_pre_ep_quant_metadata(tasks, metadata)

    torch.testing.assert_close(
        hf_metadata[hf_name].weight_amax,
        torch.tensor([4.0, 5.0]),
    )
    torch.testing.assert_close(
        hf_metadata[hf_name].weight_scale_2,
        torch.tensor([14.0, 15.0]),
    )
    torch.testing.assert_close(
        hf_metadata[hf_name].input_amax,
        torch.tensor([24.0, 25.0]),
    )


@pytest.mark.parametrize(
    ("weight_shape", "weight_scale_2", "expected"),
    [
        (
            (2, 4, 4),
            torch.tensor([1.0, 0.5]),
            torch.tensor([[[1.0]], [[0.5]]]),
        ),
        (
            (2, 4, 4),
            torch.tensor([[1.0], [0.5]]),
            torch.tensor([[[1.0]], [[0.5]]]),
        ),
        (
            (2, 4, 4),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor(
                [
                    [[1.0], [1.0], [2.0], [2.0]],
                    [[3.0], [3.0], [4.0], [4.0]],
                ]
            ),
        ),
    ],
    ids=["per-expert-vector", "singleton-column", "grouped-scales"],
)
def test_reshape_nvfp4_weight_scale_2_for_compute(weight_shape, weight_scale_2, expected):
    actual = modelopt_utils._reshape_nvfp4_weight_scale_2_for_compute(
        torch.ones(weight_shape),
        weight_scale_2,
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("weight_shape", "weight_scale_2"),
    [
        ((2, 4), torch.tensor([1.0, 0.5])),
        ((2, 4, 4), torch.tensor(1.0)),
        ((2, 5, 4), torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
        ((2, 4, 4), torch.ones(2, 1, 1)),
    ],
    ids=["non-3d-weight", "scalar", "non-divisible-groups", "already-reshaped"],
)
def test_reshape_nvfp4_weight_scale_2_for_compute_keeps_unsupported_shapes(
    weight_shape,
    weight_scale_2,
):
    actual = modelopt_utils._reshape_nvfp4_weight_scale_2_for_compute(
        torch.ones(weight_shape),
        weight_scale_2,
    )

    assert actual is weight_scale_2


def test_quantize_nvfp4_weight_uses_modelopt_scale_export_and_emits_scale_names(monkeypatch):
    captured = {}

    def fake_to_quantized_weight(
        weight,
        weight_scale,
        qformat,
        weight_scale_2,
        block_size,
    ):
        captured.update(
            weight=weight,
            weight_scale=weight_scale,
            qformat=qformat,
            weight_scale_2=weight_scale_2,
            block_size=block_size,
        )
        return torch.zeros(weight.shape, dtype=torch.uint8, device=weight.device)

    monkeypatch.setattr(quant_utils, "to_quantized_weight", fake_to_quantized_weight)

    tensors = dict(
        quantize_nvfp4_weight(
            "model.layers.0.mlp.up_proj.weight",
            torch.tensor([[-1.0, 0.25, 0.5, 2.0]], dtype=torch.float32),
            QuantMeta(
                qformat=QUANTIZATION_NVFP4,
                block_size=4,
                weight_amax=torch.tensor([2688.0]),
                weight_scale_2=torch.tensor([1.0]),
                input_amax=torch.tensor([1344.0]),
            ),
        )
    )

    assert set(tensors) == {
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.up_proj.weight_scale",
        "model.layers.0.mlp.up_proj.weight_scale_2",
        "model.layers.0.mlp.up_proj.input_scale",
    }
    assert tensors["model.layers.0.mlp.up_proj.weight"].dtype == torch.uint8
    assert tensors["model.layers.0.mlp.up_proj.weight_scale"].dtype == torch.float8_e4m3fn
    torch.testing.assert_close(
        tensors["model.layers.0.mlp.up_proj.weight_scale_2"],
        torch.tensor([1.0]),
    )
    assert tensors["model.layers.0.mlp.up_proj.input_scale"].shape == torch.Size([])
    torch.testing.assert_close(
        tensors["model.layers.0.mlp.up_proj.input_scale"],
        torch.tensor(0.5),
    )
    assert captured["qformat"] == QUANTIZATION_NVFP4
    assert captured["block_size"] == 4
    assert captured["weight_scale"].dtype == torch.float8_e4m3fn
    assert (captured["weight_scale"].to(torch.float32) >= 0).all()
    assert captured["weight_scale_2"].dim() == 0
    torch.testing.assert_close(captured["weight_scale_2"], torch.tensor(1.0))


def test_compute_nvfp4_weight_scale_uses_modelopt_fp8_clamp():
    weight_scale, weight_scale_2 = modelopt_utils.compute_nvfp4_weight_scale(
        torch.full((1, 16), 1.0e-20),
        block_size=16,
        weight_scale_2=torch.tensor(1.0),
    )

    assert weight_scale.dtype == torch.float8_e4m3fn
    assert torch.all(weight_scale.float() >= 2**-9)
    torch.testing.assert_close(weight_scale_2, torch.tensor(1.0))


def test_quantize_nvfp4_weight_exports_fused_moe_internal_names(monkeypatch):
    captured = {}

    def fake_to_quantized_weight(
        weight,
        weight_scale,
        qformat,
        weight_scale_2,
        block_size,
    ):
        captured.update(
            weight=weight,
            weight_scale=weight_scale,
            qformat=qformat,
            weight_scale_2=weight_scale_2,
            block_size=block_size,
        )
        return torch.zeros(weight.shape, dtype=torch.uint8, device=weight.device)

    monkeypatch.setattr(quant_utils, "to_quantized_weight", fake_to_quantized_weight)

    tensors = dict(
        quantize_nvfp4_weight(
            "model.layers.0.mlp.experts.gate_up_proj",
            torch.ones(2, 4, 4, dtype=torch.float32),
            QuantMeta(
                qformat=QUANTIZATION_NVFP4,
                block_size=4,
                weight_amax=torch.tensor([[2688.0], [1344.0]]),
                weight_scale_2=torch.tensor([[1.0], [0.5]]),
                input_amax=torch.tensor([[2688.0], [1344.0]]),
            ),
        )
    )

    assert set(tensors) == {
        "model.layers.0.mlp.experts.w13_weight",
        "model.layers.0.mlp.experts.w13_weight_scale",
        "model.layers.0.mlp.experts.w13_weight_scale_2",
        "model.layers.0.mlp.experts.w13_input_scale",
    }
    assert tensors["model.layers.0.mlp.experts.w13_weight"].dtype == torch.uint8
    assert tensors["model.layers.0.mlp.experts.w13_weight_scale"].dtype == torch.float8_e4m3fn
    torch.testing.assert_close(
        tensors["model.layers.0.mlp.experts.w13_weight_scale_2"],
        torch.tensor([[1.0, 1.0], [0.5, 0.5]]),
    )
    torch.testing.assert_close(
        tensors["model.layers.0.mlp.experts.w13_input_scale"],
        torch.tensor([[1.0, 1.0], [0.5, 0.5]]),
    )
    assert captured["weight_scale_2"].shape == (2, 1, 1)


def test_quantize_w4a16_weight_exports_non_gated_fused_w13_family(
    monkeypatch,
):
    w4a16_qformat = "modelopt_w4a16_nvfp4"
    captured = {}

    def fake_to_quantized_weight(
        weight,
        weight_scale,
        qformat,
        weight_scale_2,
        block_size,
    ):
        captured["weight_scale_2"] = weight_scale_2
        return torch.zeros(weight.shape, dtype=torch.uint8, device=weight.device)

    monkeypatch.setattr(quant_utils, "to_quantized_weight", fake_to_quantized_weight)

    tensors = dict(
        quantize_nvfp4_weight(
            "model.layers.0.mlp.experts.up_proj",
            torch.ones(2, 4, 4, dtype=torch.float32),
            QuantMeta(
                qformat=w4a16_qformat,
                block_size=4,
                weight_amax=torch.tensor([[2688.0], [1344.0]]),
                weight_scale_2=torch.tensor([[1.0], [0.5]]),
            ),
        )
    )

    assert set(tensors) == {
        "model.layers.0.mlp.experts.w13_weight",
        "model.layers.0.mlp.experts.w13_weight_scale",
        "model.layers.0.mlp.experts.w13_weight_scale_2",
    }
    torch.testing.assert_close(
        tensors["model.layers.0.mlp.experts.w13_weight_scale_2"],
        torch.tensor([[1.0], [0.5]]),
    )
    assert captured["weight_scale_2"].shape == (2, 1, 1)


@pytest.mark.parametrize(
    "input_amax",
    [
        torch.tensor([2688.0, 1344.0]),
        torch.tensor([[2688.0], [1344.0]]),
    ],
    ids=("one_dimensional", "column"),
)
def test_quantize_nvfp4_weight_exports_non_gated_fused_w4a4_transport(monkeypatch, input_amax):
    captured = {}

    def fake_to_quantized_weight(
        weight,
        weight_scale,
        qformat,
        weight_scale_2,
        block_size,
    ):
        captured["weight_scale_2"] = weight_scale_2
        return torch.zeros(weight.shape, dtype=torch.uint8, device=weight.device)

    monkeypatch.setattr(quant_utils, "to_quantized_weight", fake_to_quantized_weight)

    tensors = dict(
        quantize_nvfp4_weight(
            "model.layers.0.mlp.experts.up_proj",
            torch.ones(2, 4, 4, dtype=torch.float32),
            QuantMeta(
                qformat=QUANTIZATION_NVFP4,
                block_size=4,
                weight_amax=torch.tensor([[2688.0], [1344.0]]),
                weight_scale_2=torch.tensor([[1.0], [0.5]]),
                input_amax=input_amax,
            ),
        )
    )

    assert set(tensors) == {
        "model.layers.0.mlp.experts.w13_weight",
        "model.layers.0.mlp.experts.w13_weight_scale",
        "model.layers.0.mlp.experts.w13_weight_scale_2",
        "model.layers.0.mlp.experts.w13_input_scale",
    }
    torch.testing.assert_close(
        tensors["model.layers.0.mlp.experts.w13_weight_scale_2"],
        torch.tensor([[1.0], [0.5]]),
    )
    torch.testing.assert_close(
        tensors["model.layers.0.mlp.experts.w13_input_scale"],
        torch.tensor([[1.0], [0.5]]),
    )
    assert captured["weight_scale_2"].shape == (2, 1, 1)


def test_quantize_nvfp4_weight_exports_fused_moe_w2_names_and_squeezes_scale_2(monkeypatch):
    captured = {}

    def fake_to_quantized_weight(
        weight,
        weight_scale,
        qformat,
        weight_scale_2,
        block_size,
    ):
        captured["weight_scale_2"] = weight_scale_2
        return torch.zeros(weight.shape, dtype=torch.uint8, device=weight.device)

    monkeypatch.setattr(quant_utils, "to_quantized_weight", fake_to_quantized_weight)

    tensors = dict(
        quantize_nvfp4_weight(
            "model.layers.0.mlp.experts.down_proj",
            torch.ones(2, 4, 4, dtype=torch.float32),
            QuantMeta(
                qformat=QUANTIZATION_NVFP4,
                block_size=4,
                weight_amax=torch.tensor([[2688.0], [1344.0]]),
                weight_scale_2=torch.tensor([[1.0], [0.5]]),
                input_amax=torch.tensor([[2688.0], [672.0]]),
            ),
        )
    )

    assert set(tensors) == {
        "model.layers.0.mlp.experts.w2_weight",
        "model.layers.0.mlp.experts.w2_weight_scale",
        "model.layers.0.mlp.experts.w2_weight_scale_2",
        "model.layers.0.mlp.experts.w2_input_scale",
    }
    assert tensors["model.layers.0.mlp.experts.w2_weight"].dtype == torch.uint8
    assert tensors["model.layers.0.mlp.experts.w2_weight_scale"].dtype == torch.float8_e4m3fn
    torch.testing.assert_close(
        tensors["model.layers.0.mlp.experts.w2_weight_scale_2"],
        torch.tensor([1.0, 0.5]),
    )
    torch.testing.assert_close(
        tensors["model.layers.0.mlp.experts.w2_input_scale"],
        torch.tensor([1.0, 0.25]),
    )
    assert captured["weight_scale_2"].shape == (2, 1, 1)


def test_quantize_nvfp4_weight_prefers_synchronized_scale_2(monkeypatch):
    captured = {}

    def fake_to_quantized_weight(
        weight,
        weight_scale,
        qformat,
        weight_scale_2,
        block_size,
    ):
        captured["weight_scale_2"] = weight_scale_2.detach().cpu()
        return torch.zeros(weight.shape, dtype=torch.uint8, device=weight.device)

    monkeypatch.setattr(quant_utils, "to_quantized_weight", fake_to_quantized_weight)

    list(
        quantize_nvfp4_weight(
            "model.layers.0.mlp.down_proj.weight",
            torch.tensor([[0.5, 1.0, 2.0, 4.0]], dtype=torch.float32),
            QuantMeta(
                qformat=QUANTIZATION_NVFP4,
                block_size=4,
                weight_amax=torch.tensor([2688.0]),
                weight_scale_2=torch.tensor([0.5]),
                input_amax=torch.tensor([2688.0]),
            ),
        )
    )

    assert captured["weight_scale_2"].dim() == 0
    torch.testing.assert_close(captured["weight_scale_2"], torch.tensor(0.5))


def test_compute_nvfp4_input_scale_uses_modelopt_canonical_export(monkeypatch):
    from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

    captured = {}

    def fake_activation_scale(_cls, quantizer):
        captured["is_enabled"] = quantizer.is_enabled
        captured["maxbound"] = quantizer.maxbound
        captured["amax"] = quantizer.export_amax().clone()
        return quantizer.export_amax() / 100.0

    monkeypatch.setattr(
        NVFP4QTensor,
        "get_activation_scaling_factor",
        classmethod(fake_activation_scale),
    )

    input_scale = compute_nvfp4_input_scale(torch.tensor([25.0]))

    torch.testing.assert_close(input_scale, torch.tensor([0.25]))
    assert captured["is_enabled"] is True
    assert captured["maxbound"] == 6.0
    torch.testing.assert_close(captured["amax"], torch.tensor([25.0]))


@pytest.mark.parametrize(
    "input_amax",
    [
        None,
        torch.tensor([]),
        torch.tensor([0.0]),
        torch.tensor([-1.0]),
        torch.tensor([float("nan")]),
        torch.tensor([float("inf")]),
    ],
)
def test_compute_nvfp4_input_scale_rejects_missing_or_invalid_amax(input_amax):
    with pytest.raises(RuntimeError, match="ModelOpt input amax"):
        compute_nvfp4_input_scale(input_amax)


def test_quantize_nvfp4_weight_does_not_emit_input_scale_for_w4a16(monkeypatch):
    w4a16_qformat = "modelopt_w4a16_nvfp4"
    monkeypatch.setattr(
        quant_utils,
        "to_quantized_weight",
        lambda weight, *_args, **_kwargs: torch.zeros_like(weight, dtype=torch.uint8),
    )

    tensors = dict(
        quantize_nvfp4_weight(
            "model.layers.0.mlp.up_proj.weight",
            torch.ones(1, 4, dtype=torch.float32),
            QuantMeta(
                qformat=w4a16_qformat,
                block_size=4,
                weight_amax=torch.tensor([2688.0]),
                weight_scale_2=torch.tensor([1.0]),
                input_amax=torch.tensor([float("nan")]),
            ),
        )
    )

    assert set(tensors) == {
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.up_proj.weight_scale",
        "model.layers.0.mlp.up_proj.weight_scale_2",
    }


def test_quantize_nvfp4_weight_requires_quantizable_name():
    with pytest.raises(ValueError, match="Expected quantizable NVFP4 export parameter name"):
        list(
            quantize_nvfp4_weight(
                "model.layers.0.mlp.up_proj",
                torch.ones(1, 4),
                QuantMeta(
                    qformat=QUANTIZATION_NVFP4,
                    block_size=4,
                    weight_amax=torch.tensor([1.0]),
                    weight_scale_2=torch.tensor([1.0]),
                ),
            )
        )


def test_is_modelopt_quantizable_weight_name_includes_fused_moe_base_names():
    assert is_modelopt_quantizable_weight_name("model.layers.0.mlp.down_proj.weight")
    assert is_modelopt_quantizable_weight_name("model.layers.0.mlp.experts.gate_up_proj")
    assert is_modelopt_quantizable_weight_name("model.layers.0.mlp.experts.down_proj")
    assert not is_modelopt_quantizable_weight_name("model.layers.0.mlp.experts.gate_up_proj.bias")


def test_get_modelopt_quant_exporter_is_case_insensitive_and_rejects_unknown_modes():
    qformat, export_weight = get_modelopt_quant_exporter("NVFP4")

    assert qformat == QUANTIZATION_NVFP4
    assert export_weight is quantize_nvfp4_weight

    with pytest.raises(ValueError, match="Unsupported ModelOpt quant_mode"):
        get_modelopt_quant_exporter("w4a8")


def test_get_modelopt_quant_exporter_returns_supported_w4a16_format(monkeypatch):
    w4a16_qformat = "modelopt_w4a16_nvfp4"
    monkeypatch.setattr(
        quant_utils,
        "QUANTIZATION_W4A16_NVFP4",
        w4a16_qformat,
        raising=False,
    )

    qformat, export_weight = get_modelopt_quant_exporter("W4A16_NVFP4")

    assert qformat == w4a16_qformat
    assert export_weight is quantize_nvfp4_weight


def test_get_modelopt_quant_exporter_rejects_unsupported_w4a16(monkeypatch):
    monkeypatch.delattr(quant_utils, "QUANTIZATION_W4A16_NVFP4", raising=False)

    with pytest.raises(RuntimeError, match="does not support W4A16 NVFP4 export"):
        get_modelopt_quant_exporter("w4a16_nvfp4")


def test_auto_bridge_modelopt_export_quantizes_matching_weights(monkeypatch):
    conversion_tasks = [
        _task(
            "decoder.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
        )
    ]
    bridge = _bridge_for_export(
        conversion_tasks,
        [
            ("model.layers.0.mlp.up_proj.weight", torch.tensor([1.0])),
            ("model.layers.0.mlp.up_proj.bias", torch.tensor([2.0])),
            ("model.layers.0.mlp.up_proj._quantizer._amax", torch.tensor([3.0])),
        ],
    )

    def fake_export_weight(name, tensor, meta):
        assert name == "model.layers.0.mlp.up_proj.weight"
        assert meta.qformat == QUANTIZATION_NVFP4
        yield name, tensor.to(torch.uint8)
        yield "model.layers.0.mlp.up_proj.weight_scale", torch.ones(1)

    monkeypatch.setattr(
        modelopt_utils,
        "collect_modelopt_quant_metadata",
        lambda _tasks: {"decoder.layers.0.mlp.up_proj.weight": _quant_meta()},
    )
    monkeypatch.setattr(
        modelopt_utils,
        "get_modelopt_quant_exporter",
        lambda quant_mode: (QUANTIZATION_NVFP4, fake_export_weight),
    )

    output = list(
        AutoBridge.export_hf_weights_modelopt(
            bridge,
            [object()],
            cpu=True,
            conversion_tasks=conversion_tasks,
        )
    )

    assert [(name, tensor.tolist()) for name, tensor in output] == [
        ("model.layers.0.mlp.up_proj.weight", [1]),
        ("model.layers.0.mlp.up_proj.weight_scale", [1.0]),
        ("model.layers.0.mlp.up_proj.bias", [2.0]),
    ]
    assert bridge.export_calls[0][1]["cpu"] is True
    export_tasks = bridge.export_calls[0][1]["conversion_tasks"]
    assert export_tasks is not conversion_tasks
    assert isinstance(export_tasks[0], WeightConversionTask)
    assert export_tasks[0].global_param_name == conversion_tasks[0].global_param_name
    assert callable(export_tasks[0].export_hook)


@pytest.mark.parametrize("shared_group", [False, True], ids=["distinct-pp-ep", "shared-pp-ep"])
def test_auto_bridge_modelopt_export_syncs_ep_and_deduplicates_groups(monkeypatch, shared_group):
    conversion_tasks = [
        _task(
            "decoder.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
        )
    ]
    bridge = _bridge_for_export(
        conversion_tasks,
        [],
    )
    metadata = {conversion_tasks[0].global_param_name: _quant_meta()}
    pp_group = object()
    ep_group = pp_group if shared_group else object()
    group_getter_calls = []
    world_size_calls = []
    sync_calls = []

    def get_pg_size(group=None):
        world_size_calls.append(group)
        return 2 if group is ep_group else 1

    def get_pp_group(_model):
        group_getter_calls.append("pp")
        return pp_group

    def get_ep_group(_model):
        group_getter_calls.append("ep")
        return ep_group

    monkeypatch.setattr(
        "megatron.bridge.models.conversion.modelopt_utils.torch.distributed.is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.modelopt_utils.get_pg_size",
        get_pg_size,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.modelopt_utils.torch.distributed.all_gather_object",
        lambda gathered, value, *, group: gathered.__setitem__(
            slice(None),
            [value, value],
        ),
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.modelopt_utils.model_bridge_utils._get_pp_group",
        get_pp_group,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.modelopt_utils.model_bridge_utils._get_ep_group",
        get_ep_group,
    )
    monkeypatch.setattr(modelopt_utils, "collect_modelopt_quant_metadata", lambda _tasks: metadata)
    monkeypatch.setattr(
        modelopt_utils,
        "build_hf_modelopt_quant_metadata",
        lambda _tasks, _metadata: {},
    )
    monkeypatch.setattr(
        modelopt_utils,
        "sync_modelopt_quant_metadata",
        lambda metadata, group: sync_calls.append((metadata, group)),
    )

    output = list(
        AutoBridge.export_hf_weights_modelopt(
            bridge,
            [object()],
            conversion_tasks=conversion_tasks,
        )
    )

    assert output == []
    assert group_getter_calls == ["pp", "ep"]
    expected_groups = [pp_group] if shared_group else [pp_group, ep_group]
    assert world_size_calls == expected_groups
    assert [group for _, group in sync_calls] == [ep_group]
    assert sync_calls[0][0] == metadata


def test_auto_bridge_modelopt_export_keeps_regular_path_for_shared_pp_ep_group(
    monkeypatch,
):
    shared_group = object()
    conversion_tasks = [
        WeightConversionTask(
            param_name=f"decoder.layers.0.mlp.experts.linear_fc1.weight{expert}",
            global_param_name=f"decoder.layers.0.mlp.experts.linear_fc1.weight{expert}",
            mapping=AutoMapping(
                f"decoder.layers.0.mlp.experts.linear_fc1.weight{expert}",
                f"model.layers.0.mlp.experts.{expert}.up_proj.weight",
            ),
            param_weight=torch.ones(1, 2),
        )
        for expert in range(2)
    ]
    metadata = {task.global_param_name: _quant_meta() for task in conversion_tasks}
    captured_export_tasks = []
    sync_calls = []

    class FakeBridge:
        hf_pretrained = object()
        _model_bridge = SimpleNamespace()

        def export_hf_weights(self, _model, **kwargs):
            captured_export_tasks.extend(kwargs["conversion_tasks"])
            return iter(())

    monkeypatch.setattr(
        "megatron.bridge.models.conversion.modelopt_utils.model_bridge_utils._get_pp_group",
        lambda _model: shared_group,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.modelopt_utils.model_bridge_utils._get_ep_group",
        lambda _model: shared_group,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.modelopt_utils.torch.distributed.is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.modelopt_utils.get_pg_size",
        lambda group=None: 2,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.modelopt_utils.torch.distributed.all_gather_object",
        lambda gathered, value, *, group: gathered.__setitem__(
            slice(None),
            [value, value],
        ),
    )
    monkeypatch.setattr(
        modelopt_utils,
        "collect_modelopt_quant_metadata",
        lambda _tasks: metadata.copy(),
    )
    monkeypatch.setattr(
        modelopt_utils,
        "sync_modelopt_quant_metadata",
        lambda task_metadata, group: sync_calls.append((dict(task_metadata), group)),
    )
    monkeypatch.setattr(
        modelopt_utils,
        "build_hf_modelopt_quant_metadata",
        lambda *_args: {},
    )
    monkeypatch.setattr(
        modelopt_utils,
        "get_modelopt_quant_exporter",
        lambda _mode: (QUANTIZATION_NVFP4, lambda *_args: iter(())),
    )

    assert (
        list(
            AutoBridge.export_hf_weights_modelopt(
                FakeBridge(),
                [object()],
                conversion_tasks=conversion_tasks,
            )
        )
        == []
    )
    assert len(captured_export_tasks) == 2
    assert not any(getattr(task.mapping, "is_modelopt_pre_ep_export", False) for task in captured_export_tasks)
    assert sync_calls == [(metadata, shared_group)]


def test_auto_bridge_modelopt_export_ep_gathers_qwen_input_scale_in_rank_order(
    monkeypatch,
):
    ep_group = object()
    fused_name = "model.layers.0.mlp.experts.gate_up_proj"
    packed_name = "model.layers.0.mlp.experts.w13_weight"
    input_scale_name = "model.layers.0.mlp.experts.w13_input_scale"
    expert_tasks = [
        WeightConversionTask(
            param_name=f"decoder.layers.0.mlp.experts.linear_fc1.weight{expert}",
            global_param_name=f"decoder.layers.0.mlp.experts.linear_fc1.weight{expert}",
            mapping=GatedMLPMapping(
                f"decoder.layers.0.mlp.experts.linear_fc1.weight{expert}",
                gate=f"model.layers.0.mlp.experts.{expert}.gate_proj.weight",
                up=f"model.layers.0.mlp.experts.{expert}.up_proj.weight",
            ),
            param_weight=torch.ones(1, 2),
        )
        for expert in range(2)
    ]
    regular_task = _task(
        "decoder.layers.0.mlp.router.weight",
        "model.layers.0.mlp.router.weight",
    )
    conversion_tasks = [*expert_tasks, regular_task]
    metadata = {task.global_param_name: _quant_meta() for task in conversion_tasks}
    gather_inputs = []
    metadata_build_calls = []
    metadata_sync_calls = []

    class FakeBridge:
        hf_pretrained = object()
        _model_bridge = SimpleNamespace()

        def export_hf_weights(self, _model, **kwargs):
            export_tasks = kwargs["conversion_tasks"]
            assert [getattr(task.mapping, "is_modelopt_pre_ep_export", False) for task in export_tasks] == [
                True,
                True,
                False,
            ]
            for task, value in zip(export_tasks[:2], (1.0, 2.0), strict=True):
                yield from task.export_hook(fused_name, torch.full((1, 2), value))

    def fake_export_weight(name, tensor, meta):
        assert name == fused_name
        assert meta.qformat == QUANTIZATION_NVFP4
        assert meta.block_size == metadata[conversion_tasks[0].global_param_name].block_size
        yield packed_name, tensor.to(torch.uint8)
        yield f"{packed_name}_scale", (tensor[..., :1] + 10.0).to(torch.float8_e4m3fn)
        yield f"{packed_name}_scale_2", tensor[:, 0, 0] + 20.0
        yield (
            input_scale_name,
            torch.stack(
                (tensor[:, 0, 0] + 30.0, tensor[:, 0, 0] + 31.0),
                dim=1,
            ),
        )

    def fake_all_gather_into_tensor(gathered, tensor, *, group):
        assert group is ep_group
        gather_inputs.append(tensor.clone())
        assert tensor.dtype == torch.uint8
        local_numel = tensor.numel()
        gathered[:local_numel].copy_(tensor)
        if len(gather_inputs) == 1:
            remote = tensor + 100
        elif len(gather_inputs) == 2:
            remote = (tensor.view(torch.float8_e4m3fn).float() + 100).to(torch.float8_e4m3fn).view(torch.uint8)
        else:
            remote = (tensor.view(torch.float32) + 100).view(torch.uint8)
        gathered[local_numel:].copy_(remote)

    def fake_all_gather_object(gathered, value, *, group):
        assert group is ep_group
        gathered[:] = [value, value]

    def fake_build_metadata(tasks, _task_metadata):
        metadata_build_calls.append(("regular", list(tasks)))
        return {}

    original_build_pre_ep_metadata = modelopt_utils._build_hf_modelopt_pre_ep_quant_metadata

    def tracked_build_pre_ep_metadata(tasks, task_metadata):
        metadata_build_calls.append(("pre_ep", list(tasks)))
        return original_build_pre_ep_metadata(tasks, task_metadata)

    monkeypatch.setattr(
        "megatron.bridge.models.conversion.modelopt_utils.model_bridge_utils._get_pp_group",
        lambda _model: None,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.modelopt_utils.model_bridge_utils._get_ep_group",
        lambda _model: ep_group,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.modelopt_utils.torch.distributed.is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.modelopt_utils.get_pg_size",
        lambda group=None: 2 if group is ep_group else 1,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.modelopt_utils.torch.distributed.get_backend",
        lambda group: "gloo",
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.modelopt_utils.torch.distributed.all_gather_into_tensor",
        fake_all_gather_into_tensor,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.modelopt_utils.torch.distributed.all_gather_object",
        fake_all_gather_object,
    )
    monkeypatch.setattr(
        modelopt_utils,
        "collect_modelopt_quant_metadata",
        lambda _tasks: metadata.copy(),
    )
    monkeypatch.setattr(
        modelopt_utils,
        "sync_modelopt_quant_metadata",
        lambda task_metadata, group: metadata_sync_calls.append((dict(task_metadata), group)),
    )
    monkeypatch.setattr(modelopt_utils, "build_hf_modelopt_quant_metadata", fake_build_metadata)
    monkeypatch.setattr(
        modelopt_utils,
        "_build_hf_modelopt_pre_ep_quant_metadata",
        tracked_build_pre_ep_metadata,
    )
    monkeypatch.setattr(
        modelopt_utils,
        "get_modelopt_quant_exporter",
        lambda _mode: (QUANTIZATION_NVFP4, fake_export_weight),
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.modelopt_utils.model_bridge_utils.unwrap_model",
        lambda _model: [SimpleNamespace(config=SimpleNamespace(num_moe_experts=4))],
    )

    output = dict(
        AutoBridge.export_hf_weights_modelopt(
            FakeBridge(),
            [object()],
            cpu=True,
            conversion_tasks=conversion_tasks,
        )
    )

    assert len(metadata_build_calls) == 2
    assert [kind for kind, _ in metadata_build_calls] == ["regular", "pre_ep"]
    regular_metadata_tasks = metadata_build_calls[0][1]
    pre_ep_metadata_tasks = metadata_build_calls[1][1]
    assert regular_metadata_tasks == [regular_task]
    assert len(pre_ep_metadata_tasks) == 2
    assert all(getattr(task.mapping, "is_modelopt_pre_ep_export", False) for task in pre_ep_metadata_tasks)
    regular_task_ids = {id(task) for task in regular_metadata_tasks}
    pre_ep_task_ids = {id(task) for task in pre_ep_metadata_tasks}
    assert regular_task_ids.isdisjoint(pre_ep_task_ids)
    assert metadata_sync_calls == [
        ({regular_task.global_param_name: metadata[regular_task.global_param_name]}, ep_group)
    ]
    assert len(gather_inputs) == 4
    assert [tensor.dtype for tensor in gather_inputs] == [torch.uint8] * 4
    assert [tensor.shape for tensor in gather_inputs] == [
        torch.Size([4]),
        torch.Size([2]),
        torch.Size([8]),
        torch.Size([16]),
    ]
    torch.testing.assert_close(
        output[packed_name][:, 0, 0],
        torch.tensor([1, 2, 101, 102], dtype=torch.uint8),
    )
    assert torch.equal(
        output[f"{packed_name}_scale"][:, 0, 0].view(torch.uint8),
        torch.tensor([11.0, 12.0, 111.0, 112.0]).to(torch.float8_e4m3fn).view(torch.uint8),
    )
    torch.testing.assert_close(
        output[f"{packed_name}_scale_2"],
        torch.tensor([21.0, 22.0, 121.0, 122.0]),
    )
    torch.testing.assert_close(
        output[input_scale_name],
        torch.tensor(
            [
                [31.0, 32.0],
                [32.0, 33.0],
                [131.0, 132.0],
                [132.0, 133.0],
            ]
        ),
    )


@pytest.mark.parametrize("num_moe_experts", [4, None, 3])
def test_auto_bridge_modelopt_export_rejects_incomplete_pre_ep_expert_family(
    monkeypatch,
    num_moe_experts,
):
    ep_group = object()
    conversion_tasks = [
        WeightConversionTask(
            param_name="decoder.layers.0.mlp.experts.linear_fc1.weight0",
            global_param_name="decoder.layers.0.mlp.experts.linear_fc1.weight0",
            mapping=AutoMapping(
                "decoder.layers.0.mlp.experts.linear_fc1.weight0",
                "model.layers.0.mlp.experts.0.up_proj.weight",
            ),
            param_weight=torch.ones(1, 2),
        )
    ]
    metadata = {conversion_tasks[0].global_param_name: _quant_meta()}
    captured_export_tasks = []
    metadata_sync_calls = []

    class FakeBridge:
        hf_pretrained = object()
        _model_bridge = SimpleNamespace()

        def export_hf_weights(self, _model, **kwargs):
            captured_export_tasks.extend(kwargs["conversion_tasks"])
            return iter(())

    def fake_all_gather_object(gathered, value, *, group):
        assert group is ep_group
        gathered[:] = [value, value]

    monkeypatch.setattr(
        "megatron.bridge.models.conversion.modelopt_utils.model_bridge_utils._get_pp_group",
        lambda _model: None,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.modelopt_utils.model_bridge_utils._get_ep_group",
        lambda _model: ep_group,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.modelopt_utils.model_bridge_utils.unwrap_model",
        lambda _model: [SimpleNamespace(config=SimpleNamespace(num_moe_experts=num_moe_experts))],
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.modelopt_utils.torch.distributed.is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.modelopt_utils.get_pg_size",
        lambda group=None: 2 if group is ep_group else 1,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.modelopt_utils.torch.distributed.all_gather_object",
        fake_all_gather_object,
    )
    monkeypatch.setattr(
        modelopt_utils,
        "collect_modelopt_quant_metadata",
        lambda _tasks: metadata.copy(),
    )
    monkeypatch.setattr(
        modelopt_utils,
        "sync_modelopt_quant_metadata",
        lambda task_metadata, group: metadata_sync_calls.append((dict(task_metadata), group)),
    )
    monkeypatch.setattr(
        modelopt_utils,
        "build_hf_modelopt_quant_metadata",
        lambda *_args: {},
    )
    monkeypatch.setattr(
        modelopt_utils,
        "get_modelopt_quant_exporter",
        lambda _mode: (QUANTIZATION_NVFP4, lambda *_args: iter(())),
    )

    assert (
        list(
            AutoBridge.export_hf_weights_modelopt(
                FakeBridge(),
                [object()],
                cpu=True,
                conversion_tasks=conversion_tasks,
            )
        )
        == []
    )
    assert len(captured_export_tasks) == 1
    assert not getattr(
        captured_export_tasks[0].mapping,
        "is_modelopt_pre_ep_export",
        False,
    )
    assert metadata_sync_calls == [(metadata, ep_group)]


@pytest.mark.parametrize(
    ("megatron_projection", "hf_projection"),
    [("linear_fc1", "up_proj"), ("linear_fc2", "down_proj")],
)
def test_modelopt_pre_ep_mapping_supports_nemotron_h_expert_projections(
    megatron_projection,
    hf_projection,
):
    megatron_name = f"decoder.layers.0.mlp.experts.{megatron_projection}.weight7"
    hf_name = f"backbone.layers.0.mixer.experts.7.{hf_projection}.weight"
    mapping = AutoMapping(megatron_name, hf_name)
    process_groups = {
        "pp_group": object(),
        "ep_group": object(),
        "_tp_group": object(),
        "_etp_group": object(),
    }
    pg_collection = SimpleNamespace(
        pp=process_groups["pp_group"],
        ep=process_groups["ep_group"],
        tp=process_groups["_tp_group"],
        expt_tp=process_groups["_etp_group"],
    )

    replacement, original_names = _modelopt_pre_ep_mapping(mapping, pg_collection)

    assert replacement.megatron_param == megatron_name
    assert replacement.hf_param == f"backbone.layers.0.mixer.experts.{hf_projection}"
    assert replacement.is_modelopt_pre_ep_export
    assert original_names == (hf_name,)
    for attr, group in process_groups.items():
        assert getattr(replacement, attr) is group
    delegate = replacement._get_or_create_mapping("column")
    assert not delegate.is_expert
    assert delegate.tp_group is pg_collection.expt_tp


@pytest.mark.parametrize(
    ("quant_mode", "qformat"),
    [
        ("nvfp4", QUANTIZATION_NVFP4),
        ("w4a16_nvfp4", "modelopt_w4a16_nvfp4"),
    ],
    ids=("w4a4", "w4a16"),
)
def test_auto_bridge_modelopt_export_selects_nemotron_h_pre_ep_family(
    monkeypatch,
    quant_mode,
    qformat,
):
    conversion_tasks = [
        WeightConversionTask(
            param_name=f"decoder.layers.0.mlp.experts.linear_fc1.weight{expert}",
            global_param_name=f"decoder.layers.0.mlp.experts.linear_fc1.weight{expert}",
            mapping=AutoMapping(
                f"decoder.layers.0.mlp.experts.linear_fc1.weight{expert}",
                f"backbone.layers.0.mixer.experts.{expert}.up_proj.weight",
            ),
            param_weight=torch.ones(1, 2),
        )
        for expert in range(2)
    ]
    metadata = {task.global_param_name: _quant_meta(qformat) for task in conversion_tasks}
    captured_export_tasks = []

    class FakeBridge:
        hf_pretrained = object()
        _model_bridge = SimpleNamespace()

        def export_hf_weights(self, _model, **kwargs):
            captured_export_tasks.extend(kwargs["conversion_tasks"])
            return iter(())

    monkeypatch.setattr(
        "megatron.bridge.models.conversion.modelopt_utils.torch.distributed.is_initialized",
        lambda: False,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.modelopt_utils.model_bridge_utils.unwrap_model",
        lambda _model: [SimpleNamespace(config=SimpleNamespace(num_moe_experts=2))],
    )
    monkeypatch.setattr(
        modelopt_utils,
        "collect_modelopt_quant_metadata",
        lambda _tasks: metadata.copy(),
    )

    def fake_get_modelopt_quant_exporter(requested_mode):
        assert requested_mode == quant_mode
        return qformat, lambda *_args: iter(())

    monkeypatch.setattr(
        modelopt_utils,
        "get_modelopt_quant_exporter",
        fake_get_modelopt_quant_exporter,
    )

    assert (
        list(
            AutoBridge.export_hf_weights_modelopt(
                FakeBridge(),
                [object()],
                quant_mode=quant_mode,
                cpu=True,
                conversion_tasks=conversion_tasks,
            )
        )
        == []
    )
    assert len(captured_export_tasks) == 2
    assert all(getattr(task.mapping, "is_modelopt_pre_ep_export", False) for task in captured_export_tasks)
    assert {task.mapping.hf_param for task in captured_export_tasks} == {"backbone.layers.0.mixer.experts.up_proj"}


def test_modelopt_pre_ep_mapping_fuses_qwen3_gate_and_up_experts():
    megatron_name = "decoder.layers.0.mlp.experts.linear_fc1.weight7"
    gate_name = "model.layers.0.mlp.experts.7.gate_proj.weight"
    up_name = "model.layers.0.mlp.experts.7.up_proj.weight"
    mapping = GatedMLPMapping(megatron_name, gate=gate_name, up=up_name)
    process_groups = {
        "pp_group": object(),
        "ep_group": object(),
        "_tp_group": object(),
        "_etp_group": object(),
    }
    pg_collection = SimpleNamespace(
        pp=process_groups["pp_group"],
        ep=process_groups["ep_group"],
        tp=process_groups["_tp_group"],
        expt_tp=process_groups["_etp_group"],
    )

    replacement, original_names = _modelopt_pre_ep_mapping(mapping, pg_collection)

    assert replacement.megatron_param == megatron_name
    assert replacement.hf_param == "model.layers.0.mlp.experts.gate_up_proj"
    assert replacement.is_modelopt_pre_ep_export
    assert original_names == (gate_name, up_name)
    for attr, group in process_groups.items():
        assert getattr(replacement, attr) is group
        assert getattr(replacement._gated_mapping, attr) is group
    assert not replacement._gated_mapping.is_expert
    assert replacement._gated_mapping.tp_group is pg_collection.expt_tp


def test_modelopt_pre_ep_mapping_rejects_mismatched_gate_up_experts():
    mapping = GatedMLPMapping(
        "decoder.layers.0.mlp.experts.linear_fc1.weight7",
        gate="model.layers.0.mlp.experts.7.gate_proj.weight",
        up="model.layers.0.mlp.experts.8.up_proj.weight",
    )

    assert _modelopt_pre_ep_mapping(mapping) is None


def test_grouped_expert_projection_name_is_structural():
    assert _grouped_expert_projection_name("model.layers.0.mlp.experts.7.value_proj.weight") == (
        "model.layers.0.mlp.experts.value_proj",
        7,
    )


def test_fuse_grouped_projection_names_is_structural():
    assert (
        _fuse_grouped_projection_names(
            "model.layers.0.mlp.experts.left_proj",
            "model.layers.0.mlp.experts.right_proj",
        )
        == "model.layers.0.mlp.experts.left_right_proj"
    )


def test_modelopt_pre_ep_mapping_requires_an_exact_experts_path_component():
    mapping = AutoMapping(
        "decoder.layers.0.mlp.experts.linear_fc2.weight7",
        "model.layers.0.mlp.shared_experts.7.value_proj.weight",
    )

    assert _modelopt_pre_ep_mapping(mapping) is None


def test_modelopt_pre_ep_mapping_rejects_unsupported_grouped_projection():
    mapping = AutoMapping(
        "decoder.layers.0.mlp.experts.linear_fc2.weight7",
        "model.layers.0.mlp.experts.7.value_proj.weight",
    )

    assert _modelopt_pre_ep_mapping(mapping) is None


def test_stage_tensor_for_collective_moves_pinned_cpu_tensor_for_nccl(monkeypatch):
    group = object()
    cuda_tensor = object()
    tensor = SimpleNamespace(device=torch.device("cpu"))
    tensor.is_pinned = lambda: True
    calls = []

    def move_to(*, device, non_blocking):
        calls.append((device, non_blocking))
        return cuda_tensor

    tensor.to = move_to
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.modelopt_utils.torch.distributed.get_backend",
        lambda _group: "nccl",
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.modelopt_utils.torch.cuda.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.modelopt_utils.torch.cuda.current_device",
        lambda: 3,
    )

    assert _stage_tensor_for_collective(tensor, group) is cuda_tensor
    assert calls == [(torch.device("cuda", 3), True)]


def test_stage_tensor_for_collective_keeps_cpu_tensor_for_gloo(monkeypatch):
    group = object()
    tensor = torch.ones(1)
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.modelopt_utils.torch.distributed.get_backend",
        lambda _group: "gloo",
    )

    assert _stage_tensor_for_collective(tensor, group) is tensor


def test_auto_bridge_modelopt_export_falls_back_to_mapping_registry(monkeypatch):
    megatron_name = "decoder.layers.0.mlp.up_proj.weight"
    hf_name = "model.layers.0.mlp.up_proj.weight"
    conversion_tasks = [_task(megatron_name, hf_name)]
    bridge = _bridge_for_export(
        conversion_tasks,
        [(hf_name, torch.tensor([1.0]))],
    )
    quant_meta = _quant_meta()
    registry_calls = []
    lookup_calls = []

    def hf_to_megatron_lookup(name):
        lookup_calls.append(name)
        return SimpleNamespace(megatron_param=megatron_name)

    registry = SimpleNamespace(
        hf_to_megatron_lookup=hf_to_megatron_lookup,
        set_process_groups_from_pg_collection=lambda _pg_collection: None,
    )

    def get_registry():
        registry_calls.append(True)
        return registry

    bridge._model_bridge.mapping_registry = get_registry

    def fake_export_weight(name, tensor, meta):
        assert name == hf_name
        assert meta is quant_meta
        yield name, tensor.to(torch.uint8)

    monkeypatch.setattr(
        modelopt_utils,
        "collect_modelopt_quant_metadata",
        lambda _tasks: {megatron_name: quant_meta},
    )
    monkeypatch.setattr(
        modelopt_utils,
        "build_hf_modelopt_quant_metadata",
        lambda _tasks, _metadata: {},
    )
    monkeypatch.setattr(
        modelopt_utils,
        "get_modelopt_quant_exporter",
        lambda quant_mode: (QUANTIZATION_NVFP4, fake_export_weight),
    )

    output = list(
        AutoBridge.export_hf_weights_modelopt(
            bridge,
            [object()],
            conversion_tasks=conversion_tasks,
        )
    )

    assert registry_calls == [True]
    assert lookup_calls == [hf_name]
    assert [(name, tensor.dtype, tensor.tolist()) for name, tensor in output] == [
        (hf_name, torch.uint8, [1]),
    ]


def test_auto_bridge_modelopt_export_leaves_ignored_weights_unquantized(monkeypatch):
    conversion_tasks = [
        _task(
            "decoder.layers.0.self_attention.linear_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
        )
    ]
    bridge = _bridge_for_export(
        conversion_tasks,
        [("model.layers.0.self_attn.o_proj.weight", torch.tensor([1.0]))],
    )

    def fail_export_weight(*_args, **_kwargs):
        raise AssertionError("ignored weights should not be quantized")

    monkeypatch.setattr(
        modelopt_utils,
        "collect_modelopt_quant_metadata",
        lambda _tasks: {"decoder.layers.0.self_attention.linear_proj.weight": _quant_meta()},
    )
    monkeypatch.setattr(
        modelopt_utils,
        "get_modelopt_quant_exporter",
        lambda quant_mode: (QUANTIZATION_NVFP4, fail_export_weight),
    )

    output = list(
        AutoBridge.export_hf_weights_modelopt(
            bridge,
            [object()],
            conversion_tasks=conversion_tasks,
            ignore_patterns=["*self_attn*"],
        )
    )

    assert [(name, tensor.tolist()) for name, tensor in output] == [("model.layers.0.self_attn.o_proj.weight", [1.0])]


def test_auto_bridge_modelopt_export_accepts_single_model_and_builds_tasks(monkeypatch):
    model = object()
    conversion_tasks = [
        _task(
            "decoder.embedding.word_embeddings.weight",
            "model.embed_tokens.weight",
        )
    ]
    build_calls = []

    class FakeModelBridge:
        def build_conversion_tasks(self, hf_pretrained, model_arg):
            build_calls.append((hf_pretrained, model_arg))
            return conversion_tasks

    bridge = _bridge_for_export(
        conversion_tasks,
        [("model.embed_tokens.weight", torch.tensor([4.0]))],
    )
    bridge._model_bridge = FakeModelBridge()

    monkeypatch.setattr(
        modelopt_utils,
        "collect_modelopt_quant_metadata",
        lambda _tasks: {},
    )

    output = list(
        AutoBridge.export_hf_weights_modelopt(
            bridge,
            model,
            show_progress=False,
            merge_adapter_weights=False,
        )
    )

    assert [(name, tensor.tolist()) for name, tensor in output] == [("model.embed_tokens.weight", [4.0])]
    assert build_calls == [(bridge.hf_pretrained, [model])]
    assert bridge.export_calls[0][0] == [model]
    assert bridge.export_calls[0][1]["show_progress"] is False
    assert bridge.export_calls[0][1]["merge_adapter_weights"] is False
    export_tasks = bridge.export_calls[0][1]["conversion_tasks"]
    assert export_tasks is not conversion_tasks
    assert export_tasks[0].global_param_name == conversion_tasks[0].global_param_name


def test_auto_bridge_modelopt_export_streams_base_weights_lazily(monkeypatch):
    conversion_tasks = [
        _task(
            "decoder.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
        )
    ]
    events = []

    class FakeBridge:
        hf_pretrained = object()
        _model_bridge = SimpleNamespace(
            build_conversion_tasks=lambda *_args, **_kwargs: conversion_tasks,
        )

        def export_hf_weights(self, _model, **kwargs):
            export_task = kwargs["conversion_tasks"][0]
            events.append("start")
            yield from export_task.export_hook(
                "model.layers.0.mlp.up_proj.weight",
                torch.tensor([1.0]),
            )
            events.append("after-first")
            yield from export_task.export_hook(
                "model.layers.0.mlp.down_proj.weight",
                torch.tensor([2.0]),
            )

    monkeypatch.setattr(
        modelopt_utils,
        "collect_modelopt_quant_metadata",
        lambda _tasks: {},
    )

    weights = AutoBridge.export_hf_weights_modelopt(
        FakeBridge(),
        [object()],
        conversion_tasks=conversion_tasks,
    )

    assert events == []
    first = next(weights)
    assert first.param_name == "model.layers.0.mlp.up_proj.weight"
    torch.testing.assert_close(first.weight, torch.tensor([1.0]))
    assert events == ["start"]
    second = next(weights)
    assert second.param_name == "model.layers.0.mlp.down_proj.weight"
    torch.testing.assert_close(second.weight, torch.tensor([2.0]))
    assert events == ["start", "after-first"]


def test_auto_bridge_modelopt_export_rejects_mismatched_qformat(monkeypatch):
    conversion_tasks = [
        _task(
            "decoder.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
        )
    ]
    bridge = _bridge_for_export(
        conversion_tasks,
        [("model.layers.0.mlp.up_proj.weight", torch.tensor([1.0]))],
    )

    def fail_export_weight(*_args, **_kwargs):
        raise AssertionError("mismatched qformat should fail before quantization")

    monkeypatch.setattr(
        modelopt_utils,
        "collect_modelopt_quant_metadata",
        lambda _tasks: {
            "decoder.layers.0.mlp.up_proj.weight": QuantMeta(
                qformat="unexpected_qformat",
                block_size=16,
                weight_amax=torch.tensor([1.0]),
                weight_scale_2=torch.tensor([1.0]),
            )
        },
    )
    monkeypatch.setattr(
        modelopt_utils,
        "get_modelopt_quant_exporter",
        lambda quant_mode: (QUANTIZATION_NVFP4, fail_export_weight),
    )

    with pytest.raises(RuntimeError, match="Unsupported qformat"):
        list(
            AutoBridge.export_hf_weights_modelopt(
                bridge,
                [object()],
                conversion_tasks=conversion_tasks,
            )
        )
