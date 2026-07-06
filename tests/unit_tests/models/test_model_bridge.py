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
from unittest.mock import Mock

import torch

from megatron.bridge.models.conversion import model_bridge
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask


class DummyBridge(MegatronModelBridge):
    def provider_bridge(self, hf_pretrained):  # pragma: no cover - not used in tests
        return None

    def mapping_registry(self):  # pragma: no cover - not used in tests
        return MegatronMappingRegistry()


class _ExportTask:
    def __init__(self, task, exporter, finalizer=None):
        self._task = task
        self._exporter = exporter
        self._finalizer = finalizer

    def __getattr__(self, name):
        return getattr(self._task, name)

    def _export_hf_weight(self, name, tensor):
        return self._exporter(name, tensor)

    def _finalize_hf_weight(self, name, tensor):
        if self._finalizer is None:
            yield name, tensor
        else:
            yield from self._finalizer(name, tensor)


def _patch_stream_weights_megatron_to_hf_basics(
    monkeypatch,
    *,
    num_moe_experts: int = 0,
    expert_parallel_size: int = 1,
):
    monkeypatch.setattr(
        DummyBridge,
        "_with_progress_tracking",
        lambda self, tasks, *_args, **_kwargs: tasks,
    )
    monkeypatch.setattr(
        DummyBridge,
        "_share_embeddings_and_output_weights",
        lambda self, *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge.unwrap_model",
        lambda *_args, **_kwargs: [
            SimpleNamespace(
                config=SimpleNamespace(
                    num_moe_experts=num_moe_experts,
                    pipeline_model_parallel_size=1,
                )
            )
        ],
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge.parallel_state.get_expert_model_parallel_world_size",
        lambda: expert_parallel_size,
    )


def test_stream_weights_megatron_to_hf_custom_export_preserves_device_when_cpu_false(monkeypatch):
    bridge = DummyBridge()

    class TrackingTensor:
        def detach(self):
            return self

    source = TrackingTensor()

    class DummyMapping:
        def megatron_to_hf(self, weight, module):
            return {"hf.weight": weight}

    task = WeightConversionTask(
        param_name="decoder.weight",
        global_param_name="decoder.weight",
        mapping=DummyMapping(),
        pp_rank=0,
        vp_stage=0,
        megatron_module=None,
        param_weight=source,
    )

    def export(name, tensor):
        assert tensor is source
        yield name, tensor

    task = _ExportTask(task, export)
    _patch_stream_weights_megatron_to_hf_basics(monkeypatch)
    monkeypatch.setattr(
        DummyBridge,
        "maybe_modify_converted_hf_weight",
        lambda self, *_args, **_kwargs: _args[1],
    )

    weights = list(
        bridge.stream_weights_megatron_to_hf(
            [Mock()],
            SimpleNamespace(),
            cpu=False,
            show_progress=False,
            conversion_tasks=[task],
            merge_adapter_weights=False,
        )
    )

    assert weights == [("hf.weight", source)]


def test_stream_weights_megatron_to_hf_transforms_before_final_cpu_placement(monkeypatch):
    bridge = DummyBridge()
    events = []

    class TrackingTensor:
        def __init__(self, label, *, detached=False, on_cpu=False):
            self.label = label
            self.detached = detached
            self.on_cpu = on_cpu

        def detach(self):
            events.append(("detach", self.label))
            return TrackingTensor(
                self.label,
                detached=True,
                on_cpu=self.on_cpu,
            )

        def cpu(self):
            events.append(("cpu", self.label))
            return TrackingTensor(
                self.label,
                detached=self.detached,
                on_cpu=True,
            )

    source = TrackingTensor("source")

    class DummyMapping:
        def megatron_to_hf(self, weight, module):
            return {"hf.weight": weight}

    task = WeightConversionTask(
        param_name="decoder.layers.0.mlp.linear_fc1.weight",
        global_param_name="decoder.layers.0.mlp.linear_fc1.weight",
        mapping=DummyMapping(),
        pp_rank=0,
        vp_stage=0,
        megatron_module=None,
        param_weight=source,
    )

    def transform(name, tensor):
        events.append(("transform", name))
        assert tensor.detached and not tensor.on_cpu
        yield f"{name}.packed", TrackingTensor("packed")
        yield f"{name}.scale", TrackingTensor("scale")

    task = _ExportTask(task, transform)

    _patch_stream_weights_megatron_to_hf_basics(monkeypatch)
    monkeypatch.setattr(
        DummyBridge,
        "maybe_modify_converted_hf_weight",
        lambda self, *_args, **_kwargs: _args[1],
    )

    weights = list(
        bridge.stream_weights_megatron_to_hf(
            [Mock()],
            SimpleNamespace(),
            cpu=True,
            show_progress=False,
            conversion_tasks=[task],
            merge_adapter_weights=False,
        )
    )

    assert [weight.param_name for weight in weights] == [
        "hf.weight.packed",
        "hf.weight.scale",
    ]
    assert all(weight.weight.detached and weight.weight.on_cpu for weight in weights)
    transform_index = events.index(("transform", "hf.weight"))
    output_cpu_indices = [
        index for index, event in enumerate(events) if event in (("cpu", "packed"), ("cpu", "scale"))
    ]
    assert ("cpu", "source") not in events
    assert transform_index < min(output_cpu_indices)


def test_stream_weights_megatron_to_hf_transforms_grouped_tensor_once_after_accumulation(monkeypatch):
    bridge = DummyBridge()

    class GroupedMapping:
        is_grouped_export = True
        group_key = "hf.grouped"

        def megatron_to_hf(self, weight, module):
            return {self.group_key: weight}

    tasks = [
        WeightConversionTask(
            param_name=f"decoder.layers.0.mlp.experts.linear_fc2.weight{expert}",
            global_param_name=f"decoder.layers.0.mlp.experts.linear_fc2.weight{expert}",
            mapping=GroupedMapping(),
            pp_rank=0,
            vp_stage=0,
            megatron_module=None,
            param_weight=torch.full((1, 1), float(expert + 1)),
        )
        for expert in range(2)
    ]
    transform_calls = []

    def transform(name, tensor):
        transform_calls.append((name, tensor.clone()))
        yield f"{name}.packed", tensor.to(torch.uint8)
        yield f"{name}.scale", torch.ones(2, 1)
        yield f"{name}.scale_2", torch.ones(2)

    tasks = [_ExportTask(task, transform) for task in tasks]

    _patch_stream_weights_megatron_to_hf_basics(monkeypatch, num_moe_experts=2)

    weights = list(
        bridge.stream_weights_megatron_to_hf(
            [Mock()],
            SimpleNamespace(),
            cpu=True,
            show_progress=False,
            conversion_tasks=tasks,
            merge_adapter_weights=False,
        )
    )

    assert len(transform_calls) == 1
    assert transform_calls[0][0] == "hf.grouped"
    torch.testing.assert_close(
        transform_calls[0][1],
        torch.tensor([[[1.0]], [[2.0]]]),
    )
    assert [weight.param_name for weight in weights] == [
        "hf.grouped.packed",
        "hf.grouped.scale",
        "hf.grouped.scale_2",
    ]


def test_accumulate_grouped_export_uses_mapping_ep_group(monkeypatch):
    bridge = DummyBridge()
    ep_group = object()
    mapping = SimpleNamespace(ep_group=ep_group, is_grouped_export=True)
    model_config = SimpleNamespace(num_moe_experts=4)
    grouped_buffers = {}

    monkeypatch.setattr(model_bridge.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        model_bridge.torch.distributed,
        "get_world_size",
        lambda *, group: 2 if group is ep_group else 1,
    )
    monkeypatch.setattr(
        model_bridge.parallel_state,
        "get_expert_model_parallel_world_size",
        lambda: 1,
    )

    first = bridge._accumulate_grouped_export(
        SimpleNamespace(
            param_name="decoder.layers.0.mlp.experts.linear_fc2.weight0",
            mapping=mapping,
        ),
        {"hf.grouped": torch.tensor([[[1.0]], [[3.0]]])},
        model_config,
        grouped_buffers,
        {},
    )
    second = bridge._accumulate_grouped_export(
        SimpleNamespace(
            param_name="decoder.layers.0.mlp.experts.linear_fc2.weight1",
            mapping=mapping,
        ),
        {"hf.grouped": torch.tensor([[[2.0]], [[4.0]]])},
        model_config,
        grouped_buffers,
        {},
    )

    assert first is None
    assert second is not None
    torch.testing.assert_close(
        second["hf.grouped"],
        torch.tensor([[[1.0]], [[2.0]], [[3.0]], [[4.0]]]),
    )


def test_stream_weights_megatron_to_hf_batches_local_experts_before_modelopt_export(monkeypatch):
    bridge = DummyBridge()
    ep_group = object()

    class PreEPGroupedMapping:
        is_grouped_export = True
        is_modelopt_pre_ep_export = True
        group_key = "hf.grouped"

        def __init__(self):
            self.ep_group = ep_group

        def megatron_to_hf(self, weight, module):
            return {self.group_key: weight}

    tasks = [
        WeightConversionTask(
            param_name=f"decoder.layers.0.mlp.experts.linear_fc2.weight{expert}",
            global_param_name=f"decoder.layers.0.mlp.experts.linear_fc2.weight{expert}",
            mapping=PreEPGroupedMapping(),
            pp_rank=0,
            vp_stage=0,
            megatron_module=None,
            param_weight=torch.full((1, 2), float(expert + 1)),
        )
        for expert in (2, 3)
    ]
    export_calls = []

    def export_local_batch(name, tensor):
        export_calls.append((name, tensor.clone()))
        yield f"{name}.packed", tensor.to(torch.uint8)

    tasks = [_ExportTask(task, export_local_batch) for task in tasks]
    _patch_stream_weights_megatron_to_hf_basics(
        monkeypatch,
        num_moe_experts=4,
        expert_parallel_size=2,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge.torch.distributed.is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge.torch.distributed.get_world_size",
        lambda *, group: 2 if group is ep_group else 1,
    )

    weights = list(
        bridge.stream_weights_megatron_to_hf(
            [Mock()],
            SimpleNamespace(),
            cpu=True,
            show_progress=False,
            conversion_tasks=tasks,
            merge_adapter_weights=False,
        )
    )

    assert len(export_calls) == 1
    assert export_calls[0][0] == "hf.grouped"
    torch.testing.assert_close(
        export_calls[0][1],
        torch.tensor([[[3.0, 3.0]], [[4.0, 4.0]]]),
    )
    assert export_calls[0][1].shape == (2, 1, 2)
    assert [weight.param_name for weight in weights] == ["hf.grouped.packed"]


def test_stream_weights_megatron_to_hf_finalizes_exported_tensors_before_cpu(monkeypatch):
    bridge = DummyBridge()
    events = []

    class TrackingTensor:
        def __init__(self, label):
            self.label = label

        def detach(self):
            events.append(("detach", self.label))
            return self

        def cpu(self):
            events.append(("cpu", self.label))
            return self

    class DummyMapping:
        def megatron_to_hf(self, weight, module):
            return {"hf.weight": weight}

    task = WeightConversionTask(
        param_name="decoder.layers.0.mlp.linear_fc1.weight",
        global_param_name="decoder.layers.0.mlp.linear_fc1.weight",
        mapping=DummyMapping(),
        pp_rank=0,
        vp_stage=0,
        megatron_module=None,
        param_weight=TrackingTensor("source"),
    )

    def export_weight(name, tensor):
        yield f"{name}.packed", TrackingTensor("packed")
        yield f"{name}.scale", TrackingTensor("scale")
        yield f"{name}.scale_2", TrackingTensor("scale_2")

    def finalize_weight(name, tensor):
        events.append(("finalize", tensor.label))
        yield name, tensor

    task = _ExportTask(task, export_weight, finalize_weight)
    _patch_stream_weights_megatron_to_hf_basics(monkeypatch)
    monkeypatch.setattr(
        DummyBridge,
        "maybe_modify_converted_hf_weight",
        lambda self, *_args, **_kwargs: _args[1],
    )

    weights = list(
        bridge.stream_weights_megatron_to_hf(
            [Mock()],
            SimpleNamespace(),
            cpu=True,
            show_progress=False,
            conversion_tasks=[task],
            merge_adapter_weights=False,
        )
    )

    assert [weight.param_name for weight in weights] == [
        "hf.weight.packed",
        "hf.weight.scale",
        "hf.weight.scale_2",
    ]
    for label in ("packed", "scale", "scale_2"):
        assert events.index(("finalize", label)) < events.index(("cpu", label))


def test_stream_weights_megatron_to_hf_transforms_tied_aliases_independently(monkeypatch):
    bridge = DummyBridge()
    source_tensor = torch.ones(2, 2, requires_grad=True)

    class EmbeddingMapping:
        def megatron_to_hf(self, weight, module):
            return {"model.embed_tokens.weight": weight}

    task = WeightConversionTask(
        param_name="embedding.word_embeddings.weight",
        global_param_name="embedding.word_embeddings.weight",
        mapping=EmbeddingMapping(),
        pp_rank=0,
        vp_stage=0,
        megatron_module=None,
        param_weight=source_tensor,
    )
    transform_calls = []

    def transform(name, tensor):
        transform_calls.append(name)
        assert tensor.requires_grad is False
        yield f"{name}.packed", tensor
        yield f"{name}.scale", torch.ones(1)

    task = _ExportTask(task, transform)

    _patch_stream_weights_megatron_to_hf_basics(monkeypatch)
    monkeypatch.setattr(
        DummyBridge,
        "_share_embeddings_and_output_weights",
        lambda self, *_args, **_kwargs: True,
    )
    hf_pretrained = SimpleNamespace(
        state=SimpleNamespace(
            source=SimpleNamespace(
                get_all_keys=lambda: [
                    "model.embed_tokens.weight",
                    "lm_head.weight",
                ]
            )
        )
    )

    weights = list(
        bridge.stream_weights_megatron_to_hf(
            [Mock()],
            hf_pretrained,
            cpu=True,
            show_progress=False,
            conversion_tasks=[task],
            merge_adapter_weights=False,
        )
    )

    assert transform_calls == ["model.embed_tokens.weight", "lm_head.weight"]
    assert [weight.param_name for weight in weights] == [
        "model.embed_tokens.weight.packed",
        "model.embed_tokens.weight.scale",
        "lm_head.weight.packed",
        "lm_head.weight.scale",
    ]
    assert weights[0].weight.data_ptr() != weights[2].weight.data_ptr()
