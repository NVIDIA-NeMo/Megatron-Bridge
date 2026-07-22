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

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import HFWeightTuple, MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import AutoMapping


class DummyBridge(MegatronModelBridge):
    def provider_bridge(self, hf_pretrained):  # pragma: no cover - not used in tests
        return None

    def mapping_registry(self):  # pragma: no cover - not used in tests
        return MegatronMappingRegistry()


def test_build_conversion_tasks_keeps_output_bias_when_embeddings_are_tied(monkeypatch):
    """Tied embeddings remove only output_layer.weight, not an independent output bias."""
    from megatron.bridge.models.conversion import model_bridge as model_bridge_module

    class TiedOutputBridge(DummyBridge):
        def mapping_registry(self):
            return MegatronMappingRegistry(
                AutoMapping("output_layer.weight", "lm_head.weight"),
                AutoMapping("output_layer.bias", "lm_head.bias"),
            )

    class State(dict):
        def __init__(self):
            super().__init__()
            self.source = SimpleNamespace(get_all_keys=lambda: {"lm_head.weight", "lm_head.bias"})

    model_config = SimpleNamespace(
        num_moe_experts=0,
        pipeline_model_parallel_size=1,
        share_embeddings_and_output_weights=True,
    )
    parameters = {
        "output_layer.weight": torch.ones(2, 2),
        "output_layer.bias": torch.ones(2),
    }
    model = SimpleNamespace(
        config=model_config,
        named_parameters=lambda: iter(parameters.items()),
    )
    hf_pretrained = SimpleNamespace(config=SimpleNamespace(), state=State())
    bridge = TiedOutputBridge()

    monkeypatch.setattr(
        bridge,
        "_megatron_global_param_names_all_pp_ranks",
        lambda _model: ["output_layer.weight", "output_layer.bias"],
    )
    monkeypatch.setattr(model_bridge_module, "unwrap_model", lambda _model: [model])
    monkeypatch.setattr(
        model_bridge_module,
        "_megatron_local_name_to_global",
        lambda _models, _config, name, _vp_stage: name,
    )
    monkeypatch.setattr(model_bridge_module, "persistent_buffers", lambda _model: [])
    monkeypatch.setattr(
        model_bridge_module,
        "get_module_and_param_from_name",
        lambda _model, name, _vp_stage: (SimpleNamespace(config=model_config), parameters[name]),
    )
    monkeypatch.setattr(model_bridge_module.parallel_state, "get_pipeline_model_parallel_rank", lambda: 0)

    tasks = bridge.build_conversion_tasks(hf_pretrained, [model])

    assert len(tasks) == 1
    assert tasks[0] is not None
    assert tasks[0].global_param_name == "output_layer.bias"


def test_hf_weight_tuple_iter_finalized_preserves_two_field_abi():
    tensor = torch.ones(2)
    weight = HFWeightTuple("hf.weight", tensor)

    name, unpacked_tensor = weight

    assert len(weight) == 2
    assert name == "hf.weight"
    assert unpacked_tensor is tensor
    finalized = list(weight.iter_finalized(cpu=False))
    assert finalized[0].param_name == "hf.weight"
    assert finalized[0].weight.data_ptr() == tensor.data_ptr()
    assert finalized[0].weight.requires_grad is False


def test_hf_weight_tuple_iter_finalized_allows_empty_export_hook():
    weight = HFWeightTuple("hf.weight", torch.ones(2))

    assert list(weight.iter_finalized(cpu=False, export_hook=lambda *_args: iter(()))) == []


def _with_export_hook(task, exporter, finalizer=None):
    def export_hook(name, tensor):
        for exported_name, exported_tensor in exporter(name, tensor):
            if finalizer is None:
                yield exported_name, exported_tensor
            else:
                yield from finalizer(exported_name, exported_tensor)

    return replace(task, export_hook=export_hook)


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

    task = _with_export_hook(task, export)
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

    task = _with_export_hook(task, transform)

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

    tasks = [_with_export_hook(task, transform) for task in tasks]

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

    task = _with_export_hook(task, export_weight, finalize_weight)
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


@pytest.mark.parametrize(
    ("megatron_prefix", "embedding_name", "output_name"),
    [
        ("", "model.embed_tokens.weight", "lm_head.weight"),
        ("thinker.language_model.", "thinker.model.embed_tokens.weight", "thinker.lm_head.weight"),
        ("language_model.", "model.language_model.embed_tokens.weight", "lm_head.weight"),
        ("language_model.", "language_model.model.embed_tokens.weight", "language_model.lm_head.weight"),
        (
            "llava_model.language_model.",
            "language_model.backbone.embeddings.weight",
            "language_model.lm_head.weight",
        ),
    ],
    ids=[
        "plain-llm",
        "component-prefix",
        "nested-embedding-root-head",
        "nested-language-model",
        "nonstandard-embedding-name",
    ],
)
def test_stream_weights_megatron_to_hf_transforms_tied_aliases_independently(
    monkeypatch,
    megatron_prefix,
    embedding_name,
    output_name,
):
    bridge = DummyBridge()
    source_tensor = torch.ones(2, 2, requires_grad=True)

    class EmbeddingMapping:
        hf_param = embedding_name

        def megatron_to_hf(self, weight, module):
            return {embedding_name: weight}

    task = WeightConversionTask(
        param_name=f"{megatron_prefix}embedding.word_embeddings.weight",
        global_param_name=f"{megatron_prefix}embedding.word_embeddings.weight",
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

    task = _with_export_hook(task, transform)

    _patch_stream_weights_megatron_to_hf_basics(monkeypatch)
    monkeypatch.setattr(
        DummyBridge,
        "_share_embeddings_and_output_weights",
        lambda self, *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        DummyBridge,
        "mapping_registry",
        lambda self: MegatronMappingRegistry(
            AutoMapping(f"{megatron_prefix}output_layer.weight", output_name),
        ),
    )
    hf_pretrained = SimpleNamespace(
        state=SimpleNamespace(
            source=SimpleNamespace(
                get_all_keys=lambda: [
                    embedding_name,
                    output_name,
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

    assert transform_calls == [embedding_name, output_name]
    assert [weight.param_name for weight in weights] == [
        f"{embedding_name}.packed",
        f"{embedding_name}.scale",
        f"{output_name}.packed",
        f"{output_name}.scale",
    ]
    assert weights[0].weight.data_ptr() != weights[2].weight.data_ptr()


@pytest.mark.parametrize("has_output_mapping", [False, True], ids=["no-output-mapping", "output-not-in-source"])
def test_stream_weights_megatron_to_hf_does_not_invent_tied_output_alias(monkeypatch, has_output_mapping):
    bridge = DummyBridge()
    embedding_name = "model.embed_tokens.weight"

    class EmbeddingMapping:
        hf_param = embedding_name

        def megatron_to_hf(self, weight, module):
            return {embedding_name: weight}

    task = WeightConversionTask(
        param_name="embedding.word_embeddings.weight",
        global_param_name="embedding.word_embeddings.weight",
        mapping=EmbeddingMapping(),
        pp_rank=0,
        vp_stage=0,
        megatron_module=None,
        param_weight=torch.ones(2, 2),
    )

    _patch_stream_weights_megatron_to_hf_basics(monkeypatch)
    monkeypatch.setattr(
        DummyBridge,
        "_share_embeddings_and_output_weights",
        lambda self, *_args, **_kwargs: True,
    )
    output_mappings = [AutoMapping("output_layer.weight", "lm_head.weight")] if has_output_mapping else []
    monkeypatch.setattr(
        DummyBridge,
        "mapping_registry",
        lambda self: MegatronMappingRegistry(*output_mappings),
    )
    hf_pretrained = SimpleNamespace(
        state=SimpleNamespace(
            source=SimpleNamespace(get_all_keys=lambda: [embedding_name]),
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

    assert [weight.param_name for weight in weights] == [embedding_name]
