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
from transformers import PretrainedConfig

from megatron.bridge.models.conversion import model_bridge as model_bridge_module
from megatron.bridge.models.conversion import modelopt_utils
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import HFWeightTuple, MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import AutoMapping


class DummyBridge(MegatronModelBridge):
    def provider_bridge(self, hf_pretrained):  # pragma: no cover - not used in tests
        return None

    def mapping_registry(self):  # pragma: no cover - not used in tests
        return MegatronMappingRegistry()


def test_modelopt_plan_keeps_tasks_after_a_sparse_slot(monkeypatch):
    """A hole in the task list must not truncate the plan.

    `build_conversion_tasks` no longer produces holes; an unmapped parameter raises. The
    plan builder still declares `WeightConversionTask | None`, so the guard stays, and the
    sparse list is built here rather than obtained from the builder.
    """
    first_name = "first.weight"
    last_name = "last.weight"

    class MappedBridge(DummyBridge):
        def mapping_registry(self):
            return MegatronMappingRegistry(
                AutoMapping(first_name, "hf.first.weight"),
                AutoMapping(last_name, "hf.last.weight"),
            )

    model = torch.nn.Module()
    model.config = SimpleNamespace(share_embeddings_and_output_weights=False)
    model.first = torch.nn.Linear(1, 1, bias=False)
    model.last = torch.nn.Linear(1, 1, bias=False)
    bridge = MappedBridge()
    global_names = [first_name, last_name]

    monkeypatch.setattr(bridge, "_megatron_global_param_names_all_pp_ranks", lambda _model: global_names)
    monkeypatch.setattr(bridge, "_share_embeddings_and_output_weights", lambda _config: False)
    monkeypatch.setattr(model_bridge_module, "unwrap_model", lambda _model: [model])
    monkeypatch.setattr(model_bridge_module, "_get_pg_collection_from_model", lambda _model: None)
    monkeypatch.setattr(model_bridge_module, "_get_pp_rank", lambda _model: 0)
    monkeypatch.setattr(
        model_bridge_module,
        "_megatron_local_name_to_global",
        lambda _models, _config, local_name, _vp_stage: local_name,
    )

    tasks = bridge.build_conversion_tasks(PretrainedConfig(), [model])

    assert [task.global_param_name for task in tasks] == [first_name, last_name]
    sparse_tasks = [tasks[0], None, tasks[1]]

    monkeypatch.setattr(modelopt_utils, "get_modelopt_quant_exporter", lambda _mode: ("unused", lambda *_args: ()))
    monkeypatch.setattr(modelopt_utils, "get_pg_size", lambda _group: 1)
    monkeypatch.setattr(modelopt_utils.model_bridge_utils, "_get_pg_collection_from_model", lambda _model: None)

    export_tasks = modelopt_utils.build_modelopt_export_plan(
        sparse_tasks,
        model=[model],
        bridge=bridge,
        quant_mode="nvfp4",
        ignore_patterns=[],
    )

    assert [task.global_param_name for task in export_tasks] == [first_name, last_name]


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


def test_truncate_vocab_padding_handles_nested_config_and_vocab_aliases():
    bridge = DummyBridge()
    bridge.hf_config = SimpleNamespace(thinker_config=SimpleNamespace(text_config=SimpleNamespace(vocab_size=3)))
    task = SimpleNamespace(
        global_param_name="language_model.output_layer.weight",
        mapping=SimpleNamespace(hf_param="lm_head.weight"),
    )
    padded_weight = torch.arange(10).reshape(5, 2)
    weights = {
        "lm_head.weight": padded_weight,
        "model.layers.1.shared_head.head.weight": padded_weight.clone(),
        "unrelated.weight": torch.ones(4, 2),
    }

    result = bridge._truncate_vocab_padding(task, weights)

    assert result["lm_head.weight"].shape == (3, 2)
    assert result["model.layers.1.shared_head.head.weight"].shape == (3, 2)
    assert result["unrelated.weight"].shape == (4, 2)


@pytest.mark.parametrize("is_remote_pp", [False, True])
def test_truncate_vocab_padding_handles_fp8_scale_task(is_remote_pp):
    bridge = DummyBridge()
    bridge.hf_config = SimpleNamespace(vocab_size=5)
    task = SimpleNamespace(
        global_param_name="output_layer.weight_scale_inv",
        mapping=SimpleNamespace(hf_param="lm_head.weight", scale_block_size=1 if is_remote_pp else None),
        megatron_module=None if is_remote_pp else SimpleNamespace(weight=torch.ones(8, 4)),
        param_weight=None if is_remote_pp else torch.ones(8, 2),
    )

    result = bridge._truncate_vocab_padding(task, {"lm_head.weight_scale_inv": torch.ones(8, 2)})

    assert result["lm_head.weight_scale_inv"].shape == (5, 2)


@pytest.mark.parametrize("exported_vocab_size", [3, 5])
def test_quantized_stream_truncates_only_padded_vocab(monkeypatch, exported_vocab_size):
    bridge = DummyBridge()
    checker_results = []

    class VocabMapping:
        hf_param = "lm_head.weight"

        def megatron_to_hf_quant(
            self,
            weight,
            module,
            quantization_checker,
            quant_fn,
            quant_block_size,
        ):
            checker_results.append(quantization_checker("output_layer.weight"))
            return {
                self.hf_param: weight,
                f"{self.hf_param}_scale_inv": torch.ones(weight.shape[0], 1),
            }

    exported_weight = torch.arange(exported_vocab_size * 2).reshape(exported_vocab_size, 2)
    task = WeightConversionTask(
        param_name="output_layer.weight",
        global_param_name="output_layer.weight",
        mapping=VocabMapping(),
        param_weight=exported_weight,
    )
    model = SimpleNamespace(config=SimpleNamespace(share_embeddings_and_output_weights=False))
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.quant_bridge.unwrap_model",
        lambda _model: [model],
    )
    monkeypatch.setattr(bridge, "_with_progress_tracking", lambda tasks, *_args: tasks)

    exported = list(
        bridge.stream_weights_megatron_to_hf_quant(
            model,
            SimpleNamespace(config=SimpleNamespace(vocab_size=3)),
            quantization_checker=lambda _name: True,
            quant_fn=Mock(),
            quant_block_size=(1, 2),
            conversion_tasks=[task],
            show_progress=False,
        )
    )

    assert checker_results == [True]
    assert exported[0].param_name == "lm_head.weight"
    assert exported[0].weight.shape == (3, 2)
    assert exported[1].param_name == "lm_head.weight_scale_inv"
    assert exported[1].weight.shape == (3, 1)


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
        ep_size = 1

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


def test_grouped_export_uses_mapping_local_ep_size(monkeypatch):
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge.parallel_state.get_expert_model_parallel_world_size",
        lambda: 1,
    )
    mapping = SimpleNamespace(is_grouped_export=True, ep_size=2)
    model_config = SimpleNamespace(num_moe_experts=4)
    buffers = {}

    first = MegatronModelBridge._accumulate_grouped_export(
        None,
        SimpleNamespace(
            mapping=mapping,
            param_name="decoder.layers.0.mlp.experts.linear_fc2.weight0",
        ),
        {"hf.grouped": torch.tensor([[0.0], [2.0]])},
        model_config,
        buffers,
        {},
    )
    second = MegatronModelBridge._accumulate_grouped_export(
        None,
        SimpleNamespace(
            mapping=mapping,
            param_name="decoder.layers.0.mlp.experts.linear_fc2.weight1",
        ),
        {"hf.grouped": torch.tensor([[1.0], [3.0]])},
        model_config,
        buffers,
        {},
    )

    assert first is None
    torch.testing.assert_close(second["hf.grouped"], torch.tensor([[0.0], [1.0], [2.0], [3.0]]))
    assert buffers == {}


def test_grouped_export_retries_stack_on_cpu_after_cuda_oom(monkeypatch, caplog):
    class FakeCudaTensor:
        is_cuda = True

        def __init__(self, value):
            self.value = value

        def cpu(self):
            return torch.tensor([self.value])

    original_stack = torch.stack
    stack_devices = []

    def stack_with_cuda_oom(tensors, dim=0):
        stack_devices.append([type(tensor).__name__ for tensor in tensors])
        if isinstance(tensors[0], FakeCudaTensor):
            raise torch.OutOfMemoryError("simulated grouped-export CUDA OOM")
        return original_stack(tensors, dim=dim)

    monkeypatch.setattr(torch, "stack", stack_with_cuda_oom)
    mapping = SimpleNamespace(is_grouped_export=True, ep_size=1)
    model_config = SimpleNamespace(num_moe_experts=2)
    buffers = {}

    first = MegatronModelBridge._accumulate_grouped_export(
        None,
        SimpleNamespace(
            mapping=mapping,
            param_name="decoder.layers.0.mlp.experts.linear_fc2.weight0",
        ),
        {"hf.grouped": FakeCudaTensor(1.0)},
        model_config,
        buffers,
        {},
    )
    with caplog.at_level("WARNING"):
        second = MegatronModelBridge._accumulate_grouped_export(
            None,
            SimpleNamespace(
                mapping=mapping,
                param_name="decoder.layers.0.mlp.experts.linear_fc2.weight1",
            ),
            {"hf.grouped": FakeCudaTensor(2.0)},
            model_config,
            buffers,
            {},
        )

    assert first is None
    torch.testing.assert_close(second["hf.grouped"], torch.tensor([[1.0], [2.0]]))
    assert stack_devices == [["FakeCudaTensor", "FakeCudaTensor"], ["Tensor", "Tensor"]]
    assert "retrying on CPU" in caplog.text
    assert buffers == {}


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


def _patch(monkeypatch, name, value):
    """Replace a module-level name in the bridge module under test."""
    monkeypatch.setattr(f"megatron.bridge.models.conversion.model_bridge.{name}", value)


def _patch_conversion_task_context(monkeypatch, bridge, model, global_names):
    """Patch distributed task discovery while retaining registry validation."""
    _patch(monkeypatch, "_get_pg_collection_from_model", lambda *_a, **_kw: Mock())
    _patch(monkeypatch, "_get_pp_rank", lambda *_a, **_kw: 0)
    _patch(monkeypatch, "unwrap_model", lambda *_a, **_kw: [model])
    _patch(monkeypatch, "persistent_buffers", lambda *_a, **_kw: [])
    _patch(
        monkeypatch,
        "_megatron_local_name_to_global",
        lambda _models, _config, local_name, _vp_stage: local_name,
    )
    monkeypatch.setattr(
        bridge,
        "_megatron_global_param_names_all_pp_ranks",
        lambda *_a, **_kw: global_names,
    )
    monkeypatch.setattr(
        bridge,
        "_share_embeddings_and_output_weights",
        lambda *_a, **_kw: False,
    )


@pytest.mark.parametrize("owns_parameter", [True, False])
def test_build_conversion_tasks_rejects_an_unmapped_parameter(monkeypatch, owns_parameter):
    """A Megatron parameter with no registry entry must stop the conversion by name.

    Skipping it would leave the parameter at its initial value on import and drop it on
    export, which is a wrong model rather than a missing file. The message has to name the
    parameter, because that name is the only way to find which mapping is missing.
    """
    bridge = DummyBridge()
    monkeypatch.setattr(
        DummyBridge,
        "mapping_registry",
        lambda self: MegatronMappingRegistry(AutoMapping("decoder.weight", "hf.weight")),
    )

    model = Mock()
    model.named_parameters = lambda: [("orphan.weight", torch.ones(2))] if owns_parameter else []
    model.config = SimpleNamespace(share_embeddings_and_output_weights=False)

    _patch_conversion_task_context(monkeypatch, bridge, model, ["orphan.weight"])

    hf_pretrained = SimpleNamespace(state=SimpleNamespace(source=SimpleNamespace(get_all_keys=lambda: ["hf.weight"])))

    with pytest.raises(ValueError, match="orphan.weight"):
        bridge.build_conversion_tasks(hf_pretrained, [model])


def test_build_conversion_tasks_rejects_a_missing_hf_weight(monkeypatch):
    """A mapped source key must not silently become a remote-PP task."""
    bridge = DummyBridge()
    monkeypatch.setattr(
        bridge,
        "mapping_registry",
        lambda: MegatronMappingRegistry(AutoMapping("decoder.weight", "hf.weight")),
    )
    model = Mock()
    model.named_parameters = lambda: []
    model.config = SimpleNamespace(share_embeddings_and_output_weights=False)
    _patch_conversion_task_context(monkeypatch, bridge, model, ["decoder.weight"])
    hf_pretrained = SimpleNamespace(state=SimpleNamespace(source=SimpleNamespace(get_all_keys=lambda: [])))

    with pytest.raises(ValueError, match=r"decoder\.weight -> hf\.weight"):
        bridge.build_conversion_tasks(hf_pretrained, [model])


def test_build_conversion_tasks_allows_explicit_hf_name_mismatch(monkeypatch):
    """Alternate/synthesized HF names require the mapping's explicit opt-in."""

    class AlternateNameMapping(AutoMapping):
        def __init__(self, megatron_param, hf_param):
            super().__init__(megatron_param, hf_param)
            self.allow_hf_name_mismatch = True

    bridge = DummyBridge()
    monkeypatch.setattr(
        bridge,
        "mapping_registry",
        lambda: MegatronMappingRegistry(AlternateNameMapping("decoder.weight", "synthesized.weight")),
    )
    model = Mock()
    model.named_parameters = lambda: []
    model.config = SimpleNamespace(share_embeddings_and_output_weights=False)
    _patch_conversion_task_context(monkeypatch, bridge, model, ["decoder.weight"])
    hf_pretrained = SimpleNamespace(state=SimpleNamespace(source=SimpleNamespace(get_all_keys=lambda: [])))

    tasks = bridge.build_conversion_tasks(hf_pretrained, [model])

    assert len(tasks) == 1
    assert tasks[0].global_param_name == "decoder.weight"
    assert tasks[0].megatron_module is None
