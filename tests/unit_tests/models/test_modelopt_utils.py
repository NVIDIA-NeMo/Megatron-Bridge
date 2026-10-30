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
import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig


mtq = pytest.importorskip("modelopt.torch.quantization")
weight_export = pytest.importorskip("modelopt.torch.export.quantized_weight_export")

from megatron.bridge.models.conversion import modelopt_utils
from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import HFWeightTuple, MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    FusedExpertMapping,
    FusedGatedExpertMapping,
    GatedMLPMapping,
    GDNLinearMappingSeparate,
    QKVMapping,
    _fuse_gdn_separate_to_grouped,
    merge_gdn_linear_weights,
)
from megatron.bridge.models.conversion.quant_mapping import AmaxMapping
from megatron.bridge.models.deepseek.deepseek_v4_bridge import DeepSeekV4Bridge
from megatron.bridge.models.mimo_v2_flash.mimo_v2_flash_bridge import MiMoV2FlashQKVMapping


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


def _nvfp4_linear(out_features=32, in_features=16):
    module = torch.nn.Linear(in_features, out_features, bias=False)
    with torch.no_grad():
        module.weight.copy_(
            torch.arange(out_features * in_features, dtype=torch.float32).reshape(out_features, in_features) / 64 - 4
        )
    return mtq.quantize(
        module,
        copy.deepcopy(mtq.NVFP4_DEFAULT_CFG),
        lambda candidate: candidate(torch.ones(2, in_features)),
    )


def _source(state, shape, parallelism, qkv_layout=None):
    return modelopt_utils._SourceState(state, shape, parallelism, qkv_layout)


class _GroupedWeights(torch.nn.Module):
    def __init__(self, factory=_fp8_linear, num_gemms=2):
        super().__init__()
        sources = [factory() for _ in range(num_gemms)]
        for index, source in enumerate(sources):
            self.register_parameter(f"weight{index}", torch.nn.Parameter(source.weight.detach().clone()))
        self.weight_quantizer = torch.nn.ModuleList([source.weight_quantizer for source in sources])
        self.input_quantizer = sources[0].input_quantizer
        self.num_gemms = num_gemms

    def iter_weights_for_calibration(self):
        for index, quantizer in enumerate(self.weight_quantizer):
            yield getattr(self, f"weight{index}"), quantizer


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


@pytest.mark.unit
def test_fp8_export_matches_fake_quantization_of_a_bf16_weight():
    torch.manual_seed(0)
    module = torch.nn.Linear(64, 64, bias=False).to(torch.bfloat16)
    mtq.quantize(
        module,
        copy.deepcopy(mtq.FP8_DEFAULT_CFG),
        lambda candidate: candidate(torch.randn(2, 64, dtype=torch.bfloat16)),
    )
    hf_name = "model.layers.0.self_attn.o_proj.weight"
    task = _task(ColumnParallelMapping("projection.weight", hf_name), module)

    plan = modelopt_utils.build_modelopt_export_plan([task], model=[module])
    exported = dict(modelopt_utils.prepare_modelopt_export_tasks(plan)[0].export_hook(hf_name, module.weight))

    scale = exported["model.layers.0.self_attn.o_proj.weight_scale"]
    dequantized = (exported[hf_name].float() * scale).to(torch.bfloat16)
    assert torch.equal(dequantized, module.weight_quantizer(module.weight))


@pytest.mark.unit
@pytest.mark.parametrize("quant_algo", ["NVFP4", "W4A16_NVFP4"])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("remote_embedding", [False, True])
def test_public_modelopt_plan_preserves_tied_embedding_metadata(monkeypatch, quant_algo, tied, remote_embedding):
    class EmbeddingBridge(MegatronModelBridge):
        def provider_bridge(self, hf_pretrained):
            raise NotImplementedError

        def mapping_registry(self):
            return MegatronMappingRegistry(
                ColumnParallelMapping("embedding.word_embeddings.weight", "model.embed_tokens.weight"),
                ColumnParallelMapping("output_layer.weight", "lm_head.weight"),
            )

    model = torch.nn.Module()
    model.config = SimpleNamespace(share_embeddings_and_output_weights=tied)
    model.embedding = torch.nn.Embedding(8, 16, dtype=torch.bfloat16)
    model.projection = _nvfp4_linear()
    if quant_algo == "W4A16_NVFP4":
        model.projection.input_quantizer.disable()
    embedding_task = _task(
        ColumnParallelMapping("embedding.word_embeddings.weight", "model.embed_tokens.weight"),
        model.embedding,
        global_name="embedding.word_embeddings.weight",
    )
    tasks = [
        embedding_task,
        _task(ColumnParallelMapping("projection.weight", "model.projection.weight"), model.projection),
    ]
    if remote_embedding:
        embedding_source = modelopt_utils._capture_source_spec(embedding_task)
        tasks[0] = replace(embedding_task, megatron_module=None, param_weight=None)
        monkeypatch.setattr(
            modelopt_utils,
            "_sync_source_specs",
            lambda local, _group: {**local, embedding_task.global_param_name: embedding_source},
        )

    baseline = modelopt_utils.build_modelopt_export_plan(tasks, model=[model])
    bridge = SimpleNamespace(_model_bridge=EmbeddingBridge())
    plan = AutoBridge.build_hf_modelopt_export_plan(bridge, model, [None, *tasks])

    assert plan.quantization_config["quant_algo"] == quant_algo
    assert "model.embed_tokens" in plan.quantization_config["ignore"]
    assert ("lm_head" in plan.quantization_config["ignore"]) is tied
    for actual, expected in zip(plan.conversion_tasks, baseline.conversion_tasks, strict=True):
        assert actual.param_weight is expected.param_weight
        assert replace(actual, param_weight=None) == replace(expected, param_weight=None)
    assert plan.quantized_params == baseline.quantized_params
    assert plan.local_quantization_specs == baseline.local_quantization_specs
    if not tied:
        assert plan.quantization_config == baseline.quantization_config


@pytest.mark.unit
@pytest.mark.parametrize("source_quantized", [False, True])
def test_modelopt_plan_alias_copies_actual_spec(monkeypatch, source_quantized):
    module = _fp8_linear()
    source_module = _fp8_linear() if source_quantized else torch.nn.Linear(4, 4, bias=False)
    source_name = "model.embed_tokens.weight"
    alias = "lm_head.weight"
    tasks = [
        _task(ColumnParallelMapping("projection.weight", "model.projection.weight"), module),
        _task(ColumnParallelMapping("embedding.weight", source_name), source_module, global_name="embedding.weight"),
    ]
    original_builder = weight_export.build_hf_quantization_config
    captured = {}

    def capture_builder(named_specs):
        captured.update(named_specs)
        return original_builder(named_specs)

    monkeypatch.setattr(weight_export, "build_hf_quantization_config", capture_builder)
    plan = modelopt_utils.build_modelopt_export_plan(tasks, model=[module], tied_weight_aliases={alias: source_name})

    assert captured[alias] is captured[source_name]
    assert (captured[alias] is not None) is source_quantized
    assert ("lm_head" in plan.quantization_config["ignore"]) is not source_quantized
    assert len(plan.conversion_tasks) == len(tasks)


@pytest.mark.unit
@pytest.mark.parametrize("case", ["missing-source", "conflicting-alias", "matching-alias"])
def test_modelopt_plan_validates_tied_alias_spec(case):
    module = _fp8_linear()
    source_name = "model.embed_tokens.weight"
    alias = "lm_head.weight"
    embedding = torch.nn.Linear(4, 4, bias=False)
    tasks = [
        _task(ColumnParallelMapping("projection.weight", "model.projection.weight"), module),
        _task(ColumnParallelMapping("embedding.weight", source_name), embedding, global_name="embedding.weight"),
    ]
    if case != "missing-source":
        alias_module = module if case == "conflicting-alias" else embedding
        tasks.append(_task(ColumnParallelMapping("output.weight", alias), alias_module, global_name="output.weight"))
    source = "missing.weight" if case == "missing-source" else source_name
    if case == "matching-alias":
        plan = modelopt_utils.build_modelopt_export_plan(tasks, model=[module], tied_weight_aliases={alias: source})
        assert "lm_head" in plan.quantization_config["ignore"]
        assert len(plan.conversion_tasks) == len(tasks)
    else:
        message = "Missing ModelOpt export spec" if case == "missing-source" else "Inconsistent ModelOpt export spec"
        with pytest.raises(ValueError, match=message):
            modelopt_utils.build_modelopt_export_plan(tasks, model=[module], tied_weight_aliases={alias: source})


@pytest.mark.parametrize("cached_names", [False, True])
@pytest.mark.parametrize("quantization", ["fp8", "static_nvfp4"])
def test_public_modelopt_plan_excludes_quantizer_state_before_mapping_validation(
    monkeypatch, cached_names, quantization
):
    class ProjectionBridge(MegatronModelBridge):
        def provider_bridge(self, hf_pretrained):
            raise NotImplementedError

        def mapping_registry(self):
            return MegatronMappingRegistry(ColumnParallelMapping("projection.weight", "model.projection.weight"))

    implementation = ProjectionBridge()
    monkeypatch.setattr(implementation, "_megatron_global_adapters_info_all_pp_ranks", lambda _models: [])

    class ProjectionAutoBridge(AutoBridge):
        @property
        def _model_bridge(self):
            return implementation

    model = torch.nn.Module()
    model.config = SimpleNamespace(share_embeddings_and_output_weights=False)
    if quantization == "static_nvfp4":
        model.projection = mtq.quantize(
            torch.nn.Linear(16, 4, bias=False),
            {
                "quant_cfg": [
                    {"quantizer_name": "*", "enable": False},
                    {
                        "quantizer_name": "*weight_quantizer",
                        "enable": True,
                        "cfg": {
                            "num_bits": (2, 1),
                            "block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)},
                        },
                    },
                ],
                "algorithm": "max",
            },
            lambda candidate: candidate(torch.ones(2, 16)),
        )
        assert isinstance(model.projection.weight_quantizer, mtq.nn.NVFP4StaticQuantizer)
        assert model.projection.weight_quantizer.global_amax is not None
    else:
        model.projection = _fp8_linear()
    original_state = dict(model.named_buffers())
    model_bridge = modelopt_utils.model_bridge_utils
    monkeypatch.setattr(model_bridge, "_get_pg_collection_from_model", lambda _model: None)
    monkeypatch.setattr(model_bridge, "_get_pp_group", lambda _model: SimpleNamespace(size=lambda: 1))
    monkeypatch.setattr(model_bridge, "_get_pp_rank", lambda _model: 0)
    monkeypatch.setattr(model_bridge, "unwrap_model", lambda candidate: candidate)
    monkeypatch.setattr(model_bridge, "_megatron_local_name_to_global", lambda _models, _config, name, _vp: name)
    monkeypatch.setattr(
        torch.distributed, "all_gather_object", lambda output, value, group: output.__setitem__(0, value)
    )
    monkeypatch.setattr(model_bridge.parallel_state, "get_tensor_model_parallel_group", lambda: None)
    monkeypatch.setattr(model_bridge.parallel_state, "get_expert_tensor_parallel_group", lambda: None)
    monkeypatch.setattr(model_bridge.parallel_state, "get_expert_model_parallel_group", lambda: None)
    bridge = ProjectionAutoBridge(PretrainedConfig())
    if cached_names:
        implementation._megatron_global_param_names_all_pp_ranks([model])

    plan = bridge.build_hf_modelopt_export_plan(model)

    assert [task.global_param_name for task in plan.conversion_tasks] == ["projection.weight"]
    assert hasattr(implementation, "_cached_param_names") == cached_names
    exported = dict(bridge.export_hf_weights_modelopt(model, cpu=True, show_progress=False, export_plan=plan))
    if quantization == "static_nvfp4":
        assert exported["model.projection.weight"].dtype == torch.uint8
        expected_scale = model.projection.weight_quantizer.global_amax / (6 * 448)
        torch.testing.assert_close(exported["model.projection.weight_scale_2"], expected_scale)
        assert "model.projection.weight_scale" in exported
    else:
        assert exported["model.projection.weight"].dtype == torch.float8_e4m3fn
    assert dict(model.named_buffers()).keys() == original_state.keys()
    assert all(dict(model.named_buffers())[name] is value for name, value in original_state.items())
    if quantization == "static_nvfp4":
        with pytest.raises(ValueError, match="_global_amax"):
            implementation.build_conversion_tasks(bridge.hf_pretrained, [model])
    else:
        ordinary_tasks = implementation.build_conversion_tasks(bridge.hf_pretrained, [model])
        assert any(isinstance(task.mapping, AmaxMapping) for task in ordinary_tasks)
    model.register_buffer("unmapped_buffer", torch.tensor(1.0))
    with pytest.raises(ValueError, match="unmapped_buffer"):
        bridge.build_hf_modelopt_export_plan(model)


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


@pytest.mark.parametrize(
    "change", ["input_disabled", "fp8_input_disabled", "weight_disabled", "block_size", "newly_quantized"]
)
def test_reused_plan_rejects_quantization_config_changes(change):
    module = _fp8_linear() if change == "fp8_input_disabled" else _nvfp4_linear(in_features=32)
    if change == "newly_quantized":
        module.weight_quantizer.disable()
        module.input_quantizer.disable()
    tasks = [
        _task(ColumnParallelMapping("projection.weight", "model.projection.weight"), module),
        _task(
            ColumnParallelMapping("stable.weight", "model.stable.weight"),
            _fp8_linear(),
            global_name="stable.weight",
        ),
    ]
    plan = modelopt_utils.build_modelopt_export_plan(tasks, model=[module])
    original_config = copy.deepcopy(plan.quantization_config)

    if change in ("input_disabled", "fp8_input_disabled"):
        module.input_quantizer.disable()
    elif change == "weight_disabled":
        module.weight_quantizer.disable()
    elif change == "block_size":
        module.weight_quantizer.block_sizes = {**module.weight_quantizer.block_sizes, -1: 32}
    else:
        module.weight_quantizer.enable()
        module.input_quantizer.enable()

    with pytest.raises(RuntimeError, match="ModelOpt export plan validation failed"):
        modelopt_utils.prepare_modelopt_export_tasks(plan)
    assert plan.quantization_config == original_config


def test_direct_weight_name_rejects_partial_storage_views():
    module = torch.nn.Linear(4, 4, bias=False)

    assert modelopt_utils._direct_weight_name(module, module.weight.detach()) == "weight"
    assert modelopt_utils._direct_weight_name(module, module.weight[:2]) is None


@pytest.mark.parametrize("method", ["export_hf_weights_modelopt", "export_hf_weight_groups_modelopt"])
def test_modelopt_export_options_are_keyword_only(method):
    bridge = object.__new__(AutoBridge)
    with pytest.raises(TypeError):
        getattr(bridge, method)(torch.nn.Linear(1, 1), "nvfp4")


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


@pytest.mark.parametrize("method", ["megatron_to_hf", "_megatron_to_hf"])
def test_build_plan_rejects_custom_quantized_qkv_layout(monkeypatch, method):
    class CustomQKVMapping(QKVMapping):
        pass

    monkeypatch.setattr(CustomQKVMapping, method, lambda *_args, **_kwargs: {})
    for mapping_cls in (CustomQKVMapping, MiMoV2FlashQKVMapping):
        mapping = mapping_cls("projection.weight", "q.weight", "k.weight", "v.weight")
        module = _fp8_linear()
        with pytest.raises(RuntimeError, match="custom row layout"):
            modelopt_utils.build_modelopt_export_plan([_task(mapping, module)], model=[module])


def test_build_plan_accepts_inherited_standard_qkv_layout():
    class StandardQKVMapping(QKVMapping):
        pass

    module = _fp8_linear(out_features=8)
    module.tensor_model_parallel = True
    module.partition_dim = 0
    module.config = SimpleNamespace(num_attention_heads=2, num_query_groups=1, kv_channels=2, hidden_size=4)
    mapping = StandardQKVMapping("projection.weight", "q.weight", "k.weight", "v.weight")

    plan = modelopt_utils.build_modelopt_export_plan([_task(mapping, module)], model=[module])

    assert plan.quantized_params == {"decoder.layers.0.projection.weight"}


@pytest.mark.parametrize("weight_dtype", [None, torch.bfloat16])
def test_source_dtype_skips_checkpoint_requantization_and_keeps_key_fixes(monkeypatch, weight_dtype):
    quantized = _fp8_linear()
    excluded = torch.nn.Linear(4, 4, bias=False)
    tasks = [
        _task(ColumnParallelMapping("quantized.weight", "layers.0.attn.wo.weight"), quantized),
        _task(
            ColumnParallelMapping("excluded.weight", "layers.0.attn.wq.weight"),
            excluded,
            global_name="excluded.weight",
        ),
    ]
    tasks = [replace(task, weight_dtype=weight_dtype) for task in tasks]
    plan = modelopt_utils.build_modelopt_export_plan(tasks, model=[quantized])
    native_key = "layers.0.indexer.scorer.weights_proj.weight"
    legacy_key = "layers.0.indexer.weights_proj.weight"
    hf_source = {legacy_key: torch.ones(4, 4)}
    for task in tasks:
        hf_source[task.mapping.hf_param] = task.param_weight.to(torch.float8_e4m3fn)
        hf_source[task.mapping.hf_param.removesuffix(".weight") + ".weight_scale_inv"] = torch.ones(1)

    monkeypatch.setattr(
        "megatron.bridge.models.deepseek.deepseek_v4_bridge.quantization_utils.requantize_hf_weight_scale_pairs",
        lambda *_args, **_kwargs: pytest.fail("ModelOpt export must not restore the source quantization format"),
    )
    bridge = object.__new__(DeepSeekV4Bridge)
    for task in plan.conversion_tasks:
        assert task.weight_dtype == (weight_dtype or task.param_weight.dtype)
        hf_name = task.mapping.hf_param
        result = bridge.maybe_modify_converted_hf_weight(
            task, {hf_name: task.param_weight, native_key: hf_source[legacy_key]}, hf_source
        )
        assert result[hf_name] is task.param_weight
        assert native_key not in result and legacy_key in result
        assert not any(name.endswith("weight_scale_inv") for name in result)


def test_source_dtype_reaches_non_owning_pp_task(monkeypatch):
    module = _fp8_linear()
    task = _task(ColumnParallelMapping("projection.weight", "projection.weight"), module)
    remote_source = modelopt_utils._capture_source_spec(task)
    task = replace(task, megatron_module=None, param_weight=None)
    monkeypatch.setattr(modelopt_utils, "_sync_source_specs", lambda *_args: {task.global_param_name: remote_source})

    plan = modelopt_utils.build_modelopt_export_plan([task], model=[module])

    assert plan.conversion_tasks[0].weight_dtype == module.weight.dtype


@pytest.mark.parametrize("quantized", [False, True])
def test_legacy_renamed_weight_fails_closed_only_when_quantized(quantized):
    module = _fp8_linear() if quantized else torch.nn.Linear(4, 4, bias=False)
    dense = _fp8_linear()
    native_key = "layers.0.attn.indexer.scorer.weights_proj.weight"
    legacy_key = "layers.0.attn.indexer.weights_proj.weight"
    task = _task(ColumnParallelMapping("scorer.weight", native_key), module)
    plan = modelopt_utils.build_modelopt_export_plan(
        [task, _task(ColumnParallelMapping("dense.weight", "dense.weight"), dense, global_name="dense.weight")],
        model=[dense],
    )
    prepared = modelopt_utils.prepare_modelopt_export_tasks(plan)[0]
    bridge = object.__new__(DeepSeekV4Bridge)
    result = bridge.maybe_modify_converted_hf_weight(
        prepared, {native_key: module.weight}, {legacy_key: module.weight}
    )
    output = HFWeightTuple(legacy_key, result[legacy_key]).iter_finalized(cpu=True, export_hook=prepared.export_hook)

    if quantized:
        with pytest.raises(RuntimeError, match=f"Missing ModelOpt export state for {legacy_key}"):
            list(output)
    else:
        assert [weight.param_name for weight in output] == [legacy_key]


@pytest.mark.parametrize("remote_error", [None, "ValueError: rank-local failure"], ids=["success", "remote-failure"])
def test_distributed_error_check_gathers_messages_only_on_failure(monkeypatch, remote_error):
    group = object()
    monkeypatch.setattr(modelopt_utils, "get_pg_size", lambda _group: 2)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda _group: "gloo")

    def reduce_status(failed, **_kwargs):
        if remote_error is not None:
            failed.fill_(1)

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce_status)

    def gather_errors(_error, _group):
        assert remote_error is not None, "success path used object gather"
        return [None, remote_error]

    monkeypatch.setattr(modelopt_utils, "_all_gather_objects", gather_errors)

    if remote_error is not None:
        with pytest.raises(RuntimeError, match="rank-local failure"):
            modelopt_utils._raise_distributed_errors(None, group, "capture")
    else:
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

    monkeypatch.setattr(weight_export, "select_quantized_weight_export_state", select)
    monkeypatch.setattr(weight_export, "merge_quantized_weight_export_states", merge)

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

    monkeypatch.setattr(weight_export, "select_quantized_weight_export_state", select)

    transformed = modelopt_utils._transform_source_state(task, source)

    assert set(transformed) == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
    }
    assert [len(indices) for _, _, indices in selections] == [8, 4, 4]
    assert sorted(index for _, _, indices in selections for index in indices) == list(range(16))


@pytest.mark.parametrize("tp_size", [1, 2])
def test_gdn_state_follows_the_in_proj_layout_and_keeps_gates_unquantized(monkeypatch, tp_size):
    prefix = "model.layers.0.linear_attn.in_proj"
    mapping = GDNLinearMappingSeparate(
        "decoder.layers.0.self_attention.in_proj.weight",
        f"{prefix}_qkv.weight",
        f"{prefix}_z.weight",
        f"{prefix}_b.weight",
        f"{prefix}_a.weight",
    )
    monkeypatch.setattr(GDNLinearMappingSeparate, "tp_size", property(lambda _self: tp_size))
    layout = (2, 4, 3, 5)  # key heads, value heads, key head dim, value head dim
    config = SimpleNamespace(
        hidden_size=1,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=3,
        linear_value_head_dim=5,
    )
    # Label each HF row, then lay the rows out as the gathered Megatron in_proj at this TP size.
    hf_rows = dict(zip(("qkv", "z", "b", "a"), torch.arange(60.0).unsqueeze(1).split([32, 20, 4, 4])))
    in_proj = merge_gdn_linear_weights(config, *_fuse_gdn_separate_to_grouped(config, *hf_rows.values()), tp_size)
    task = SimpleNamespace(mapping=mapping, global_param_name="in_proj.weight")
    source = modelopt_utils._SourceState("in_proj", (60, 1), "column", gdn_layout=layout)
    monkeypatch.setattr(modelopt_utils, "_gather_source_states", lambda value, _mapping, _error=None: [value])
    monkeypatch.setattr(weight_export, "select_quantized_weight_export_state", lambda _state, _dim, rows: rows)

    transformed = modelopt_utils._transform_source_state(task, source)

    for key in ("qkv", "z"):
        assert torch.equal(in_proj[transformed[f"{prefix}_{key}.weight"]], hf_rows[key])
    assert transformed[f"{prefix}_b.weight"] is None and transformed[f"{prefix}_a.weight"] is None


def test_export_hook_passes_unquantized_outputs_through(monkeypatch):
    task = SimpleNamespace(mapping=ColumnParallelMapping("projection.weight", "model.projection.weight"))
    weight = torch.randn(4, 4, dtype=torch.bfloat16)
    monkeypatch.setattr(modelopt_utils, "_capture_current_source_state", lambda _task: (None, None))
    monkeypatch.setattr(modelopt_utils, "_transform_source_state", lambda *_args: {"model.projection.weight": None})

    exported = list(modelopt_utils._make_export_hook(task)("model.projection.weight", weight))

    assert exported == [HFWeightTuple("model.projection.weight", weight)]


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


@pytest.mark.parametrize("transpose_on_export", [False, True])
def test_grouped_hf_expert_mapping_exports_one_expert_before_ep_gather(transpose_on_export):
    module = _GroupedWeights()
    module.tensor_model_parallel = True
    module.partition_dim = 1
    mapping = FusedExpertMapping(
        "decoder.layers.0.mlp.experts.linear_fc2.weight1",
        "model.layers.0.mlp.experts.down_proj.weight",
        transpose_on_export=transpose_on_export,
    )
    task = _task(mapping, module, global_name=mapping.megatron_param, weight_name="weight1")

    plan = modelopt_utils.build_modelopt_export_plan([task], model=[module])
    export_task = modelopt_utils.prepare_modelopt_export_tasks(plan)[0]
    mapped = export_task.mapping.megatron_to_hf(module.weight1, module)
    exported = {
        name: value for hf_name, weight in mapped.items() for name, value in export_task.export_hook(hf_name, weight)
    }

    hf_name = "model.layers.0.mlp.experts.1.down_proj.weight"
    assert not export_task.mapping.is_expert
    assert not getattr(export_task.mapping, "is_grouped_export", False)
    assert exported[hf_name].dtype == torch.float8_e4m3fn
    expected = weight_export.export_quantized_weight_tensors(
        module.weight1, weight_export.capture_quantized_weight_export_state(module, "weight1"), module.weight1.dtype
    )
    torch.testing.assert_close(exported[hf_name].float(), expected["weight"].float())
    torch.testing.assert_close(exported[f"{hf_name.removesuffix('.weight')}.weight_scale"], expected["weight_scale"])


def test_grouped_gated_expert_preserves_shared_fp8_input_scale():
    module = _GroupedWeights(lambda: _fp8_linear(out_features=8))
    mapping = FusedGatedExpertMapping(
        "decoder.layers.0.mlp.experts.linear_fc1.weight0",
        "model.layers.0.mlp.experts.gate_up_proj",
    )
    task = _task(mapping, module, global_name=mapping.megatron_param, weight_name="weight0")

    plan = modelopt_utils.build_modelopt_export_plan([task], model=[module])
    export_task = modelopt_utils.prepare_modelopt_export_tasks(plan)[0]
    mapped = export_task.mapping.megatron_to_hf(module.weight0, module)
    exported = {
        name: value for hf_name, weight in mapped.items() for name, value in export_task.export_hook(hf_name, weight)
    }

    gate_prefix = "model.layers.0.mlp.experts.0.gate_proj"
    up_prefix = "model.layers.0.mlp.experts.0.up_proj"
    assert exported[f"{gate_prefix}.weight"].dtype == torch.float8_e4m3fn
    assert exported[f"{up_prefix}.weight"].dtype == torch.float8_e4m3fn
    assert torch.equal(exported[f"{gate_prefix}.input_scale"], exported[f"{up_prefix}.input_scale"])


@pytest.mark.parametrize("transpose_on_export", [False, True])
def test_grouped_gated_expert_preserves_shared_nvfp4_weight_scale_2(transpose_on_export):
    module = _GroupedWeights(_nvfp4_linear)
    mapping = FusedGatedExpertMapping(
        "decoder.layers.0.mlp.experts.linear_fc1.weight1",
        "model.layers.0.mlp.experts.gate_up_proj.weight",
        transpose_on_export=transpose_on_export,
    )
    task = _task(mapping, module, global_name=mapping.megatron_param, weight_name="weight1")

    plan = modelopt_utils.build_modelopt_export_plan([task], model=[module])
    export_task = modelopt_utils.prepare_modelopt_export_tasks(plan)[0]
    mapped = export_task.mapping.megatron_to_hf(module.weight1, module)
    exported = {
        name: value for hf_name, weight in mapped.items() for name, value in export_task.export_hook(hf_name, weight)
    }

    gate_prefix = "model.layers.0.mlp.experts.1.gate_proj"
    up_prefix = "model.layers.0.mlp.experts.1.up_proj"
    assert torch.equal(exported[f"{gate_prefix}.weight_scale_2"], exported[f"{up_prefix}.weight_scale_2"])
    source_state = weight_export.capture_quantized_weight_export_state(module, "weight1")
    for prefix, indices in ((gate_prefix, torch.arange(16)), (up_prefix, torch.arange(16, 32))):
        state = weight_export.select_quantized_weight_export_state(source_state, 0, indices)
        expected = weight_export.export_quantized_weight_tensors(module.weight1[indices], state, module.weight1.dtype)
        for suffix, value in expected.items():
            torch.testing.assert_close(exported[f"{prefix}.{suffix}"].float(), value.float())


def test_grouped_expert_module_rejects_inconsistent_projection_specs():
    gate_up = _fp8_linear(out_features=8)
    down = torch.nn.Linear(4, 4, bias=False)
    down.tensor_model_parallel = True
    down.partition_dim = 1
    gate_up_mapping = FusedGatedExpertMapping(
        "decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight",
        "model.layers.0.mlp.experts.gate_up_proj.weight",
    )
    down_mapping = FusedExpertMapping(
        "decoder.layers.0.mlp.experts.local_experts.0.linear_fc2.weight",
        "model.layers.0.mlp.experts.down_proj.weight",
    )

    with pytest.raises(RuntimeError, match="Inconsistent ModelOpt export specs for fused expert module"):
        modelopt_utils.build_modelopt_export_plan(
            [
                _task(gate_up_mapping, gate_up, global_name=gate_up_mapping.megatron_param),
                _task(down_mapping, down, global_name=down_mapping.megatron_param),
            ],
            model=[gate_up],
        )


def test_unquantized_grouped_transpose_mapping_keeps_offline_export_path():
    expert = torch.nn.Linear(4, 4, bias=False)
    expert_mapping = FusedExpertMapping(
        "decoder.layers.0.mlp.experts.local_experts.0.linear_fc2.weight",
        "model.layers.0.mlp.experts.down_proj",
        transpose_on_export=True,
    )
    dense = _fp8_linear()
    dense_mapping = ColumnParallelMapping(
        "decoder.layers.0.mlp.shared_expert.linear_fc2.weight",
        "model.layers.0.mlp.shared_expert.down_proj.weight",
    )

    plan = modelopt_utils.build_modelopt_export_plan(
        [
            _task(expert_mapping, expert, global_name=expert_mapping.megatron_param),
            _task(dense_mapping, dense, global_name=dense_mapping.megatron_param),
        ],
        model=[dense],
    )

    assert plan.conversion_tasks[0].mapping is expert_mapping
    assert plan.conversion_tasks[0].export_hook is None


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
            weight_export.capture_quantized_weight_export_state(module),
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

        local_groups = [torch.distributed.new_group([member]) for member in range(world_size)]
        local_group = local_groups[rank]
        module = _GroupedWeights(num_gemms=1)
        module.tensor_model_parallel = True
        module.partition_dim = 1
        module.pg_collection = SimpleNamespace(
            pp=local_group,
            ep=torch.distributed.group.WORLD,
            tp=local_group,
            expt_tp=local_group,
        )
        global_name = f"decoder.layers.0.mlp.experts.linear_fc2.weight{rank}"
        mapping = FusedExpertMapping(
            global_name,
            "model.layers.0.mlp.experts.down_proj.weight",
        )
        mapping.set_process_groups_from_pg_collection(module.pg_collection)
        task = _task(mapping, module, global_name=global_name, weight_name="weight0")

        plan = modelopt_utils.build_modelopt_export_plan([task], model=[module])
        export_task = modelopt_utils.prepare_modelopt_export_tasks(plan)[0]
        mapped = export_task.mapping.megatron_to_hf(module.weight0, module)
        exported = {
            name: value
            for hf_name, weight in mapped.items()
            for name, value in export_task.export_hook(hf_name, weight)
        }
        expected_weights = {f"model.layers.0.mlp.experts.{expert}.down_proj.weight" for expert in range(world_size)}
        assert expected_weights.issubset(exported)
        assert {f"{name.removesuffix('.weight')}.weight_scale" for name in expected_weights}.issubset(exported)

        if rank == 0:
            module.weight_quantizer[0].disable()
        with pytest.raises(RuntimeError, match="ModelOpt export plan validation failed"):
            modelopt_utils.prepare_modelopt_export_tasks(plan)
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
            weight_export.capture_quantized_weight_export_state(module),
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
    task = SimpleNamespace(name="task", global_param_name="model.weight", export_hook=None)
    plan = modelopt_utils.ModelOptExportPlan([task], {"quantized_layers": {}}, frozenset(), {})

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


@pytest.mark.unit
def test_plan_config_lists_unquantized_matrices_but_not_vectors():
    norm = torch.nn.LayerNorm(32)
    model = torch.nn.Module()
    model.projection, model.unquantized, model.norm = _nvfp4_linear(), _nvfp4_linear(), norm
    model.unquantized.weight_quantizer.disable()
    model.unquantized.input_quantizer.disable()
    tasks = [
        _task(ColumnParallelMapping("projection.weight", "model.layers.0.mlp.up_proj.weight"), model.projection),
        _task(
            ColumnParallelMapping("unquantized.weight", "model.layers.0.mlp.gate.weight"),
            model.unquantized,
            global_name="decoder.layers.0.unquantized.weight",
        ),
        _task(
            ColumnParallelMapping("norm.weight", "model.layers.0.input_layernorm.weight"),
            norm,
            global_name="decoder.layers.0.norm.weight",
        ),
    ]

    plan = modelopt_utils.build_modelopt_export_plan(tasks, model=[model])

    assert plan.quantization_config["ignore"] == ["model.layers.0.mlp.gate"]


@pytest.mark.unit
@pytest.mark.parametrize("max_shard_size", [1, 1 << 20])
def test_shard_writer_accepts_tensors_the_source_lacks(tmp_path, max_shard_size):
    from safetensors.torch import load_file

    (tmp_path / "model.safetensors").write_bytes(b"stale")
    tensors = {
        "model.weight": torch.arange(4, dtype=torch.uint8),
        "model.weight_scale": torch.ones(2, dtype=torch.float8_e4m3fn),
        "model.input_scale": torch.tensor(0.5),
    }

    saved = modelopt_utils.save_safetensors_shards(iter(tensors.items()), tmp_path, max_shard_size=max_shard_size)

    loaded = {}
    for filename in saved - {"model.safetensors.index.json"}:
        loaded.update(load_file(tmp_path / filename))
    assert loaded.keys() == tensors.keys()
    for name, tensor in tensors.items():
        assert loaded[name].dtype == tensor.dtype and torch.equal(loaded[name].float(), tensor.float()), name
    if max_shard_size == 1:
        index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
        assert set(index["weight_map"]) == tensors.keys() and set(index["weight_map"].values()) == saved - {
            "model.safetensors.index.json"
        }
        assert len(saved) == 4
    else:
        assert saved == {"model.safetensors"}
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.unit
def test_shard_writer_rejects_duplicate_names(tmp_path):
    tensors = [("model.weight", torch.ones(1)), ("model.weight", torch.ones(1))]
    with pytest.raises(ValueError, match="Duplicate"):
        modelopt_utils.save_safetensors_shards(iter(tensors), tmp_path, max_shard_size=1 << 20)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("projections", "scale", "matching"),
    [
        (("q_proj", "k_proj", "v_proj"), "input_scale", True),
        (("q_proj", "k_proj", "v_proj"), "input_scale", False),
        (("gate_proj", "up_proj"), "weight_scale_2", False),
        (("q_a_proj", "kv_a_proj_with_mqa"), "input_scale", False),
        (("q_proj", "o_proj"), "input_scale", True),  # Not fused: different scales are fine.
    ],
)
def test_fused_sibling_scales_must_match(projections, scale, matching):
    tensors = {
        f"model.layers.0.self_attn.{projection}.{scale}": torch.tensor(1.0 if matching or index else 2.0)
        for index, projection in enumerate(projections)
    }
    tensors["model.layers.1.self_attn.q_proj.input_scale"] = torch.tensor(3.0)  # Another layer.
    if "o_proj" in projections:
        tensors["model.layers.0.self_attn.o_proj.input_scale"] = torch.tensor(4.0)

    if matching:
        modelopt_utils.check_fused_sibling_scales(tensors)
    else:
        with pytest.raises(ValueError, match=projections[0]):
            modelopt_utils.check_fused_sibling_scales(tensors)


@pytest.mark.unit
def test_export_rejects_enabled_quantizers_it_would_drop():
    model = torch.nn.Module()
    model.projection = _nvfp4_linear()
    model.attention = torch.nn.Module()
    model.attention.k_bmm_quantizer = mtq.nn.TensorQuantizer()
    model.attention.k_bmm_quantizer.disable()
    modelopt_utils.check_exportable_quantizers([model])

    model.attention.k_bmm_quantizer.enable()
    model.projection.output_quantizer.enable()
    with pytest.raises(RuntimeError, match=r"attention\.k_bmm_quantizer.*projection\.output_quantizer"):
        modelopt_utils.check_exportable_quantizers([model])

    # An activation-only layer would be exported unquantized.
    model.attention.k_bmm_quantizer.disable()
    model.projection.output_quantizer.disable()
    model.projection.weight_quantizer.disable()
    with pytest.raises(RuntimeError, match=r"\['projection\.input_quantizer'\]"):
        modelopt_utils.check_exportable_quantizers([model])


class _StreamingBridge(AutoBridge):
    """Streams one tensor from every rank, or fails like a stale export plan or a rank-0 artifact save."""

    def __init__(self, case):
        self.case = case

    @property
    def _model_bridge(self):
        return MegatronModelBridge()

    def _save_hf_artifacts(self, path, source_path):
        if self.case == "artifacts" and torch.distributed.get_rank() == 0:
            raise OSError("tokenizer save failed")
        if self.case == "success" and torch.distributed.get_rank() == 0:
            (Path(path) / "config.json").write_text("{}")

    def export_hf_weights_modelopt(self, model, **kwargs):
        if self.case == "stale_plan":
            raise ValueError("ModelOpt export plan validation failed")
        yield HFWeightTuple("model.weight", torch.ones(2, 2))


def _save_failure_worker(rank, world_size, init_file, out_dir, case):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:

        def save():
            _StreamingBridge(case).save_hf_pretrained_modelopt(
                [torch.nn.Linear(2, 2)],
                out_dir,
                export_plan=SimpleNamespace(quantization_config={"quant_method": "modelopt"}),
                show_progress=False,
            )

        if case == "success":
            save()
            config = json.loads((Path(out_dir) / "config.json").read_text())
            assert config["quantization_config"] == {"quant_method": "modelopt"}
        else:
            # Artifact saving and a config.json missing at finalization fail rank 0 only; a stale plan
            # fails every rank.
            errors = {
                "stale_plan": "plan validation failed",
                "artifacts": "OSError",
                "finalization": "FileNotFoundError",
            }
            with pytest.raises(RuntimeError, match=errors[case]):
                save()
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.unit
@pytest.mark.parametrize("case", ["success", "stale_plan", "artifacts", "finalization"])
def test_save_fails_on_every_rank_when_any_rank_fails(tmp_path, case):
    torch.multiprocessing.spawn(
        _save_failure_worker,
        args=(2, str(tmp_path / "dist-init"), str(tmp_path), case),
        nprocs=2,
        join=True,
    )
