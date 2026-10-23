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

import weakref
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import PretrainedConfig, Qwen2Config

from megatron.bridge.models.conversion import model_bridge as model_bridge_module
from megatron.bridge.models.conversion import modelopt_utils
from megatron.bridge.models.conversion import param_mapping as param_mapping_module
from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import (
    HFSourcedWeightTuple,
    HFWeightTuple,
    MegatronModelBridge,
    WeightConversionTask,
)
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    DirectMapping,
    FusedExpertMapping,
    FusedGatedExpertMapping,
    GatedMLPMapping,
    QKVMapping,
    RowParallelMapping,
)
from megatron.bridge.models.conversion.transformers_compat import partial_rotary_factor_from_hf
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


class DummyBridge(MegatronModelBridge):
    def provider_bridge(self, hf_pretrained):  # pragma: no cover - not used in tests
        return None

    def mapping_registry(self):  # pragma: no cover - not used in tests
        return MegatronMappingRegistry()


@pytest.mark.parametrize(
    ("current_pp_stage_only", "pp_rank", "expected"),
    [(False, 1, True), (True, 0, True), (True, 1, False)],
)
def test_hf_passthrough_has_canonical_pipeline_owner(monkeypatch, current_pp_stage_only, pp_rank, expected):
    monkeypatch.setattr(model_bridge_module, "_get_pp_rank", lambda _model: pp_rank)
    assert DummyBridge()._should_emit_hf_passthrough([], current_pp_stage_only=current_pp_stage_only) is expected


@pytest.mark.parametrize(
    ("pre_process", "mtp_process", "expected"),
    [(False, True, True), (True, True, False), (False, False, False)],
)
def test_mtp_duplicate_embedding_source_is_excluded(pre_process, mtp_process, expected):
    model = SimpleNamespace(
        # MTP input embeddings are synchronized copies even when output embeddings are untied.
        config=SimpleNamespace(pipeline_model_parallel_size=2, share_embeddings_and_output_weights=False),
        pre_process=pre_process,
        language_model=SimpleNamespace(mtp_process=mtp_process),
    )

    assert (
        DummyBridge._should_skip_mtp_duplicate_embedding_export(
            "embedding.word_embeddings.weight",
            model,
        )
        is expected
    )
    assert not DummyBridge._should_skip_mtp_duplicate_embedding_export("output_layer.weight", model)


def test_weight_conversion_task_round_trips_local_hf_views():
    mapping = GatedMLPMapping(
        "decoder.mlp.linear_fc1.weight",
        gate="model.mlp.gate_proj.weight",
        up="model.mlp.up_proj.weight",
    )
    task = WeightConversionTask(
        param_name="decoder.mlp.linear_fc1.weight",
        global_param_name="decoder.mlp.linear_fc1.weight",
        mapping=mapping,
    )
    logical = torch.arange(32).reshape(8, 4)
    local_weights = {spec.name: spec.select(logical) for spec in task.local_hf_param_specs()}

    assert task.hf_param_names == (
        "model.mlp.gate_proj.weight",
        "model.mlp.up_proj.weight",
    )
    assert torch.equal(task.combine_local_hf_weights(local_weights), logical)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mapping_type", "projection"),
    [(FusedExpertMapping, "down_proj"), (FusedGatedExpertMapping, "gate_up_proj")],
)
@pytest.mark.parametrize("transpose_on_export", [False, True])
def test_fused_expert_local_views_match_full_export(monkeypatch, mapping_type, projection, transpose_on_export):
    """Local expert views and normal grouped export must describe the same weights."""
    _patch_stream_weights_megatron_to_hf_basics(monkeypatch, num_moe_experts=2)
    group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    groups = SimpleNamespace(tp=group, expt_tp=group, pp=group, ep=group)
    module = torch.nn.Module()
    module.tensor_model_parallel = True
    module.partition_dim = 1 if projection == "down_proj" else 0
    hf_name = f"model.mlp.experts.{projection}"
    tasks, local_experts = [], {}

    # Export must order experts numerically even when the tasks arrive in reverse.
    for expert in (1, 0):
        param_name = f"decoder.mlp.experts.linear_fc{2 if projection == 'down_proj' else 1}.weight{expert}"
        mapping = mapping_type(param_name, hf_name, transpose_on_export=transpose_on_export)
        mapping.set_process_groups_from_pg_collection(groups)
        weight = torch.arange(32, dtype=torch.bfloat16).reshape(8, 4) + expert * 32
        task = WeightConversionTask(
            param_name=param_name,
            global_param_name=param_name,
            mapping=mapping,
            megatron_module=module,
            param_weight=weight,
        )
        tasks.append(task)
        views = {spec.name: spec.select(weight) for spec in task.local_hf_param_specs()}
        projections = ["down_proj"] if projection == "down_proj" else ["gate_proj", "up_proj"]
        names = [f"model.mlp.experts.{expert}.{name}.weight" for name in projections]
        assert list(views) == names
        local_experts[expert] = torch.cat([views[name] for name in names], dim=0)

    [exported] = DummyBridge().stream_weights_megatron_to_hf(
        [module], SimpleNamespace(), conversion_tasks=tasks, show_progress=False, merge_adapter_weights=False
    )
    expected = torch.stack([local_experts[expert] for expert in range(2)])
    if transpose_on_export:
        expected = expected.transpose(-1, -2)
    assert exported.param_name == hf_name
    torch.testing.assert_close(exported.weight, expected, rtol=0, atol=0)


def test_iter_local_hf_params_yields_identity_bf16_view():
    weight = torch.nn.Parameter(torch.arange(8, dtype=torch.bfloat16).reshape(2, 4))
    task = WeightConversionTask(
        param_name="decoder.proj.weight",
        global_param_name="decoder.proj.weight",
        mapping=DirectMapping("decoder.proj.weight", "model.proj.weight"),
        megatron_module=torch.nn.Module(),
        param_weight=weight,
    )

    [param] = DummyBridge().iter_local_hf_params([task])

    assert param.name == "model.proj.weight"
    assert param.weight is weight
    assert param.global_weight_shape == torch.Size((2, 4))
    assert param.shard_group == "replicated"
    assert param.shard_dim is None


def test_iter_local_hf_params_splits_gate_up_with_tp_metadata():
    class TwoWayGatedMLPMapping(GatedMLPMapping):
        @property
        def tp_size(self):
            return 2

    weight = torch.nn.Parameter(torch.arange(32, dtype=torch.bfloat16).reshape(8, 4))
    task = WeightConversionTask(
        param_name="decoder.mlp.linear_fc1.weight",
        global_param_name="decoder.mlp.linear_fc1.weight",
        mapping=TwoWayGatedMLPMapping(
            "decoder.mlp.linear_fc1.weight",
            gate="model.mlp.gate_proj.weight",
            up="model.mlp.up_proj.weight",
        ),
        megatron_module=torch.nn.Module(),
        param_weight=weight,
    )

    gate, up = DummyBridge().iter_local_hf_params([task])

    assert [gate.name, up.name] == ["model.mlp.gate_proj.weight", "model.mlp.up_proj.weight"]
    assert gate.weight.data_ptr() == weight.data_ptr()
    assert torch.equal(gate.weight, weight[:4])
    assert torch.equal(up.weight, weight[4:])
    assert gate.global_weight_shape == up.global_weight_shape == torch.Size((8, 4))
    assert gate.shard_group == up.shard_group == "tp"
    assert gate.shard_dim == up.shard_dim == 0


def test_iter_local_hf_params_reports_row_parallel_shard_metadata():
    class FourWayRowParallelMapping(RowParallelMapping):
        @property
        def tp_size(self):
            return 4

    weight = torch.nn.Parameter(torch.zeros((4, 2), dtype=torch.bfloat16))
    task = WeightConversionTask(
        param_name="decoder.mlp.linear_fc2.weight",
        global_param_name="decoder.mlp.linear_fc2.weight",
        mapping=FourWayRowParallelMapping("decoder.mlp.linear_fc2.weight", "model.mlp.down_proj.weight"),
        megatron_module=torch.nn.Module(),
        param_weight=weight,
    )

    [param] = DummyBridge().iter_local_hf_params([task])

    assert param.global_weight_shape == torch.Size((4, 8))
    assert param.shard_group == "tp"
    assert param.shard_dim == 1


def test_iter_local_hf_params_reports_fused_expert_etp_metadata(monkeypatch):
    etp_group = object()
    monkeypatch.setattr(param_mapping_module, "get_pg_size", lambda group: 2 if group is etp_group else 1)
    mapping = FusedGatedExpertMapping(
        "decoder.layers.0.mlp.experts.linear_fc1.weight2",
        "model.layers.0.mlp.experts.gate_up_proj",
    )
    mapping.set_process_groups_from_pg_collection(SimpleNamespace(expt_tp=etp_group))
    module = type("TEColumnParallelGroupedLinear", (torch.nn.Module,), {})()
    weight = torch.nn.Parameter(torch.arange(32, dtype=torch.bfloat16).reshape(8, 4))
    task = WeightConversionTask(
        param_name=mapping.megatron_param,
        global_param_name=mapping.megatron_param,
        mapping=mapping,
        megatron_module=module,
        param_weight=weight,
    )

    gate, up = DummyBridge().iter_local_hf_params([task])

    assert [gate.name, up.name] == [
        "model.layers.0.mlp.experts.2.gate_proj.weight",
        "model.layers.0.mlp.experts.2.up_proj.weight",
    ]
    assert gate.global_weight_shape == up.global_weight_shape == torch.Size((8, 4))
    assert gate.shard_group == up.shard_group == "etp"
    assert gate.shard_dim == up.shard_dim == 0


def test_iter_local_hf_params_skips_remote_pp_placeholders():
    remote = WeightConversionTask(
        param_name="decoder.remote.weight",
        global_param_name="decoder.remote.weight",
        mapping=DirectMapping("decoder.remote.weight", "model.remote.weight"),
        param_weight=None,
        megatron_module=None,
    )
    local_weight = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    local = WeightConversionTask(
        param_name="decoder.local.weight",
        global_param_name="decoder.local.weight",
        mapping=DirectMapping("decoder.local.weight", "model.local.weight"),
        param_weight=local_weight,
        megatron_module=torch.nn.Module(),
    )

    params = list(DummyBridge().iter_local_hf_params([remote, local]))

    assert [param.name for param in params] == ["model.local.weight"]


def test_iter_local_hf_params_rejects_unsupported_qkv_mapping():
    task = WeightConversionTask(
        param_name="decoder.self_attention.linear_qkv.weight",
        global_param_name="decoder.self_attention.linear_qkv.weight",
        mapping=QKVMapping(
            "decoder.self_attention.linear_qkv.weight",
            q="model.self_attn.q_proj.weight",
            k="model.self_attn.k_proj.weight",
            v="model.self_attn.v_proj.weight",
        ),
        megatron_module=torch.nn.Module(),
        param_weight=torch.nn.Parameter(torch.zeros((8, 4), dtype=torch.bfloat16)),
    )

    with pytest.raises(ValueError, match="QKVMapping.*cannot be represented as canonical local HF views"):
        list(DummyBridge().iter_local_hf_params([task]))


def test_iter_local_hf_params_preflights_before_yielding():
    supported = WeightConversionTask(
        param_name="decoder.proj.weight",
        global_param_name="decoder.proj.weight",
        mapping=DirectMapping("decoder.proj.weight", "model.proj.weight"),
        megatron_module=torch.nn.Module(),
        param_weight=torch.nn.Parameter(torch.zeros((2, 2), dtype=torch.bfloat16)),
    )
    unsupported = WeightConversionTask(
        param_name="decoder.qkv.weight",
        global_param_name="decoder.qkv.weight",
        mapping=QKVMapping("decoder.qkv.weight", q="hf.q", k="hf.k", v="hf.v"),
        megatron_module=torch.nn.Module(),
        param_weight=torch.nn.Parameter(torch.zeros((4, 2), dtype=torch.bfloat16)),
    )
    exposed = []

    with pytest.raises(ValueError, match="QKVMapping.*cannot be represented as canonical local HF views"):
        exposed.extend(DummyBridge().iter_local_hf_params([supported, unsupported]))

    assert exposed == []


def test_iter_local_hf_params_preserves_task_and_mapping_order():
    first_weight = torch.nn.Parameter(torch.ones((2, 2), dtype=torch.bfloat16))
    fused_weight = torch.nn.Parameter(torch.arange(16, dtype=torch.bfloat16).reshape(4, 4))
    tasks = [
        WeightConversionTask(
            param_name="decoder.first.weight",
            global_param_name="decoder.first.weight",
            mapping=DirectMapping("decoder.first.weight", "hf.first.weight"),
            megatron_module=torch.nn.Module(),
            param_weight=first_weight,
        ),
        WeightConversionTask(
            param_name="decoder.fused.weight",
            global_param_name="decoder.fused.weight",
            mapping=GatedMLPMapping(
                "decoder.fused.weight",
                gate="hf.gate.weight",
                up="hf.up.weight",
            ),
            megatron_module=torch.nn.Module(),
            param_weight=fused_weight,
        ),
    ]

    params = list(DummyBridge().iter_local_hf_params(tasks))

    assert [param.name for param in params] == ["hf.first.weight", "hf.gate.weight", "hf.up.weight"]


def test_iter_local_hf_params_recaptures_live_weights_from_reused_tasks():
    weight = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
    task = WeightConversionTask(
        param_name="decoder.weight",
        global_param_name="decoder.weight",
        mapping=DirectMapping("decoder.weight", "hf.weight"),
        megatron_module=torch.nn.Module(),
        param_weight=weight,
    )
    bridge = DummyBridge()

    [first] = bridge.iter_local_hf_params([task])
    with torch.no_grad():
        weight.fill_(7)
    [second] = bridge.iter_local_hf_params([task])

    assert first.weight.data_ptr() == second.weight.data_ptr() == weight.data_ptr()
    assert torch.equal(second.weight, torch.full_like(weight, 7))


def test_iter_local_hf_params_keeps_auto_conversion_cache_unmodified():
    mapping = AutoMapping("decoder.weight", "hf.weight")
    module = torch.nn.Module()
    module.tensor_model_parallel = False
    task = WeightConversionTask(
        param_name="decoder.weight",
        global_param_name="decoder.weight",
        mapping=mapping,
        megatron_module=module,
        param_weight=torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16)),
    )

    list(DummyBridge().iter_local_hf_params([task]))

    assert mapping._mapping is None
    assert mapping._detected_type is None


def test_iter_local_hf_params_rejects_non_bf16_storage():
    task = WeightConversionTask(
        param_name="decoder.weight",
        global_param_name="decoder.weight",
        mapping=DirectMapping("decoder.weight", "hf.weight"),
        megatron_module=torch.nn.Module(),
        param_weight=torch.nn.Parameter(torch.zeros(4, dtype=torch.float32)),
    )

    with pytest.raises(ValueError, match="requires unquantized BF16 storage"):
        list(DummyBridge().iter_local_hf_params([task]))


def test_stream_weights_hf_to_megatron_uses_external_state_and_bridge_preprocessing(monkeypatch):
    bridge = DummyBridge()
    mapping = Mock()
    mapping.hf_param = "hf.weight"
    mapping.hf_to_megatron.return_value = torch.ones(2)
    task = WeightConversionTask(
        param_name="weight",
        global_param_name="weight",
        mapping=mapping,
        megatron_module=torch.nn.Module(),
    )
    configured_state = {"hf.weight": torch.full((2,), -1.0)}
    external_state = {"hf.weight": torch.zeros(2)}
    hf_pretrained = SimpleNamespace(state=configured_state)
    preprocess = Mock(wraps=bridge.maybe_modify_loaded_hf_weight)
    monkeypatch.setattr(bridge, "maybe_modify_loaded_hf_weight", preprocess)

    converted = list(
        bridge.stream_weights_hf_to_megatron(
            hf_pretrained,
            [torch.nn.Module()],
            [task],
            hf_state_dict=external_state,
        )
    )

    preprocess.assert_called_once_with(task.mapping.hf_param, external_state)
    mapping.hf_to_megatron.assert_called_once_with(external_state["hf.weight"], task.megatron_module)
    assert len(converted) == 1
    assert converted[0].weight is mapping.hf_to_megatron.return_value


@pytest.mark.parametrize(
    ("available_names", "expected_names"),
    [
        (
            {"hf.weight", "hf.weight_scale_inv"},
            ("hf.weight", "hf.weight_scale_inv"),
        ),
        (
            {"hf.weight_packed", "hf.weight_scale", "hf.weight_shape"},
            ("hf.weight_packed", "hf.weight_scale", "hf.weight_shape"),
        ),
        (
            {"hf.weight_blocks", "hf.weight_scales"},
            ("hf.weight_blocks", "hf.weight_scales"),
        ),
    ],
)
def test_get_hf_import_param_names_declares_quantized_companions(available_names, expected_names):
    assert DummyBridge.get_hf_import_param_names("hf.weight", available_names) == expected_names


def test_finalize_hf_import_broadcasts_tied_weights_and_refreshes_caches(
    monkeypatch,
):
    bridge = DummyBridge()
    model = [torch.nn.Sequential()]
    broadcast = Mock()
    refresh = Mock()
    monkeypatch.setattr(bridge, "_broadcast_shared_embeddings", broadcast)
    import megatron.core.resharding as resharding

    monkeypatch.setattr(resharding, "refresh_module_caches", refresh, raising=False)

    bridge.finalize_hf_import(model)

    broadcast.assert_called_once_with(model)
    refresh.assert_called_once_with(model)


def test_finalize_hf_import_allows_mcore_without_cache_refresh(monkeypatch):
    """Older MCore refs do not expose the optional resharding cache refresher."""
    import megatron.core.resharding as resharding

    bridge = DummyBridge()
    model = torch.nn.Module()
    broadcast = Mock()
    monkeypatch.setattr(bridge, "_broadcast_shared_embeddings", broadcast)
    monkeypatch.delattr(resharding, "refresh_module_caches", raising=False)

    bridge.finalize_hf_import(model)

    broadcast.assert_called_once_with(model)


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

    from modelopt.torch.export import quantized_weight_export

    fake_spec = object()
    monkeypatch.setattr(
        modelopt_utils,
        "_capture_source_spec",
        lambda task: modelopt_utils._SourceState(
            fake_spec if task.global_param_name == first_name else None,
            tuple(task.param_weight.shape),
            "replicated",
        ),
    )
    monkeypatch.setattr(
        modelopt_utils,
        "_sync_source_specs",
        lambda local_specs, _group: local_specs,
    )
    monkeypatch.setattr(
        modelopt_utils,
        "_transform_source_spec",
        lambda task, source: {task.mapping.hf_param: source.state},
    )
    monkeypatch.setattr(modelopt_utils, "get_pg_size", lambda _group: 1)
    monkeypatch.setattr(modelopt_utils.model_bridge_utils, "_get_pg_collection_from_model", lambda _model: None)
    monkeypatch.setattr(
        quantized_weight_export,
        "build_hf_quantization_config",
        lambda _specs: {},
    )

    export_plan = modelopt_utils.build_modelopt_export_plan(
        sparse_tasks,
        model=[model],
    )

    assert [task.global_param_name for task in export_plan.conversion_tasks] == [first_name, last_name]


def _setup_tied_output_bridge(
    monkeypatch, *, share_embeddings_and_output_weights, global_param_names=None, flag_on_model=False
):
    """Build a minimal single-rank bridge whose model has both an output weight and bias.

    Returns ``(bridge, model, hf_pretrained)`` with every distributed/Megatron helper
    stubbed out so the task-construction paths can be exercised without a process group.
    With ``flag_on_model``, the tie flag lives on the model instead of ``model.config``,
    as for builder-backed models whose ``config`` is the nested ``TransformerConfig``.
    """

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

    tie_flag = {"share_embeddings_and_output_weights": share_embeddings_and_output_weights}
    model_config = SimpleNamespace(
        num_moe_experts=0,
        pipeline_model_parallel_size=1,
        **({} if flag_on_model else tie_flag),
    )
    parameters = {
        "output_layer.weight": torch.ones(2, 2),
        "output_layer.bias": torch.ones(2),
    }
    model = SimpleNamespace(
        config=model_config,
        named_parameters=lambda: iter(parameters.items()),
        **(tie_flag if flag_on_model else {}),
    )
    hf_pretrained = SimpleNamespace(config=SimpleNamespace(), state=State())
    bridge = TiedOutputBridge()

    monkeypatch.setattr(
        bridge,
        "_megatron_global_param_names_all_pp_ranks",
        lambda _model: list(global_param_names if global_param_names is not None else parameters),
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
    monkeypatch.setattr(model_bridge_module.parallel_state, "get_pipeline_model_parallel_group", lambda: None)

    return bridge, model, hf_pretrained


def test_build_conversion_tasks_keeps_output_bias_when_embeddings_are_tied(monkeypatch):
    """Tied embeddings remove only output_layer.weight, not an independent output bias."""
    bridge, model, hf_pretrained = _setup_tied_output_bridge(monkeypatch, share_embeddings_and_output_weights=True)

    tasks = bridge.build_conversion_tasks(hf_pretrained, [model])

    assert len(tasks) == 1
    assert tasks[0] is not None
    assert tasks[0].global_param_name == "output_layer.bias"


@pytest.mark.parametrize("share_embeddings_and_output_weights", [True, False])
def test_build_conversion_tasks_reads_tie_flag_from_builder_model(monkeypatch, share_embeddings_and_output_weights):
    """A builder-backed model's config lacks the tie flag; the flag on the built model decides."""
    bridge, model, hf_pretrained = _setup_tied_output_bridge(
        monkeypatch,
        share_embeddings_and_output_weights=share_embeddings_and_output_weights,
        flag_on_model=True,
    )

    tasks = bridge.build_conversion_tasks(hf_pretrained, [model])

    expected = ["output_layer.weight", "output_layer.bias"]
    if share_embeddings_and_output_weights:
        expected = ["output_layer.bias"]
    assert [task.global_param_name for task in tasks] == expected


@pytest.mark.parametrize(
    ("config", "model_flag", "language_model_flag", "expected"),
    [
        # Provider-built models: the config flag wins, even over a wrapper attribute.
        (SimpleNamespace(share_embeddings_and_output_weights=True), False, None, True),
        (SimpleNamespace(share_embeddings_and_output_weights=False), True, None, False),
        # Builder-backed models: the nested TransformerConfig has no tie flag.
        (SimpleNamespace(), True, None, True),
        (SimpleNamespace(), False, None, False),
        # A wrapper defers to the language model that owns the output layer.
        (SimpleNamespace(), False, True, True),
        (SimpleNamespace(), None, None, False),
    ],
)
def test_share_embeddings_and_output_weights_resolution(config, model_flag, language_model_flag, expected):
    model = SimpleNamespace(config=config)
    if model_flag is not None:
        model.share_embeddings_and_output_weights = model_flag
    if language_model_flag is not None:
        model.language_model = SimpleNamespace(share_embeddings_and_output_weights=language_model_flag)

    assert DummyBridge()._share_embeddings_and_output_weights(model) is expected


def test_build_conversion_tasks_keeps_output_weight_when_embeddings_are_untied(monkeypatch):
    """Without weight tying the output weight is a real parameter and must be converted."""
    bridge, model, hf_pretrained = _setup_tied_output_bridge(monkeypatch, share_embeddings_and_output_weights=False)

    tasks = bridge.build_conversion_tasks(hf_pretrained, [model])

    assert [task.global_param_name for task in tasks] == ["output_layer.weight", "output_layer.bias"]


@pytest.mark.parametrize(
    ("name", "is_filtered"),
    [
        ("output_layer.weight", True),
        ("decoder.output_layer.weight", True),
        ("output_layer.bias", False),
        # Substring matches that the previous `"output_layer" not in name` filter wrongly dropped.
        ("output_layer.weight_scale_inv", False),
        ("lm_head.output_layer_norm.weight", False),
    ],
)
def test_build_conversion_tasks_tied_filter_only_targets_output_layer_weight(monkeypatch, name, is_filtered):
    """Only a trailing `output_layer.weight` is removed under weight tying."""
    bridge, model, hf_pretrained = _setup_tied_output_bridge(
        monkeypatch, share_embeddings_and_output_weights=True, global_param_names=[name]
    )
    monkeypatch.setattr(
        bridge,
        "mapping_registry",
        lambda: MegatronMappingRegistry(AutoMapping(name, "lm_head.weight")),
    )

    tasks = bridge.build_conversion_tasks(hf_pretrained, [model])

    assert (len(tasks) == 0) is is_filtered


def test_build_export_fp8_tasks_keeps_output_bias_when_embeddings_are_tied(monkeypatch):
    """The FP8 export path applies the same narrowed tied-weight filter as the import path."""
    bridge, model, hf_pretrained = _setup_tied_output_bridge(monkeypatch, share_embeddings_and_output_weights=True)

    detected_names = []

    def _fake_detect_fp8_params(_megatron_model, _model_config, sorted_names, _pp_group, _fp8_scale_inv_attr):
        detected_names.append(list(sorted_names))
        return {}

    monkeypatch.setattr(bridge, "_detect_fp8_params", _fake_detect_fp8_params)

    tasks = bridge.build_export_fp8_tasks(hf_pretrained, [model])

    # The filtered name list is what drives both FP8 detection and the final task ordering.
    assert detected_names == [["output_layer.bias"]]
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


def test_truncate_vocab_padding_handles_output_bias_and_alias():
    """Padded vocabulary biases and their tied HF aliases are restored to the HF vocabulary size."""
    bridge = DummyBridge()
    bridge.hf_config = SimpleNamespace(vocab_size=3)
    task = SimpleNamespace(
        global_param_name="output_layer.bias",
        mapping=SimpleNamespace(hf_param="cls.predictions.bias"),
    )
    padded_bias = torch.arange(5)

    result = bridge._truncate_vocab_padding(
        task,
        {
            "cls.predictions.bias": padded_bias,
            "cls.predictions.decoder.bias": padded_bias.clone(),
        },
    )

    assert result["cls.predictions.bias"].shape == (3,)
    assert result["cls.predictions.decoder.bias"].shape == (3,)


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


@pytest.mark.unit
def test_stream_weights_megatron_to_hf_honors_replaced_task_subset(monkeypatch):
    """Reusable export plans must honor filtered, caller-materialized logical weights."""
    bridge = DummyBridge()
    module = torch.nn.Linear(4, 2, bias=False)
    mapping = DirectMapping("decoder.proj.weight", "model.proj.weight")
    group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    mapping.set_process_groups_from_pg_collection(SimpleNamespace(pp=group))
    task = WeightConversionTask(
        param_name="decoder.proj.weight",
        global_param_name="decoder.proj.weight",
        mapping=mapping,
        megatron_module=module,
        param_weight=module.weight,
    )
    omitted = replace(task, mapping=DirectMapping("decoder.omitted.weight", "model.omitted.weight"))
    logical = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    tasks = [omitted, replace(task, param_weight=logical)]
    monkeypatch.setattr(
        bridge, "build_conversion_tasks", Mock(side_effect=AssertionError("must reuse supplied tasks"))
    )
    _patch_stream_weights_megatron_to_hf_basics(monkeypatch)

    for offset in (0, 16):
        logical.copy_(torch.arange(8, dtype=torch.bfloat16).reshape(2, 4) + offset)
        [exported] = bridge.stream_weights_megatron_to_hf(
            [module],
            SimpleNamespace(),
            cpu=True,
            show_progress=False,
            conversion_tasks=(item for item in tasks if item is not omitted),
            merge_adapter_weights=False,
        )

        assert exported.param_name == "model.proj.weight"
        assert exported.weight.dtype == torch.bfloat16
        assert exported.weight.shape == logical.shape
        torch.testing.assert_close(exported.weight, logical)
    assert task.param_weight is module.weight


def test_stream_weights_megatron_to_hf_current_pp_stage_only_filters_non_owned_tasks(monkeypatch):
    bridge = DummyBridge()
    source = torch.ones(2, 2)
    calls = []

    class DummyMapping:
        is_grouped_export = False
        hf_param = "hf.weight"

        def __init__(self, hf_name):
            self.hf_name = hf_name

        def megatron_to_hf(self, weight, module):
            calls.append(
                (
                    self.hf_name,
                    weight,
                    module,
                    param_mapping_module._SKIP_PP_BROADCAST.get(),
                )
            )
            return {self.hf_name: weight}

    owned_task = WeightConversionTask(
        param_name="decoder.layers.0.weight",
        global_param_name="decoder.layers.0.weight",
        mapping=DummyMapping("hf.layers.0.weight"),
        pp_rank=0,
        vp_stage=0,
        megatron_module=object(),
        param_weight=source,
    )
    remote_task = WeightConversionTask(
        param_name="decoder.layers.1.weight",
        global_param_name="decoder.layers.1.weight",
        mapping=DummyMapping("hf.layers.1.weight"),
        pp_rank=1,
        vp_stage=0,
        megatron_module=None,
        param_weight=None,
    )
    bridge._cached_param_owners = {
        owned_task.global_param_name: 0,
        remote_task.global_param_name: 1,
    }
    monkeypatch.setattr(model_bridge_module, "_get_pp_rank", lambda _model: 0)
    _patch_stream_weights_megatron_to_hf_basics(monkeypatch)
    monkeypatch.setattr(
        DummyBridge,
        "maybe_modify_converted_hf_weight",
        lambda self, *_args, **_kwargs: _args[1],
    )

    weights = bridge.stream_weights_megatron_to_hf(
        [Mock()],
        SimpleNamespace(),
        cpu=False,
        show_progress=False,
        conversion_tasks=[owned_task, remote_task],
        merge_adapter_weights=False,
        current_pp_stage_only=True,
    )
    weight = next(weights)

    # The local policy is active while the mapping executes, but must be reset
    # before the streaming generator yields control back to its caller.
    assert calls[0][3] is True
    assert param_mapping_module._SKIP_PP_BROADCAST.get() is False
    with pytest.raises(StopIteration):
        next(weights)

    assert weight[0] == "hf.layers.0.weight"
    assert torch.equal(weight[1], source)
    assert len(calls) == 1
    assert calls[0][0] == "hf.layers.0.weight"
    assert torch.equal(calls[0][1], source)
    assert calls[0][2] is owned_task.megatron_module


@pytest.mark.parametrize(("pp_rank", "expected_names"), [(0, ["hf.shared.weight"]), (1, [])])
def test_current_pp_stage_only_uses_canonical_owner_for_duplicate_parameters(
    monkeypatch,
    pp_rank,
    expected_names,
):
    bridge = DummyBridge()
    bridge._cached_param_owners = {"decoder.shared.weight": 0}
    source = torch.ones(2, 2)
    calls = []

    class DummyMapping:
        is_grouped_export = False
        hf_param = "hf.shared.weight"

        def megatron_to_hf(self, weight, module):
            calls.append((weight, module))
            return {self.hf_param: weight}

    task = WeightConversionTask(
        param_name="decoder.shared.weight",
        global_param_name="decoder.shared.weight",
        mapping=DummyMapping(),
        pp_rank=pp_rank,
        vp_stage=0,
        megatron_module=object(),
        param_weight=source,
    )
    monkeypatch.setattr(model_bridge_module, "_get_pp_rank", lambda _model: pp_rank)
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
            current_pp_stage_only=True,
        )
    )

    assert [weight.param_name for weight in weights] == expected_names
    assert len(calls) == (1 if pp_rank == 0 else 0)


def test_stream_weights_megatron_to_hf_current_pp_stage_only_covers_adapter_materialization(monkeypatch):
    bridge = DummyBridge()
    source = torch.ones(2, 2)
    observed_context = []

    class DummyMapping:
        is_grouped_export = False
        hf_param = "hf.weight"

        def megatron_to_hf(self, weight, module):
            return {"hf.weight": weight}

    task = WeightConversionTask(
        param_name="decoder.layers.0.linear.to_wrap.weight",
        global_param_name="decoder.layers.0.linear.to_wrap.weight",
        mapping=DummyMapping(),
        pp_rank=0,
        vp_stage=0,
        megatron_module=object(),
        param_weight=source,
    )
    bridge._cached_param_owners = {task.global_param_name: 0}
    monkeypatch.setattr(model_bridge_module, "_get_pp_rank", lambda _model: 0)
    _patch_stream_weights_megatron_to_hf_basics(monkeypatch)
    monkeypatch.setattr(
        DummyBridge,
        "build_adapter_conversion_tasks",
        lambda self, *_args: {"decoder.layers.0.linear": [object()]},
    )

    def materialize_adapter_weights(self, adapter_tasks):
        observed_context.append(param_mapping_module._SKIP_PP_BROADCAST.get())
        return []

    monkeypatch.setattr(DummyBridge, "materialize_adapter_weights", materialize_adapter_weights)
    monkeypatch.setattr(
        DummyBridge,
        "_merge_lora_adapter_weights",
        lambda self, _model, converted, _adapters: converted,
    )
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
            merge_adapter_weights=True,
            current_pp_stage_only=True,
        )
    )

    assert [weight.param_name for weight in weights] == ["hf.weight"]
    assert observed_context == [True]
    assert param_mapping_module._SKIP_PP_BROADCAST.get() is False


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


@pytest.mark.parametrize("grouped", [False, True])
def test_stream_weights_megatron_to_hf_with_megatron_names_reports_source_param(monkeypatch, grouped):
    bridge = DummyBridge()
    source = torch.ones(2, 2)

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
    _patch_stream_weights_megatron_to_hf_basics(monkeypatch)
    monkeypatch.setattr(
        DummyBridge,
        "maybe_modify_converted_hf_weight",
        lambda self, *_args, **_kwargs: _args[1],
    )

    def stream(**kwargs):
        stream_method = (
            bridge._stream_weight_groups_megatron_to_hf if grouped else bridge.stream_weights_megatron_to_hf
        )
        result = list(
            stream_method(
                [Mock()],
                SimpleNamespace(),
                cpu=False,
                show_progress=False,
                conversion_tasks=[task],
                merge_adapter_weights=False,
                **kwargs,
            )
        )
        if grouped:
            assert len(result) == 1
            return list(result[0])
        return result

    # Default output stays a two-field tuple so ``for name, weight in ...`` keeps working.
    (plain,) = stream()
    assert type(plain) is HFWeightTuple
    name, weight = plain
    assert name == "hf.weight"
    assert torch.equal(weight, source)

    (sourced,) = stream(with_megatron_names=True)
    assert type(sourced) is HFSourcedWeightTuple
    assert sourced.param_name == "hf.weight"
    assert torch.equal(sourced.weight, source)
    assert sourced.megatron_param_names == ("decoder.layers.0.mlp.linear_fc1.weight",)
    assert sourced.megatron_param_name == "decoder.layers.0.mlp.linear_fc1.weight"


@pytest.mark.parametrize("grouped", [False, True])
def test_stream_weights_megatron_to_hf_with_megatron_names_lists_every_grouped_source(monkeypatch, grouped):
    """A packed grouped-expert tensor names all contributing per-expert params, not the last one."""
    bridge = DummyBridge()

    class GroupedMapping:
        is_grouped_export = True
        group_key = "hf.grouped"
        ep_size = 1

        def megatron_to_hf(self, weight, module):
            return {self.group_key: weight}

    expert_names = [f"decoder.layers.0.mlp.experts.linear_fc2.weight{expert}" for expert in range(3)]
    tasks = [
        WeightConversionTask(
            param_name=name,
            global_param_name=name,
            mapping=GroupedMapping(),
            pp_rank=0,
            vp_stage=0,
            megatron_module=None,
            param_weight=torch.full((1, 1), float(index + 1)),
        )
        for index, name in enumerate(expert_names)
    ]

    def transform(name, tensor):
        yield f"{name}.packed", tensor.to(torch.uint8)
        yield f"{name}.scale", torch.ones(3, 1)

    tasks = [_with_export_hook(task, transform) for task in tasks]
    _patch_stream_weights_megatron_to_hf_basics(monkeypatch, num_moe_experts=3)

    def stream(**kwargs):
        stream_method = (
            bridge._stream_weight_groups_megatron_to_hf if grouped else bridge.stream_weights_megatron_to_hf
        )
        result = list(
            stream_method(
                [Mock()],
                SimpleNamespace(),
                cpu=True,
                show_progress=False,
                conversion_tasks=tasks,
                merge_adapter_weights=False,
                **kwargs,
            )
        )
        if grouped:
            assert len(result) == 3 and result[:2] == [(), ()]
            assert len(result[2]) == 2
            return list(result[2])
        return result

    plain = stream()
    assert [type(weight) for weight in plain] == [HFWeightTuple, HFWeightTuple]

    sourced = stream(with_megatron_names=True)
    assert [weight.param_name for weight in sourced] == ["hf.grouped.packed", "hf.grouped.scale"]
    assert [type(weight) for weight in sourced] == [HFSourcedWeightTuple, HFSourcedWeightTuple]
    for weight in sourced:
        assert weight.megatron_param_names == tuple(expert_names)
        assert weight.megatron_param_name is None


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


def test_stream_weight_groups_materializes_one_complete_task(monkeypatch):
    bridge = DummyBridge()
    events = []
    converted_refs = []

    class DummyMapping:
        def __init__(self, index):
            self.index = index

        def megatron_to_hf(self, weight, module):
            events.append(("map", self.index))
            converted = weight.clone()
            converted_refs.append(weakref.ref(converted))
            return {f"hf.weight_{self.index}": converted}

    def make_export_hook(index, output_count):
        def export(name, tensor):
            events.append(("export_start", index))
            for output_index in range(output_count):
                yield f"{name}.{output_index}", tensor + output_index
            events.append(("export_done", index))

        return export

    tasks = []
    for index, output_count in enumerate((2, 0, 1)):
        task = WeightConversionTask(
            param_name=f"decoder.weight_{index}",
            global_param_name=f"decoder.weight_{index}",
            mapping=DummyMapping(index),
            pp_rank=0,
            vp_stage=0,
            megatron_module=None,
            param_weight=torch.ones(1),
        )
        tasks.append(_with_export_hook(task, make_export_hook(index, output_count)))

    _patch_stream_weights_megatron_to_hf_basics(monkeypatch)
    monkeypatch.setattr(
        DummyBridge,
        "maybe_modify_converted_hf_weight",
        lambda self, *_args, **_kwargs: _args[1],
    )

    groups = iter(
        bridge._stream_weight_groups_megatron_to_hf(
            [Mock()],
            SimpleNamespace(),
            cpu=False,
            show_progress=False,
            conversion_tasks=tasks,
            merge_adapter_weights=False,
        )
    )

    first = next(groups)
    assert [weight.param_name for weight in first] == ["hf.weight_0.0", "hf.weight_0.1"]
    assert events == [("map", 0), ("export_start", 0), ("export_done", 0)]
    assert converted_refs[0]() is None
    assert next(groups) == ()
    assert events[-3:] == [("map", 1), ("export_start", 1), ("export_done", 1)]
    assert converted_refs[1]() is None
    assert [weight.param_name for weight in next(groups)] == ["hf.weight_2.0"]
    with pytest.raises(StopIteration):
        next(groups)


def _distributed_weight_group_stream_worker(rank, world_size, init_file):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=10),
    )
    try:
        model = SimpleNamespace(
            config=SimpleNamespace(
                num_moe_experts=0,
                pipeline_model_parallel_size=world_size,
            )
        )
        model_bridge_module.unwrap_model = lambda *_args, **_kwargs: [model]
        model_bridge_module.parallel_state.get_expert_model_parallel_world_size = lambda: 1
        DummyBridge._with_progress_tracking = lambda self, tasks, *_args, **_kwargs: tasks
        DummyBridge._share_embeddings_and_output_weights = lambda self, *_args, **_kwargs: False

        tasks = []
        for owner_rank, output_count in enumerate((2, 0)):
            mapping = DirectMapping(
                f"decoder.weight_{owner_rank}",
                f"hf.weight_{owner_rank}",
            )
            mapping.pp_group = torch.distributed.group.WORLD
            source = torch.tensor([owner_rank + 1.0]) if rank == owner_rank else None

            def export(name, tensor, count=output_count):
                for output_index in range(count):
                    yield f"{name}.{output_index}", tensor + output_index

            task = WeightConversionTask(
                param_name=f"decoder.weight_{owner_rank}",
                global_param_name=f"decoder.weight_{owner_rank}",
                mapping=mapping,
                pp_rank=owner_rank,
                megatron_module=torch.nn.Module() if rank == owner_rank else None,
                param_weight=source,
            )
            tasks.append(_with_export_hook(task, export))

        groups = list(
            DummyBridge()._stream_weight_groups_megatron_to_hf(
                [model],
                SimpleNamespace(),
                cpu=False,
                show_progress=False,
                conversion_tasks=tasks,
                merge_adapter_weights=False,
            )
        )

        assert [[weight.param_name for weight in group] for group in groups] == [
            ["hf.weight_0.0", "hf.weight_0.1"],
            [],
        ]
        torch.testing.assert_close(groups[0][0].weight.cpu(), torch.tensor([1.0]))
        torch.testing.assert_close(groups[0][1].weight.cpu(), torch.tensor([2.0]))
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.timeout(60)
def test_weight_groups_preserve_pp_placeholder_epochs(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    torch.multiprocessing.spawn(
        _distributed_weight_group_stream_worker,
        args=(2, str(tmp_path / "weight-group-stream")),
        nprocs=2,
        join=True,
    )


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


@pytest.mark.unit
def test_build_conversion_tasks_preserves_pp_vp_and_expert_ownership(monkeypatch):
    """Task topology retains global names and live local sources across VP chunks."""
    config = SimpleNamespace(share_embeddings_and_output_weights=False, num_moe_experts=4)
    group = SimpleNamespace(size=lambda: 2, rank=lambda: 1)
    groups = SimpleNamespace(pp=group, ep=group)
    models = []
    for layer_number in (2, 4):
        layer = model_bridge_module.MegatronModule(config)
        layer.layer_number = layer_number
        layer.mlp = torch.nn.Module()
        layer.mlp.experts = torch.nn.Module()
        layer.mlp.experts.linear_fc1 = torch.nn.Module()
        layer.mlp.experts.linear_fc1.register_parameter("weight0", torch.nn.Parameter(torch.ones(2, 2)))
        model = torch.nn.Module()
        model.config = config
        model.pg_collection = groups
        model.decoder = torch.nn.Module()
        model.decoder.layers = torch.nn.ModuleList([layer])
        models.append(model)

    local_name = "decoder.layers.0.mlp.experts.linear_fc1.weight0"
    global_names = [f"decoder.layers.{layer}.mlp.experts.linear_fc1.weight2" for layer in (1, 2, 3)]
    bridge = DummyBridge()
    monkeypatch.setattr(
        bridge,
        "mapping_registry",
        lambda: MegatronMappingRegistry(
            DirectMapping(
                "decoder.layers.*.mlp.experts.linear_fc1.weight*",
                "model.layers.*.mlp.experts.*.gate_proj.weight",
            )
        ),
    )
    monkeypatch.setattr(bridge, "_megatron_global_param_names_all_pp_ranks", lambda _models: global_names)
    monkeypatch.setattr(model_bridge_module, "get_pg_size", lambda pg: pg.size())
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: group.rank() if group is not None else 0)

    tasks = bridge.build_conversion_tasks(PretrainedConfig(), models)

    assert [task.global_param_name for task in tasks] == global_names
    for vp_stage, task in enumerate((tasks[0], tasks[2])):
        source_module = models[vp_stage].decoder.layers[0].mlp.experts.linear_fc1
        assert task.param_name == local_name
        assert (task.pp_rank, task.vp_stage) == (1, vp_stage)
        assert task.megatron_module is source_module
        assert task.param_weight is source_module.weight0
        assert task.mapping.hf_param == f"model.layers.{2 * vp_stage + 1}.mlp.experts.2.gate_proj.weight"
    remote = tasks[1]
    assert remote.param_name == global_names[1]
    assert (remote.param_weight, remote.megatron_module, remote.vp_stage) == (None, None, None)
    assert remote.mapping.hf_param == "model.layers.2.mlp.experts.2.gate_proj.weight"


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


def test_conversion_discovery_excludes_complete_module_subtrees():
    class Excluded(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.register_buffer("amax", torch.ones(1))
            self.child = torch.nn.Linear(1, 1, bias=False)

    model = torch.nn.Module()
    model.quantizer = Excluded()
    model.quantizer_neighbor = torch.nn.Linear(1, 1, bias=False)
    ordinary = dict(model_bridge_module._named_conversion_parameters(model))

    filtered = dict(model_bridge_module._named_conversion_parameters(model, (Excluded,)))

    assert set(filtered) == {"quantizer_neighbor.weight"}
    assert "quantizer.amax" in ordinary
    assert "quantizer.child.weight" in ordinary
    assert dict(model_bridge_module._named_conversion_parameters(model)).keys() == ordinary.keys()
    assert list(model_bridge_module._named_conversion_parameters(model.quantizer, (Excluded,))) == []


@pytest.mark.parametrize("cached_names", [False, True])
def test_fake_refit_excludes_weight_quantizer_state_before_validation(monkeypatch, cached_names):
    implementation = DummyBridge()
    model = torch.nn.Module()
    model.config = SimpleNamespace(share_embeddings_and_output_weights=False)
    model.projection = torch.nn.Module()
    for index in range(2):
        model.projection.register_parameter(f"weight{index}", torch.nn.Parameter(torch.ones(2, 2)))
    model.projection.weight_quantizer = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
    for index, quantizer in enumerate(model.projection.weight_quantizer):
        quantizer.register_buffer("_amax", torch.tensor(float(index + 1)))
    for name in ("input_quantizer", "k_bmm_quantizer", "v_bmm_quantizer"):
        quantizer = torch.nn.Module()
        quantizer.register_buffer("_amax", torch.ones(1))
        model.projection.add_module(name, quantizer)

    retained_names = {
        "projection.weight0",
        "projection.weight1",
        "projection.input_quantizer._amax",
        "projection.k_bmm_quantizer._amax",
        "projection.v_bmm_quantizer._amax",
    }
    monkeypatch.setenv("ENABLE_BRIDGE_QUANT_MAPPING", "0")
    monkeypatch.setattr(
        implementation,
        "mapping_registry",
        lambda: MegatronMappingRegistry(*(AutoMapping(name, f"hf.{name}") for name in sorted(retained_names))),
    )
    _patch(monkeypatch, "_get_pg_collection_from_model", lambda _models: None)
    _patch(monkeypatch, "_get_pp_group", lambda _models: SimpleNamespace(size=lambda: 1))
    _patch(monkeypatch, "_get_pp_rank", lambda _models: 0)
    _patch(monkeypatch, "unwrap_model", lambda models: models)
    _patch(monkeypatch, "_megatron_local_name_to_global", lambda _models, _config, name, _vp: name)
    monkeypatch.setattr(
        torch.distributed, "all_gather_object", lambda output, value, group: output.__setitem__(0, value)
    )
    pretrained = Mock(spec=PreTrainedCausalLM, config=PretrainedConfig())
    pretrained.state.source.get_all_keys.return_value = [f"hf.{name}" for name in retained_names]

    class ProjectionAutoBridge(AutoBridge):
        @property
        def _model_bridge(self):
            return implementation

    bridge = ProjectionAutoBridge(pretrained)
    if cached_names:
        implementation._megatron_global_param_names_all_pp_ranks([model])
    original_buffers = dict(model.named_buffers())
    original_cache = getattr(implementation, "_cached_param_names", None)
    original_owners = getattr(implementation, "_cached_param_owners", None)

    tasks = bridge.get_conversion_tasks(model, exclude_param_patterns=("*weight_quantizer*",))
    assert {task.global_param_name for task in tasks} == retained_names
    assert getattr(implementation, "_cached_param_names", None) is original_cache
    assert getattr(implementation, "_cached_param_owners", None) is original_owners
    assert dict(model.named_buffers()).keys() == original_buffers.keys()
    assert all(dict(model.named_buffers())[name] is tensor for name, tensor in original_buffers.items())

    with pytest.raises(ValueError, match=r"weight_quantizer\.0\._amax"):
        bridge.get_conversion_tasks(model)
    model.register_parameter("unmapped_weight", torch.nn.Parameter(torch.ones(1)))
    with pytest.raises(ValueError, match="unmapped_weight"):
        bridge.get_conversion_tasks(model, exclude_param_patterns=("*weight_quantizer*",))


@pytest.mark.parametrize("model_family", ["kimi_k3", "kimi_k25"])
def test_kimi_conversion_tasks_forward_module_exclusions(monkeypatch, model_family):
    from megatron.bridge.models.kimi.kimi_k3_bridge import KimiK3Bridge
    from megatron.bridge.models.kimi_vl.kimi_k25_vl_bridge import KimiK25VLBridge

    bridge_cls = KimiK3Bridge if model_family == "kimi_k3" else KimiK25VLBridge
    keys = ["expert.weight_packed", "expert.weight_scale", "expert.weight_shape"]
    get_all_keys = lambda: keys
    source = SimpleNamespace(get_all_keys=get_all_keys)
    pretrained = SimpleNamespace(state=SimpleNamespace(source=source))
    tasks = []

    def build_tasks(_self, hf_pretrained, model, *, weight_dtype, exclude_module_types, exclude_param_patterns):
        assert model == []
        assert weight_dtype == torch.bfloat16
        assert exclude_module_types == (torch.nn.ReLU,)
        assert exclude_param_patterns == ("*weight_quantizer*",)
        assert "expert.weight" in hf_pretrained.state.source.get_all_keys()
        return tasks

    monkeypatch.setattr(MegatronModelBridge, "build_conversion_tasks", build_tasks)
    bridge = bridge_cls()

    assert (
        bridge.build_conversion_tasks(
            pretrained,
            [],
            weight_dtype=torch.bfloat16,
            exclude_module_types=(torch.nn.ReLU,),
            exclude_param_patterns=("*weight_quantizer*",),
        )
        is tasks
    )
    assert source.get_all_keys is get_all_keys


def _yarn_qwen2_config(rope_scaling):
    return Qwen2Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16384,
        vocab_size=128,
        rope_scaling=rope_scaling,
    )


def test_yarn_mapping_fills_hugging_face_defaults_for_omitted_keys():
    """Omitted YaRN keys must not reach Megatron Core as None."""
    config = _yarn_qwen2_config({"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 4096})

    kwargs = DummyBridge().hf_config_to_provider_kwargs(config)

    assert kwargs["position_embedding_type"] == "yarn"
    assert kwargs["yarn_rotary_scaling_factor"] == 4.0
    assert kwargs["yarn_original_max_position_embeddings"] == 4096
    assert kwargs["yarn_beta_fast"] == 32.0
    assert kwargs["yarn_beta_slow"] == 1.0
    assert kwargs["yarn_correction_range_round_to_int"] is True


def test_yarn_mapping_preserves_explicit_keys():
    config = _yarn_qwen2_config(
        {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 16.0,
            "beta_slow": 2.0,
            "truncate": False,
        }
    )

    kwargs = DummyBridge().hf_config_to_provider_kwargs(config)

    assert kwargs["yarn_beta_fast"] == 16.0
    assert kwargs["yarn_beta_slow"] == 2.0
    assert kwargs["yarn_correction_range_round_to_int"] is False


def test_non_yarn_mapping_adds_no_yarn_defaults():
    kwargs = DummyBridge().hf_config_to_provider_kwargs(_yarn_qwen2_config(None))

    assert "position_embedding_type" not in kwargs
    assert not any(key.startswith("yarn_") for key in kwargs)


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(partial_rotary_factor=0.25, rope_parameters={"rope_theta": 10000.0}),
        SimpleNamespace(rope_parameters={"rope_theta": 10000.0, "partial_rotary_factor": 0.25}),
    ],
)
def test_partial_rotary_factor_from_hf_reads_attribute_or_rope_parameters(config):
    assert partial_rotary_factor_from_hf(config) == 0.25


def test_partial_rotary_factor_from_hf_raises_when_absent():
    with pytest.raises(ValueError, match="partial_rotary_factor not found"):
        partial_rotary_factor_from_hf(SimpleNamespace(rope_parameters={"rope_theta": 10000.0}))


def test_mapping_reads_partial_rotary_factor_from_hf_config():
    """Transformers 5.20+ keeps partial_rotary_factor only in rope_parameters, not as an attribute."""
    config = Qwen2Config(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, partial_rotary_factor=0.5)

    kwargs = DummyBridge().hf_config_to_provider_kwargs(config)

    assert kwargs["rotary_percent"] == 0.5
