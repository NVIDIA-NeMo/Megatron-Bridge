# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Regression tests for GLM-5.3-Flash's section-wise KDA TP weight layout."""

from math import prod
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import pytest
import torch
from megatron.core.models.hybrid.layers.utils import create_layer_config

from megatron.bridge.models.conversion.param_mapping import HCAlphaMapping, MegatronParamMapping
from megatron.bridge.models.glm5next import glm53flash_bridge
from megatron.bridge.models.glm5next.glm53flash_bridge import (
    GLM53FlashBridge,
    _ColumnParallelConcatMapping,
    _EhProjSplitMapping,
)


pytestmark = pytest.mark.unit


def test_vlm_provider_keeps_vision_trainable_and_copies_multimodal_config():
    config_module = pytest.importorskip("transformers.models.glm5_next.configuration_glm5_next")
    config = config_module.Glm5NextConfig(
        text_config={
            "scoring_func": "sigmoid",
            "linear_attn_config": {"num_heads": 64, "head_dim": 128, "short_conv_kernel_size": 4},
        }
    )
    provider = GLM53FlashBridge().provider_bridge(SimpleNamespace(config=config))
    assert provider.vision_config is config.vision_config
    assert provider.spatial_merge_size == config.vision_config.spatial_merge_size
    assert not provider.freeze_vision_model
    assert not provider.freeze_vision_projection
    assert not provider.scatter_embedding_sequence_parallel
    for field in (
        "image_token_id",
        "video_token_id",
        "image_start_token_id",
        "image_end_token_id",
        "video_start_token_id",
        "video_end_token_id",
    ):
        assert getattr(provider, field) == getattr(config, field)


def test_vlm_mapping_prefixes_text_weights_and_nested_tp_mappings():
    config_module = pytest.importorskip("transformers.models.glm5_next.configuration_glm5_next")
    config = config_module.Glm5NextConfig(text_config={"scoring_func": "sigmoid"})
    bridge = GLM53FlashBridge()
    bridge.hf_config = config
    mappings = bridge.mapping_registry().mappings
    # The unified bridge prefixes every language-model weight with ``language_model.``
    # and appends a single replicated vision mapping at the end.
    assert mappings[-1].megatron_param == "visual.**"
    assert mappings[-1].hf_param == "model.visual.**"
    for mapping in mappings[:-1]:
        assert mapping.megatron_param.startswith("language_model.")
        if isinstance(mapping, _ColumnParallelConcatMapping):
            assert mapping._tp_mapping.megatron_param == mapping.megatron_param


def test_mtp_final_layernorm_round_trips_through_the_mtp_shared_head_norm():
    config_module = pytest.importorskip("transformers.models.glm5_next.configuration_glm5_next")
    config = config_module.Glm5NextConfig(
        text_config={
            "scoring_func": "sigmoid",
            "num_nextn_predict_layers": 1,
        }
    )
    bridge = GLM53FlashBridge()
    bridge.hf_config = config
    registry = bridge.mapping_registry()

    megatron_name = "language_model.mtp.layers.0.final_layernorm.weight"
    hf_name = f"model.language_model.layers.{config.text_config.num_hidden_layers}.shared_head.norm.weight"

    imported = registry.megatron_to_hf_lookup(megatron_name)
    assert imported is not None
    assert imported.hf_param == hf_name

    exported = registry.hf_to_megatron_lookup(hf_name)
    assert exported is not None
    assert exported.megatron_param == megatron_name


@pytest.mark.parametrize("path", ["model_config", "provider"])
@pytest.mark.parametrize("gate_lower_bound", [-5.0, -1.0])
@pytest.mark.parametrize("kda_disable_fp8", [False, True])
def test_glm5next_provider_precision_contract(gate_lower_bound, kda_disable_fp8, path):
    config_module = pytest.importorskip("transformers.models.glm5_next.configuration_glm5_next")
    config = config_module.Glm5NextConfig(
        text_config={
            "scoring_func": "sigmoid",
            "index_kpool": 4,
            "linear_attn_config": {
                "num_heads": 64,
                "head_dim": 128,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": gate_lower_bound,
            },
        }
    )
    if path == "model_config":
        provider = GLM53FlashBridge().hf_config_to_model_config(config)
        layer_config_source = provider.transformer
    else:
        provider = GLM53FlashBridge().provider_bridge(SimpleNamespace(config=config))
        layer_config_source = provider
    assert provider.layernorm_epsilon == 1e-5
    assert not provider.mla_proj_use_quantization
    assert provider.dsa_indexer_k_norm_epsilon == 1e-6
    assert not provider.dsa_indexer_qk_proj_use_quantization
    assert provider.dsa_indexer_kpool_use_quantization
    assert not provider.dsa_indexer_weights_proj_use_quantization
    assert provider.dsa_indexer_weights_proj_output_dtype == "fp32"
    assert create_layer_config(layer_config_source, "D").dsa_indexer_kpool_use_quantization
    assert provider.dsa_indexer_kpool_always_select_tail
    assert provider.mhc_norm_eps_inside_sqrt
    assert provider.mhc_keep_mappings_in_fp32
    assert not provider.mhc_learned_output_contract
    assert provider.kda_two_stage_gates
    assert provider.kda_disable_fp8
    assert provider.kda_lower_bound == gate_lower_bound
    provider.kda_disable_fp8 = kda_disable_fp8
    provider.finalize()
    kda_config = create_layer_config(layer_config_source, "K")
    assert kda_config.kda_disable_fp8 is kda_disable_fp8
    assert kda_config.kda_two_stage_gates
    assert kda_config.kda_safe_gate
    assert kda_config.kda_lower_bound == gate_lower_bound


@pytest.mark.parametrize("index", [0, 1, 2])
def test_hc_alpha_import_export_and_wildcard_resolution(index):
    attr = ("alpha_pre", "alpha_post", "alpha_res")[index]
    mapping = HCAlphaMapping(f"decoder.layers.*.hyper_connection.{attr}", "model.layers.*.hc_scale", index)
    mapping = mapping.resolve(("7",))
    assert mapping.megatron_param == f"decoder.layers.7.hyper_connection.{attr}"
    assert mapping.hf_param == "model.layers.7.hc_scale"
    module = torch.nn.Module()
    source = torch.tensor([0.1, 0.2, 0.3])
    for i, name in enumerate(("alpha_pre", "alpha_post", "alpha_res")):
        module.register_parameter(name, torch.nn.Parameter(source[i : i + 1].clone()))
    torch.testing.assert_close(mapping.hf_to_megatron(source, module), source[index : index + 1])
    with patch.object(mapping, "broadcast_from_pp_rank", side_effect=lambda weight, **_: weight):
        exported = mapping.megatron_to_hf(getattr(module, attr), module)
    if index == 0:
        torch.testing.assert_close(exported[mapping.hf_param], source)
    else:
        assert exported == {}


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("trailing_shape", [(3,), (1, 4)])
def test_kda_import_shards_each_projection_independently(tp_size, trailing_shape):
    mapping = _ColumnParallelConcatMapping("weight", ["q.weight", "k.weight", "v.weight"])
    sections = [
        torch.arange(rows * prod(trailing_shape)).reshape(rows, *trailing_shape) + offset
        for rows, offset in [(16, 0), (16, 1000), (32, 2000)]
    ]
    merged = torch.cat(sections)

    with patch.object(MegatronParamMapping, "tp_size", new_callable=PropertyMock, return_value=tp_size):
        shards = mapping._shard_per_rank(merged, [16, 16, 32])

    assert len(shards) == tp_size
    for rank, shard in enumerate(shards):
        expected = torch.cat(
            [
                section[rank * (len(section) // tp_size) : (rank + 1) * (len(section) // tp_size)]
                for section in sections
            ]
        )
        torch.testing.assert_close(shard, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("trailing_shape", [(3,), (1, 4)])
def test_kda_export_reconstructs_hf_order_from_rank_local_sections(tp_size, trailing_shape):
    mapping = _ColumnParallelConcatMapping("weight", ["q.weight", "k.weight", "v.weight"])
    module = torch.nn.Module()
    module.config = SimpleNamespace(
        linear_key_head_dim=2,
        linear_value_head_dim=4,
        linear_num_key_heads=8,
        linear_num_value_heads=8,
    )
    sections = [torch.randn(rows, *trailing_shape) for rows in (16, 16, 32)]
    # Build the distributed input independently of the import implementation.
    shards = [
        torch.cat(
            [section.narrow(0, rank * (len(section) // tp_size), len(section) // tp_size) for section in sections]
        )
        for rank in range(tp_size)
    ]

    with (
        patch.object(MegatronParamMapping, "tp_size", new_callable=PropertyMock, return_value=tp_size),
        patch.object(mapping._tp_mapping, "gather_from_tp_ranks", return_value=shards),
    ):
        exported = mapping.megatron_to_hf(shards[0], module)

    assert list(exported) == ["q.weight", "k.weight", "v.weight"]
    for name, expected in zip(exported, sections):
        torch.testing.assert_close(exported[name], expected, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Export under PP replication and Megatron-FSDP
# ---------------------------------------------------------------------------


def _kda_module():
    """KDA config with Q/K sections of 4 rows and an 8-row V section."""
    module = torch.nn.Module()
    module.config = SimpleNamespace(
        linear_key_head_dim=2,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
    )
    return module


def _kda_sections(trailing_shape=(1,)):
    return [
        (torch.arange(rows * prod(trailing_shape), dtype=torch.float32).reshape(rows, *trailing_shape) + offset)
        for rows, offset in [(4, 100), (4, 200), (8, 300)]
    ]


def _kda_rank_pieces(sections, tp_size):
    return [torch.cat([section.chunk(tp_size, dim=0)[rank] for section in sections], dim=0) for rank in range(tp_size)]


@pytest.mark.parametrize("trailing_shape", [(3,), (1, 4)])
def test_kda_export_fsdp_tp2_regroups_materialized_rank_pieces(trailing_shape):
    mapping = _ColumnParallelConcatMapping("weight", ["q.weight", "k.weight", "v.weight"])
    sections = _kda_sections(trailing_shape)
    # Megatron-FSDP materializes the rank-local [q_r | k_r | v_r] pieces in rank order.
    materialized = torch.cat(_kda_rank_pieces(sections, 2), dim=0)

    with (
        patch.object(MegatronParamMapping, "tp_size", new_callable=PropertyMock, return_value=2),
        patch.object(glm53flash_bridge, "_module_uses_fsdp", return_value=True),
        patch.object(mapping._tp_mapping, "gather_from_tp_ranks", side_effect=AssertionError("gathered twice")),
    ):
        exported = mapping.megatron_to_hf(materialized, _kda_module())

    assert list(exported) == ["q.weight", "k.weight", "v.weight"]
    for name, expected in zip(exported, sections):
        torch.testing.assert_close(exported[name], expected, rtol=0, atol=0)


def test_kda_pp_export_replicates_owner_tensor_and_layout():
    owner = _ColumnParallelConcatMapping("weight", ["q.weight", "k.weight", "v.weight"])
    non_owner = _ColumnParallelConcatMapping("weight", ["q.weight", "k.weight", "v.weight"])
    sections = _kda_sections()
    pieces = _kda_rank_pieces(sections, 2)
    broadcast = {}

    def owner_tensor(tensor, cache_key):
        broadcast["tensor"] = tensor
        return tensor

    def owner_obj(obj, cache_key):
        broadcast["layout"] = obj
        return obj

    with patch.object(MegatronParamMapping, "tp_size", new_callable=PropertyMock, return_value=2):
        with (
            patch.object(owner, "broadcast_from_pp_rank", side_effect=owner_tensor),
            patch.object(owner, "broadcast_obj_from_pp_rank", side_effect=owner_obj),
            patch.object(owner._tp_mapping, "gather_from_tp_ranks", return_value=pieces),
        ):
            owner_export = owner.megatron_to_hf(pieces[0], _kda_module())
        with (
            patch.object(
                non_owner, "broadcast_from_pp_rank", side_effect=lambda tensor, cache_key: broadcast["tensor"]
            ),
            patch.object(
                non_owner, "broadcast_obj_from_pp_rank", side_effect=lambda obj, cache_key: broadcast["layout"]
            ),
            patch.object(non_owner._tp_mapping, "gather_from_tp_ranks", return_value=pieces),
        ):
            non_owner_export = non_owner.megatron_to_hf(None, None)

    assert broadcast["layout"] == ([4, 4, 8], False)
    assert list(non_owner_export) == list(owner_export) == ["q.weight", "k.weight", "v.weight"]
    for name, expected in zip(owner_export, sections):
        torch.testing.assert_close(owner_export[name], expected, rtol=0, atol=0)
        torch.testing.assert_close(non_owner_export[name], expected, rtol=0, atol=0)


def test_custom_mappings_stay_stage_local_when_pp_broadcast_is_skipped():
    kda = _ColumnParallelConcatMapping("weight", ["q.weight", "k.weight", "v.weight"])
    eh = _EhProjSplitMapping("mtp.layers.0.e_proj.weight", "model.layers.9.eh_proj.weight", 0)
    with (
        patch.object(MegatronParamMapping, "pp_size", new_callable=PropertyMock, return_value=2),
        patch("torch.distributed.all_gather_object", side_effect=AssertionError("PP collective")),
        patch("torch.distributed.broadcast", side_effect=AssertionError("PP collective")),
        MegatronParamMapping.skip_pp_broadcast(),
    ):
        assert kda.megatron_to_hf(None, None) == {}
        assert eh.megatron_to_hf(None, None) == {}
        exported = kda.megatron_to_hf(torch.cat(_kda_sections()), _kda_module())
    assert list(exported) == ["q.weight", "k.weight", "v.weight"]


def _export_eh(order, e, h, *, tp_size, materialized, gather=None):
    mappings = {
        index: _EhProjSplitMapping(f"mtp.layers.0.{name}_proj.weight", "model.layers.9.eh_proj.weight", index)
        for index, name in ((0, "e"), (1, "h"))
    }
    weights = {0: e, 1: h}
    _EhProjSplitMapping._export_cache.clear()
    results = []
    with (
        patch.object(MegatronParamMapping, "tp_size", new_callable=PropertyMock, return_value=tp_size),
        patch.object(glm53flash_bridge, "_module_uses_fsdp", return_value=materialized),
    ):
        for index in order:
            mapping = mappings[index]
            local = weights[index] if gather is None else gather[index][0]
            side_effect = AssertionError("gathered twice") if gather is None else None
            with patch.object(
                mapping,
                "gather_from_tp_ranks",
                side_effect=side_effect,
                return_value=None if gather is None else gather[index],
            ):
                results.append(mapping.megatron_to_hf(local, torch.nn.Linear(1, 1)))
    return results


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_mtp_eh_export_fsdp_tp2_reconstructs_tp_once(order):
    hidden = 4
    e = torch.arange(hidden * hidden, dtype=torch.float32).reshape(hidden, hidden)
    h = e + 100
    first, second = _export_eh(order, e, h, tp_size=2, materialized=True)

    assert first == {}
    exported = second["model.layers.9.eh_proj.weight"]
    assert exported.shape == (hidden, 2 * hidden)
    torch.testing.assert_close(exported, torch.cat([e, h], dim=1), rtol=0, atol=0)


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_mtp_eh_export_tp2_gathers_plain_row_shards(order):
    hidden = 4
    e = torch.arange(hidden * hidden, dtype=torch.float32).reshape(hidden, hidden)
    h = e + 100
    gather = {0: list(e.chunk(2, dim=0)), 1: list(h.chunk(2, dim=0))}
    first, second = _export_eh(order, e, h, tp_size=2, materialized=False, gather=gather)

    assert first == {}
    torch.testing.assert_close(second["model.layers.9.eh_proj.weight"], torch.cat([e, h], dim=1), rtol=0, atol=0)


def test_mtp_eh_pp_export_replicates_owner_tensor():
    hidden = 4
    e = torch.randn(hidden, hidden)
    h = torch.randn(hidden, hidden)
    _EhProjSplitMapping._export_cache.clear()
    exported = {}
    for index, weight in ((0, e), (1, h)):
        mapping = _EhProjSplitMapping(f"mtp.layers.0.p{index}.weight", "model.layers.9.eh_proj.weight", index)
        with (
            patch.object(mapping, "broadcast_from_pp_rank", return_value=weight),
            patch.object(mapping, "broadcast_obj_from_pp_rank", return_value=False),
        ):
            exported = mapping.megatron_to_hf(None, None)
    torch.testing.assert_close(exported["model.layers.9.eh_proj.weight"], torch.cat([e, h], dim=1))


# ---------------------------------------------------------------------------
# FP8 source checkpoints
# ---------------------------------------------------------------------------


def test_fp8_source_import_dequantizes_and_export_drops_inverse_scales():
    bridge = GLM53FlashBridge()
    weight = torch.ones(128, 128).to(torch.float8_e4m3fn)
    state = {
        "attn.q_proj.weight": weight,
        "attn.q_proj.weight_scale_inv": torch.tensor([[0.5]]),
        "attn.k_proj.weight": weight,
        "attn.k_proj.weight_scale_inv": torch.tensor([[0.25]]),
        "attn.v_proj.weight": torch.full((128, 128), 2.0, dtype=torch.bfloat16),
    }
    single = bridge.maybe_modify_loaded_hf_weight("attn.q_proj.weight", state)
    assert single.dtype == torch.bfloat16
    torch.testing.assert_close(single, torch.full((128, 128), 0.5, dtype=torch.bfloat16))

    compound = bridge.maybe_modify_loaded_hf_weight(
        {"0": "attn.q_proj.weight", "1": "attn.k_proj.weight", "2": "attn.v_proj.weight"}, state
    )
    for key, value in (("0", 0.5), ("1", 0.25), ("2", 2.0)):
        torch.testing.assert_close(compound[key], torch.full((128, 128), value, dtype=torch.bfloat16))

    # Saved artifacts drop quantization_config, so export writes plain weights and
    # strict saving must not expect the source inverse scales.
    assert GLM53FlashBridge.DEQUANTIZED_SOURCE_KEY_SUFFIXES == ("_scale_inv",)
