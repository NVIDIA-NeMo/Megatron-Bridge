# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Regression tests for GLM-5.3-Flash's section-wise KDA TP weight layout."""

from math import prod
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import pytest
import torch
from megatron.core.models.hybrid.layers.utils import create_layer_config

from megatron.bridge.models.conversion.param_mapping import HCAlphaMapping, MegatronParamMapping
from megatron.bridge.models.glm5next.glm53flash_bridge import GLM53FlashBridge, _ColumnParallelConcatMapping


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


@pytest.mark.parametrize("gate_lower_bound", [-5.0, -1.0])
def test_glm53_provider_precision_contract(gate_lower_bound):
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
    provider = GLM53FlashBridge().provider_bridge(SimpleNamespace(config=config))
    assert provider.layernorm_epsilon == 1e-5
    assert provider.dsa_indexer_k_norm_epsilon == 1e-6
    assert provider.dsa_indexer_kpool_fp8
    assert create_layer_config(provider, "D").dsa_indexer_kpool_fp8
    assert provider.mhc_norm_eps_inside_sqrt
    assert provider.mhc_keep_mappings_in_fp32
    assert not provider.mhc_learned_output_contract
    assert provider.kda_two_stage_gates
    assert provider.kda_lower_bound == gate_lower_bound


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
