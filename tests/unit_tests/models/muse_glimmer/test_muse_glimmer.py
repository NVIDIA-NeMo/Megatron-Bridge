# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Unit tests for the builder-backed Muse Glimmer implementation."""

from __future__ import annotations

import math
from collections.abc import Iterator
from unittest.mock import patch

import pytest
import torch
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.training.models.hybrid import HybridModelConfig

from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion.param_mapping import merge_qkv_weights, split_qkv_weights
from megatron.bridge.models.muse_glimmer import (
    MuseGlimmerConfig,
    MuseGlimmerModel,
    MuseGlimmerModelBuilder,
    MuseGlimmerModelConfig,
    MuseGlimmerTextConfig,
    MuseGlimmerVisionConfig,
)
from megatron.bridge.models.muse_glimmer.modeling_muse_glimmer import (
    MuseGlimmerCenteredRMSNorm,
    MuseGlimmerRMSNorm,
    MuseGlimmerVisionModel,
)
from megatron.bridge.models.muse_glimmer.muse_glimmer_bridge import (
    MuseGlimmerBridge,
    MuseGlimmerQKVGMapping,
)


pytestmark = pytest.mark.unit


def _tiny_hf_config() -> MuseGlimmerConfig:
    text_config = MuseGlimmerTextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        sliding_window=32,
        layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
        layer_rope_theta=[500_000.0, 500_000.0, 500_000.0, 0.0],
        rope_parameters={"rope_theta": 500_000.0, "rope_type": "default"},
        bos_token_id=100,
        eos_token_id=101,
        pad_token_id=0,
    )
    vision_config = MuseGlimmerVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        patch_size=2,
        patch_temporal=2,
        merge_size=2,
        pos_emb_height=4,
        pos_emb_width=4,
        max_position_embeddings=16,
        layer_types=["window_attention", "window_attention", "window_attention", "full_attention"],
        rope_parameters={"rope_theta": 10_000.0, "rope_type": "default"},
    )
    return MuseGlimmerConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=120,
        video_token_id=121,
        out_hidden_size=64,
        projector_hidden_size=32,
        projector_hidden_act="gelu",
        architectures=["MuseGlimmerForConditionalGeneration"],
        torch_dtype=torch.bfloat16,
    )


def test_config_conversion_uses_model_config_path_only() -> None:
    bridge = MuseGlimmerBridge()
    hf_config = _tiny_hf_config()

    with (
        patch.object(bridge, "provider_bridge", side_effect=AssertionError("legacy provider path used")),
        patch.object(
            bridge,
            "hf_config_to_provider_kwargs",
            side_effect=AssertionError("legacy provider kwargs path used"),
        ),
    ):
        model_config = bridge.hf_config_to_model_config(hf_config)

    assert isinstance(model_config, MuseGlimmerModelConfig)
    assert isinstance(model_config, HybridModelConfig)
    assert model_config.get_builder_cls() is MuseGlimmerModelBuilder
    assert model_config.num_layers == 4
    assert model_config.hidden_size == 64
    assert model_config.ffn_hidden_size == 128
    assert model_config.num_query_groups == 2
    assert model_config.kv_channels == 8
    assert model_config.softmax_scale == pytest.approx(3.87 / math.sqrt(8))
    assert model_config.window_size == (31, 0)
    assert model_config.window_attn_skip_freq == [True, True, True, False]
    assert model_config.no_rope_freq == [False, False, False, True]
    assert model_config.attention_output_gate is True
    assert model_config.qk_layernorm is True
    assert model_config.hybrid_layer_pattern == "****"
    assert model_config.special_token_ids == {"images": 120, "videos": 121}


def test_autobridge_selects_string_registration_and_serializes_config() -> None:
    auto_bridge = AutoBridge.from_hf_config(_tiny_hf_config())
    model_config = auto_bridge.get_model_config()

    assert isinstance(auto_bridge._model_bridge, MuseGlimmerBridge)
    assert isinstance(model_config, MuseGlimmerModelConfig)
    assert model_config.builder == "megatron.bridge.models.muse_glimmer.MuseGlimmerModelBuilder"

    restored = MuseGlimmerModelConfig.from_dict(model_config.as_dict())
    assert isinstance(restored, MuseGlimmerModelConfig)
    assert restored.vision.layer_types == model_config.vision.layer_types
    assert restored.transformer.softmax_scale == model_config.transformer.softmax_scale
    assert restored.get_builder_cls() is MuseGlimmerModelBuilder


def test_config_export_preserves_nested_muse_architecture() -> None:
    model_config = MuseGlimmerBridge().hf_config_to_model_config(_tiny_hf_config())

    exported = MuseGlimmerBridge.megatron_to_hf_config(model_config)

    assert exported["architectures"] == ["MuseGlimmerForConditionalGeneration"]
    assert exported["model_type"] == "muse_glimmer"
    assert exported["text_config"]["layer_types"] == [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]
    assert exported["text_config"]["layer_rope_theta"] == [500_000.0, 500_000.0, 500_000.0, 0]
    assert exported["vision_config"]["layer_types"][-1] == "full_attention"
    assert exported["out_hidden_size"] == 64


def test_full_head_gate_qkv_layout_round_trips() -> None:
    config = MuseGlimmerBridge().hf_config_to_model_config(_tiny_hf_config()).transformer
    query = torch.arange(32 * 64, dtype=torch.float32).reshape(32, 64)
    gate = query + 10_000
    key = torch.arange(16 * 64, dtype=torch.float32).reshape(16, 64) + 20_000
    value = key + 30_000

    query_with_gate = MuseGlimmerQKVGMapping._combine_query_and_gate(config, query, gate)
    packed = merge_qkv_weights(config, query_with_gate, key, value)
    restored_query_with_gate, restored_key, restored_value = split_qkv_weights(config, packed)
    restored_query, restored_gate = restored_query_with_gate.view(4, 16, 64).split(8, dim=1)

    torch.testing.assert_close(restored_query.reshape_as(query), query)
    torch.testing.assert_close(restored_gate.reshape_as(gate), gate)
    torch.testing.assert_close(restored_key, key)
    torch.testing.assert_close(restored_value, value)


def test_centered_and_final_norms_use_their_native_expressions() -> None:
    config = MuseGlimmerBridge().hf_config_to_model_config(_tiny_hf_config()).transformer
    hidden_states = torch.tensor([[0.25, -0.5, 1.0, -2.0]], dtype=torch.float32)
    centered = MuseGlimmerCenteredRMSNorm(config, hidden_size=4, eps=1e-5)
    final = MuseGlimmerRMSNorm(config, hidden_size=4, eps=1e-5)
    with torch.no_grad():
        centered.weight.fill_(0.25)
        final.weight.fill_(1.25)

    mean_square = hidden_states.pow(2).mean(dim=-1, keepdim=True)
    expected_centered = hidden_states * torch.rsqrt(mean_square + 1e-5) * 1.25
    expected_final = hidden_states * torch.pow(mean_square + 1e-5, -0.5) * 1.25

    torch.testing.assert_close(centered(hidden_states), expected_centered)
    torch.testing.assert_close(final(hidden_states), expected_final)


def test_centered_norm_sharded_state_dict_identifies_tensor_parallel_replicas() -> None:
    config = MuseGlimmerBridge().hf_config_to_model_config(_tiny_hf_config()).transformer
    norm = MuseGlimmerCenteredRMSNorm(config, hidden_size=4)
    tp_group = object()
    dp_cp_group = object()

    with (
        patch(
            "megatron.bridge.models.muse_glimmer.modeling_muse_glimmer.parallel_state.get_tensor_model_parallel_group",
            return_value=tp_group,
        ),
        patch(
            "megatron.bridge.models.muse_glimmer.modeling_muse_glimmer.make_sharded_tensors_for_checkpoint",
            return_value={"norm.weight": object()},
        ) as make_sharded,
    ):
        result = norm.sharded_state_dict(prefix="norm.", metadata={"dp_cp_group": dp_cp_group})

    assert set(result) == {"norm.weight"}
    make_sharded.assert_called_once_with(
        norm.state_dict(keep_vars=True),
        "norm.",
        sharded_offsets=(),
        tp_group=tp_group,
        dp_cp_group=dp_cp_group,
    )


def test_mapping_registry_covers_complete_checkpoint() -> None:
    registry = MuseGlimmerBridge().mapping_registry()

    qkvg = registry.megatron_to_hf_lookup("decoder.layers.2.self_attention.linear_qkv.weight")
    assert isinstance(qkvg, MuseGlimmerQKVGMapping)
    assert qkvg.hf_param["gate"] == "model.language_model.layers.2.self_attn.gate_proj.weight"
    assert (
        registry.megatron_to_hf_lookup("vision_tower.layers.1.attn.q_proj.bias").hf_param
        == "model.vision_tower.layers.1.attn.q_proj.bias"
    )
    assert registry.megatron_to_hf_lookup("vision_adapter.fc2.weight").hf_param == "model.vision_adapter.fc2.weight"


def test_tiny_vision_model_preserves_expected_token_count() -> None:
    vision_config = MuseGlimmerBridge().hf_config_to_model_config(_tiny_hf_config()).vision
    model = MuseGlimmerVisionModel(vision_config)
    pixel_values = torch.randn(16, 24)
    grid_thw = torch.tensor([[1, 4, 4]])

    output = model(pixel_values, grid_thw)

    assert output.shape == (4, 64)


@pytest.fixture(scope="module")
def tiny_hybrid_model() -> Iterator[MuseGlimmerModel]:
    auto_bridge = AutoBridge.from_hf_config(_tiny_hf_config())
    model_config = auto_bridge.get_model_config()
    model_config.use_cpu_initialization = True
    model_config.params_dtype = torch.float32
    model_config.bf16 = False
    model_config.bias_activation_fusion = False
    model_config.masked_softmax_fusion = False
    model_config.persist_layer_norm = False
    model_config.bias_dropout_fusion = False
    model_config.apply_rope_fusion = False
    model_config.gradient_accumulation_fusion = False
    model_config.cross_entropy_loss_fusion = False

    models = auto_bridge.get_model(
        model_config,
        load_weights=False,
        wrap_with_ddp=False,
        mixed_precision_wrapper=None,
    )
    assert len(models) == 1
    model = models[0]
    try:
        yield model
    finally:
        from megatron.core import parallel_state

        if parallel_state.is_initialized():
            parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def test_builder_constructs_native_hybrid_model_on_cpu(tiny_hybrid_model: MuseGlimmerModel) -> None:
    model = tiny_hybrid_model

    assert isinstance(model, MuseGlimmerModel)
    assert isinstance(model, HybridModel)
    assert not isinstance(model, GPTModel)
    assert model.hybrid_layer_pattern == "****"
    assert model.decoder.layer_type_list == ["*", "*", "*", "*"]
    names = dict(model.named_parameters())
    assert "vision_tower.patch_embedder.patch_embedding.weight" in names
    assert "decoder.layers.0.self_attention.post_layernorm.weight" in names
    assert "decoder.layers.0.mlp.post_layernorm.weight" in names
    assert not any(name.startswith("language_model.") for name in names)
    assert isinstance(model.decoder.final_norm, MuseGlimmerRMSNorm)
    assert names["vision_tower.patch_embedder.patch_embedding.weight"].dtype == torch.float32
    assert all(
        getattr(parameter, "average_gradients_across_tp_domain", False)
        for name, parameter in names.items()
        if name.startswith(("vision_tower.", "vision_adapter.", "vision_projection."))
    )

    inputs_embeds = torch.randn(2, 5, 64)
    input_ids = torch.randint(0, 100, (2, 5))
    expected_output = torch.randn(5, 2, 64)
    with (
        patch(
            "megatron.bridge.models.muse_glimmer.modeling_muse_glimmer.slice_batch_for_context_parallel",
            side_effect=lambda **kwargs: (
                kwargs["inputs_embeds"],
                kwargs["labels"],
                kwargs["loss_mask"],
                kwargs["position_ids"],
                kwargs["attention_mask"],
            ),
        ),
        patch.object(HybridModel, "forward", return_value=expected_output) as hybrid_forward,
    ):
        output = model(input_ids=input_ids, inputs_embeds=inputs_embeds)

    assert output is expected_output
    torch.testing.assert_close(hybrid_forward.call_args.kwargs["decoder_input"], inputs_embeds.transpose(0, 1))

    media_input_ids = input_ids.clone()
    media_input_ids[0, 1] = model.model_config.image_token_id
    media_input_ids[1, 3] = model.model_config.video_token_id
    embedding_output = torch.randn(5, 2, 64)
    with (
        patch.object(model.embedding, "forward", return_value=embedding_output) as embedding_forward,
        patch.object(HybridModel, "forward", return_value=expected_output),
        patch(
            "megatron.bridge.models.muse_glimmer.modeling_muse_glimmer.slice_batch_for_context_parallel",
            side_effect=lambda **kwargs: (
                kwargs["inputs_embeds"],
                kwargs["labels"],
                kwargs["loss_mask"],
                kwargs["position_ids"],
                kwargs["attention_mask"],
            ),
        ),
    ):
        model(input_ids=media_input_ids)

    expected_embedding_ids = media_input_ids.clone()
    expected_embedding_ids[expected_embedding_ids == model.model_config.image_token_id] = 0
    expected_embedding_ids[expected_embedding_ids == model.model_config.video_token_id] = 0
    torch.testing.assert_close(embedding_forward.call_args.kwargs["input_ids"], expected_embedding_ids)


def test_qkvg_mapping_executes_against_hybrid_qkv_module(tiny_hybrid_model: MuseGlimmerModel) -> None:
    model = tiny_hybrid_model
    registry = MuseGlimmerBridge().mapping_registry()
    for parameter_name, _ in model.named_parameters():
        assert registry.megatron_to_hf_lookup(parameter_name) is not None

    mapping = registry.megatron_to_hf_lookup("decoder.layers.0.self_attention.linear_qkv.weight")
    qkv_module = model.decoder.layers[0].self_attention.linear_qkv
    query = torch.arange(32 * 64, dtype=torch.float32).reshape(32, 64)
    gate = query + 10_000
    key = torch.arange(16 * 64, dtype=torch.float32).reshape(16, 64) + 20_000
    value = key + 30_000

    packed = mapping.hf_to_megatron({"q": query, "k": key, "v": value, "gate": gate}, qkv_module)
    exported = mapping.megatron_to_hf(packed, qkv_module)

    torch.testing.assert_close(exported["model.language_model.layers.0.self_attn.q_proj.weight"], query)
    torch.testing.assert_close(exported["model.language_model.layers.0.self_attn.k_proj.weight"], key)
    torch.testing.assert_close(exported["model.language_model.layers.0.self_attn.v_proj.weight"], value)
    torch.testing.assert_close(exported["model.language_model.layers.0.self_attn.gate_proj.weight"], gate)
