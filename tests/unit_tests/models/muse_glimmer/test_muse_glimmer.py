# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Unit tests for the builder-backed Muse Glimmer implementation."""

from __future__ import annotations

import math
import os
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TEDotProductAttention, TENorm
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.training.models.hybrid import HybridModelConfig
from transformer_engine.pytorch import LayerNorm as TELayerNorm
from transformer_engine.pytorch import RMSNorm as TERMSNorm

from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion.param_mapping import (
    RMSNorm2ZeroCenteredRMSNormMapping,
    merge_qkv_weights,
    split_qkv_weights,
)
from megatron.bridge.models.muse_glimmer import (
    MuseGlimmerConfig,
    MuseGlimmerModel,
    MuseGlimmerModelBuilder,
    MuseGlimmerModelConfig,
    MuseGlimmerTextConfig,
    MuseGlimmerTransformerConfig,
    MuseGlimmerVisionConfig,
)
from megatron.bridge.models.muse_glimmer.modeling_muse_glimmer import (
    MuseGlimmerVisionAttention,
    MuseGlimmerVisionModel,
    get_muse_glimmer_hybrid_stack_spec,
)
from megatron.bridge.models.muse_glimmer.muse_glimmer_bridge import (
    MuseGlimmerBridge,
    MuseGlimmerQKVGMapping,
)
from megatron.bridge.training.utils.flop_utils import num_floating_point_operations, vit_flops_from_grid_thw


pytestmark = pytest.mark.unit

_NVTE_ATTENTION_ENV_VARS = ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN")


@pytest.fixture(scope="module", autouse=True)
def restore_nvte_attention_environment() -> Iterator[None]:
    """Keep MCore attention-backend selection local to this test module."""
    original_values = {name: os.environ.get(name) for name in _NVTE_ATTENTION_ENV_VARS}
    yield
    for name, value in original_values.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


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
        tie_word_embeddings=False,
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
    assert model_config.transformer_impl == "local"
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
    assert model_config.vision_config is model_config.vision
    assert model_config.vision_config.depth == 4
    assert model_config.vision_config.spatial_merge_size == 2


def test_model_config_owns_transformer_type_and_routes_flat_overrides() -> None:
    model_config = MuseGlimmerBridge().hf_config_to_model_config(_tiny_hf_config())

    assert MuseGlimmerModelConfig.transformer_config_class is MuseGlimmerTransformerConfig
    assert "TRANSFORMER_CONFIG_CLASS" not in MuseGlimmerBridge.__dict__
    assert MuseGlimmerBridge.USE_MODEL_CONFIG_FOR_CONVERSION is True

    model_config.tensor_model_parallel_size = 8
    model_config.seq_length = 512

    assert model_config.transformer.tensor_model_parallel_size == 8
    assert "tensor_model_parallel_size" not in model_config.__dict__
    assert model_config.seq_length == 512
    assert "seq_length" in model_config.__dict__

    with pytest.raises(AttributeError, match="declares a field"):
        model_config.tesnor_model_parallel_size = 4


def test_config_conversion_uses_top_level_embedding_sharing_contract() -> None:
    """The VLM config, not its nested text config, owns HF weight tying."""
    hf_config = _tiny_hf_config()
    hf_config.tie_word_embeddings = True
    hf_config.text_config.tie_word_embeddings = False

    model_config = MuseGlimmerBridge().hf_config_to_model_config(hf_config)

    assert model_config.share_embeddings_and_output_weights is True


def test_config_conversion_falls_back_to_nested_embedding_sharing_contract() -> None:
    """Released Muse configs without a top-level field retain text weight tying."""
    hf_config = _tiny_hf_config()
    del hf_config.tie_word_embeddings
    hf_config.text_config.tie_word_embeddings = True

    model_config = MuseGlimmerBridge().hf_config_to_model_config(hf_config)

    assert model_config.share_embeddings_and_output_weights is True


def test_vision_config_contributes_to_runtime_flops() -> None:
    model_config = MuseGlimmerBridge().hf_config_to_model_config(_tiny_hf_config())
    cfg = SimpleNamespace(model=model_config)

    vision_flops = vit_flops_from_grid_thw(cfg, torch.tensor([[1, 4, 4]], dtype=torch.int64))

    assert vision_flops > 0


def test_hybrid_attention_layers_use_full_transformer_flops() -> None:
    model_config = MuseGlimmerBridge().hf_config_to_model_config(_tiny_hf_config())
    cfg = SimpleNamespace(model=model_config)
    batch_size = 8
    sequence_length = model_config.seq_length

    muse_flops = num_floating_point_operations(
        cfg,
        batch_size=batch_size,
        seqlen_sum=batch_size * sequence_length,
        seqlen_squared_sum=batch_size * sequence_length**2,
    )

    model_config.hybrid_attention_layers_include_mlp = False
    model_config.hybrid_layer_pattern = None
    transformer_flops = num_floating_point_operations(
        cfg,
        batch_size=batch_size,
        seqlen_sum=batch_size * sequence_length,
        seqlen_squared_sum=batch_size * sequence_length**2,
    )

    assert muse_flops == transformer_flops


@pytest.mark.parametrize(
    ("transformer_impl", "expected_qkv", "expected_attention", "expected_norm"),
    [
        ("local", ColumnParallelLinear, DotProductAttention, TENorm),
        ("transformer_engine", TEColumnParallelLinear, TEDotProductAttention, TENorm),
    ],
)
def test_decoder_stack_honors_transformer_backend(
    transformer_impl: str,
    expected_qkv: type,
    expected_attention: type,
    expected_norm: type,
) -> None:
    config = MuseGlimmerBridge().hf_config_to_model_config(_tiny_hf_config()).transformer
    config.transformer_impl = transformer_impl

    stack_spec = get_muse_glimmer_hybrid_stack_spec(config)
    attention = stack_spec.submodules.attention_layer.submodules.self_attention.submodules

    assert attention.linear_qkv is expected_qkv
    assert attention.core_attention is expected_attention
    assert stack_spec.submodules.attention_layer.submodules.input_layernorm is expected_norm
    assert stack_spec.submodules.attention_layer.submodules.pre_mlp_layernorm is expected_norm


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


def test_model_config_restores_persisted_dtype_and_enum_values() -> None:
    serialized = MuseGlimmerBridge().hf_config_to_model_config(_tiny_hf_config()).as_dict()
    serialized["transformer"]["params_dtype"] = {
        "_target_": "torch.bfloat16",
        "_call_": False,
    }
    serialized["transformer"]["attention_backend"] = {
        "_target_": "megatron.core.transformer.enums.AttnBackend",
        "_call_": True,
        "_args_": [5],
        "_name_": "auto",
    }

    restored = MuseGlimmerModelConfig.from_dict(serialized)

    assert restored.transformer.params_dtype is torch.bfloat16
    assert restored.transformer.attention_backend.name == "auto"


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


def test_affine_norms_use_transformer_engine_directly() -> None:
    model_config = MuseGlimmerBridge().hf_config_to_model_config(_tiny_hf_config())
    config = model_config.transformer
    config.use_cpu_initialization = True
    config.params_dtype = torch.float32

    layer_submodules = get_muse_glimmer_hybrid_stack_spec(config).submodules.attention_layer.submodules
    centered_norm = layer_submodules.input_layernorm(config, config.hidden_size, eps=config.layernorm_epsilon)
    vision_model = MuseGlimmerVisionModel(model_config.vision, config)

    assert layer_submodules.input_layernorm is TENorm
    assert layer_submodules.pre_mlp_layernorm is TENorm
    assert isinstance(centered_norm, TERMSNorm)
    assert centered_norm.zero_centered_gamma is True
    assert isinstance(vision_model.ln_pre, TELayerNorm)
    assert isinstance(vision_model.ln_post, TELayerNorm)
    assert all(isinstance(layer.norm1, TELayerNorm) for layer in vision_model.layers)
    assert all(isinstance(layer.norm2, TELayerNorm) for layer in vision_model.layers)
    assert set(vision_model.ln_pre.state_dict()) == {"weight", "bias", "_extra_state"}

    registry = MuseGlimmerBridge().mapping_registry()
    assert isinstance(
        registry.megatron_to_hf_lookup("decoder.final_norm.weight"),
        RMSNorm2ZeroCenteredRMSNormMapping,
    )
    for name, _ in vision_model.named_parameters():
        assert registry.megatron_to_hf_lookup(f"vision_tower.{name}") is not None


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


@pytest.mark.run_only_on("GPU")
def test_tiny_vision_model_preserves_expected_token_count() -> None:
    model_config = MuseGlimmerBridge().hf_config_to_model_config(_tiny_hf_config())
    model_config.transformer_impl = "local"
    model_config.params_dtype = torch.float32
    vision_config = model_config.vision
    pixel_values = torch.randn(16, 24, device="cuda")
    grid_thw = torch.tensor([[1, 4, 4]], device="cuda")

    model = MuseGlimmerVisionModel(vision_config, model_config.transformer).cuda()
    output = model(pixel_values, grid_thw)

    assert output.shape == (4, 64)


def test_vision_attention_matches_eager_reference_with_finite_backward() -> None:
    vision_config = MuseGlimmerBridge().hf_config_to_model_config(_tiny_hf_config()).vision
    attention = MuseGlimmerVisionAttention(vision_config)
    hidden_states = torch.randn(5, vision_config.hidden_size, requires_grad=True)
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
    cosine = torch.ones(5, attention.head_dim)
    sine = torch.zeros_like(cosine)
    real_softmax = torch.nn.functional.softmax

    with (
        patch(
            "megatron.bridge.models.muse_glimmer.modeling_muse_glimmer.F.softmax",
            wraps=real_softmax,
        ) as softmax,
        patch(
            "megatron.bridge.models.muse_glimmer.modeling_muse_glimmer.F.scaled_dot_product_attention",
            side_effect=AssertionError("Muse vision attention must use the HF eager path"),
        ),
    ):
        output = attention(hidden_states, cu_seqlens, (cosine, sine))

    query = attention.q_proj(hidden_states).reshape(5, attention.num_heads, attention.head_dim)
    key = attention.k_proj(hidden_states).reshape(5, attention.num_heads, attention.head_dim)
    value = attention.v_proj(hidden_states).reshape(5, attention.num_heads, attention.head_dim)
    expected_chunks = []
    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        q = query[start:end].transpose(0, 1).unsqueeze(0)
        k = key[start:end].transpose(0, 1).unsqueeze(0)
        v = value[start:end].transpose(0, 1).unsqueeze(0)
        weights = torch.matmul(q, k.transpose(-2, -1)) * attention.scaling
        weights = real_softmax(weights, dim=-1, dtype=torch.float32).to(q.dtype)
        expected_chunks.append(torch.matmul(weights, v).squeeze(0).transpose(0, 1))
    expected = attention.proj(torch.cat(expected_chunks, dim=0).reshape(5, -1))

    assert softmax.call_count == 2
    assert all(call.kwargs["dtype"] is torch.float32 for call in softmax.call_args_list)
    torch.testing.assert_close(output, expected)
    output.square().mean().backward()
    assert torch.isfinite(hidden_states.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in attention.parameters()
    )


@pytest.mark.run_only_on("GPU")
def test_vision_layer_recomputation_is_training_only() -> None:
    model_config = MuseGlimmerBridge().hf_config_to_model_config(_tiny_hf_config())
    model_config.transformer_impl = "local"
    model_config.params_dtype = torch.float32
    vision_config = model_config.vision
    pixel_values = torch.randn(16, 24, device="cuda")
    grid_thw = torch.tensor([[1, 4, 4]], device="cuda")

    with patch(
        "megatron.bridge.models.muse_glimmer.modeling_muse_glimmer.torch_checkpoint",
        side_effect=lambda function, *args, **kwargs: function(*args),
    ) as checkpoint:
        model = MuseGlimmerVisionModel(vision_config, model_config.transformer, recompute_layers=True).cuda()
        training_output = model(pixel_values, grid_thw)
        model.eval()
        inference_output = model(pixel_values, grid_thw)

    assert checkpoint.call_count == vision_config.num_hidden_layers
    torch.testing.assert_close(training_output, inference_output)


@pytest.fixture(scope="module")
def tiny_hybrid_model() -> Iterator[MuseGlimmerModel]:
    auto_bridge = AutoBridge.from_hf_config(_tiny_hf_config())
    model_config = auto_bridge.get_model_config()
    model_config.use_cpu_initialization = True
    model_config.transformer_impl = "local"
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
    assert isinstance(model.decoder.layers[0].self_attention.q_layernorm, torch.nn.RMSNorm)
    assert isinstance(model.decoder.layers[0].self_attention.k_layernorm, torch.nn.RMSNorm)
    assert dict(model.decoder.layers[0].self_attention.q_layernorm.named_parameters()) == {}
    assert dict(model.decoder.layers[0].self_attention.k_layernorm.named_parameters()) == {}
    assert isinstance(model.decoder.final_norm, TERMSNorm)
    assert model.decoder.final_norm.zero_centered_gamma is True
    assert isinstance(model.vision_tower.ln_pre, TELayerNorm)
    assert isinstance(model.vision_tower.layers[0].norm1, TELayerNorm)
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


def test_checkpoint_schema_omits_only_empty_backend_norm_state(tiny_hybrid_model: MuseGlimmerModel) -> None:
    norm_weight = SimpleNamespace(replica_id=(0, 0, 0))
    state = {
        "decoder.layers.0.input_layernorm._extra_state": SimpleNamespace(data=torch.empty(0)),
        "decoder.layers.0.self_attention.post_layernorm._extra_state": SimpleNamespace(data=None),
        "decoder.layers.0.pre_mlp_layernorm._extra_state": SimpleNamespace(data=None),
        "decoder.layers.0.mlp.post_layernorm._extra_state": SimpleNamespace(data=None),
        "decoder.final_norm._extra_state": SimpleNamespace(data=None),
        "vision_tower.ln_pre._extra_state": SimpleNamespace(data=None),
        "vision_tower.layers.0.norm1._extra_state": SimpleNamespace(data=None),
        "decoder.layers.0.self_attention.linear_qkv._extra_state": SimpleNamespace(data={"scale": 1}),
        "decoder.layers.0.input_layernorm.weight": norm_weight,
    }

    with (
        patch.object(HybridModel, "sharded_state_dict", return_value=state),
        patch(
            "megatron.bridge.models.muse_glimmer.modeling_muse_glimmer.get_pg_rank",
            side_effect=[3, 4],
        ),
    ):
        result = tiny_hybrid_model.sharded_state_dict()

    assert "decoder.layers.0.input_layernorm._extra_state" not in result
    assert "decoder.layers.0.self_attention.post_layernorm._extra_state" not in result
    assert "decoder.layers.0.pre_mlp_layernorm._extra_state" not in result
    assert "decoder.layers.0.mlp.post_layernorm._extra_state" not in result
    assert "decoder.final_norm._extra_state" not in result
    assert "vision_tower.ln_pre._extra_state" not in result
    assert "vision_tower.layers.0.norm1._extra_state" not in result
    assert "decoder.layers.0.self_attention.linear_qkv._extra_state" in result
    assert "decoder.layers.0.input_layernorm.weight" in result
    assert norm_weight.replica_id == (0, 3, 4)


def test_checkpoint_schema_rejects_stateful_backend_norm_state(tiny_hybrid_model: MuseGlimmerModel) -> None:
    state = {
        "decoder.layers.0.input_layernorm._extra_state": SimpleNamespace(data=torch.ones(1)),
    }

    with (
        patch.object(HybridModel, "sharded_state_dict", return_value=state),
        patch(
            "megatron.bridge.models.muse_glimmer.modeling_muse_glimmer.get_pg_rank",
            side_effect=[3, 4],
        ),
        pytest.raises(ValueError, match="Transformer Engine norm extra state must be empty"),
    ):
        tiny_hybrid_model.sharded_state_dict()


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
