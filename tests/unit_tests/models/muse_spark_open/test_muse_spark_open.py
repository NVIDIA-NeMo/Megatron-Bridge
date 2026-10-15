# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Weight-independent checks for Hybrid schedules and the missing operations."""

import copy
import math
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F
from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.utils import is_layer_window_attention
from torch import nn
from transformers import PretrainedConfig

from megatron.bridge.models.muse_spark_open import (
    MuseSparkBridge,
    MuseSparkModel,
    MuseSparkModelBuilder,
    MuseSparkModelConfig,
)
from megatron.bridge.models.muse_spark_open.modeling_muse_spark_open.blocks import (
    InputScaledRMSNorm,
    MuseSparkExperts,
    MuseSparkResidualGate,
    MuseSparkRouter,
    MuseSparkSelfAttention,
)


pytestmark = pytest.mark.unit


def test_fp32_router_casts_bf16_activations_for_native_gemm():
    router = MuseSparkRouter.__new__(MuseSparkRouter)
    nn.Module.__init__(router)
    router.weight = nn.Parameter(torch.randn(4, 8, dtype=torch.float32))
    router.config = SimpleNamespace(output_multiplier=0.25)
    inputs = torch.randn(3, 8, dtype=torch.bfloat16, requires_grad=True)

    def native_gating(module, values):
        assert values.dtype == module.weight.dtype
        return F.linear(values, module.weight)

    with patch.object(TopKRouter, "gating", native_gating):
        actual = router.gating(inputs)
    expected = F.linear(inputs.float(), router.weight) * 0.25
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    assert inputs.grad is not None


def test_router_correction_buffer_retains_fp32_values_during_conversion():
    router = MuseSparkRouter.__new__(MuseSparkRouter)
    nn.Module.__init__(router)
    values = torch.tensor([0.1234567, -0.9876543], dtype=torch.float32)
    router.register_buffer("expert_bias", values.clone())
    # A device move replaces the buffer object before Float16Module converts it.
    router._apply(lambda tensor: tensor.clone())
    convert_module_to_dtype_except_fp32_marked(router, torch.bfloat16)
    assert router.expert_bias.dtype == torch.float32
    torch.testing.assert_close(router.expert_bias, values, rtol=0, atol=0)


@pytest.fixture
def hf_config():
    text = PretrainedConfig(
        num_hidden_layers=6,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=128,
        max_position_embeddings=256,
        layer_types=["sliding_attention", "full_attention"] + ["sliding_attention"] * 3 + ["full_attention"],
        layer_rope_theta=[500000, 0, 500000, 500000, 500000, 0],
        mlp_layer_types=["sparse"] * 6,
        rope_parameters={"rope_type": "default", "rope_theta": 500000},
        sliding_window=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=0,
        moe_hidden_size=16,
        moe_intermediate_size=32,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        post_norm_eps=1e-8,
        residual_gate_temperature=0.3,
        attention_bias=False,
        attention_dropout=0.0,
        qk_scale_factor=2.0,
        routed_scaling_factor=1.0,
        output_multiplier=0.25,
        final_logit_softcapping=20.0,
        initializer_range=0.02,
        tie_word_embeddings=False,
    )
    return PretrainedConfig(
        architectures=["MuseSparkForConditionalGeneration"],
        text_config=text,
        vision_config={"hidden_size": 16},
        audio_config={"out_hidden_size": 32},
        image_token_id=120,
        video_token_id=121,
        audio_token_id=122,
        tie_word_embeddings=False,
    )


def test_hybrid_schedule_uses_attention_positions(hf_config):
    config = MuseSparkBridge().hf_config_to_model_config(hf_config)
    config.finalize()
    assert config.num_layers == 12
    assert config.hybrid_layer_pattern == "*E" * 6
    assert config.num_query_groups == 2
    assert config.attention_output_gate
    assert config.position_embedding_type == "rope"
    assert config.softmax_scale == pytest.approx(2 / math.sqrt(8))
    assert config.window_attn_skip_freq == [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0]
    attention_positions = range(1, 13, 2)
    actual = [
        is_layer_window_attention(config.window_size, config.window_attn_skip_freq, i) for i in attention_positions
    ]
    assert actual == [True, False, True, True, True, False]
    assert config.no_rope_freq[::2] == [0, 1, 0, 0, 0, 1]
    # MCore constructs new per-layer config types; custom fields must survive.
    layers = validate_segment_layers(config.hybrid_layer_pattern, config.transformer)
    assert all(layer.output_multiplier == 0.25 for layer in layers)
    assert all(layer.post_norm_epsilon == 1e-8 for layer in layers)


def test_config_roundtrip_preserves_perception_and_builder(hf_config):
    config = MuseSparkBridge().hf_config_to_model_config(hf_config)
    original = copy.deepcopy(config.hf_config)
    restored = MuseSparkModelConfig.from_dict(config.as_dict())
    assert restored.get_builder_cls() is MuseSparkModelBuilder
    assert restored.transformer.activation_func is F.silu
    assert restored.special_token_ids == {"images": 120, "videos": 121, "audio": 122}
    exported = MuseSparkBridge.megatron_to_hf_config(restored)
    for key in ("layer_types", "layer_rope_theta", "num_hidden_layers", "moe_hidden_size", "qk_scale_factor"):
        assert exported["text_config"][key] == original["text_config"][key]
    assert exported["vision_config"] == original["vision_config"]
    assert exported["audio_config"] == original["audio_config"]
    assert config.hf_config == original


@pytest.mark.parametrize("field,value", [("expert_tensor_parallel_size", 2), ("cuda_graph_impl", "local")])
def test_unsupported_execution_modes_fail_before_allocation(hf_config, field, value):
    config = MuseSparkBridge().hf_config_to_model_config(hf_config)
    assert hasattr(config.transformer, field)
    setattr(config.transformer, field, value)
    with pytest.raises(ValueError):
        config.finalize()


def test_bad_decoder_schedule_is_rejected(hf_config):
    hf_config.text_config.layer_types.pop()
    with pytest.raises(ValueError, match="schedules"):
        MuseSparkBridge().hf_config_to_model_config(hf_config)


def test_mappings_separate_attention_and_moe_positions(hf_config):
    bridge = MuseSparkBridge()
    bridge.hf_config = hf_config
    registry = bridge.mapping_registry()
    attention = registry.megatron_to_hf_lookup("decoder.layers.2.self_attention.linear_qkv.weight")
    assert attention.hf_param["q"] == "model.language_model.layers.1.self_attn.q_proj.weight"
    expert_norm = registry.megatron_to_hf_lookup("decoder.layers.3.mlp.experts.post_expert_norm.weight")
    assert expert_norm.hf_param == "model.language_model.layers.1.mlp.experts.post_expert_norm.weight"
    expert = registry.megatron_to_hf_lookup("decoder.layers.3.mlp.experts.linear_fc1.weight2")
    assert expert is not None
    audio = registry.megatron_to_hf_lookup("audio_tower.downsample_layers.0.0.weight")
    assert audio.hf_param == "model.audio_tower.downsample_layers.0.0.weight"
    assert registry.megatron_to_hf_lookup("decoder.layers.2.mlp.router.weight") is None


def test_input_scaling_is_inside_normalization():
    config = SimpleNamespace(params_dtype=torch.float64)
    norm = InputScaledRMSNorm(2, eps=1e-8, config=config, sequence_parallel=True)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, 0.0]))
    x = torch.tensor([[1.0, 1.0]], dtype=torch.float64, requires_grad=True)
    expected = torch.tensor([[2.0, 1.0]], dtype=torch.float64) / math.sqrt(2.5 + 1e-8)
    torch.testing.assert_close(norm(x), expected)
    assert not torch.allclose(norm(x), F.rms_norm(x, (2,), eps=1e-8) * (1 + norm.weight))
    assert norm.weight.sequence_parallel
    assert torch.autograd.gradcheck(norm, (x,))


def test_residual_gate_zero_and_saturation():
    config = SimpleNamespace(
        residual_gate_temperature=0.3,
        hidden_size=2,
        sequence_parallel=True,
        perform_initialization=False,
        post_norm_epsilon=1e-8,
        params_dtype=torch.float32,
    )
    gate = MuseSparkResidualGate(config, input_scaled_norm=False)
    residual = torch.tensor([[2.0, -3.0]], requires_grad=True)
    branch = torch.tensor([[1.0, 2.0]], requires_grad=True)
    normalized = F.rms_norm(branch, (2,), eps=1e-8)
    actual = gate(True, False)((branch, None), residual, 0.0)
    torch.testing.assert_close(actual, math.sqrt(0.75) * residual + 0.5 * normalized)
    actual.sum().backward()
    assert gate.gate.grad is not None and torch.isfinite(gate.gate.grad).all()
    assert gate.gate.keep_in_fp32
    with torch.no_grad():
        gate.gate.fill_(100.0)
    torch.testing.assert_close(
        gate(False, False)((branch, None), residual, 0.0), math.sqrt(0.001) * residual + normalized
    )


def test_expert_normalization_precedes_routing_weight():
    experts = MuseSparkExperts.__new__(MuseSparkExperts)
    nn.Module.__init__(experts)
    experts.post_expert_norm = InputScaledRMSNorm(
        2, eps=1e-8, config=SimpleNamespace(params_dtype=torch.float32), sequence_parallel=True
    )
    raw = torch.tensor([[3.0, 4.0], [3.0, 4.0]], requires_grad=True)
    probs = torch.tensor([0.01, 0.9], requires_grad=True)
    tokens = torch.tensor([1, 1])
    with patch.object(TEGroupedMLP, "forward", return_value=(raw, None)) as grouped:
        actual, bias = experts(raw, tokens, probs)
    torch.testing.assert_close(grouped.call_args.args[2], torch.ones_like(probs))
    expected = F.rms_norm(raw, (2,), eps=1e-8) * probs[:, None]
    torch.testing.assert_close(actual, expected)
    assert bias is None
    actual.sum().backward()
    assert probs.grad is not None and (probs.grad != 0).all()
    assert experts.post_expert_norm.weight.grad is not None


def test_attention_output_norm_precedes_sigmoid_gate():
    attention = MuseSparkSelfAttention.__new__(MuseSparkSelfAttention)
    nn.Module.__init__(attention)
    attention.config = SimpleNamespace(kv_channels=2)
    attention.output_norm = nn.RMSNorm(2, eps=1e-5, elementwise_affine=False)
    x = torch.tensor([[[3.0, 4.0, 0.0, 2.0]]])
    gate = torch.tensor([[[0.0, 1.0, 2.0, -2.0]]])
    expected = F.rms_norm(x.reshape(1, 1, 2, 2), (2,), eps=1e-5).reshape_as(x) * torch.sigmoid(gate)
    torch.testing.assert_close(attention._apply_output_gate(x, gate), expected)


def test_shared_expert_norm_checkpoint_uses_ordinary_tp_replicas():
    experts = MuseSparkExperts.__new__(MuseSparkExperts)
    nn.Module.__init__(experts)
    experts.post_expert_norm = nn.Identity()
    experts.attention_tp_group = object()
    metadata = {"dp_cp_group": object()}
    key = "decoder.layers.1.mlp.experts.post_expert_norm.weight"
    with (
        patch.object(TEGroupedMLP, "sharded_state_dict", return_value={key: "expert-tp-replica"}),
        patch(
            "megatron.bridge.models.muse_spark_open.modeling_muse_spark_open.blocks.sharded_state_dict_default",
            return_value={key: "ordinary-tp-replica"},
        ) as shared_state,
    ):
        state = experts.sharded_state_dict("decoder.layers.1.mlp.experts.", (), metadata)
    assert state[key] == "ordinary-tp-replica"
    assert shared_state.call_args.kwargs["tp_group"] is experts.attention_tp_group
    assert shared_state.call_args.args[3] is metadata


def test_media_features_call_hf_towers_and_trim_audio_padding():
    model = MuseSparkModel.__new__(MuseSparkModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=2, layernorm_epsilon=1e-5)
    model.audio_samples_per_token = 4
    audio = torch.arange(1, 13, dtype=torch.float32).reshape(2, 3, 2)
    model.audio_tower = Mock(return_value=SimpleNamespace(last_hidden_state=audio))
    waveforms = torch.zeros(2, 12)
    mask = torch.tensor([[1] * 5 + [0] * 7, [1] * 9 + [0] * 3])
    actual = model._audio_features(waveforms, mask)
    expected = torch.cat([audio[0, :2], audio[1, :3]])
    torch.testing.assert_close(actual, F.rms_norm(expected, (2,), eps=1e-5))
    model.audio_tower.assert_called_once_with(input_values=waveforms)
    pixels = torch.ones(3, 2)
    grid = torch.tensor([[1, 2, 2]])
    model.vision_tower = Mock(return_value=SimpleNamespace(last_hidden_state=pixels))
    model._restore_hf_vision_rope_buffers = Mock()
    model.vision_adapter = nn.Identity()
    model.vision_projection = nn.Identity()
    torch.testing.assert_close(model._vision_features(pixels, grid), F.rms_norm(pixels, (2,), eps=1e-5))
    model.vision_tower.assert_called_once_with(pixel_values=pixels, grid_thw=grid)
    model._restore_hf_vision_rope_buffers.assert_called_once_with()


def test_vision_frequencies_are_rebuilt_by_hf_after_bf16_cast():
    model = MuseSparkModel.__new__(MuseSparkModel)
    nn.Module.__init__(model)
    expected = torch.tensor([1.0, 0.1234567, 0.0012345], dtype=torch.float32)
    rotary = SimpleNamespace(
        config=object(), inv_freq=expected.bfloat16(), compute_axial_rope_parameters=Mock(return_value=(expected, 1.0))
    )
    model.vision_tower = SimpleNamespace(rotary_emb=rotary)
    model._restore_hf_vision_rope_buffers()
    torch.testing.assert_close(rotary.inv_freq, expected, rtol=0, atol=0)
    torch.testing.assert_close(rotary.original_inv_freq, expected, rtol=0, atol=0)
    assert rotary.inv_freq.dtype == torch.float32
    assert rotary.attention_scaling == 1.0
    rotary.compute_axial_rope_parameters.assert_called_once_with(rotary.config, device=torch.device("cpu"))


def test_media_placeholder_count_is_checked():
    with pytest.raises(ValueError, match="placeholders"):
        MuseSparkModel._insert_features(torch.zeros(1, 3, 2), torch.tensor([[1, 2, 3]]), torch.ones(2, 2), 2)


def test_media_ids_are_replaced_before_text_embedding():
    model = MuseSparkModel.__new__(MuseSparkModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=2, layernorm_epsilon=1e-5, sequence_parallel=False)
    model.model_config = SimpleNamespace(
        hf_model_text_only=False, special_token_ids={"images": 120, "videos": 121, "audio": 122}
    )
    model.pre_process = True
    model.pg_collection = SimpleNamespace()
    model.embedding = Mock(return_value=torch.ones(4, 1, 2))
    ids = torch.tensor([[1, 120, 121, 122]])
    with (
        patch(
            "megatron.bridge.models.muse_spark_open.modeling_muse_spark_open.model.slice_batch_for_context_parallel",
            side_effect=lambda **kwargs: tuple(
                kwargs[key] for key in ("inputs_embeds", "labels", "loss_mask", "position_ids", "attention_mask")
            ),
        ),
        patch.object(HybridModel, "forward", return_value=torch.ones(1, 4, 2)),
    ):
        model.forward(input_ids=ids)
    torch.testing.assert_close(model.embedding.call_args.kwargs["input_ids"], torch.tensor([[1, 0, 0, 0]]))
    torch.testing.assert_close(ids, torch.tensor([[1, 120, 121, 122]]))


def test_text_config_retains_hybrid_schedules_and_serializes_selection(hf_config):
    from megatron.bridge.models.muse_spark_open import MuseSparkTextBridge

    text = copy.deepcopy(hf_config.text_config)
    text.architectures = ["MuseSparkForCausalLM"]
    text.tie_word_embeddings = hf_config.tie_word_embeddings
    bridge = MuseSparkTextBridge()
    config = bridge.hf_config_to_model_config(text)
    assert config.hf_model_text_only
    assert config.special_token_ids == {}
    assert config.hybrid_layer_pattern == "*E" * 6
    assert config.num_query_groups == 2
    restored = MuseSparkModelConfig.from_dict(config.as_dict())
    assert restored.hf_model_text_only
    exported = bridge.megatron_to_hf_config(restored)
    assert "text_config" not in exported and "vision_config" not in exported
    assert exported["layer_types"] == text.layer_types
    assert exported["layer_rope_theta"] == text.layer_rope_theta


def test_text_model_rejects_perception_inputs_before_decoder():
    model = MuseSparkModel.__new__(MuseSparkModel)
    nn.Module.__init__(model)
    model.model_config = SimpleNamespace(hf_model_text_only=True)
    with pytest.raises(ValueError, match="does not accept perception inputs"):
        model.forward(input_ids=torch.ones(1, 2, dtype=torch.long), pixel_values=torch.zeros(1))
