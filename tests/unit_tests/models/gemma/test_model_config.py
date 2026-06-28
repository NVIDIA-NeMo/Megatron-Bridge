# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.models.gpt import GPTModelBuilder

from megatron.bridge.models.common.base import ModelConfig
from megatron.bridge.models.gemma.gemma2_bridge import Gemma2Bridge
from megatron.bridge.models.gemma.gemma3_bridge import Gemma3ModelBridge
from megatron.bridge.models.gemma.gemma_bridge import GemmaBridge
from megatron.bridge.models.gemma.model_config import (
    Gemma2ModelBuilder,
    Gemma2ModelConfig,
    Gemma3ModelBuilder,
    Gemma3ModelConfig,
    GemmaModelBuilder,
    GemmaModelConfig,
)
from megatron.bridge.models.gemma.modeling_gemma2 import Gemma2OutputLayer, gemma2_layer_spec
from megatron.bridge.models.gemma.modeling_gemma3 import gemma3_layer_spec
from megatron.bridge.models.gemma.modules import EmbeddingScalingMixin


def _common_hf_config(**overrides: object) -> SimpleNamespace:
    values = {
        "num_hidden_layers": 2,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "vocab_size": 128,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "initializer_range": 0.02,
        "attention_dropout": 0.0,
        "hidden_dropout": 0.0,
        "tie_word_embeddings": True,
        "attention_bias": False,
        "mlp_bias": False,
        "rope_theta": 10_000.0,
        "rope_scaling": None,
        "hidden_act": "gelu_pytorch_tanh",
        "torch_dtype": torch.float32,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_gemma_model_config_bridge_uses_pure_config_and_builder() -> None:
    model_config = GemmaBridge().model_config_bridge(SimpleNamespace(config=_common_hf_config()))

    assert isinstance(model_config, GemmaModelConfig)
    assert model_config.get_builder_cls() is GemmaModelBuilder
    assert model_config.transformer.normalization == "RMSNorm"
    assert model_config.transformer.gated_linear_unit is True
    assert model_config.transformer.layernorm_zero_centered_gamma is True
    assert model_config.transformer.attention_backend == AttnBackend.flash
    assert model_config.share_embeddings_and_output_weights is True


def test_gemma2_model_config_bridge_keeps_custom_fields_in_outer_config() -> None:
    hf_config = _common_hf_config(
        query_pre_attn_scalar=256,
        attn_logit_softcapping=50.0,
        final_logit_softcapping=30.0,
        sliding_window=4096,
    )

    model_config = Gemma2Bridge().model_config_bridge(SimpleNamespace(config=hf_config))

    assert isinstance(model_config, Gemma2ModelConfig)
    assert type(model_config.transformer) is TransformerConfig
    assert model_config.get_builder_cls() is Gemma2ModelBuilder
    assert model_config.query_pre_attn_scalar == 256
    assert model_config.attn_logit_softcapping == 50.0
    assert model_config.final_logit_softcapping == 30.0
    assert model_config.transformer.window_size == (4095, 0)
    assert model_config.transformer_layer_spec is gemma2_layer_spec


@patch("megatron.bridge.models.gemma.gemma3_bridge.AutoConfig.from_pretrained")
def test_gemma3_model_config_bridge_maps_rope_and_parent_precision(mock_auto_config: Mock) -> None:
    mock_auto_config.return_value = SimpleNamespace(torch_dtype=torch.bfloat16)
    hf_config = _common_hf_config(
        query_pre_attn_scalar=256,
        sliding_window=512,
        rope_local_base_freq=10_000.0,
        rope_theta=1_000_000.0,
        rope_scaling={"factor": 8.0, "type": "linear"},
    )
    pretrained = SimpleNamespace(config=hf_config, _model_name_or_path="google/gemma-3-test")

    model_config = Gemma3ModelBridge().model_config_bridge(pretrained)

    assert isinstance(model_config, Gemma3ModelConfig)
    assert type(model_config.transformer) is TransformerConfig
    assert model_config.get_builder_cls() is Gemma3ModelBuilder
    assert model_config.rotary_base_local == 10_000
    assert model_config.rotary_base == 1_000_000
    assert model_config.rope_scaling_factor == 8.0
    assert model_config.transformer.window_size == (511, 0)
    assert model_config.transformer.softmax_scale == 1.0 / 16.0
    assert model_config.transformer.params_dtype == torch.bfloat16
    assert model_config.transformer_layer_spec is gemma3_layer_spec


def test_gemma2_model_config_serialization_restores_family_defaults() -> None:
    hf_config = _common_hf_config(
        query_pre_attn_scalar=256,
        attn_logit_softcapping=50.0,
        final_logit_softcapping=30.0,
        sliding_window=4096,
    )
    original = Gemma2Bridge().model_config_bridge(SimpleNamespace(config=hf_config))

    restored = ModelConfig.from_dict(original.as_dict())

    assert isinstance(restored, Gemma2ModelConfig)
    assert type(restored.transformer) is TransformerConfig
    assert restored.query_pre_attn_scalar == 256
    assert restored.get_builder_cls() is Gemma2ModelBuilder
    assert restored.transformer_layer_spec is gemma2_layer_spec
    assert restored.transformer.activation_func is original.transformer.activation_func


def test_gemma3_model_config_serialization_restores_family_defaults() -> None:
    original = Gemma3ModelConfig(
        transformer=TransformerConfig(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=2,
            activation_func=torch.nn.functional.gelu,
        ),
        vocab_size=128,
        rotary_base=1_000_000,
        rotary_base_local=10_000,
    )

    restored = ModelConfig.from_dict(original.as_dict())

    assert isinstance(restored, Gemma3ModelConfig)
    assert type(restored.transformer) is TransformerConfig
    assert restored.get_builder_cls() is Gemma3ModelBuilder
    assert restored.transformer_layer_spec is gemma3_layer_spec
    assert restored.rotary_base == 1_000_000
    assert restored.rotary_base_local == 10_000


def test_gemma_and_gemma2_builders_install_stage_customizations() -> None:
    gemma_config = GemmaBridge().model_config_bridge(SimpleNamespace(config=_common_hf_config()))
    gemma2_config = Gemma2Bridge().model_config_bridge(
        SimpleNamespace(
            config=_common_hf_config(
                query_pre_attn_scalar=256,
                attn_logit_softcapping=50.0,
                final_logit_softcapping=30.0,
                sliding_window=4096,
            )
        )
    )
    gemma_model = SimpleNamespace(embedding=object())
    gemma2_model = SimpleNamespace(embedding=object(), output_layer=SimpleNamespace())

    with (
        patch.object(GPTModelBuilder, "build_model", side_effect=[gemma_model, gemma2_model]),
        patch("megatron.bridge.models.gemma.model_config.extend_instance") as extend,
    ):
        assert GemmaModelBuilder(gemma_config).build_model(Mock()) is gemma_model
        assert Gemma2ModelBuilder(gemma2_config).build_model(Mock()) is gemma2_model

    assert extend.call_args_list == [
        call(gemma_model.embedding, EmbeddingScalingMixin),
        call(gemma2_model.embedding, EmbeddingScalingMixin),
        call(gemma2_model.output_layer, Gemma2OutputLayer),
    ]


def test_gemma3_builder_replaces_embedding_and_rope() -> None:
    transformer = TransformerConfig(
        num_layers=2,
        hidden_size=16,
        num_attention_heads=2,
        num_query_groups=1,
        kv_channels=8,
    )
    config = Gemma3ModelConfig(
        transformer=transformer,
        vocab_size=128,
        seq_length=64,
        rotary_base=1_000_000,
        rotary_base_local=10_000,
        rope_scaling_factor=8.0,
    )
    model = SimpleNamespace(embedding=object(), output_layer=object(), setup_embeddings_and_output_layer=Mock())
    embedding = object()
    rotary = object()

    with (
        patch.object(GPTModelBuilder, "build_model", return_value=model),
        patch(
            "megatron.bridge.models.gemma.model_config.Gemma3LanguageModelEmbedding", return_value=embedding
        ) as embedding_cls,
        patch("megatron.bridge.models.gemma.model_config.Gemma3RotaryEmbedding", return_value=rotary) as rotary_cls,
    ):
        result = Gemma3ModelBuilder(config).build_model(Mock())

    assert result is model
    assert model.embedding is embedding
    assert model.rotary_pos_emb is rotary
    embedding_cls.assert_called_once_with(
        config=transformer,
        vocab_size=128,
        max_sequence_length=64,
        position_embedding_type="learned_absolute",
        scatter_to_sequence_parallel=True,
    )
    assert rotary_cls.call_args.kwargs["rotary_base"] == 1_000_000
    assert rotary_cls.call_args.kwargs["rotary_base_local"] == 10_000
    assert rotary_cls.call_args.kwargs["rope_scaling_factor"] == 8.0
    model.setup_embeddings_and_output_layer.assert_called_once_with()
