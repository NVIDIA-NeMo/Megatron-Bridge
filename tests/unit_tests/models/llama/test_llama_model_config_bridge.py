# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace
from unittest.mock import patch

import torch
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.models.gpt import GPTModelConfig

from megatron.bridge.models.llama.llama_bridge import LlamaBridge


def _hf_config(*, rope_scaling: dict[str, object] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=2,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        tie_word_embeddings=False,
        attention_bias=False,
        mlp_bias=False,
        rope_theta=10_000.0,
        rope_scaling=rope_scaling,
        hidden_act="silu",
        torch_dtype=torch.float32,
    )


def test_model_config_bridge_constructs_specialized_llama_config():
    pretrained = SimpleNamespace(config=_hf_config())

    result = LlamaBridge().model_config_bridge(pretrained)

    assert isinstance(result, GPTModelConfig)
    assert type(result.transformer) is TransformerConfig
    assert result.rotary_percent == 1.0
    assert result.transformer.normalization == "RMSNorm"
    assert result.transformer.gated_linear_unit is True
    assert result.transformer.hidden_dropout == 0.0
    assert result.transformer.bias_activation_fusion is True
    assert result.transformer.masked_softmax_fusion is True
    assert result.transformer.persist_layer_norm is True
    assert result.transformer.bias_dropout_fusion is True
    assert isinstance(result.transformer.apply_rope_fusion, bool)
    assert "normalization" not in result.__dict__
    assert "rotary_percent" not in result.transformer.__dict__


def test_model_config_bridge_disables_unavailable_rope_fusion():
    pretrained = SimpleNamespace(config=_hf_config())

    with patch("megatron.bridge.models.llama.llama_bridge.fusions.can_enable_rope_fusion", return_value=False):
        result = LlamaBridge().model_config_bridge(pretrained)

    assert result.transformer.apply_rope_fusion is False


def test_model_config_bridge_constructs_llama3_rope_scaling():
    rope_scaling = {"rope_type": "llama3", "factor": 32.0}
    pretrained = SimpleNamespace(config=_hf_config(rope_scaling=rope_scaling))

    result = LlamaBridge().model_config_bridge(pretrained)

    assert result.rope_scaling is True
    assert result.rope_scaling_factor == 32.0
    assert "rope_scaling" not in result.transformer.__dict__
    assert "rope_scaling_factor" not in result.transformer.__dict__
