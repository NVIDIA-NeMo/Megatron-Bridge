# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import torch
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.models.gpt import GPTModelConfig

from megatron.bridge.models.qwen.qwen2_bridge import Qwen2Bridge


def test_model_config_bridge_constructs_specialized_qwen2_config():
    hf_config = SimpleNamespace(
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
        attention_bias=True,
        mlp_bias=False,
        rope_theta=1_000_000.0,
        rope_scaling=None,
        hidden_act="silu",
        torch_dtype=torch.float32,
    )
    pretrained = SimpleNamespace(config=hf_config)

    result = Qwen2Bridge().model_config_bridge(pretrained)

    assert isinstance(result, GPTModelConfig)
    assert type(result.transformer) is TransformerConfig
    assert result.transformer.normalization == "RMSNorm"
    assert result.transformer.gated_linear_unit is True
    assert result.transformer.add_bias_linear is False
    assert result.transformer.add_qkv_bias is True
    assert result.transformer.hidden_dropout == 0.0
    assert result.transformer.autocast_dtype == torch.bfloat16
    for field_name in (
        "normalization",
        "gated_linear_unit",
        "add_bias_linear",
        "add_qkv_bias",
        "hidden_dropout",
        "autocast_dtype",
    ):
        assert field_name not in result.__dict__
