# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from megatron.core.activations import squared_relu
from megatron.core.transformer import TransformerConfig

from megatron.bridge.models.common.base import ModelConfig
from megatron.bridge.models.falcon_h1.model_config import FalconH1ModelConfig
from megatron.bridge.models.nemotronh.model_config import NemotronHModelConfig


def test_falcon_h1_model_config_uses_exact_config_and_roundtrips():
    config = FalconH1ModelConfig(
        transformer=TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4),
        vocab_size=256,
    )

    restored = ModelConfig.from_dict(config.as_dict())

    assert type(restored.transformer) is TransformerConfig
    assert restored.builder == "megatron.bridge.models.falcon_h1.model_config.FalconH1ModelBuilder"


def test_nemotron_h_model_config_uses_exact_mcore_config_and_preserves_activation():
    config = NemotronHModelConfig(
        transformer=TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            activation_func=squared_relu,
        ),
        vocab_size=256,
        hybrid_layer_pattern="MM",
    )

    restored = ModelConfig.from_dict(config.as_dict())

    assert type(restored.transformer) is TransformerConfig
    assert restored.transformer.activation_func is squared_relu
