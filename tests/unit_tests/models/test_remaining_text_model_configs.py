# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import Mock, patch

import pytest
from megatron.core.activations import squared_relu
from megatron.core.transformer import ModuleSpec, TransformerConfig

from megatron.bridge.models.common.base import ModelConfig
from megatron.bridge.models.falcon_h1.model_config import FalconH1ModelConfig
from megatron.bridge.models.nemotronh.model_config import (
    NemotronHModelBuilder,
    NemotronHModelConfig,
    _configure_mamba_chunk_size,
)


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
    assert restored.get_builder_cls() is NemotronHModelBuilder


def test_nemotron_h_model_config_validates_mamba_chunk_size():
    config = NemotronHModelConfig(
        transformer=TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4),
        vocab_size=256,
        hybrid_layer_pattern="MM",
        mamba_chunk_size=0,
    )

    with pytest.raises(ValueError, match="mamba_chunk_size must be at least 1"):
        config.finalize()


def test_nemotron_h_builder_applies_chunk_size_without_mutating_default_spec():
    mixer = Mock(params={"chunk_size": 128})
    stack_spec = Mock(spec=ModuleSpec)
    stack_spec.submodules.mamba_layer.submodules.mixer = mixer
    configured_spec = _configure_mamba_chunk_size(stack_spec, 64)

    assert configured_spec is not stack_spec
    assert configured_spec.submodules.mamba_layer.submodules.mixer.params["chunk_size"] == 64
    assert stack_spec.submodules.mamba_layer.submodules.mixer.params == {"chunk_size": 128}


def test_nemotron_h_builder_restores_custom_stack_spec():
    stack_spec = ModuleSpec(module=object)
    config = NemotronHModelConfig(
        transformer=TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4),
        vocab_size=256,
        hybrid_layer_pattern="MM",
        hybrid_stack_spec=stack_spec,
        mamba_chunk_size=64,
    )
    builder = NemotronHModelBuilder(config)

    with (
        patch(
            "megatron.bridge.models.nemotronh.model_config._configure_mamba_chunk_size",
            return_value=Mock(spec=ModuleSpec),
        ) as configure,
        patch(
            "megatron.training.models.hybrid.HybridModelBuilder.build_model",
            return_value=Mock(),
        ),
    ):
        builder.build_model(Mock())

    configure.assert_called_once_with(stack_spec, 64)
    assert config.hybrid_stack_spec is stack_spec
