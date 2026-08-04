# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import Mock, patch

from megatron.core.transformer import TransformerConfig

from megatron.bridge.models.common.base import ModelConfig
from megatron.bridge.models.ernie_vl.model_config import Ernie45VLModelBuilder, Ernie45VLModelConfig, _namespace
from megatron.bridge.models.ernie_vl.modeling_ernie45_vl.ernie_decoder_layer_spec import (
    get_ernie45_vl_decoder_block_spec,
)


def test_namespace_preserves_nested_non_string_key_mappings():
    result = _namespace({"rope_scaling": {"factor_by_layer": {0: 1.0, 1: 2.0}}})

    assert result.rope_scaling.factor_by_layer == {0: 1.0, 1: 2.0}


def test_ernie_vl_config_roundtrip_preserves_exact_mcore_configs():
    config = Ernie45VLModelConfig(
        transformer=TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4),
        vision_transformer=TransformerConfig(num_layers=2, hidden_size=64, num_attention_heads=4),
        vocab_size=256,
        patch_size=16,
        in_channels=3,
        spatial_merge_size=4,
        moe_intermediate_size=(2048, 768),
    )

    restored = ModelConfig.from_dict(config.as_dict())

    assert type(restored.transformer) is TransformerConfig
    assert restored.moe_intermediate_size == (2048, 768)
    assert type(restored.vision_transformer) is TransformerConfig
    assert restored.patch_size == 16
    assert restored.spatial_merge_size == 4
    assert restored.transformer_layer_spec is get_ernie45_vl_decoder_block_spec
    assert restored.builder == "megatron.bridge.models.ernie_vl.model_config.Ernie45VLModelBuilder"


def test_builder_materializes_nested_hf_vision_config():
    config = Ernie45VLModelConfig(
        transformer=TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4),
        vision_transformer=TransformerConfig(num_layers=2, hidden_size=64, num_attention_heads=4),
        vocab_size=256,
        hf_config={"text_config": {"hidden_size": 128}, "vision_config": {"hidden_size": 64}},
        vision_config={"hidden_size": 64},
    )
    built_model = Mock()

    with (
        patch("megatron.training.models.gpt.GPTModelBuilder.build_model", return_value=Mock()),
        patch("megatron.bridge.models.ernie_vl.model_config.Ernie45VLModel", return_value=built_model) as model_cls,
    ):
        result = Ernie45VLModelBuilder(config).build_model(Mock())

    runtime_config = model_cls.call_args.kwargs["config"]
    assert runtime_config.vision_config.hidden_size == 64
    assert runtime_config.hf_config.vision_config is runtime_config.vision_config
    assert result is built_model
