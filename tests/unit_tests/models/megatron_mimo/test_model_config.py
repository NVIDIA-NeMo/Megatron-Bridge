# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import yaml
from megatron.core.transformer import TransformerConfig
from megatron.training.config.yaml_utils import safe_yaml_representers

from megatron.bridge.models.common.base import ModelConfig
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)
from megatron.bridge.models.megatron_mimo.model_config import MegatronMIMOModelBuilder, MegatronMIMOModelConfig
from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.bridge.training.config import ConfigContainer


def _model_config() -> MegatronMIMOModelConfig:
    source = BridgeGPTModelConfig(
        transformer=TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=1),
        vocab_size=32,
    )
    parallelism = MegatronMIMOParallelismConfig(
        module_parallelisms={
            "language": ModuleParallelismConfig(data_parallel_size=1),
            "images": ModuleParallelismConfig(data_parallel_size=1, rank_offset=1),
        }
    )
    return MegatronMIMOModelConfig(
        source_model_config=source,
        megatron_mimo_parallelism_config=parallelism,
        language_spec_builder="pkg.language_spec",
        modality_spec_builder="pkg.modality_specs",
        modality_keys={"images": "vision"},
        special_token_ids={"images": 7},
    )


def test_model_config_roundtrip_is_provider_neutral() -> None:
    config = _model_config()
    serialized = config.as_dict()
    restored = MegatronMIMOModelConfig.from_dict(serialized)

    assert not isinstance(config, ModelProviderMixin)
    assert restored.as_dict() == serialized
    assert restored.special_token_ids == {"images": 7}


def test_config_container_yaml_representation_restores_parallelism_dataclasses() -> None:
    config = _model_config()
    model_payload = ConfigContainer._convert_value_to_dict(config)
    with safe_yaml_representers():
        yaml_payload = yaml.safe_load(yaml.safe_dump({"model": model_payload}))

    restored = ModelConfig.from_dict(yaml_payload["model"])

    assert isinstance(restored, MegatronMIMOModelConfig)
    assert isinstance(restored.megatron_mimo_parallelism_config, MegatronMIMOParallelismConfig)
    assert all(
        isinstance(module_config, ModuleParallelismConfig)
        for module_config in restored.megatron_mimo_parallelism_config.module_parallelisms.values()
    )
    assert restored.megatron_mimo_parallelism_config.module_parallelisms["images"].rank_offset == 1
    builder = MegatronMIMOModelBuilder(restored)
    assert isinstance(
        builder._model_config.megatron_mimo_parallelism_config.module_parallelisms["language"],
        ModuleParallelismConfig,
    )
