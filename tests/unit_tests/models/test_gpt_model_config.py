# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch.nn.functional as F

from megatron.bridge.models.common.base import ModelConfig
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, ModelConfigNotSupportedError
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.llama.llama_bridge import LlamaBridge
from megatron.bridge.models.transformer_config import TransformerConfig


pytestmark = pytest.mark.unit


def _make_model_config() -> BridgeGPTModelConfig:
    transformer = TransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        ffn_hidden_size=256,
        transformer_impl="local",
        use_cpu_initialization=True,
        activation_func=F.silu,
    )
    return BridgeGPTModelConfig(
        transformer=transformer,
        vocab_size=256,
        seq_length=128,
        position_embedding_type="rope",
    )


def test_flat_assignment_routes_to_declared_field_owner() -> None:
    config = _make_model_config()

    config.hidden_size = 256
    config.rotary_base = 500_000

    assert config.transformer.hidden_size == 256
    assert "hidden_size" not in config.__dict__
    assert config.rotary_base == 500_000
    assert "rotary_base" in config.__dict__

    with pytest.raises(AttributeError, match="declares a field"):
        config.phantom_override = True


def test_model_config_round_trip_restores_runtime_activation() -> None:
    config = _make_model_config()

    serialized = config.as_dict()
    restored = ModelConfig.from_dict(serialized)

    assert serialized["transformer"]["activation_func"] == "silu"
    assert isinstance(restored, BridgeGPTModelConfig)
    assert restored.transformer.activation_func == "silu"
    assert restored.get_builder_cls().__name__ == "GPTModelBuilder"

    restored.finalize()
    assert restored.transformer.activation_func is F.silu


def test_bridge_model_config_from_dict_restores_activation_immediately() -> None:
    restored = BridgeGPTModelConfig.from_dict(_make_model_config().as_dict())

    assert restored.transformer.activation_func is F.silu


def test_model_config_rejects_unregistered_activation() -> None:
    config = _make_model_config()
    config.transformer.activation_func = lambda value: value

    with pytest.raises(ValueError, match="Cannot serialize unregistered activation"):
        config.as_dict()


def test_model_config_mapping_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="phantom_field"):
        LlamaBridge._partition_model_config_kwargs(
            {"phantom_field": True},
            BridgeGPTModelConfig,
            TransformerConfig,
        )


def test_gpt_model_config_is_the_bridge_default() -> None:
    assert MegatronModelBridge.MODEL_CONFIG_CLASS is BridgeGPTModelConfig
    assert LlamaBridge.MODEL_CONFIG_CLASS is BridgeGPTModelConfig
    assert "MODEL_CONFIG_CLASS" not in LlamaBridge.__dict__


def test_bridge_can_explicitly_disable_model_config() -> None:
    class UnsupportedBridge(LlamaBridge):
        MODEL_CONFIG_CLASS = None

    with pytest.raises(ModelConfigNotSupportedError, match="sets MODEL_CONFIG_CLASS to None"):
        UnsupportedBridge().hf_config_to_model_config(object())
