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

import inspect
import warnings
from dataclasses import dataclass
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F
from megatron.core.models.gpt import GPTModel
from megatron.core.transformer import ModuleSpec
from transformers import Qwen2Config

import megatron.bridge.models  # noqa: F401  (registers every model-family bridge)
from megatron.bridge.models.common.base import ModelConfig
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, ModelConfigNotSupportedError
from megatron.bridge.models.gpt.gpt_builder import GPTModelBuilder
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.llama.llama_bridge import LlamaBridge
from megatron.bridge.models.muse_glimmer.muse_glimmer_bridge import MuseGlimmerBridge
from megatron.bridge.models.qwen.qwen2_bridge import Qwen2Bridge
from megatron.bridge.models.qwen.qwen3_bridge import Qwen3Bridge
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


@pytest.mark.skipif(
    hasattr(TransformerConfig, "dynamic_context_parallel"),
    reason="The pinned Megatron-Core declares the dynamic context parallel fields.",
)
def test_dynamic_cp_fields_name_the_dev_pin_on_pins_without_them() -> None:
    config = _make_model_config()

    with pytest.raises(AttributeError, match="switch_mcore.sh dev"):
        config.dynamic_context_parallel = True
    with pytest.raises(AttributeError, match="switch_mcore.sh dev"):
        config.min_dynamic_context_parallel_size = 1
    with pytest.raises(AttributeError, match="declares a field") as excinfo:
        config.phantom_override = True
    assert "switch_mcore.sh" not in str(excinfo.value)


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


def test_model_config_class_selects_nested_transformer_type(monkeypatch: pytest.MonkeyPatch) -> None:
    @dataclass
    class SpecializedTransformerConfig(TransformerConfig):
        specialized_field: bool = False

    class SpecializedModelConfig(BridgeGPTModelConfig):
        transformer_config_class = SpecializedTransformerConfig

    class SpecializedBridge(LlamaBridge):
        MODEL_CONFIG_CLASS = SpecializedModelConfig

    bridge = SpecializedBridge()
    monkeypatch.setattr(
        bridge,
        "hf_config_to_model_config_kwargs",
        lambda _hf_config: {
            "num_layers": 2,
            "hidden_size": 128,
            "num_attention_heads": 4,
            "ffn_hidden_size": 256,
            "activation_func": F.silu,
            "specialized_field": True,
            "vocab_size": 256,
        },
    )

    model_config = bridge.hf_config_to_model_config(object())

    assert isinstance(model_config, SpecializedModelConfig)
    assert isinstance(model_config.transformer, SpecializedTransformerConfig)
    assert model_config.specialized_field is True


def _bridge_subclasses(bridge_class: type[MegatronModelBridge]) -> set[type[MegatronModelBridge]]:
    subclasses = set(bridge_class.__subclasses__())
    return subclasses.union(*(_bridge_subclasses(subclass) for subclass in subclasses))


def _all_bridge_classes() -> list[type[MegatronModelBridge]]:
    """Return every bridge class registered by importing ``megatron.bridge.models``."""
    return sorted(
        (
            bridge_class
            for bridge_class in _bridge_subclasses(MegatronModelBridge)
            if bridge_class.__module__.startswith("megatron.bridge.")
        ),
        key=lambda bridge_class: bridge_class.__name__,
    )


def _has_model_config_mapping(bridge_class: type[MegatronModelBridge]) -> bool:
    return (
        bridge_class.hf_config_to_model_config_kwargs is not MegatronModelBridge.hf_config_to_model_config_kwargs
        or bridge_class.hf_config_to_model_config is not MegatronModelBridge.hf_config_to_model_config
    )


def test_model_config_is_opt_in_per_family() -> None:
    assert MegatronModelBridge.MODEL_CONFIG_CLASS is None
    assert MegatronModelBridge.USE_MODEL_CONFIG_FOR_CONVERSION is False
    for bridge_class in (LlamaBridge, Qwen3Bridge):
        assert bridge_class.__dict__["MODEL_CONFIG_CLASS"] is BridgeGPTModelConfig
    assert MuseGlimmerBridge.__dict__["MODEL_CONFIG_CLASS"] is not None
    assert LlamaBridge.USE_MODEL_CONFIG_FOR_CONVERSION is False
    assert BridgeGPTModelConfig.transformer_config_class is TransformerConfig
    assert not hasattr(MegatronModelBridge, "TRANSFORMER_CONFIG_CLASS")


def test_bridge_subclass_inherits_model_config_opt_in() -> None:
    class DerivedBridge(LlamaBridge):
        pass

    assert DerivedBridge.MODEL_CONFIG_CLASS is BridgeGPTModelConfig


@pytest.mark.parametrize("bridge_class", _all_bridge_classes(), ids=lambda bridge_class: bridge_class.__name__)
def test_bridge_can_explicitly_disable_model_config(bridge_class: type[MegatronModelBridge]) -> None:
    if bridge_class.MODEL_CONFIG_CLASS is None:
        # Unmigrated family: the builder path is rejected instead of returning generic defaults.
        assert not _has_model_config_mapping(bridge_class)
        assert bridge_class.USE_MODEL_CONFIG_FOR_CONVERSION is False
        with pytest.raises(ModelConfigNotSupportedError, match=f"{bridge_class.__name__}.*to_megatron_provider"):
            bridge_class().hf_config_to_model_config(object())
    else:
        # Opted-in family: it must map its Hugging Face config for the builder path.
        assert _has_model_config_mapping(bridge_class)

    # A bridge that replaces hf_config_to_model_config() wholesale does not read MODEL_CONFIG_CLASS.
    if bridge_class.hf_config_to_model_config is MegatronModelBridge.hf_config_to_model_config:

        class UnsupportedBridge(bridge_class):
            MODEL_CONFIG_CLASS = None

        with pytest.raises(ModelConfigNotSupportedError, match="UnsupportedBridge.*to_megatron_provider"):
            UnsupportedBridge().hf_config_to_model_config(object())


def test_unmigrated_bridge_keeps_provider_without_warning() -> None:
    hf_config = Qwen2Config(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
    )

    class UnmigratedQwen2Bridge(Qwen2Bridge):
        MODEL_CONFIG_CLASS = None

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        provider = UnmigratedQwen2Bridge().provider_bridge(Mock(config=hf_config))

    assert provider.position_embedding_type == "rope"
    assert provider.add_qkv_bias is True
    assert provider.add_bias_linear is False


@pytest.mark.skipif(
    "logit_dtype" not in inspect.signature(GPTModel).parameters,
    reason="Installed MCore predates logit_dtype",
)
@patch("megatron.training.models.gpt.GPTModel")
def test_requested_logit_dtype_reaches_new_mcore_builder(mock_model) -> None:
    config = _make_model_config()
    config.logit_dtype = torch.float32
    config.transformer_layer_spec = ModuleSpec(module=object)
    pg_collection = Mock()

    GPTModelBuilder(config).build_model(pg_collection, pre_process=True, post_process=True)

    assert mock_model.call_args.kwargs["logit_dtype"] is torch.float32


@pytest.mark.skipif(
    "logit_dtype" in inspect.signature(GPTModel).parameters,
    reason="Installed MCore supports logit_dtype",
)
def test_requested_logit_dtype_fails_before_old_mcore_builder() -> None:
    config = _make_model_config()
    config.logit_dtype = torch.float32

    with pytest.raises(RuntimeError, match="Megatron-LM PR #6252"):
        GPTModelBuilder(config).build_model(Mock(), pre_process=True, post_process=True)
