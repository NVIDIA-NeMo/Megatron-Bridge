# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
from megatron.core.transformer.spec_utils import ModuleSpec
from transformers.configuration_utils import PretrainedConfig

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.megatron_mimo.conversion import (
    MegatronMIMOBridge,
    MIMOComponent,
    get_mimo_conversion_spec,
    register_mimo_conversion_spec,
    supports_mimo_conversion,
    validate_route_table,
)
from megatron.bridge.models.megatron_mimo.conversion.orchestrator import _reset_registry_for_tests
from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)
from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import MegatronMIMOProvider


def _two_component_config() -> MegatronMIMOParallelismConfig:
    return MegatronMIMOParallelismConfig(
        module_parallelisms={
            "language": ModuleParallelismConfig(tensor_model_parallel_size=1),
            "vision": ModuleParallelismConfig(tensor_model_parallel_size=1),
        }
    )


class TestValidateRouteTable:
    def test_valid_two_component(self):
        config = _two_component_config()
        routes = [
            MIMOComponent("language", "language_model.", "language_model"),
            MIMOComponent("vision", "vision_model.", "modality_submodules.images.encoders.qwen_visual"),
        ]
        validate_route_table(routes, parallelism_config=config)

    def test_duplicate_route_names_rejected(self):
        config = _two_component_config()
        routes = [
            MIMOComponent("language", "language_model.", "language_model"),
            MIMOComponent("language", "lm.", "lm"),
        ]
        with pytest.raises(ValueError, match="Duplicate route names"):
            validate_route_table(routes, parallelism_config=config)

    def test_route_name_not_in_parallelism_config_rejected(self):
        config = _two_component_config()
        routes = [
            MIMOComponent("language", "language_model.", "language_model"),
            MIMOComponent("audio", "audio_model.", "modality_submodules.audio.encoders.whisper"),
        ]
        with pytest.raises(ValueError, match="not present in parallelism_config"):
            validate_route_table(routes, parallelism_config=config)

    def test_parallelism_config_entry_without_route_rejected(self):
        config = _two_component_config()
        routes = [MIMOComponent("language", "language_model.", "language_model")]
        with pytest.raises(ValueError, match="without a route"):
            validate_route_table(routes, parallelism_config=config)

    def test_nested_prefix_rejected(self):
        config = _two_component_config()
        routes = [
            MIMOComponent("language", "language_model.", "language_model"),
            MIMOComponent("vision", "language_model.mtp.", "language_model.mtp"),
        ]
        with pytest.raises(ValueError, match="nests inside"):
            validate_route_table(routes, parallelism_config=config)

    def test_identical_prefix_rejected(self):
        config = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(tensor_model_parallel_size=1),
                "language_alt": ModuleParallelismConfig(tensor_model_parallel_size=1),
            }
        )
        routes = [
            MIMOComponent("language", "language_model.", "language_model"),
            MIMOComponent("language_alt", "language_model.", "language_model_alt"),
        ]
        with pytest.raises(ValueError, match="share source_prefix"):
            validate_route_table(routes, parallelism_config=config)

    def test_modality_alignment_passes_when_keys_match(self):
        config = _two_component_config()
        routes = [
            MIMOComponent("language", "language_model.", "language_model"),
            MIMOComponent("vision", "vision_model.", "modality_submodules.vision.encoders.x"),
        ]
        # Use any falsy-but-dict sentinel for ModuleSpec — only keys are read.
        modality_specs = {"vision": object()}
        validate_route_table(
            routes,
            parallelism_config=config,
            modality_submodules_spec=modality_specs,
        )

    def test_modality_alignment_rejects_route_name_not_in_modality_dict(self):
        config = _two_component_config()
        routes = [
            MIMOComponent("language", "language_model.", "language_model"),
            MIMOComponent("vision", "vision_model.", "modality_submodules.images.encoders.x"),
        ]
        modality_specs = {"images": object()}
        with pytest.raises(ValueError, match="do not align with modality_submodules_spec"):
            validate_route_table(
                routes,
                parallelism_config=config,
                modality_submodules_spec=modality_specs,
            )

    def test_modality_alignment_rejects_modality_key_without_route(self):
        config = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(tensor_model_parallel_size=1),
                "images": ModuleParallelismConfig(tensor_model_parallel_size=1),
                "audio": ModuleParallelismConfig(tensor_model_parallel_size=1),
            }
        )
        routes = [
            MIMOComponent("language", "language_model.", "language_model"),
            MIMOComponent("images", "vision_model.", "modality_submodules.images.encoders.x"),
            MIMOComponent("audio", "audio_model.", "modality_submodules.audio.encoders.x"),
        ]
        modality_specs = {"images": object()}  # 'audio' modality not declared
        with pytest.raises(ValueError, match="missing from routes:|do not align"):
            validate_route_table(
                routes,
                parallelism_config=config,
                modality_submodules_spec=modality_specs,
            )


class _FakeBridgeA:
    pass


class _FakeBridgeB:
    pass


class _FakeStandardProvider:
    modality_keys = {"vision": "fake_visual"}
    special_token_ids = {"vision": 7}

    def build_language_model_spec(self):
        return ModuleSpec(module=object)

    def build_vision_encoder_spec(self):
        return ModuleSpec(module=object)


class _FakeSourceBridgeWithProvider:
    def __init__(self):
        self.provider = _FakeStandardProvider()
        self.hf_pretrained = None

    def provider_bridge(self, hf_pretrained):
        self.hf_pretrained = hf_pretrained
        return self.provider


class _FakeSourceBridgeWithDefaultRoutes(_FakeSourceBridgeWithProvider):
    mimo_source_prefixes = {"language": "language_model.", "vision": "vision_model."}


class TestConversionSpecRegistry:
    def setup_method(self):
        _reset_registry_for_tests()

    def teardown_method(self):
        _reset_registry_for_tests()

    def test_register_and_lookup(self):
        @register_mimo_conversion_spec(_FakeBridgeA)
        def conversion_spec(source_bridge, hf_pretrained, parallelism_config):
            return None, []

        assert get_mimo_conversion_spec(_FakeBridgeA) is conversion_spec
        assert supports_mimo_conversion(_FakeBridgeA)

    def test_duplicate_registration_rejected(self):
        @register_mimo_conversion_spec(_FakeBridgeA)
        def first(source_bridge, hf_pretrained, parallelism_config):
            return None, []

        with pytest.raises(ValueError, match="already registered"):

            @register_mimo_conversion_spec(_FakeBridgeA)
            def second(source_bridge, hf_pretrained, parallelism_config):
                return None, []

    def test_different_bridges_independent(self):
        @register_mimo_conversion_spec(_FakeBridgeA)
        def conversion_spec_a(source_bridge, hf_pretrained, parallelism_config):
            return "A", []

        @register_mimo_conversion_spec(_FakeBridgeB)
        def conversion_spec_b(source_bridge, hf_pretrained, parallelism_config):
            return "B", []

        assert get_mimo_conversion_spec(_FakeBridgeA) is conversion_spec_a
        assert get_mimo_conversion_spec(_FakeBridgeB) is conversion_spec_b

    def test_unregistered_bridge_with_mimo_metadata_uses_default_conversion_spec(self):
        assert supports_mimo_conversion(_FakeSourceBridgeWithDefaultRoutes)

        source_bridge = _FakeSourceBridgeWithDefaultRoutes()
        hf_pretrained = object()
        provider, route_table = get_mimo_conversion_spec(_FakeSourceBridgeWithDefaultRoutes)(
            source_bridge,
            hf_pretrained,
            _two_component_config(),
        )

        assert isinstance(provider, MegatronMIMOProvider)
        assert provider.standard_provider is source_bridge.provider
        assert source_bridge.hf_pretrained is hf_pretrained
        assert route_table == [
            MIMOComponent("language", "language_model.", "language_model"),
            MIMOComponent("vision", "vision_model.", "modality_submodules.vision.encoders.fake_visual"),
        ]

    def test_unregistered_bridge_without_mimo_metadata_rejected(self):
        assert not supports_mimo_conversion(_FakeBridgeA)
        assert not supports_mimo_conversion(MegatronModelBridge)

        with pytest.raises(KeyError, match="mimo_source_prefixes"):
            get_mimo_conversion_spec(_FakeBridgeA)

    def test_validate_mimo_conversion_support_resolves_provider_and_routes(self):
        source_bridge = _FakeSourceBridgeWithDefaultRoutes()
        bridge = MegatronMIMOBridge(
            PretrainedConfig(),
            parallelism_config=_two_component_config(),
            source_bridge=source_bridge,
        )

        bridge.validate_mimo_conversion_support()

        assert isinstance(bridge.to_megatron_mimo_provider(load_weights=False), MegatronMIMOProvider)
        assert bridge.routes == [
            MIMOComponent("language", "language_model.", "language_model"),
            MIMOComponent("vision", "vision_model.", "modality_submodules.vision.encoders.fake_visual"),
        ]
