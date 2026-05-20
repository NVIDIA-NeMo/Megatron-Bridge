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

from megatron.bridge.models.megatron_mimo.conversion import (
    MIMOComponent,
    get_mimo_adapter,
    register_mimo_conversion,
    validate_route_table,
)
from megatron.bridge.models.megatron_mimo.conversion.orchestrator import _reset_registry_for_tests
from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)


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


class TestAdapterRegistry:
    def setup_method(self):
        _reset_registry_for_tests()

    def teardown_method(self):
        _reset_registry_for_tests()

    def test_register_and_lookup(self):
        @register_mimo_conversion(_FakeBridgeA)
        def adapter(source_bridge, hf_pretrained, parallelism_config):
            return None, []

        assert get_mimo_adapter(_FakeBridgeA) is adapter

    def test_duplicate_registration_rejected(self):
        @register_mimo_conversion(_FakeBridgeA)
        def first(source_bridge, hf_pretrained, parallelism_config):
            return None, []

        with pytest.raises(ValueError, match="already registered"):

            @register_mimo_conversion(_FakeBridgeA)
            def second(source_bridge, hf_pretrained, parallelism_config):
                return None, []

    def test_different_bridges_independent(self):
        @register_mimo_conversion(_FakeBridgeA)
        def adapter_a(source_bridge, hf_pretrained, parallelism_config):
            return "A", []

        @register_mimo_conversion(_FakeBridgeB)
        def adapter_b(source_bridge, hf_pretrained, parallelism_config):
            return "B", []

        assert get_mimo_adapter(_FakeBridgeA) is adapter_a
        assert get_mimo_adapter(_FakeBridgeB) is adapter_b
