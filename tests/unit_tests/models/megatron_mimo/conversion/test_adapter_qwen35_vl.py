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

"""Unit tests for the Qwen3.5-VL MegatronMIMO conversion adapter."""

import importlib
from unittest.mock import MagicMock

import pytest

from megatron.bridge.models.megatron_mimo.conversion import MIMOComponent, get_mimo_adapter, validate_route_table
from megatron.bridge.models.megatron_mimo.conversion.adapters.qwen35_vl import qwen35_vl_mimo_adapter
from megatron.bridge.models.megatron_mimo.conversion.orchestrator import _reset_registry_for_tests
from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)
from megatron.bridge.models.megatron_mimo.qwen35_vl_provider import Qwen35VLMegatronMIMOProvider
from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import Qwen35VLBridge
from megatron.bridge.models.qwen_vl.qwen35_vl_provider import _TRANSFORMERS_HAS_QWEN3_5, Qwen35VLModelProvider


pytestmark = pytest.mark.skipif(not _TRANSFORMERS_HAS_QWEN3_5, reason="transformers does not have qwen3_5 support")


def _make_parallelism_config() -> MegatronMIMOParallelismConfig:
    """Two-component parallelism config matching the Qwen3.5-VL MIMO route table."""
    return MegatronMIMOParallelismConfig(
        module_parallelisms={
            "language": ModuleParallelismConfig(tensor_model_parallel_size=1),
            "images": ModuleParallelismConfig(tensor_model_parallel_size=1),
        }
    )


def _make_language_provider() -> Qwen35VLModelProvider:
    """Small Qwen35VLModelProvider; ``deepstack_visual_indexes`` patched in.

    Bare ``Qwen3_5VisionConfig()`` constructor lacks ``deepstack_visual_indexes``;
    real loaded HF configs supply it (defaulted to empty). Setting it here so
    the MIMO provider's eager ``_build_vision_modality_spec`` doesn't trip
    inside ``get_vision_model_config``.
    """
    provider = Qwen35VLModelProvider(
        num_layers=64,
        hidden_size=5120,
        num_attention_heads=24,
        vocab_size=128,
    )
    provider.vision_config.deepstack_visual_indexes = []
    return provider


class TestQwen35VLAdapterRegistration:
    def test_registered_for_qwen35_vl_bridge(self):
        _reset_registry_for_tests()
        import megatron.bridge.models.megatron_mimo.conversion.adapters.qwen35_vl as adapter_module

        importlib.reload(adapter_module)

        assert get_mimo_adapter(Qwen35VLBridge) is adapter_module.qwen35_vl_mimo_adapter


class TestQwen35VLAdapter:
    def test_returns_provider_and_routes(self, monkeypatch):
        language_provider = _make_language_provider()
        parallelism_config = _make_parallelism_config()

        source_bridge = MagicMock(spec=Qwen35VLBridge)
        source_bridge.provider_bridge = MagicMock(return_value=language_provider)
        hf_pretrained = MagicMock()

        provider, routes = qwen35_vl_mimo_adapter(source_bridge, hf_pretrained, parallelism_config)

        # Adapter must consult the source bridge to derive the language provider
        # rather than constructing one from scratch.
        source_bridge.provider_bridge.assert_called_once_with(hf_pretrained)

        assert isinstance(provider, Qwen35VLMegatronMIMOProvider)
        assert provider.language_provider is language_provider
        assert provider.megatron_mimo_parallelism_config is parallelism_config

        assert len(routes) == 2
        assert all(isinstance(r, MIMOComponent) for r in routes)

    def test_forces_mtp_off(self):
        """MIMO v1 doesn't support MTP. Adapter must force ``mtp_num_layers=None``
        so ``Qwen35VLMegatronMIMOProvider`` doesn't reject the model.
        """
        language_provider = _make_language_provider()
        language_provider.mtp_num_layers = 4  # would otherwise be rejected
        parallelism_config = _make_parallelism_config()

        source_bridge = MagicMock(spec=Qwen35VLBridge)
        source_bridge.provider_bridge = MagicMock(return_value=language_provider)
        hf_pretrained = MagicMock()

        provider, _ = qwen35_vl_mimo_adapter(source_bridge, hf_pretrained, parallelism_config)
        assert provider.language_provider.mtp_num_layers is None

    def test_route_table_contents(self):
        """Routes match the parameter prefixes used by Qwen35VLBridge.mapping_registry.

        Source bridge mapping registry uses ``language_model.`` and ``vision_model.``
        as the two top-level prefixes (see ``qwen35_vl_bridge.py:212-268``).
        The route table strips these and dispatches to:
          - ``mimo_model.language_model``
          - ``mimo_model.modality_submodules.images.encoders.qwen_visual``
        """
        language_provider = _make_language_provider()
        parallelism_config = _make_parallelism_config()
        source_bridge = MagicMock(spec=Qwen35VLBridge)
        source_bridge.provider_bridge = MagicMock(return_value=language_provider)

        _, routes = qwen35_vl_mimo_adapter(source_bridge, MagicMock(), parallelism_config)

        names = {route.name: route for route in routes}
        assert set(names.keys()) == {"language", "images"}

        assert names["language"].source_prefix == "language_model."
        assert names["language"].target_module_path == "language_model"

        # ``"images"`` matches both the parallelism config component key AND
        # the modality_submodules_spec key (MimoModelConfig validates these
        # align with module_to_grid_map).
        assert names["images"].source_prefix == "vision_model."
        assert names["images"].target_module_path == "modality_submodules.images.encoders.qwen_visual"

    def test_route_table_validates_against_parallelism_config(self):
        """The returned route table must pass ``validate_route_table`` against
        the same parallelism config — guarantees orchestrator can drive both
        routes without spurious "unmapped component" errors.
        """
        language_provider = _make_language_provider()
        parallelism_config = _make_parallelism_config()
        source_bridge = MagicMock(spec=Qwen35VLBridge)
        source_bridge.provider_bridge = MagicMock(return_value=language_provider)

        _, routes = qwen35_vl_mimo_adapter(source_bridge, MagicMock(), parallelism_config)

        validate_route_table(routes, parallelism_config=parallelism_config)
