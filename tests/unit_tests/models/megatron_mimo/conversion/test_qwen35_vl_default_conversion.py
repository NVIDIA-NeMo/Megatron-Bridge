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

"""Unit tests for Qwen3.5-VL default MegatronMIMO conversion metadata."""

from unittest.mock import MagicMock

import pytest

from megatron.bridge.models.megatron_mimo.conversion import (
    MIMOComponent,
    get_mimo_conversion_spec,
    supports_mimo_conversion,
    validate_route_table,
)
from megatron.bridge.models.megatron_mimo.conversion.orchestrator import _reset_registry_for_tests
from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)
from megatron.bridge.models.megatron_mimo.model_config import MegatronMIMOModelConfig
from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import Qwen35VLBridge
from megatron.bridge.models.qwen_vl.qwen35_vl_provider import _TRANSFORMERS_HAS_QWEN3_5


pytestmark = pytest.mark.skipif(not _TRANSFORMERS_HAS_QWEN3_5, reason="transformers does not have qwen3_5 support")


def _make_parallelism_config() -> MegatronMIMOParallelismConfig:
    """Two-component parallelism config matching the Qwen3.5-VL MIMO route table."""
    return MegatronMIMOParallelismConfig(
        module_parallelisms={
            "language": ModuleParallelismConfig(tensor_model_parallel_size=1),
            "images": ModuleParallelismConfig(tensor_model_parallel_size=1),
        }
    )


def _get_qwen35_conversion_spec():
    _reset_registry_for_tests()
    return get_mimo_conversion_spec(Qwen35VLBridge)


def _run_qwen35_conversion_spec(
    monkeypatch,
    parallelism_config: MegatronMIMOParallelismConfig,
) -> tuple[MegatronMIMOModelConfig, list[MIMOComponent], Qwen35VLBridge, object]:
    source_bridge = Qwen35VLBridge()
    source_model_config = MagicMock(image_token_id=248056)
    monkeypatch.setattr(source_bridge, "model_config_bridge", MagicMock(return_value=source_model_config))
    monkeypatch.setattr(source_bridge, "provider_bridge", MagicMock(side_effect=AssertionError("provider path used")))
    hf_pretrained = object()
    config, routes = _get_qwen35_conversion_spec()(source_bridge, hf_pretrained, parallelism_config)
    return config, routes, source_bridge, hf_pretrained


class TestQwen35VLDefaultMIMOConversion:
    def test_default_conversion_spec_available_for_qwen35_vl_bridge(self):
        conversion_spec = _get_qwen35_conversion_spec()

        assert callable(conversion_spec)
        assert supports_mimo_conversion(Qwen35VLBridge)

    def test_returns_config_and_routes(self, monkeypatch):
        parallelism_config = _make_parallelism_config()

        config, routes, source_bridge, hf_pretrained = _run_qwen35_conversion_spec(monkeypatch, parallelism_config)

        source_bridge.model_config_bridge.assert_called_once_with(hf_pretrained)
        source_bridge.provider_bridge.assert_not_called()
        assert isinstance(config, MegatronMIMOModelConfig)
        assert config.megatron_mimo_parallelism_config is parallelism_config

        assert len(routes) == 2
        assert all(isinstance(r, MIMOComponent) for r in routes)

    def test_route_table_contents(self, monkeypatch):
        """Routes match the parameter prefixes used by Qwen35VLBridge.mapping_registry.

        Source bridge mapping registry uses ``language_model.`` and ``vision_model.``
        as the two top-level prefixes.
        The route table strips these and dispatches to:
          - ``mimo_model.language_model``
          - ``mimo_model.modality_submodules.images.encoders.qwen_visual``
        """
        _, routes, _, _ = _run_qwen35_conversion_spec(monkeypatch, _make_parallelism_config())

        names = {route.name: route for route in routes}
        assert set(names.keys()) == {"language", "images"}

        assert names["language"].source_prefix == "language_model."
        assert names["language"].target_module_path == "language_model"

        # ``"images"`` matches both the parallelism config component key AND
        # the modality_submodules_spec key (MimoModelConfig validates these
        # align with module_to_grid_map).
        assert names["images"].source_prefix == "vision_model."
        assert names["images"].target_module_path == "modality_submodules.images.encoders.qwen_visual"

    def test_route_table_validates_against_parallelism_config(self, monkeypatch):
        """The returned route table must pass ``validate_route_table`` against
        the same parallelism config — guarantees orchestrator can drive both
        routes without spurious "unmapped component" errors.
        """
        parallelism_config = _make_parallelism_config()
        _, routes, _, _ = _run_qwen35_conversion_spec(monkeypatch, parallelism_config)

        validate_route_table(routes, parallelism_config=parallelism_config)
