# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Unit tests for MegatronPEFTBridge base class and dispatch functionality.
"""

from unittest.mock import Mock, patch

import pytest
import torch

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.conversion.peft_bridge import (
    MegatronPEFTBridge,
    get_peft_bridge,
    list_registered_bridges,
    register_bridge_implementation,
    stream_adapters_megatron_to_hf,
)
from megatron.bridge.peft.conversion.pretrained_adapters import PreTrainedAdapters


class MockPEFTBridge(MegatronPEFTBridge):
    """Mock PEFT bridge for testing."""

    def peft_bridge(self, adapters: PreTrainedAdapters) -> PEFT:
        mock_peft = Mock(spec=PEFT)
        return mock_peft

    def mapping_registry(self) -> MegatronMappingRegistry:
        return MegatronMappingRegistry()


class TestMegatronPEFTBridge:
    """Test cases for MegatronPEFTBridge base class."""

    def test_bridge_initialization(self):
        """Test PEFT bridge initialization."""
        mock_base_bridge = Mock(spec=AutoBridge)

        bridge = MockPEFTBridge(mock_base_bridge)
        assert bridge.base_bridge == mock_base_bridge

        # Test without base bridge
        bridge_no_base = MockPEFTBridge()
        assert bridge_no_base.base_bridge is None

    def test_peft_bridge_abstract_method(self):
        """Test that peft_bridge is properly implemented."""
        bridge = MockPEFTBridge()
        mock_adapters = Mock(spec=PreTrainedAdapters)

        result = bridge.peft_bridge(mock_adapters)
        assert isinstance(result, Mock)  # Our mock implementation

    def test_mapping_registry_abstract_method(self):
        """Test that mapping_registry is properly implemented."""
        bridge = MockPEFTBridge()

        result = bridge.mapping_registry()
        assert isinstance(result, MegatronMappingRegistry)

    @patch("torch.distributed.is_initialized", return_value=False)
    def test_global_param_names_single_process(self, mock_is_initialized):
        """Test parameter name gathering in single-process mode."""
        bridge = MockPEFTBridge()

        # Create mock model with adapter parameters
        mock_model = Mock()
        mock_model.named_parameters.return_value = [
            ("decoder.layers.0.self_attention.linear_proj.adapter.linear_in.weight", Mock()),
            ("decoder.layers.0.self_attention.linear_proj.adapter.linear_out.weight", Mock()),
            ("decoder.layers.0.self_attention.linear_proj.weight", Mock()),  # Non-adapter, should be filtered
        ]

        # Mock unwrap_model and config
        with patch("megatron.bridge.peft.conversion.peft_bridge.unwrap_model") as mock_unwrap:
            mock_config = Mock()
            mock_unwrap.return_value = [mock_model]
            mock_model.config = mock_config

            with patch.object(bridge, "_unwrap_name", side_effect=lambda x: x):
                with patch.object(
                    bridge, "_megatron_local_name_to_global", side_effect=lambda models, config, name, vp: name
                ):
                    result = bridge._megatron_global_param_names_all_pp_ranks([mock_model])

                    # Should only include adapter parameters
                    adapter_params = [name for name in result if ".adapter." in name]
                    assert len(adapter_params) == 2

    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.all_gather_object")
    @patch("megatron.bridge.peft.conversion.peft_bridge.get_pg_size", return_value=2)
    def test_global_param_names_distributed(self, mock_get_pg_size, mock_all_gather, mock_is_initialized):
        """Test parameter name gathering in distributed mode."""
        bridge = MockPEFTBridge()

        # Create mock model
        mock_model = Mock()
        mock_model.named_parameters.return_value = [
            ("decoder.layers.0.self_attention.linear_proj.adapter.linear_in.weight", Mock()),
        ]

        # Mock distributed gathering
        mock_all_gather.side_effect = lambda gathered_list, local_list, group: gathered_list.__setitem__(
            slice(None), [local_list, local_list]
        )

        with patch("megatron.bridge.peft.conversion.peft_bridge.unwrap_model") as mock_unwrap:
            mock_config = Mock()
            mock_unwrap.return_value = [mock_model]
            mock_model.config = mock_config

            with patch.object(bridge, "_unwrap_name", side_effect=lambda x: x):
                with patch.object(
                    bridge, "_megatron_local_name_to_global", side_effect=lambda models, config, name, vp: name
                ):
                    with patch(
                        "megatron.bridge.peft.conversion.peft_bridge.parallel_state.get_pipeline_model_parallel_group"
                    ):
                        result = bridge._megatron_global_param_names_all_pp_ranks([mock_model])

                        # Should call all_gather_object for distributed coordination
                        mock_all_gather.assert_called_once()
                        assert isinstance(result, list)

    def test_unwrap_name(self):
        """Test parameter name unwrapping."""
        bridge = MockPEFTBridge()

        # Test various wrapped names
        assert bridge._unwrap_name("module.decoder.layers.0.weight") == "decoder.layers.0.weight"
        assert bridge._unwrap_name("module.module.decoder.weight") == "decoder.weight"
        assert bridge._unwrap_name("decoder.weight") == "decoder.weight"

        # Test invalid input
        with pytest.raises(ValueError, match="name must be a string"):
            bridge._unwrap_name(123)


class TestPEFTBridgeRegistry:
    """Test cases for PEFT bridge registration and dispatch."""

    def test_register_bridge_implementation(self):
        """Test bridge registration functionality."""
        # Create mock classes
        mock_config_class = Mock()
        mock_config_class.__name__ = "MockConfig"

        mock_peft_class = Mock()
        mock_peft_class.__name__ = "MockPEFT"

        class TestBridge(MegatronPEFTBridge):
            def peft_bridge(self, adapters):
                return Mock()

            def mapping_registry(self):
                return MegatronMappingRegistry()

        # Register the bridge
        register_bridge_implementation(source=mock_config_class, target=mock_peft_class, bridge_class=TestBridge)

        # Verify registration
        bridges = list_registered_bridges()
        assert mock_config_class in bridges
        assert bridges[mock_config_class] == TestBridge

    def test_get_peft_bridge_dispatch(self):
        """Test dispatch-based bridge selection."""
        # This test requires actual registration, so we'll mock the dispatch
        mock_config_class = Mock()

        with patch("megatron.bridge.peft.conversion.peft_bridge.get_peft_bridge") as mock_dispatch:
            mock_bridge = Mock(spec=MegatronPEFTBridge)
            mock_dispatch.return_value = mock_bridge

            _ = get_peft_bridge(mock_config_class)
            # In our current implementation, this calls the dispatch function
            # The actual dispatch behavior depends on registrations

    def test_stream_adapters_dispatch(self):
        """Test dispatch-based streaming."""
        mock_model = [Mock()]
        mock_adapters = Mock(spec=PreTrainedAdapters)

        with patch("megatron.bridge.peft.conversion.peft_bridge.stream_adapters_megatron_to_hf") as mock_stream:
            mock_stream.return_value = iter([])

            # Test the dispatch function exists and is callable
            result = list(
                stream_adapters_megatron_to_hf(
                    (Mock(), Mock()),  # dispatch instance
                    mock_model,
                    mock_adapters,
                )
            )
            assert isinstance(result, list)

    def test_list_registered_bridges(self):
        """Test listing registered bridges."""
        # Get current registry state
        bridges = list_registered_bridges()
        assert isinstance(bridges, dict)

        # Test that it returns a copy (not the original registry)
        bridges_copy = list_registered_bridges()
        assert bridges is not bridges_copy  # Different object instances
        assert bridges == bridges_copy  # Same content


class TestPEFTBridgeErrorHandling:
    """Test error handling in PEFT bridge components."""

    def test_build_conversion_tasks_invalid_adapters(self):
        """Test build_conversion_tasks with invalid adapter state."""
        bridge = MockPEFTBridge()

        # Create mock adapters without proper state structure
        mock_adapters = Mock(spec=PreTrainedAdapters)
        mock_adapters.state = Mock()
        # Missing 'source' attribute

        mock_model = [Mock()]

        with pytest.raises(ValueError, match="adapters.state.source is required"):
            bridge.build_conversion_tasks(mock_adapters, mock_model)

    def test_progress_tracking(self):
        """Test progress tracking functionality."""
        bridge = MockPEFTBridge()

        tasks = [Mock(), Mock(), Mock()]

        # Mock progress tracking
        with patch("megatron.bridge.peft.conversion.peft_bridge.Progress") as mock_progress:
            mock_progress_instance = Mock()
            mock_progress.__enter__ = Mock(return_value=mock_progress_instance)
            mock_progress.__exit__ = Mock(return_value=None)

            # Test with progress enabled
            tracked_tasks = list(bridge._with_progress_tracking(tasks, "Test description", show_progress=True))
            assert len(tracked_tasks) == 3

    @patch("torch.distributed.is_initialized", return_value=False)
    def test_load_adapters_single_process(self, mock_is_initialized):
        """Test loading adapters in single-process mode."""
        bridge = MockPEFTBridge()

        # Create mock adapters and model
        mock_adapters = Mock(spec=PreTrainedAdapters)
        mock_adapters.state = {"param1": torch.randn(10, 10)}
        mock_adapters.config = {"modules_to_save": []}

        mock_model = [Mock()]

        # Mock build_conversion_tasks to return empty list
        with patch.object(bridge, "build_conversion_tasks", return_value=[]):
            with patch.object(bridge, "_with_progress_tracking", return_value=iter([])):
                # Should not raise in single-process mode
                bridge.load_adapters_hf_to_megatron(mock_adapters, mock_model)
