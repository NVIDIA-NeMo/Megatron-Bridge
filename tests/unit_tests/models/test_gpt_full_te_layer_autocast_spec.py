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
Unit tests for gpt_full_te_layer_autocast_spec.py

Tests AutocastTransformerLayer, TETransformerLayerAutocast, and related utilities
for GPT models with Transformer Engine autocast functionality.
"""

from unittest.mock import Mock, patch

import pytest
import torch

from megatron.bridge.models.gpt_full_te_layer_autocast_spec import (
    AutocastTransformerLayer,
    TETransformerLayerAutocast,
    get_gpt_full_te_layer_autocast_spec,
    torch_dtype_from_precision,
)


class TestTorchDtypeFromPrecision:
    """Test the torch_dtype_from_precision function."""

    @pytest.mark.parametrize(
        "precision,expected_dtype",
        [
            ("bf16", torch.bfloat16),
            ("bf16-mixed", torch.bfloat16),
            (16, torch.float16),
            ("16", torch.float16),
            ("16-mixed", torch.float16),
            (32, torch.float32),
            ("32", torch.float32),
            ("32-true", torch.float32),
        ],
    )
    def test_valid_precision_mappings(self, precision, expected_dtype):
        """Test valid precision to dtype mappings."""
        result = torch_dtype_from_precision(precision)
        assert result == expected_dtype

    @pytest.mark.parametrize(
        "invalid_precision",
        [
            "invalid",
            64,
            "64",
            None,
            [],
            {},
        ],
    )
    def test_invalid_precision_raises_error(self, invalid_precision):
        """Test that invalid precision values raise ValueError."""
        with pytest.raises(ValueError, match="Could not parse the precision"):
            torch_dtype_from_precision(invalid_precision)

    def test_torch_dtype_from_precision_edge_cases(self):
        """Test edge cases for torch_dtype_from_precision."""
        # Test case sensitivity doesn't matter for string inputs
        assert torch_dtype_from_precision("16") == torch.float16
        assert torch_dtype_from_precision("32") == torch.float32

        # Test that we handle both int and string versions
        assert torch_dtype_from_precision(16) == torch_dtype_from_precision("16")
        assert torch_dtype_from_precision(32) == torch_dtype_from_precision("32")


class TestAutocastTransformerLayer:
    """Test the AutocastTransformerLayer class."""

    @pytest.fixture
    def mock_init_method(self):
        """Mock initialization method."""
        return lambda x: torch.nn.init.normal_(x, 0.0, 0.02)

    @pytest.fixture
    def basic_config(self, mock_init_method):
        """Basic configuration for AutocastTransformerLayer."""
        return {
            "hidden_size": 512,
            "ffn_hidden_size": 2048,
            "layernorm_epsilon": 1e-5,
            "num_attention_heads": 8,
            "init_method": mock_init_method,
            "output_layer_init_method": mock_init_method,
            "hidden_dropout": 0.1,
            "attention_dropout": 0.1,
            "layer_number": 1,
            "kv_channels": 64,
            "tp_size": 1,
            "params_dtype": torch.float32,
        }

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_autocast_transformer_layer_init(self, basic_config):
        """Test AutocastTransformerLayer initialization."""
        layer = AutocastTransformerLayer(**basic_config)

        assert isinstance(layer, AutocastTransformerLayer)
        assert layer.dtype == torch.float16  # Default autocast_dtype is 16

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_autocast_transformer_layer_with_bf16(self, basic_config):
        """Test AutocastTransformerLayer with bf16 precision."""
        basic_config["autocast_dtype"] = "bf16"
        layer = AutocastTransformerLayer(**basic_config)

        assert layer.dtype == torch.bfloat16

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_autocast_transformer_layer_with_fp32(self, basic_config):
        """Test AutocastTransformerLayer with fp32 precision."""
        basic_config["autocast_dtype"] = 32
        layer = AutocastTransformerLayer(**basic_config)

        assert layer.dtype == torch.float32

    @patch("transformer_engine.pytorch.TransformerLayer.__init__")
    def test_autocast_transformer_layer_te_version_handling(self, mock_te_init, basic_config):
        """Test handling of different Transformer Engine versions."""
        mock_te_init.return_value = None

        with patch("megatron.bridge.models.gpt_full_te_layer_autocast_spec.version") as mock_version:
            # Test version > 1.5.0
            mock_version.return_value = "1.6.0"
            _ = AutocastTransformerLayer(**basic_config)

            # Check that the init was called with appropriate arguments
            mock_te_init.assert_called_once()
            args, kwargs = mock_te_init.call_args
            assert "ub_overlap_ag" in kwargs
            assert "ub_overlap_rs" in kwargs

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_autocast_transformer_layer_forward_with_autocast(self, basic_config):
        """Test forward pass with autocast enabled."""
        layer = AutocastTransformerLayer(**basic_config)

        # Mock the parent forward method to avoid complex TE dependencies
        with patch.object(layer.__class__.__bases__[0], "forward") as mock_forward:
            mock_forward.return_value = torch.randn(2, 4, 512)

            hidden_states = torch.randn(2, 4, 512)
            attention_mask = torch.ones(2, 4, 4).bool()

            result = layer.forward(hidden_states, attention_mask)

            assert isinstance(result, torch.Tensor)
            mock_forward.assert_called_once()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_autocast_transformer_layer_forward_fp32_no_autocast(self, basic_config):
        """Test forward pass with fp32 (no autocast)."""
        basic_config["autocast_dtype"] = 32
        layer = AutocastTransformerLayer(**basic_config)

        with patch.object(layer.__class__.__bases__[0], "forward") as mock_forward:
            mock_forward.return_value = torch.randn(2, 4, 512)

            hidden_states = torch.randn(2, 4, 512)
            attention_mask = torch.ones(2, 4, 4).bool()

            result = layer.forward(hidden_states, attention_mask)

            assert isinstance(result, torch.Tensor)
            mock_forward.assert_called_once()


class TestTETransformerLayerAutocast:
    """Test the TETransformerLayerAutocast class."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for TETransformerLayerAutocast."""
        config = Mock()
        config.hidden_size = 512
        config.ffn_hidden_size = 2048
        config.layernorm_epsilon = 1e-5
        config.num_attention_heads = 8
        config.init_method = lambda x: torch.nn.init.normal_(x, 0.0, 0.02)
        config.output_layer_init_method = lambda x: torch.nn.init.normal_(x, 0.0, 0.02)
        config.hidden_dropout = 0.1
        config.attention_dropout = 0.1
        config.kv_channels = 64
        config.params_dtype = torch.float32
        config.gradient_accumulation_fusion = False
        config.sequence_parallel = False
        config.apply_residual_connection_post_layernorm = False
        config.tp_comm_overlap = False
        config.tp_comm_bulk_wgrad = True
        config.tp_comm_bulk_dgrad = True
        config.tp_comm_split_ag = True
        config.tp_comm_split_rs = True
        config.tp_comm_atomic_ag = False
        config.tp_comm_atomic_rs = False
        config.layernorm_zero_centered_gamma = False
        config.use_cpu_initialization = False
        config.bf16 = False
        config.num_layers = 12
        config.cuda_graph_impl = "none"
        config.cuda_graph_scope = []
        config.cpu_offloading = False
        config.recompute_granularity = None
        config.virtual_pipeline_model_parallel_size = None
        return config

    @patch("megatron.core.parallel_state.get_tensor_model_parallel_world_size")
    @patch("megatron.core.parallel_state.get_pipeline_model_parallel_rank")
    @patch("megatron.core.parallel_state.get_pipeline_model_parallel_world_size")
    def test_te_transformer_layer_autocast_init(
        self, mock_pp_world_size, mock_pp_rank, mock_tp_world_size, mock_config
    ):
        """Test TETransformerLayerAutocast initialization."""
        mock_tp_world_size.return_value = 1
        mock_pp_rank.return_value = 0
        mock_pp_world_size.return_value = 1

        with patch("megatron.bridge.models.gpt_full_te_layer_autocast_spec.AutocastTransformerLayer"):
            layer = TETransformerLayerAutocast(mock_config, layer_number=0)

            assert isinstance(layer, TETransformerLayerAutocast)
            assert layer.layer_number == 0
            assert layer.is_first_microbatch is True

    @patch("megatron.core.parallel_state.get_tensor_model_parallel_world_size")
    @patch("megatron.core.parallel_state.get_pipeline_model_parallel_rank")
    @patch("megatron.core.parallel_state.get_pipeline_model_parallel_world_size")
    def test_te_transformer_layer_autocast_forward(
        self, mock_pp_world_size, mock_pp_rank, mock_tp_world_size, mock_config
    ):
        """Test TETransformerLayerAutocast forward pass."""
        mock_tp_world_size.return_value = 1
        mock_pp_rank.return_value = 0
        mock_pp_world_size.return_value = 1

        # Ensure external_cuda_graph is False so we get tuple return
        mock_config.cuda_graph_impl = "none"

        with patch("megatron.bridge.models.gpt_full_te_layer_autocast_spec.AutocastTransformerLayer") as mock_autocast:
            mock_transformer = Mock()
            mock_transformer.forward.return_value = torch.randn(2, 4, 512)
            mock_autocast.return_value = mock_transformer

            layer = TETransformerLayerAutocast(mock_config, layer_number=0)
            # Ensure layer is not in training mode to avoid external_cuda_graph path
            layer.training = False

            hidden_states = torch.randn(2, 4, 512)
            attention_mask = torch.ones(2, 4, 4).bool()

            # Test that forward returns (hidden_states, None)
            forward_result = layer.forward(hidden_states, attention_mask=attention_mask)

            assert isinstance(forward_result, tuple)
            assert len(forward_result) == 2
            result, context = forward_result

            assert isinstance(result, torch.Tensor)
            assert context is None
            assert layer.is_first_microbatch is False

    @patch("megatron.core.parallel_state.get_tensor_model_parallel_world_size")
    @patch("megatron.core.parallel_state.get_pipeline_model_parallel_rank")
    @patch("megatron.core.parallel_state.get_pipeline_model_parallel_world_size")
    def test_te_transformer_layer_autocast_get_layer_offset(
        self, mock_pp_world_size, mock_pp_rank, mock_tp_world_size, mock_config
    ):
        """Test _get_layer_offset method."""
        mock_tp_world_size.return_value = 1
        mock_pp_rank.return_value = 1  # Second pipeline rank
        mock_pp_world_size.return_value = 2  # Two pipeline ranks

        with patch("megatron.bridge.models.gpt_full_te_layer_autocast_spec.AutocastTransformerLayer"):
            layer = TETransformerLayerAutocast(mock_config, layer_number=0)

            offset = layer._get_layer_offset()

            expected_offset = 1 * (12 // 2)  # rank * (num_layers // pp_size)
            assert offset == expected_offset

    @patch("megatron.core.parallel_state.get_tensor_model_parallel_world_size")
    @patch("megatron.core.parallel_state.get_pipeline_model_parallel_rank")
    @patch("megatron.core.parallel_state.get_pipeline_model_parallel_world_size")
    def test_te_transformer_layer_autocast_with_cuda_graph(
        self, mock_pp_world_size, mock_pp_rank, mock_tp_world_size, mock_config
    ):
        """Test TETransformerLayerAutocast with CUDA graph enabled."""
        mock_tp_world_size.return_value = 1
        mock_pp_rank.return_value = 0
        mock_pp_world_size.return_value = 1
        mock_config.cuda_graph_impl = "local"
        mock_config.cuda_graph_scope = []  # Empty list means layerwise graph

        with patch("megatron.bridge.models.gpt_full_te_layer_autocast_spec.AutocastTransformerLayer"):
            with patch("megatron.bridge.models.gpt_full_te_layer_autocast_spec.CudaGraphManager") as mock_cuda_manager:
                # Create a proper mock instance that inherits from nn.Module
                class MockCudaGraphManager(torch.nn.Module):
                    def __init__(self, config):
                        super().__init__()

                mock_cuda_manager.return_value = MockCudaGraphManager(mock_config)

                layer = TETransformerLayerAutocast(mock_config, layer_number=0)
                layer.training = True

                assert hasattr(layer, "cudagraph_manager")

    @patch("megatron.core.parallel_state.get_tensor_model_parallel_world_size")
    @patch("megatron.core.parallel_state.get_pipeline_model_parallel_rank")
    @patch("megatron.core.parallel_state.get_pipeline_model_parallel_world_size")
    def test_te_transformer_layer_autocast_with_full_iteration_cuda_graph(
        self, mock_pp_world_size, mock_pp_rank, mock_tp_world_size, mock_config
    ):
        """Test TETransformerLayerAutocast with full_iteration CUDA graph (cudagraph_manager should NOT be created)."""
        mock_tp_world_size.return_value = 1
        mock_pp_rank.return_value = 0
        mock_pp_world_size.return_value = 1
        mock_config.cuda_graph_impl = "local"
        mock_config.cuda_graph_scope = ["full_iteration"]  # Full iteration graph

        with patch("megatron.bridge.models.gpt_full_te_layer_autocast_spec.AutocastTransformerLayer"):
            with patch("megatron.bridge.models.gpt_full_te_layer_autocast_spec.CudaGraphManager") as mock_cuda_manager:
                layer = TETransformerLayerAutocast(mock_config, layer_number=0)
                layer.training = True

                # cudagraph_manager should NOT be created for full_iteration scope
                assert not hasattr(layer, "cudagraph_manager")
                mock_cuda_manager.assert_not_called()

    @patch("megatron.core.parallel_state.get_tensor_model_parallel_world_size")
    @patch("megatron.core.parallel_state.get_pipeline_model_parallel_rank")
    @patch("megatron.core.parallel_state.get_pipeline_model_parallel_world_size")
    def test_te_transformer_layer_autocast_external_cuda_graph(
        self, mock_pp_world_size, mock_pp_rank, mock_tp_world_size, mock_config
    ):
        """Test TETransformerLayerAutocast with external CUDA graph (transformer_engine impl)."""
        mock_tp_world_size.return_value = 1
        mock_pp_rank.return_value = 0
        mock_pp_world_size.return_value = 1
        mock_config.cuda_graph_impl = "transformer_engine"
        mock_config.cuda_graph_scope = ["attn", "mlp"]  # TE supports multi-scope

        with patch("megatron.bridge.models.gpt_full_te_layer_autocast_spec.AutocastTransformerLayer") as mock_autocast:
            mock_transformer = Mock()
            mock_transformer.forward.return_value = torch.randn(2, 4, 512)
            mock_autocast.return_value = mock_transformer

            layer = TETransformerLayerAutocast(mock_config, layer_number=0)
            layer.training = True

            hidden_states = torch.randn(2, 4, 512)
            result = layer.forward(hidden_states)

            # Should return tensor directly for external CUDA graph
            assert isinstance(result, torch.Tensor)

    @patch("megatron.core.parallel_state.get_tensor_model_parallel_world_size")
    @patch("megatron.core.parallel_state.get_pipeline_model_parallel_rank")
    @patch("megatron.core.parallel_state.get_pipeline_model_parallel_world_size")
    def test_te_transformer_layer_autocast_sharded_state_dict(
        self, mock_pp_world_size, mock_pp_rank, mock_tp_world_size, mock_config
    ):
        """Test sharded_state_dict method."""
        mock_tp_world_size.return_value = 1
        mock_pp_rank.return_value = 0
        mock_pp_world_size.return_value = 1

        with patch("megatron.bridge.models.gpt_full_te_layer_autocast_spec.AutocastTransformerLayer"):
            with patch(
                "megatron.bridge.models.gpt_full_te_layer_autocast_spec.make_sharded_tensors_for_checkpoint"
            ) as mock_make_sharded:
                mock_make_sharded.return_value = {"test_key": "test_value"}

                layer = TETransformerLayerAutocast(mock_config, layer_number=0)

                result = layer.sharded_state_dict()

                assert isinstance(result, dict)
                mock_make_sharded.assert_called_once()


class TestGetGPTFullTELayerAutocastSpec:
    """Test the get_gpt_full_te_layer_autocast_spec function."""

    @pytest.fixture
    def mock_transformer_config(self):
        """Mock transformer configuration."""
        config = Mock()
        config.num_layers = 12
        return config

    @patch("megatron.bridge.models.gpt_full_te_layer_autocast_spec.get_num_layers_to_build")
    def test_get_gpt_full_te_layer_autocast_spec_basic(self, mock_get_num_layers, mock_transformer_config):
        """Test basic functionality of get_gpt_full_te_layer_autocast_spec."""
        mock_get_num_layers.return_value = 12

        spec = get_gpt_full_te_layer_autocast_spec(mock_transformer_config)

        # Check that we get a TransformerBlockSubmodules object
        from megatron.core.transformer.transformer_block import TransformerBlockSubmodules

        assert isinstance(spec, TransformerBlockSubmodules)

        # Check that layer_specs has correct number of layers
        assert len(spec.layer_specs) == 12

        # Check that each spec is for TETransformerLayerAutocast
        from megatron.core.transformer.spec_utils import ModuleSpec

        for layer_spec in spec.layer_specs:
            assert isinstance(layer_spec, ModuleSpec)
            assert layer_spec.module == TETransformerLayerAutocast

    @patch("megatron.bridge.models.gpt_full_te_layer_autocast_spec.get_num_layers_to_build")
    def test_get_gpt_full_te_layer_autocast_spec_different_num_layers(
        self, mock_get_num_layers, mock_transformer_config
    ):
        """Test with different number of layers."""
        mock_get_num_layers.return_value = 24

        spec = get_gpt_full_te_layer_autocast_spec(mock_transformer_config)

        assert len(spec.layer_specs) == 24


class TestVersionCompatibility:
    """Test version compatibility handling."""

    @patch("megatron.bridge.models.gpt_full_te_layer_autocast_spec.version")
    def test_te_version_compatibility_old_version(self, mock_version):
        """Test handling of older TE versions."""
        mock_version.return_value = "1.4.0"

        basic_config = {
            "hidden_size": 512,
            "ffn_hidden_size": 2048,
            "layernorm_epsilon": 1e-5,
            "num_attention_heads": 8,
            "init_method": lambda x: torch.nn.init.normal_(x, 0.0, 0.02),
            "output_layer_init_method": lambda x: torch.nn.init.normal_(x, 0.0, 0.02),
            "hidden_dropout": 0.1,
            "attention_dropout": 0.1,
            "tp_size": 1,
            "params_dtype": torch.float32,
        }

        with patch("transformer_engine.pytorch.TransformerLayer.__init__") as mock_te_init:
            mock_te_init.return_value = None
            _ = AutocastTransformerLayer(**basic_config)

            # Should use old-style parameters for older versions
            args, kwargs = mock_te_init.call_args
            assert "ub_split_ag" in kwargs
            assert "ub_split_rs" in kwargs

    @patch("megatron.bridge.models.gpt_full_te_layer_autocast_spec.version")
    def test_te_version_compatibility_new_version(self, mock_version):
        """Test handling of newer TE versions."""
        mock_version.return_value = "1.7.0"

        basic_config = {
            "hidden_size": 512,
            "ffn_hidden_size": 2048,
            "layernorm_epsilon": 1e-5,
            "num_attention_heads": 8,
            "init_method": lambda x: torch.nn.init.normal_(x, 0.0, 0.02),
            "output_layer_init_method": lambda x: torch.nn.init.normal_(x, 0.0, 0.02),
            "hidden_dropout": 0.1,
            "attention_dropout": 0.1,
            "tp_size": 1,
            "params_dtype": torch.float32,
        }

        with patch("transformer_engine.pytorch.TransformerLayer.__init__") as mock_te_init:
            mock_te_init.return_value = None
            _ = AutocastTransformerLayer(**basic_config)

            # Should use new-style parameters for newer versions
            args, kwargs = mock_te_init.call_args
            assert "ub_overlap_ag" in kwargs
            assert "ub_overlap_rs" in kwargs
            assert "ub_overlap_rs_dgrad" in kwargs
