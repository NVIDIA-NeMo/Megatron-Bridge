# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for MIMO Model Provider."""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.bridge.models.mimo import MimoModelProvider, MimoModelProviderResult
from megatron.bridge.training.mimo_config import MimoParallelismConfig, ModuleParallelismConfig


class TestMimoModelProvider:
    """Test cases for MimoModelProvider."""
    
    def test_provider_initialization_minimal(self):
        """Test provider initializes with minimal required fields."""
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        
        provider = MimoModelProvider(
            language_model_spec=language_spec,
        )
        
        assert provider.language_model_spec == language_spec
        assert provider.modality_submodules_spec == {}
        assert provider.special_token_ids == {}
        assert provider.mimo_parallelism_config is None
    
    def test_provider_initialization_full(self):
        """Test provider initializes with all fields."""
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        modality_spec = ModuleSpec(module=Mock, params={})
        mimo_parallelism_config = MimoParallelismConfig(
            llm_module_name="llm",
            module_parallelisms={
                "llm": ModuleParallelismConfig(tensor_parallel=2),
            },
        )
        
        provider = MimoModelProvider(
            language_model_spec=language_spec,
            modality_submodules_spec={"images": modality_spec},
            special_token_ids={"images": 32000},
            mimo_parallelism_config=mimo_parallelism_config,
            freeze_language_model=True,
            freeze_modality_encoders={"images": True},
        )
        
        assert provider.language_model_spec == language_spec
        assert "images" in provider.modality_submodules_spec
        assert provider.special_token_ids == {"images": 32000}
        assert provider.mimo_parallelism_config == mimo_parallelism_config
        assert provider.freeze_language_model is True
        assert provider.freeze_modality_encoders == {"images": True}
    
    @patch('megatron.bridge.models.mimo.mimo_provider.MimoModel')
    @patch('megatron.bridge.models.mimo.mimo_provider.build_hypercomm_grids')
    def test_provide_without_parallelism(self, mock_build_grids, mock_mimo_model):
        """Test provide() without mimo_parallelism_config (like llava_vlm.py)."""
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        
        provider = MimoModelProvider(
            language_model_spec=language_spec,
            special_token_ids={"images": 32000},
        )
        
        # Mock MimoModel
        mock_model_instance = MagicMock()
        mock_mimo_model.return_value = mock_model_instance
        
        result = provider.provide()
        
        # Should not build grids when no parallelism config
        mock_build_grids.assert_not_called()
        
        # Should create model
        assert result.model == mock_model_instance
        assert result.module_to_grid_map == {}
        assert result.topology == {}
        assert result.pg_collections == {}
    
    @patch('torch.distributed.get_rank')
    @patch('megatron.bridge.models.mimo.mimo_provider.MimoModel')
    @patch('megatron.bridge.models.mimo.mimo_provider.build_hypercomm_grids')
    @patch('megatron.bridge.models.mimo.mimo_provider._default_topology')
    def test_provide_with_parallelism(
        self, mock_topology, mock_build_grids, mock_mimo_model, mock_get_rank
    ):
        """Test provide() with parallelism config."""
        # Setup
        mock_get_rank.return_value = 0
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        
        mimo_parallelism_config = MimoParallelismConfig(
            llm_module_name="llm",
            module_parallelisms={
                "llm": ModuleParallelismConfig(
                    tensor_parallel=2,
                    data_parallel=2,
                ),
            },
        )
        
        # Mock grid
        mock_grid = MagicMock()
        mock_grid.rank_offset = 0
        mock_grid.size = 4
        mock_grid.get_pg.return_value = MagicMock()
        mock_build_grids.return_value = {"llm": mock_grid}
        
        mock_topology.return_value = {"llm": []}
        
        provider = MimoModelProvider(
            language_model_spec=language_spec,
            mimo_parallelism_config=mimo_parallelism_config,
        )
        
        # Mock MimoModel
        mock_model_instance = MagicMock()
        mock_mimo_model.return_value = mock_model_instance
        
        result = provider.provide()
        
        # Should build grids
        mock_build_grids.assert_called_once_with(mimo_parallelism_config)
        
        # Should create model
        assert result.model == mock_model_instance
        assert "llm" in result.module_to_grid_map
        assert "llm" in result.pg_collections
    
    @patch('torch.distributed.get_rank')
    @patch('megatron.bridge.models.mimo.mimo_provider.build_hypercomm_grids')
    def test_non_participating_rank(self, mock_build_grids, mock_get_rank):
        """Test that non-participating ranks get model=None."""
        # Rank 10 doesn't participate
        mock_get_rank.return_value = 10
        
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        mimo_parallelism_config = MimoParallelismConfig(
            llm_module_name="llm",
            module_parallelisms={
                "llm": ModuleParallelismConfig(
                    tensor_parallel=2,
                    data_parallel=2,
                    rank_offset=0,
                ),
            },
        )
        
        # Mock grid (only ranks 0-3)
        mock_grid = MagicMock()
        mock_grid.rank_offset = 0
        mock_grid.size = 4
        mock_build_grids.return_value = {"llm": mock_grid}
        
        provider = MimoModelProvider(
            language_model_spec=language_spec,
            mimo_parallelism_config=mimo_parallelism_config,
        )
        
        result = provider.provide()
        
        # Non-participating rank should get None
        assert result.model is None
        assert result.pg_collections["llm"] is None
    
    def test_inject_pg_collection_into_language_spec(self):
        """Test that pg_collection is injected into language specs."""
        language_spec = ModuleSpec(module=Mock, params={})
        
        provider = MimoModelProvider(language_model_spec=language_spec)
        
        mock_pg_collection = MagicMock()
        injected_spec = provider._inject_pg_collection_into_language_spec(
            language_spec, mock_pg_collection
        )
        
        assert injected_spec.params["pg_collection"] == mock_pg_collection
        # Should be a deep copy, not the same object
        assert injected_spec is not language_spec
    
    def test_inject_pg_collection_into_modality_spec(self):
        """Test pg_collection injection into modality submodule specs."""
        encoder_spec = ModuleSpec(module=Mock, params={})
        modality_spec = ModuleSpec(
            module=Mock,
            params={},
            submodules={"encoders": {"clip": encoder_spec}},
        )
        
        provider = MimoModelProvider(
            language_model_spec=ModuleSpec(module=Mock, params={})
        )
        
        mock_pg_collection = MagicMock()
        mock_pg_collection.tp = MagicMock()
        
        injected_spec = provider._inject_pg_collection_into_modality_spec(
            modality_spec, mock_pg_collection
        )
        
        # Check encoder has pg_collection
        assert (
            injected_spec.submodules["encoders"]["clip"].params["pg_collection"]
            == mock_pg_collection
        )
    
    @patch('megatron.bridge.models.mimo.mimo_provider.MimoModel')
    def test_freezing_language_model(self, mock_mimo_model):
        """Test freeze_language_model works."""
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        
        # Create mock model with parameters
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.requires_grad = True
        mock_model.language_model.parameters.return_value = [mock_param]
        mock_mimo_model.return_value = mock_model
        
        provider = MimoModelProvider(
            language_model_spec=language_spec,
            freeze_language_model=True,
        )
        
        result = provider.provide()
        
        # Check parameter was frozen
        assert mock_param.requires_grad is False
    
    @patch('torch.distributed.get_rank')
    @patch('megatron.bridge.models.mimo.mimo_provider.MimoModel')
    @patch('megatron.bridge.models.mimo.mimo_provider.build_hypercomm_grids')
    @patch('megatron.bridge.models.mimo.mimo_provider._default_topology')
    def test_per_encoder_parallelism(
        self, mock_topology, mock_build_grids, mock_mimo_model, mock_get_rank
    ):
        """Test per-encoder parallelism with different TP per encoder."""
        # Setup
        mock_get_rank.return_value = 0
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        clip_spec = ModuleSpec(module=Mock, params={})
        dino_spec = ModuleSpec(module=Mock, params={})
        
        mimo_parallelism_config = MimoParallelismConfig(
            llm_module_name="llm",
            module_parallelisms={
                "llm": ModuleParallelismConfig(tensor_parallel=8, data_parallel=1),
                "clip_encoder": ModuleParallelismConfig(tensor_parallel=2, data_parallel=1),
                "dino_encoder": ModuleParallelismConfig(tensor_parallel=4, data_parallel=1),
            },
        )
        
        # Mock grids - each encoder gets different grid
        llm_grid = MagicMock()
        llm_grid.rank_offset = 0
        llm_grid.size = 8
        llm_grid.get_pg.return_value = MagicMock()
        
        clip_grid = MagicMock()
        clip_grid.rank_offset = 0
        clip_grid.size = 2
        clip_grid.get_pg.return_value = MagicMock()
        
        dino_grid = MagicMock()
        dino_grid.rank_offset = 0
        dino_grid.size = 4
        dino_grid.get_pg.return_value = MagicMock()
        
        mock_build_grids.return_value = {
            "llm": llm_grid,
            "clip_encoder": clip_grid,
            "dino_encoder": dino_grid,
        }
        
        mock_topology.return_value = {
            "clip_encoder": ["llm"],
            "dino_encoder": ["llm"],
            "llm": [],
        }
        
        provider = MimoModelProvider(
            language_model_spec=language_spec,
            modality_submodules_spec={
                "clip_encoder": clip_spec,
                "dino_encoder": dino_spec,
            },
            special_token_ids={
                "clip_encoder": 32000,
                "dino_encoder": 32001,
            },
            mimo_parallelism_config=mimo_parallelism_config,
        )
        
        # Mock MimoModel
        mock_model_instance = MagicMock()
        mock_mimo_model.return_value = mock_model_instance
        
        result = provider.provide()
        
        # Should build grids with all three modules
        mock_build_grids.assert_called_once_with(mimo_parallelism_config)
        
        # Should have pg_collections for all modules
        assert "llm" in result.pg_collections
        assert "clip_encoder" in result.pg_collections
        assert "dino_encoder" in result.pg_collections
        
        # Should create model
        assert result.model == mock_model_instance


class TestMimoModelProviderResult:
    """Test cases for MimoModelProviderResult."""
    
    def test_result_initialization(self):
        """Test result container initializes correctly."""
        mock_model = MagicMock()
        grids = {"llm": MagicMock()}
        topology = {"llm": []}
        pg_collections = {"llm": MagicMock()}
        
        result = MimoModelProviderResult(
            model=mock_model,
            module_to_grid_map=grids,
            topology=topology,
            pg_collections=pg_collections,
        )
        
        assert result.model == mock_model
        assert result.module_to_grid_map == grids
        assert result.topology == topology
        assert result.pg_collections == pg_collections
