# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for MIMO Model Provider."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.mimo import (
    MimoModelInfra,
    MimoModelProvider,
)
from megatron.bridge.models.mimo.mimo_config import MimoParallelismConfig, ModuleParallelismConfig


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
            module_parallelisms={
                "llm": ModuleParallelismConfig(tensor_model_parallel_size=2),
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

    def test_provider_has_mixin_fields(self):
        """Test provider has fields required by ModelProviderMixin."""
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        provider = MimoModelProvider(language_model_spec=language_spec)

        # Check mixin-required fields exist with defaults
        assert hasattr(provider, "fp16")
        assert hasattr(provider, "bf16")
        assert hasattr(provider, "use_cpu_initialization")
        assert hasattr(provider, "init_model_with_meta_device")
        assert hasattr(provider, "virtual_pipeline_model_parallel_size")

        # Check defaults
        assert provider.fp16 is False
        assert provider.bf16 is True
        assert provider.use_cpu_initialization is False

    @patch("megatron.bridge.models.mimo.mimo_provider.MimoModel")
    @patch("megatron.bridge.models.mimo.mimo_provider.build_hypercomm_grids")
    def test_provide_returns_model_directly(self, mock_build_grids, mock_mimo_model):
        """Test provide() returns model directly, not a wrapper."""
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})

        provider = MimoModelProvider(
            language_model_spec=language_spec,
            special_token_ids={"images": 32000},
        )

        # Mock MimoModel
        mock_model_instance = MagicMock()
        mock_mimo_model.return_value = mock_model_instance

        result = provider.provide()

        assert result == mock_model_instance

        # Should not build grids when no parallelism config
        mock_build_grids.assert_not_called()

    @patch("megatron.bridge.models.mimo.mimo_provider.MimoModel")
    @patch("megatron.bridge.models.mimo.mimo_provider.build_hypercomm_grids")
    def test_provide_signature_matches_mixin(self, mock_build_grids, mock_mimo_model):
        """Test provide() accepts standard mixin signature arguments."""
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        provider = MimoModelProvider(language_model_spec=language_spec)

        mock_mimo_model.return_value = MagicMock()

        # Should accept pre_process, post_process, vp_stage (even if unused)
        result = provider.provide(pre_process=True, post_process=False, vp_stage=0)

        # Should still return a model
        assert result is not None

    @patch("megatron.bridge.models.mimo.mimo_provider.build_hypercomm_grids")
    def test_build_infra_without_parallelism(self, mock_build_grids):
        """Test build_infra() without parallelism config."""
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        provider = MimoModelProvider(language_model_spec=language_spec)

        infra = provider.build_infra()

        # Should return empty infrastructure
        assert isinstance(infra, MimoModelInfra)
        assert infra.module_to_grid_map == {}
        assert infra.topology == {}
        assert infra.pg_collections == {}
        assert infra.participating_modules == []

        # Should not build grids
        mock_build_grids.assert_not_called()

    @patch("torch.distributed.get_rank")
    @patch("megatron.bridge.models.mimo.mimo_provider.build_hypercomm_grids")
    @patch("megatron.bridge.models.mimo.mimo_provider._default_topology")
    def test_build_infra_with_parallelism(self, mock_topology, mock_build_grids, mock_get_rank):
        """Test build_infra() with parallelism config."""
        mock_get_rank.return_value = 0
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})

        mimo_parallelism_config = MimoParallelismConfig(
            module_parallelisms={
                "llm": ModuleParallelismConfig(
                    tensor_model_parallel_size=2,
                    data_parallel_size=2,
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

        infra = provider.build_infra()

        # Should build grids
        mock_build_grids.assert_called_once_with(mimo_parallelism_config)

        # Should return populated infrastructure
        assert isinstance(infra, MimoModelInfra)
        assert "llm" in infra.module_to_grid_map
        assert "llm" in infra.pg_collections
        assert "llm" in infra.participating_modules

    @patch("torch.distributed.get_rank")
    @patch("megatron.bridge.models.mimo.mimo_provider.build_hypercomm_grids")
    @patch("megatron.bridge.models.mimo.mimo_provider._default_topology")
    def test_build_infra_is_idempotent(self, mock_topology, mock_build_grids, mock_get_rank):
        """Test build_infra() can be called multiple times."""
        mock_get_rank.return_value = 0
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})

        mimo_parallelism_config = MimoParallelismConfig(
            module_parallelisms={
                "llm": ModuleParallelismConfig(tensor_model_parallel_size=2),
            },
        )

        mock_grid = MagicMock()
        mock_grid.rank_offset = 0
        mock_grid.size = 2
        mock_grid.get_pg.return_value = MagicMock()
        mock_build_grids.return_value = {"llm": mock_grid}
        mock_topology.return_value = {"llm": []}

        provider = MimoModelProvider(
            language_model_spec=language_spec,
            mimo_parallelism_config=mimo_parallelism_config,
        )

        # Call multiple times
        infra1 = provider.build_infra()
        infra2 = provider.build_infra()

        # Should return equivalent results (not cached, but same structure)
        assert infra1.participating_modules == infra2.participating_modules

    @patch("torch.distributed.get_rank")
    @patch("megatron.bridge.models.mimo.mimo_provider.MimoModel")
    @patch("megatron.bridge.models.mimo.mimo_provider.build_hypercomm_grids")
    @patch("megatron.bridge.models.mimo.mimo_provider._default_topology")
    def test_provide_with_parallelism(self, mock_topology, mock_build_grids, mock_mimo_model, mock_get_rank):
        """Test provide() with parallelism config."""
        mock_get_rank.return_value = 0
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})

        mimo_parallelism_config = MimoParallelismConfig(
            module_parallelisms={
                "llm": ModuleParallelismConfig(
                    tensor_model_parallel_size=2,
                    data_parallel_size=2,
                ),
            },
        )

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

        mock_model_instance = MagicMock()
        mock_mimo_model.return_value = mock_model_instance

        model = provider.provide()

        # Should return model directly
        assert model == mock_model_instance

        # Infrastructure should be available via build_infra()
        infra = provider.build_infra()
        assert "llm" in infra.module_to_grid_map
        assert "llm" in infra.pg_collections

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_rank")
    @patch("megatron.bridge.models.mimo.mimo_provider.build_hypercomm_grids")
    def test_non_participating_rank_raises_error(
        self, mock_build_grids, mock_get_rank, mock_get_world_size, mock_is_initialized
    ):
        """Test that non-participating ranks raise ValueError during finalize().

        This tests the gap scenario: world_size=12, but modules only cover
        ranks 0-3 and 8-11, leaving ranks 4-7 as non-participating.
        """
        mock_is_initialized.return_value = True
        mock_get_world_size.return_value = 12  # World has 12 ranks
        mock_get_rank.return_value = 5  # Rank 5 is in the gap

        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        # Create config with a gap: llm at 0-3, encoder at 8-11, gap at 4-7
        mimo_parallelism_config = MimoParallelismConfig(
            module_parallelisms={
                "llm": ModuleParallelismConfig(
                    tensor_model_parallel_size=2,
                    data_parallel_size=2,
                    rank_offset=0,  # ranks 0-3
                ),
                "encoder": ModuleParallelismConfig(
                    tensor_model_parallel_size=2,
                    data_parallel_size=2,
                    rank_offset=8,  # ranks 8-11
                ),
            },
        )

        # Mock grids with the gap (ranks 4-7 not covered)
        llm_grid = MagicMock()
        llm_grid.rank_offset = 0
        llm_grid.size = 4  # ranks 0-3

        encoder_grid = MagicMock()
        encoder_grid.rank_offset = 8
        encoder_grid.size = 4  # ranks 8-11

        mock_build_grids.return_value = {"llm": llm_grid, "encoder": encoder_grid}

        provider = MimoModelProvider(
            language_model_spec=language_spec,
            mimo_parallelism_config=mimo_parallelism_config,
        )

        # Should raise ValueError because ranks 4-7 don't participate
        with pytest.raises(ValueError, match="do not participate in any MIMO module"):
            provider.finalize()

    def test_inject_pg_collection_into_language_spec(self):
        """Test that pg_collection is injected into language specs."""
        language_spec = ModuleSpec(module=Mock, params={})

        provider = MimoModelProvider(language_model_spec=language_spec)

        mock_pg_collection = MagicMock()
        injected_spec = provider._inject_pg_collection_into_language_spec(language_spec, mock_pg_collection)

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

        provider = MimoModelProvider(language_model_spec=ModuleSpec(module=Mock, params={}))

        mock_pg_collection = MagicMock()
        mock_pg_collection.tp = MagicMock()

        injected_spec = provider._inject_pg_collection_into_modality_spec(modality_spec, mock_pg_collection)

        # Check encoder has pg_collection
        assert injected_spec.submodules["encoders"]["clip"].params["pg_collection"] == mock_pg_collection

    @patch("megatron.bridge.models.mimo.mimo_provider.MimoModel")
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

        _ = provider.provide()

        # Check parameter was frozen
        assert mock_param.requires_grad is False

    @patch("torch.distributed.get_rank")
    @patch("megatron.bridge.models.mimo.mimo_provider.MimoModel")
    @patch("megatron.bridge.models.mimo.mimo_provider.build_hypercomm_grids")
    @patch("megatron.bridge.models.mimo.mimo_provider._default_topology")
    def test_per_encoder_parallelism(self, mock_topology, mock_build_grids, mock_mimo_model, mock_get_rank):
        """Test per-encoder parallelism with different TP per encoder."""
        mock_get_rank.return_value = 0
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        clip_spec = ModuleSpec(module=Mock, params={})
        dino_spec = ModuleSpec(module=Mock, params={})

        mimo_parallelism_config = MimoParallelismConfig(
            module_parallelisms={
                "llm": ModuleParallelismConfig(tensor_model_parallel_size=8, data_parallel_size=1),
                "clip_encoder": ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=1),
                "dino_encoder": ModuleParallelismConfig(tensor_model_parallel_size=4, data_parallel_size=1),
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

        mock_model_instance = MagicMock()
        mock_mimo_model.return_value = mock_model_instance

        model = provider.provide()
        infra = provider.build_infra()

        # Should build grids with all three modules
        mock_build_grids.assert_called_with(mimo_parallelism_config)

        # Should have pg_collections for all modules
        assert "llm" in infra.pg_collections
        assert "clip_encoder" in infra.pg_collections
        assert "dino_encoder" in infra.pg_collections

        # Should return model directly
        assert model == mock_model_instance

    def test_initialize_model_parallel_is_noop(self):
        """Test that initialize_model_parallel() is a no-op for MIMO."""
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        provider = MimoModelProvider(language_model_spec=language_spec)

        # Should not raise, should be a no-op
        provider.initialize_model_parallel(seed=42)
        provider.initialize_model_parallel()


class TestMimoModelInfra:
    """Test cases for MimoModelInfra dataclass."""

    def test_infra_initialization(self):
        """Test infrastructure dataclass initializes correctly."""
        grids = {"llm": MagicMock()}
        topology = {"llm": []}
        pg_collections = {"llm": MagicMock()}
        participating = ["llm"]

        infra = MimoModelInfra(
            module_to_grid_map=grids,
            topology=topology,
            pg_collections=pg_collections,
            participating_modules=participating,
        )

        assert infra.module_to_grid_map == grids
        assert infra.topology == topology
        assert infra.pg_collections == pg_collections
        assert infra.participating_modules == participating
