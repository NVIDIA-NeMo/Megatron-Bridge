# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for MIMO Model Provider."""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch.nn as nn
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.mimo import (
    MimoModelInfra,
    MimoModelProvider,
)
from megatron.bridge.models.mimo.mimo_config import MimoParallelismConfig, ModuleParallelismConfig


class MockModule(nn.Module):
    """Mock Module for testing that is a proper torch.nn.Module subclass."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        # Add config attribute that Float16Module looks for
        self.config = MagicMock()

    def forward(self, *args, **kwargs):
        return None

    def cuda(self, device=None):
        """Mock cuda() method."""
        # Return self to avoid actual CUDA calls
        return self

    def bfloat16(self):
        """Mock bfloat16() method."""
        return self

    def half(self):
        """Mock half() method."""
        return self


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

    @patch("megatron.bridge.models.mimo.mimo_provider.MimoModel")
    def test_freezing_modality_encoders(self, mock_mimo_model):
        """Test freeze_modality_encoders works."""
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})

        # Create mock model with modality submodules
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.requires_grad = True

        # Create mock encoder with parameters
        mock_encoder = MagicMock()
        mock_encoder.parameters.return_value = [mock_param]

        # Create mock modality submodule with encoders
        mock_submodule = MagicMock()
        mock_submodule.encoders = mock_encoder

        mock_model.modality_submodules = {"images": mock_submodule}
        mock_mimo_model.return_value = mock_model

        provider = MimoModelProvider(
            language_model_spec=language_spec,
            freeze_modality_encoders={"images": True},
        )

        _ = provider.provide()

        # Check encoder parameters were frozen
        assert mock_param.requires_grad is False

    @patch("megatron.bridge.models.mimo.mimo_provider.MimoModel")
    def test_freezing_modality_projections(self, mock_mimo_model):
        """Test freeze_modality_projections works."""
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})

        # Create mock model with modality submodules
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.requires_grad = True

        # Create mock projection with parameters
        mock_projection = MagicMock()
        mock_projection.parameters.return_value = [mock_param]

        # Create mock modality submodule with projections
        mock_submodule = MagicMock()
        mock_submodule.input_projections = mock_projection

        mock_model.modality_submodules = {"images": mock_submodule}
        mock_mimo_model.return_value = mock_model

        provider = MimoModelProvider(
            language_model_spec=language_spec,
            freeze_modality_projections={"images": True},
        )

        _ = provider.provide()

        # Check projection parameters were frozen
        assert mock_param.requires_grad is False

    @patch("megatron.bridge.models.mimo.mimo_provider.MimoModel")
    def test_combined_freezing(self, mock_mimo_model):
        """Test freezing language model, encoders, and projections together."""
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})

        # Create mock model with all components
        mock_model = MagicMock()

        # Language model param
        mock_lang_param = MagicMock()
        mock_lang_param.requires_grad = True
        mock_model.language_model.parameters.return_value = [mock_lang_param]

        # Encoder param
        mock_enc_param = MagicMock()
        mock_enc_param.requires_grad = True
        mock_encoder = MagicMock()
        mock_encoder.parameters.return_value = [mock_enc_param]

        # Projection param
        mock_proj_param = MagicMock()
        mock_proj_param.requires_grad = True
        mock_projection = MagicMock()
        mock_projection.parameters.return_value = [mock_proj_param]

        # Modality submodule
        mock_submodule = MagicMock()
        mock_submodule.encoders = mock_encoder
        mock_submodule.input_projections = mock_projection

        mock_model.modality_submodules = {"images": mock_submodule}
        mock_mimo_model.return_value = mock_model

        provider = MimoModelProvider(
            language_model_spec=language_spec,
            freeze_language_model=True,
            freeze_modality_encoders={"images": True},
            freeze_modality_projections={"images": True},
        )

        _ = provider.provide()

        # Check all parameters were frozen
        assert mock_lang_param.requires_grad is False
        assert mock_enc_param.requires_grad is False
        assert mock_proj_param.requires_grad is False

    @patch("megatron.bridge.models.mimo.mimo_provider.MimoModel")
    def test_partial_modality_freezing(self, mock_mimo_model):
        """Test freezing only specific modalities."""
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})

        # Create mock model with multiple modalities
        mock_model = MagicMock()

        # Images modality (frozen)
        mock_images_param = MagicMock()
        mock_images_param.requires_grad = True
        mock_images_encoder = MagicMock()
        mock_images_encoder.parameters.return_value = [mock_images_param]
        mock_images_submodule = MagicMock()
        mock_images_submodule.encoders = mock_images_encoder

        # Audio modality (not frozen)
        mock_audio_param = MagicMock()
        mock_audio_param.requires_grad = True
        mock_audio_encoder = MagicMock()
        mock_audio_encoder.parameters.return_value = [mock_audio_param]
        mock_audio_submodule = MagicMock()
        mock_audio_submodule.encoders = mock_audio_encoder

        mock_model.modality_submodules = {
            "images": mock_images_submodule,
            "audio": mock_audio_submodule,
        }
        mock_mimo_model.return_value = mock_model

        provider = MimoModelProvider(
            language_model_spec=language_spec,
            freeze_modality_encoders={"images": True},  # Only freeze images
        )

        _ = provider.provide()

        # Check only images parameters were frozen
        assert mock_images_param.requires_grad is False
        assert mock_audio_param.requires_grad is True  # Should remain unfrozen

    @patch("megatron.bridge.models.mimo.mimo_provider.MimoModel")
    def test_freezing_with_missing_attributes(self, mock_mimo_model):
        """Test freezing handles missing attributes gracefully."""
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})

        # Create mock model without expected attributes
        mock_model = MagicMock()
        # No language_model attribute
        del mock_model.language_model
        # No modality_submodules attribute
        del mock_model.modality_submodules

        mock_mimo_model.return_value = mock_model

        provider = MimoModelProvider(
            language_model_spec=language_spec,
            freeze_language_model=True,
            freeze_modality_encoders={"images": True},
            freeze_modality_projections={"images": True},
        )

        # Should not raise an error
        _ = provider.provide()

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

    def test_tensor_model_parallel_size_property_with_config(self):
        """Test tensor_model_parallel_size property returns LLM's TP size."""
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        mimo_config = MimoParallelismConfig(
            module_parallelisms={
                "llm": ModuleParallelismConfig(tensor_model_parallel_size=8),
            }
        )

        provider = MimoModelProvider(
            language_model_spec=language_spec,
            mimo_parallelism_config=mimo_config,
        )

        assert provider.tensor_model_parallel_size == 8

    def test_tensor_model_parallel_size_property_without_config(self):
        """Test tensor_model_parallel_size property returns 1 without config."""
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        provider = MimoModelProvider(language_model_spec=language_spec)

        assert provider.tensor_model_parallel_size == 1

    def test_pipeline_model_parallel_size_property_with_config(self):
        """Test pipeline_model_parallel_size property returns LLM's PP size."""
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        mimo_config = MimoParallelismConfig(
            module_parallelisms={
                "llm": ModuleParallelismConfig(pipeline_model_parallel_size=4),
            }
        )

        provider = MimoModelProvider(
            language_model_spec=language_spec,
            mimo_parallelism_config=mimo_config,
        )

        assert provider.pipeline_model_parallel_size == 4

    def test_pipeline_model_parallel_size_property_without_config(self):
        """Test pipeline_model_parallel_size property returns 1 without config."""
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        provider = MimoModelProvider(language_model_spec=language_spec)

        assert provider.pipeline_model_parallel_size == 1

    def test_context_parallel_size_property_with_config(self):
        """Test context_parallel_size property returns LLM's CP size."""
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        mimo_config = MimoParallelismConfig(
            module_parallelisms={
                "llm": ModuleParallelismConfig(context_parallel_size=2),
            }
        )

        provider = MimoModelProvider(
            language_model_spec=language_spec,
            mimo_parallelism_config=mimo_config,
        )

        assert provider.context_parallel_size == 2

    def test_context_parallel_size_property_without_config(self):
        """Test context_parallel_size property returns 1 without config."""
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        provider = MimoModelProvider(language_model_spec=language_spec)

        assert provider.context_parallel_size == 1


class TestMimoModelProviderDistributed:
    """Test cases for MimoModelProvider.provide_distributed_model()."""

    @patch("torch.cuda.current_device")
    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_rank")
    @patch("megatron.bridge.models.mimo.mimo_provider.MimoModel")
    @patch("megatron.bridge.models.mimo.mimo_provider.build_hypercomm_grids")
    @patch("megatron.bridge.models.mimo.mimo_provider._default_topology")
    def test_basic_provide_distributed_model_flow(
        self,
        mock_topology,
        mock_build_grids,
        mock_mimo_model,
        mock_get_rank,
        mock_get_world_size,
        mock_is_initialized,
        mock_current_device,
    ):
        """Test basic provide_distributed_model() flow without DDP."""
        mock_is_initialized.return_value = True
        mock_get_world_size.return_value = 4
        mock_get_rank.return_value = 0
        mock_current_device.return_value = 0

        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        mimo_config = MimoParallelismConfig(
            module_parallelisms={
                "llm": ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=2),
            }
        )

        mock_grid = MagicMock()
        mock_grid.rank_offset = 0
        mock_grid.size = 4
        mock_grid.get_pg.return_value = MagicMock()
        mock_build_grids.return_value = {"llm": mock_grid}
        mock_topology.return_value = {"llm": []}

        mock_model = MockModule()
        mock_mimo_model.return_value = mock_model

        provider = MimoModelProvider(
            language_model_spec=language_spec,
            mimo_parallelism_config=mimo_config,
        )

        result = provider.provide_distributed_model(wrap_with_ddp=False)

        # Should return list with model
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] is not None

    @patch("torch.cuda.stream")
    @patch("torch.cuda.is_available")
    @patch("torch.cuda.set_device")
    @patch("torch.cuda.device_count")
    @patch("torch.cuda.Stream")
    @patch("torch.cuda.current_stream")
    @patch("torch.cuda.current_device")
    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_rank")
    @patch("megatron.bridge.models.mimo.mimo_provider.MimoModel")
    @patch("megatron.bridge.models.mimo.mimo_provider.DistributedDataParallel")
    @patch("megatron.bridge.models.mimo.mimo_provider.build_hypercomm_grids")
    @patch("megatron.bridge.models.mimo.mimo_provider._default_topology")
    @patch("megatron.bridge.models.mimo.mimo_provider.get_model_config")
    def test_with_ddp_wrapping(
        self,
        mock_get_config,
        mock_topology,
        mock_build_grids,
        mock_ddp,
        mock_mimo_model,
        mock_get_rank,
        mock_get_world_size,
        mock_is_initialized,
        mock_current_device,
        mock_current_stream,
        mock_stream_class,
        mock_device_count,
        mock_set_device,
        mock_is_available,
        mock_stream_ctx,
    ):
        """Test DDP wrapping with data_parallel_random_init=True."""
        from megatron.core.distributed import DistributedDataParallelConfig

        mock_is_initialized.return_value = True
        mock_get_world_size.return_value = 4
        mock_get_rank.return_value = 0
        mock_current_device.return_value = 0
        mock_device_count.return_value = 8  # Mock sufficient GPUs
        mock_set_device.return_value = None  # Mock set_device to avoid CUDA calls
        mock_is_available.return_value = True  # Mock CUDA availability

        # Mock the stream context manager
        mock_stream_ctx.return_value.__enter__ = MagicMock(return_value=None)
        mock_stream_ctx.return_value.__exit__ = MagicMock(return_value=None)

        # Mock streams
        mock_stream = MagicMock()
        mock_stream_class.return_value = mock_stream
        mock_current_stream.return_value = MagicMock()

        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        mimo_config = MimoParallelismConfig(
            module_parallelisms={
                "llm": ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=2),
            }
        )

        mock_grid = MagicMock()
        mock_grid.rank_offset = 0
        mock_grid.size = 4
        mock_grid.get_pg.return_value = MagicMock()
        mock_build_grids.return_value = {"llm": mock_grid}
        mock_topology.return_value = {"llm": []}

        mock_model = MockModule()
        mock_mimo_model.return_value = mock_model
        mock_get_config.return_value = Mock()

        # Mock DDP wrapper
        mock_ddp_model = MagicMock()
        mock_ddp_model.broadcast_params = MagicMock()
        mock_ddp.return_value = mock_ddp_model

        provider = MimoModelProvider(
            language_model_spec=language_spec,
            mimo_parallelism_config=mimo_config,
        )

        ddp_config = DistributedDataParallelConfig()
        provider.provide_distributed_model(ddp_config=ddp_config, wrap_with_ddp=True, data_parallel_random_init=True)

        # Should wrap with DDP
        assert mock_ddp.called
        # Should broadcast params
        mock_ddp_model.broadcast_params.assert_called_once()

    @patch("torch.cuda.current_device")
    @patch("torch.distributed.is_initialized")
    @patch("megatron.bridge.models.mimo.mimo_provider.MimoModel")
    def test_ddp_config_required_when_wrap_with_ddp_true(
        self, mock_mimo_model, mock_is_initialized, mock_current_device
    ):
        """Test ValueError raised when wrap_with_ddp=True but ddp_config=None."""
        mock_is_initialized.return_value = False
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})

        provider = MimoModelProvider(language_model_spec=language_spec)

        with pytest.raises(ValueError, match="ddp_config is required when wrap_with_ddp is True"):
            provider.provide_distributed_model(wrap_with_ddp=True, ddp_config=None)

    @patch("torch.cuda.current_device")
    @patch("torch.distributed.is_initialized")
    @patch("megatron.bridge.models.mimo.mimo_provider.MimoModel")
    def test_cpu_initialization(self, mock_mimo_model, mock_is_initialized, mock_current_device):
        """Test model stays on CPU when use_cpu_initialization=True."""
        mock_is_initialized.return_value = False
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})

        mock_model = MockModule()
        mock_mimo_model.return_value = mock_model

        provider = MimoModelProvider(language_model_spec=language_spec)

        result = provider.provide_distributed_model(wrap_with_ddp=False, use_cpu_initialization=True)

        # Should NOT move to CUDA (we can't easily assert on MockModule methods, so just check result)
        assert len(result) == 1

    @patch("torch.cuda.current_device")
    @patch("torch.distributed.is_initialized")
    @patch("megatron.bridge.models.mimo.mimo_provider.MimoModel")
    def test_meta_device_initialization(self, mock_mimo_model, mock_is_initialized, mock_current_device):
        """Test model stays on meta device when init_model_with_meta_device=True."""
        mock_is_initialized.return_value = False
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})

        mock_model = MockModule()
        mock_mimo_model.return_value = mock_model

        provider = MimoModelProvider(language_model_spec=language_spec)

        result = provider.provide_distributed_model(wrap_with_ddp=False, init_model_with_meta_device=True)

        # Should NOT move to CUDA (we can't easily assert on MockModule methods, so just check result)
        assert len(result) == 1

    @patch("torch.cuda.current_device")
    @patch("torch.distributed.is_initialized")
    @patch("megatron.bridge.models.mimo.mimo_provider.MimoModel")
    @patch("megatron.bridge.models.mimo.mimo_provider.get_model_config")
    def test_fp16_handling(self, mock_get_config, mock_mimo_model, mock_is_initialized, mock_current_device):
        """Test FP16 mixed precision wrapper."""
        mock_is_initialized.return_value = False
        mock_current_device.return_value = 0
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})

        mock_model = MockModule()
        mock_mimo_model.return_value = mock_model

        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        provider = MimoModelProvider(language_model_spec=language_spec)

        with patch("megatron.core.transformer.module.Float16Module") as mock_float16:
            mock_wrapped = MagicMock()
            mock_float16.return_value = mock_wrapped

            result = provider.provide_distributed_model(wrap_with_ddp=False, fp16=True)

            # Should wrap with Float16Module
            mock_float16.assert_called_once()
            assert mock_config.fp16 is True
            assert result[0] == mock_wrapped

    @patch("torch.cuda.current_device")
    @patch("torch.distributed.is_initialized")
    @patch("megatron.bridge.models.mimo.mimo_provider.MimoModel")
    @patch("megatron.bridge.models.mimo.mimo_provider.get_model_config")
    def test_bf16_handling(self, mock_get_config, mock_mimo_model, mock_is_initialized, mock_current_device):
        """Test BF16 mixed precision wrapper (default)."""
        mock_is_initialized.return_value = False
        mock_current_device.return_value = 0
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})

        mock_model = MockModule()
        mock_mimo_model.return_value = mock_model

        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        provider = MimoModelProvider(language_model_spec=language_spec, bf16=True)

        with patch("megatron.core.transformer.module.Float16Module") as mock_float16:
            mock_wrapped = MagicMock()
            mock_float16.return_value = mock_wrapped

            provider.provide_distributed_model(wrap_with_ddp=False)

            # Should wrap with Float16Module for BF16
            mock_float16.assert_called_once()
            assert mock_config.bf16 is True

    @patch("torch.cuda.current_device")
    @patch("torch.distributed.is_initialized")
    @patch("megatron.bridge.models.mimo.mimo_provider.MimoModel")
    @patch("megatron.bridge.models.mimo.mimo_provider.get_model_config")
    def test_custom_mixed_precision_wrapper(
        self, mock_get_config, mock_mimo_model, mock_is_initialized, mock_current_device
    ):
        """Test custom mixed precision wrapper is used."""
        mock_is_initialized.return_value = False
        mock_current_device.return_value = 0
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})

        mock_model = MagicMock()
        mock_model.cuda = MagicMock(return_value=mock_model)
        mock_mimo_model.return_value = mock_model

        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        # Custom wrapper
        mock_custom_wrapper = MagicMock()
        mock_wrapped = MagicMock()
        mock_custom_wrapper.return_value = mock_wrapped

        provider = MimoModelProvider(language_model_spec=language_spec, fp16=True)

        result = provider.provide_distributed_model(wrap_with_ddp=False, mixed_precision_wrapper=mock_custom_wrapper)

        # Should use custom wrapper instead of Float16Module
        mock_custom_wrapper.assert_called_once_with(mock_config, mock_model)
        assert result[0] == mock_wrapped

    @patch("torch.cuda.current_device")
    @patch("torch.distributed.is_initialized")
    @patch("megatron.bridge.models.mimo.mimo_provider.MimoModel")
    def test_pre_wrap_hook_single(self, mock_mimo_model, mock_is_initialized, mock_current_device):
        """Test single pre-wrap hook is called."""
        mock_is_initialized.return_value = False
        mock_current_device.return_value = 0
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})

        mock_model = MockModule()
        mock_mimo_model.return_value = mock_model

        # Pre-wrap hook
        hook_called = []

        def pre_hook(models):
            hook_called.append(True)
            return models

        provider = MimoModelProvider(language_model_spec=language_spec)

        provider.provide_distributed_model(wrap_with_ddp=False, pre_wrap_hook=pre_hook)

        # Hook should be called
        assert len(hook_called) == 1

    @patch("torch.cuda.current_device")
    @patch("torch.distributed.is_initialized")
    @patch("megatron.bridge.models.mimo.mimo_provider.MimoModel")
    def test_pre_wrap_hooks_multiple(self, mock_mimo_model, mock_is_initialized, mock_current_device):
        """Test multiple pre-wrap hooks are called in order."""
        mock_is_initialized.return_value = False
        mock_current_device.return_value = 0
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})

        mock_model = MockModule()
        mock_mimo_model.return_value = mock_model

        # Track hook execution order
        hook_order = []

        def hook1(models):
            hook_order.append(1)
            return models

        def hook2(models):
            hook_order.append(2)
            return models

        provider = MimoModelProvider(language_model_spec=language_spec)

        provider.provide_distributed_model(wrap_with_ddp=False, pre_wrap_hook=[hook1, hook2])

        # Hooks should be called in order
        assert hook_order == [1, 2]

    @patch("torch.cuda.current_device")
    @patch("torch.distributed.is_initialized")
    @patch("megatron.bridge.models.mimo.mimo_provider.MimoModel")
    def test_post_wrap_hook(self, mock_mimo_model, mock_is_initialized, mock_current_device):
        """Test post-wrap hook is called after everything."""
        mock_is_initialized.return_value = False
        mock_current_device.return_value = 0
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})

        mock_model = MockModule()
        mock_mimo_model.return_value = mock_model

        # Post-wrap hook
        hook_called = []

        def post_hook(models):
            hook_called.append(True)
            return models

        provider = MimoModelProvider(language_model_spec=language_spec)

        provider.provide_distributed_model(wrap_with_ddp=False, post_wrap_hook=post_hook)

        # Hook should be called
        assert len(hook_called) == 1

    @patch("torch.cuda.stream")
    @patch("torch.cuda.is_available")
    @patch("torch.cuda.set_device")
    @patch("torch.cuda.device_count")
    @patch("torch.cuda.Stream")
    @patch("torch.cuda.current_stream")
    @patch("torch.cuda.current_device")
    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_rank")
    @patch("megatron.bridge.models.mimo.mimo_provider.MimoModel")
    @patch("megatron.bridge.models.mimo.mimo_provider.DistributedDataParallel")
    @patch("megatron.bridge.models.mimo.mimo_provider.build_hypercomm_grids")
    @patch("megatron.bridge.models.mimo.mimo_provider._default_topology")
    @patch("megatron.bridge.models.mimo.mimo_provider.get_model_config")
    def test_overlap_param_gather(
        self,
        mock_get_config,
        mock_topology,
        mock_build_grids,
        mock_ddp,
        mock_mimo_model,
        mock_get_rank,
        mock_get_world_size,
        mock_is_initialized,
        mock_current_device,
        mock_current_stream,
        mock_stream_class,
        mock_device_count,
        mock_set_device,
        mock_is_available,
        mock_stream_ctx,
    ):
        """Test overlap_param_gather_with_optimizer_step sets disable_bucketing."""
        from megatron.core.distributed import DistributedDataParallelConfig

        mock_is_initialized.return_value = True
        mock_get_world_size.return_value = 4
        mock_get_rank.return_value = 0
        mock_current_device.return_value = 0
        mock_device_count.return_value = 8  # Mock sufficient GPUs
        mock_set_device.return_value = None  # Mock set_device to avoid CUDA calls
        mock_is_available.return_value = True  # Mock CUDA availability

        # Mock the stream context manager
        mock_stream_ctx.return_value.__enter__ = MagicMock(return_value=None)
        mock_stream_ctx.return_value.__exit__ = MagicMock(return_value=None)

        # Mock streams
        mock_stream = MagicMock()
        mock_stream_class.return_value = mock_stream
        mock_current_stream.return_value = MagicMock()

        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        mimo_config = MimoParallelismConfig(
            module_parallelisms={
                "llm": ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=2),
            }
        )

        mock_grid = MagicMock()
        mock_grid.rank_offset = 0
        mock_grid.size = 4
        mock_grid.get_pg.return_value = MagicMock()
        mock_build_grids.return_value = {"llm": mock_grid}
        mock_topology.return_value = {"llm": []}

        mock_model = MockModule()
        mock_mimo_model.return_value = mock_model
        mock_get_config.return_value = Mock()

        mock_ddp_model = MagicMock()
        mock_ddp_model.broadcast_params = MagicMock()
        mock_ddp.return_value = mock_ddp_model

        provider = MimoModelProvider(
            language_model_spec=language_spec,
            mimo_parallelism_config=mimo_config,
        )

        ddp_config = DistributedDataParallelConfig()
        provider.provide_distributed_model(
            ddp_config=ddp_config,
            wrap_with_ddp=True,
            overlap_param_gather_with_optimizer_step=True,
            data_parallel_random_init=False,
        )

        # Check disable_bucketing was set correctly
        call_kwargs = mock_ddp.call_args[1]
        assert call_kwargs["disable_bucketing"] is True


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
