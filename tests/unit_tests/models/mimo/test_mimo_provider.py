# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for MIMO Model Provider."""

from unittest.mock import MagicMock, Mock, patch

import torch
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

        # Check stack-compatibility fields
        assert hasattr(provider, "vocab_size")
        assert hasattr(provider, "seq_length")

        # Check defaults
        assert provider.fp16 is False
        assert provider.bf16 is True
        assert provider.use_cpu_initialization is False
        assert provider.vocab_size is None
        assert provider.seq_length == 1024

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
        config_arg = mock_mimo_model.call_args[0][0]
        assert config_arg.module_to_grid_map is None
        assert config_arg.language_module_key is None

    @patch("megatron.bridge.models.mimo.mimo_provider.MimoModel")
    @patch("megatron.bridge.models.mimo.mimo_provider.build_hypercomm_grids")
    def test_provide_signature_matches_mixin(self, _mock_build_grids, mock_mimo_model):
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

        # Should return infrastructure with auto-derived topology
        assert isinstance(infra, MimoModelInfra)
        assert infra.module_to_grid_map == {}
        assert infra.topology == {"llm": []}
        assert infra.pg_collections == {}
        assert infra.participating_modules == []

        # Should not build grids
        mock_build_grids.assert_not_called()

    @patch("torch.distributed.new_group")
    @patch("torch.distributed.get_process_group_ranks")
    @patch("torch.distributed.get_rank")
    @patch("megatron.bridge.models.mimo.mimo_provider.build_hypercomm_grids")
    def test_build_infra_with_parallelism(self, mock_build_grids, mock_get_rank, mock_get_pg_ranks, mock_new_group):
        """Test build_infra() with parallelism config."""
        mock_get_rank.return_value = 0
        mock_get_pg_ranks.return_value = [0, 1, 2, 3]
        mock_new_group.return_value = MagicMock()
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

    @patch("torch.distributed.new_group")
    @patch("torch.distributed.get_process_group_ranks")
    @patch("torch.distributed.get_rank")
    @patch("megatron.bridge.models.mimo.mimo_provider.build_hypercomm_grids")
    def test_build_infra_is_idempotent(self, mock_build_grids, mock_get_rank, mock_get_pg_ranks, mock_new_group):
        """Test build_infra() can be called multiple times."""
        mock_get_rank.return_value = 0
        mock_get_pg_ranks.return_value = [0, 1]
        mock_new_group.return_value = MagicMock()
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})

        mimo_parallelism_config = MimoParallelismConfig(
            module_parallelisms={
                "llm": ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=1, rank_offset=0),
            },
        )

        mock_grid = MagicMock()
        mock_grid.rank_offset = 0
        mock_grid.size = 2
        mock_grid.get_pg.return_value = MagicMock()
        mock_build_grids.return_value = {"llm": mock_grid}

        provider = MimoModelProvider(
            language_model_spec=language_spec,
            mimo_parallelism_config=mimo_parallelism_config,
        )

        # Call multiple times
        infra1 = provider.build_infra()
        infra2 = provider.build_infra()

        # Should return equivalent results (not cached, but same structure)
        assert infra1.participating_modules == infra2.participating_modules

    @patch("torch.distributed.new_group")
    @patch("torch.distributed.get_process_group_ranks")
    @patch("torch.distributed.get_rank")
    @patch("megatron.bridge.models.mimo.mimo_provider.MimoModel")
    @patch("megatron.bridge.models.mimo.mimo_provider.build_hypercomm_grids")
    def test_provide_with_parallelism(
        self, mock_build_grids, mock_mimo_model, mock_get_rank, mock_get_pg_ranks, mock_new_group
    ):
        """Test provide() with parallelism config."""
        mock_get_rank.return_value = 0
        mock_get_pg_ranks.return_value = [0, 1, 2, 3]
        mock_new_group.return_value = MagicMock()
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

        provider = MimoModelProvider(
            language_model_spec=language_spec,
            mimo_parallelism_config=mimo_parallelism_config,
        )

        mock_model_instance = MagicMock()
        mock_mimo_model.return_value = mock_model_instance

        model = provider.provide()

        # Should return model directly
        assert model == mock_model_instance
        config_arg = mock_mimo_model.call_args[0][0]
        assert config_arg.module_to_grid_map == {"llm": mock_grid}
        assert config_arg.language_module_key == "llm"

        # Infrastructure should be available via build_infra()
        infra = provider.build_infra()
        assert "llm" in infra.module_to_grid_map
        assert "llm" in infra.pg_collections

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

        provider.provide()

        # Check parameter was frozen
        assert mock_param.requires_grad is False

    @patch("torch.distributed.new_group")
    @patch("torch.distributed.get_process_group_ranks")
    @patch("torch.distributed.get_rank")
    @patch("megatron.bridge.models.mimo.mimo_provider.MimoModel")
    @patch("megatron.bridge.models.mimo.mimo_provider.build_hypercomm_grids")
    def test_per_encoder_parallelism(
        self, mock_build_grids, mock_mimo_model, mock_get_rank, mock_get_pg_ranks, mock_new_group
    ):
        """Test per-encoder parallelism with different TP per encoder."""
        mock_get_rank.return_value = 0
        mock_get_pg_ranks.return_value = [0, 1, 2, 3]
        mock_new_group.return_value = MagicMock()
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

    def test_initialize_model_parallel_raises_heterogeneous(self):
        """Test that initialize_model_parallel() raises NotImplementedError in heterogeneous mode."""
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        mimo_parallelism_config = MimoParallelismConfig(
            module_parallelisms={
                "llm": ModuleParallelismConfig(tensor_model_parallel_size=2),
            },
        )
        provider = MimoModelProvider(
            language_model_spec=language_spec,
            mimo_parallelism_config=mimo_parallelism_config,
        )

        import pytest

        with pytest.raises(NotImplementedError, match="Heterogeneous MIMO does not use global"):
            provider.initialize_model_parallel(seed=42)

    @patch("megatron.bridge.models.model_provider.ModelProviderMixin.initialize_model_parallel")
    def test_initialize_model_parallel_delegates_homogeneous(self, mock_parent_init):
        """Test that initialize_model_parallel() delegates to parent in homogeneous mode."""
        from megatron.core.transformer.transformer_config import TransformerConfig

        config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            tensor_model_parallel_size=2,
        )
        language_spec = ModuleSpec(module=Mock, params={"config": config})
        provider = MimoModelProvider(
            language_model_spec=language_spec,
            mimo_parallelism_config=None,
        )

        provider.initialize_model_parallel(seed=42)
        mock_parent_init.assert_called_once_with(seed=42, seed_kwargs=None)

    @patch("megatron.core.transformer.module.Float16Module")
    @patch("megatron.bridge.models.mimo.mimo_provider.get_model_config")
    @patch("megatron.bridge.models.mimo.mimo_provider.MimoModel")
    @patch.object(MimoModelProvider, "build_infra")
    @patch.object(MimoModelProvider, "finalize")
    def test_provide_distributed_model_sets_variable_seq_lengths(
        self, mock_finalize, mock_build_infra, mock_mimo_model, mock_get_config, mock_float16
    ):
        """Test that provide_distributed_model sets variable_seq_lengths=True (heterogeneous path)."""
        mock_build_infra.return_value = MimoModelInfra(
            module_to_grid_map={},
            topology={"llm": []},
            pg_collections={},
            participating_modules=[],
        )
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})

        mimo_parallelism_config = MimoParallelismConfig(
            module_parallelisms={
                "llm": ModuleParallelismConfig(tensor_model_parallel_size=1),
            },
        )

        provider = MimoModelProvider(
            language_model_spec=language_spec,
            mimo_parallelism_config=mimo_parallelism_config,
            bf16=False,
            fp16=False,
        )

        mock_model_instance = MagicMock()
        mock_model_instance.cuda = MagicMock(return_value=None)
        mock_mimo_model.return_value = mock_model_instance

        mock_config = MagicMock()
        mock_config.variable_seq_lengths = False
        mock_get_config.return_value = mock_config

        provider.provide_distributed_model(wrap_with_ddp=False)

        assert mock_config.variable_seq_lengths is True

    def test_getattr_proxies_in_homogeneous_mode(self):
        """Test __getattr__ proxies reads to nested TransformerConfig in homogeneous mode."""
        from megatron.core.transformer.transformer_config import TransformerConfig

        config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            tensor_model_parallel_size=4,
            pipeline_model_parallel_size=1,
            context_parallel_size=2,
        )
        language_spec = ModuleSpec(module=Mock, params={"config": config})
        provider = MimoModelProvider(
            language_model_spec=language_spec,
            mimo_parallelism_config=None,
        )

        assert provider.tensor_model_parallel_size == 4
        assert provider.pipeline_model_parallel_size == 1
        assert provider.context_parallel_size == 2
        assert provider.num_layers == 2
        assert provider.hidden_size == 64

    def test_getattr_blocks_in_heterogeneous_mode(self):
        """Test __getattr__ raises AttributeError with actionable message in heterogeneous mode."""
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        mimo_parallelism_config = MimoParallelismConfig(
            module_parallelisms={
                "llm": ModuleParallelismConfig(tensor_model_parallel_size=2),
            },
        )
        provider = MimoModelProvider(
            language_model_spec=language_spec,
            mimo_parallelism_config=mimo_parallelism_config,
        )

        import pytest

        with pytest.raises(AttributeError, match="pretrain_mimo"):
            _ = provider.tensor_model_parallel_size

    def test_setattr_syncs_to_nested_config_in_homogeneous_mode(self):
        """Test __setattr__ syncs writes to nested TransformerConfig in homogeneous mode."""
        from megatron.core.transformer.transformer_config import TransformerConfig

        config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
        )
        language_spec = ModuleSpec(module=Mock, params={"config": config})
        provider = MimoModelProvider(
            language_model_spec=language_spec,
            mimo_parallelism_config=None,
        )

        provider.fp16 = True
        assert config.fp16 is True

        provider.bf16 = False
        assert config.bf16 is False

    def test_setattr_does_not_sync_in_heterogeneous_mode(self):
        """Test __setattr__ does not sync to nested config in heterogeneous mode."""
        from megatron.core.transformer.transformer_config import TransformerConfig

        config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
        )
        original_fp16 = config.fp16
        language_spec = ModuleSpec(module=Mock, params={"config": config})
        mimo_parallelism_config = MimoParallelismConfig(
            module_parallelisms={
                "llm": ModuleParallelismConfig(tensor_model_parallel_size=2),
            },
        )
        provider = MimoModelProvider(
            language_model_spec=language_spec,
            mimo_parallelism_config=mimo_parallelism_config,
        )

        provider.fp16 = True
        # Provider gets the value, but nested config should be untouched
        assert provider.fp16 is True
        assert config.fp16 == original_fp16

    def test_finalize_enforces_pp1_homogeneous(self):
        """Test finalize() raises ValueError when PP > 1 in homogeneous mode."""
        from megatron.core.transformer.transformer_config import TransformerConfig

        config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.bfloat16,
        )
        language_spec = ModuleSpec(module=Mock, params={"config": config})
        provider = MimoModelProvider(
            language_model_spec=language_spec,
            mimo_parallelism_config=None,
        )

        import pytest

        with pytest.raises(ValueError, match="pipeline_model_parallel_size=1"):
            provider.finalize()

    def test_finalize_pp1_passes_homogeneous(self):
        """Test finalize() succeeds when PP == 1 in homogeneous mode."""
        from megatron.core.transformer.transformer_config import TransformerConfig

        config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            pipeline_model_parallel_size=1,
        )
        language_spec = ModuleSpec(module=Mock, params={"config": config})
        provider = MimoModelProvider(
            language_model_spec=language_spec,
            mimo_parallelism_config=None,
        )

        # Should not raise
        provider.finalize()

    def test_getattr_raises_for_nonexistent_attr_homogeneous(self):
        """Test __getattr__ raises AttributeError for truly missing attributes."""
        from megatron.core.transformer.transformer_config import TransformerConfig

        config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
        )
        language_spec = ModuleSpec(module=Mock, params={"config": config})
        provider = MimoModelProvider(
            language_model_spec=language_spec,
            mimo_parallelism_config=None,
        )

        import pytest

        with pytest.raises(AttributeError, match="has no attribute 'totally_fake_attr'"):
            _ = provider.totally_fake_attr


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


class TestEmbeddingGroupHelpers:
    """Test cases for embedding group helper functions."""

    @patch("torch.distributed.new_group")
    @patch("torch.distributed.get_process_group_ranks")
    def test_populate_embedding_groups_single_pp_rank(self, mock_get_ranks, mock_new_group):
        """Test embedding groups with single PP rank (PP=1)."""
        from megatron.bridge.models.mimo.mimo_builder import (
            populate_embedding_and_position_groups,
        )

        mock_pp_group = MagicMock()
        mock_get_ranks.return_value = [0]  # Single PP rank
        mock_new_group.return_value = MagicMock()

        populate_embedding_and_position_groups(mock_pp_group)

        # Should create groups for both position and word embeddings
        assert mock_new_group.call_count == 2
        # Both groups should include only rank 0
        calls = mock_new_group.call_args_list
        assert calls[0].kwargs["ranks"] == [0]
        assert calls[1].kwargs["ranks"] == [0]

    @patch("torch.distributed.new_group")
    @patch("torch.distributed.get_process_group_ranks")
    def test_populate_embedding_groups_multiple_pp_ranks(self, mock_get_ranks, mock_new_group):
        """Test embedding groups with multiple PP ranks (PP>1)."""
        from megatron.bridge.models.mimo.mimo_builder import (
            populate_embedding_and_position_groups,
        )

        mock_pp_group = MagicMock()
        mock_get_ranks.return_value = [0, 4, 8, 12]  # PP=4
        mock_new_group.return_value = MagicMock()

        populate_embedding_and_position_groups(mock_pp_group)

        # Should create two groups
        assert mock_new_group.call_count == 2
        calls = mock_new_group.call_args_list
        # pos_embd only on first rank
        assert calls[0].kwargs["ranks"] == [0]
        # embd on first and last ranks
        assert calls[1].kwargs["ranks"] == [0, 12]

    def test_populate_embedding_groups_none_pp_group(self):
        """Test embedding groups with None PP group."""
        from megatron.bridge.models.mimo.mimo_builder import (
            populate_embedding_and_position_groups,
        )

        pos_embd_pg, embd_pg = populate_embedding_and_position_groups(None)

        assert pos_embd_pg is None
        assert embd_pg is None

    @patch("torch.distributed.get_process_group_ranks")
    @patch("torch.distributed.get_rank")
    def test_is_pp_first_stage_true(self, mock_get_rank, mock_get_ranks):
        """Test is_pp_first_stage returns True for first stage."""
        from megatron.bridge.models.mimo.mimo_builder import is_pp_first_stage

        mock_pp_group = MagicMock()
        mock_get_ranks.return_value = [0, 4, 8, 12]
        mock_get_rank.return_value = 0

        assert is_pp_first_stage(mock_pp_group) is True

    @patch("torch.distributed.get_process_group_ranks")
    @patch("torch.distributed.get_rank")
    def test_is_pp_first_stage_false(self, mock_get_rank, mock_get_ranks):
        """Test is_pp_first_stage returns False for non-first stage."""
        from megatron.bridge.models.mimo.mimo_builder import is_pp_first_stage

        mock_pp_group = MagicMock()
        mock_get_ranks.return_value = [0, 4, 8, 12]
        mock_get_rank.return_value = 4

        assert is_pp_first_stage(mock_pp_group) is False

    def test_is_pp_first_stage_none_group(self):
        """Test is_pp_first_stage returns True for None group (no PP)."""
        from megatron.bridge.models.mimo.mimo_builder import is_pp_first_stage

        assert is_pp_first_stage(None) is True

    @patch("torch.distributed.get_process_group_ranks")
    @patch("torch.distributed.get_rank")
    def test_is_pp_last_stage_true(self, mock_get_rank, mock_get_ranks):
        """Test is_pp_last_stage returns True for last stage."""
        from megatron.bridge.models.mimo.mimo_builder import is_pp_last_stage

        mock_pp_group = MagicMock()
        mock_get_ranks.return_value = [0, 4, 8, 12]
        mock_get_rank.return_value = 12

        assert is_pp_last_stage(mock_pp_group) is True

    @patch("torch.distributed.get_process_group_ranks")
    @patch("torch.distributed.get_rank")
    def test_is_pp_last_stage_false(self, mock_get_rank, mock_get_ranks):
        """Test is_pp_last_stage returns False for non-last stage."""
        from megatron.bridge.models.mimo.mimo_builder import is_pp_last_stage

        mock_pp_group = MagicMock()
        mock_get_ranks.return_value = [0, 4, 8, 12]
        mock_get_rank.return_value = 4

        assert is_pp_last_stage(mock_pp_group) is False

    def test_is_pp_last_stage_none_group(self):
        """Test is_pp_last_stage returns True for None group (no PP)."""
        from megatron.bridge.models.mimo.mimo_builder import is_pp_last_stage

        assert is_pp_last_stage(None) is True


class TestProcessGroupCollectionWithEmbeddingGroups:
    """Test that ProcessGroupCollection includes embedding groups."""

    @patch("megatron.bridge.models.mimo.mimo_provider.is_pp_last_stage")
    @patch("megatron.bridge.models.mimo.mimo_provider.is_pp_first_stage")
    @patch("megatron.bridge.models.mimo.mimo_provider.populate_embedding_and_position_groups")
    @patch("torch.distributed.get_rank")
    def test_pg_collection_includes_embedding_groups_first_stage(
        self, mock_get_rank, mock_populate, mock_is_first, mock_is_last
    ):
        """Test that pg_collection includes embedding groups for first PP stage."""
        mock_get_rank.return_value = 0
        mock_pos_embd = MagicMock()
        mock_embd = MagicMock()
        mock_populate.return_value = (mock_pos_embd, mock_embd)
        mock_is_first.return_value = True
        mock_is_last.return_value = False

        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        mimo_parallelism_config = MimoParallelismConfig(
            module_parallelisms={
                "llm": ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=2),
            },
        )

        # Mock grid
        mock_grid = MagicMock()
        mock_grid.rank_offset = 0
        mock_grid.size = 4
        mock_grid.get_pg.return_value = MagicMock()

        provider = MimoModelProvider(
            language_model_spec=language_spec,
            mimo_parallelism_config=mimo_parallelism_config,
        )

        pg_collections = provider._get_pg_collections_from_grids({"llm": mock_grid})

        # First stage should have pos_embd but not embd (not last stage)
        assert pg_collections["llm"].pos_embd == mock_pos_embd
        assert pg_collections["llm"].embd == mock_embd  # First stage gets embd too

    @patch("megatron.bridge.models.mimo.mimo_provider.is_pp_last_stage")
    @patch("megatron.bridge.models.mimo.mimo_provider.is_pp_first_stage")
    @patch("megatron.bridge.models.mimo.mimo_provider.populate_embedding_and_position_groups")
    @patch("torch.distributed.get_rank")
    def test_pg_collection_middle_stage_no_embedding_groups(
        self, mock_get_rank, mock_populate, mock_is_first, mock_is_last
    ):
        """Test that middle PP stages don't get embedding groups."""
        mock_get_rank.return_value = 4
        mock_pos_embd = MagicMock()
        mock_embd = MagicMock()
        mock_populate.return_value = (mock_pos_embd, mock_embd)
        mock_is_first.return_value = False
        mock_is_last.return_value = False

        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        mimo_parallelism_config = MimoParallelismConfig(
            module_parallelisms={
                "llm": ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=2),
            },
        )

        # Mock grid
        mock_grid = MagicMock()
        mock_grid.rank_offset = 0
        mock_grid.size = 8
        mock_grid.get_pg.return_value = MagicMock()

        provider = MimoModelProvider(
            language_model_spec=language_spec,
            mimo_parallelism_config=mimo_parallelism_config,
        )

        pg_collections = provider._get_pg_collections_from_grids({"llm": mock_grid})

        # Middle stage should have neither embedding group
        assert pg_collections["llm"].pos_embd is None
        assert pg_collections["llm"].embd is None

    @patch("megatron.bridge.models.mimo.mimo_provider.is_pp_last_stage")
    @patch("megatron.bridge.models.mimo.mimo_provider.is_pp_first_stage")
    @patch("megatron.bridge.models.mimo.mimo_provider.populate_embedding_and_position_groups")
    @patch("torch.distributed.get_rank")
    def test_pg_collection_includes_composite_groups(self, mock_get_rank, mock_populate, mock_is_first, mock_is_last):
        """Test that pg_collection includes mp, tp_ep_pp, and expt_dp composite groups."""
        mock_get_rank.return_value = 0
        mock_populate.return_value = (MagicMock(), MagicMock())
        mock_is_first.return_value = True
        mock_is_last.return_value = True

        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        mimo_parallelism_config = MimoParallelismConfig(
            module_parallelisms={
                "llm": ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=2),
            },
        )

        mock_tp = MagicMock(name="tp_pg")
        mock_dp = MagicMock(name="dp_pg")
        mock_pp = MagicMock(name="pp_pg")
        mock_cp = MagicMock(name="cp_pg")
        mock_ep = MagicMock(name="ep_pg")
        mock_dp_cp = MagicMock(name="dp_cp_pg")
        mock_mp = MagicMock(name="mp_pg")
        mock_tp_ep_pp = MagicMock(name="tp_ep_pp_pg")

        pg_map = {
            ("tp",): mock_tp,
            ("dp",): mock_dp,
            ("pp",): mock_pp,
            ("cp",): mock_cp,
            ("ep",): mock_ep,
            ("dp", "cp"): mock_dp_cp,
            ("tp", "pp"): mock_mp,
            ("tp", "ep", "pp"): mock_tp_ep_pp,
        }

        mock_grid = MagicMock()
        mock_grid.rank_offset = 0
        mock_grid.size = 4
        mock_grid.get_pg.side_effect = lambda dims: pg_map[tuple(dims)]

        provider = MimoModelProvider(
            language_model_spec=language_spec,
            mimo_parallelism_config=mimo_parallelism_config,
        )

        pg_collections = provider._get_pg_collections_from_grids({"llm": mock_grid})

        pgc = pg_collections["llm"]
        assert pgc.tp == mock_tp
        assert pgc.dp == mock_dp
        assert pgc.pp == mock_pp
        assert pgc.cp == mock_cp
        assert pgc.ep == mock_ep
        assert pgc.dp_cp == mock_dp_cp
        assert pgc.mp == mock_mp
        assert pgc.tp_ep_pp == mock_tp_ep_pp
