# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for MegatronMIMO Model Provider."""

from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.megatron_mimo import (
    MegatronMIMOInfra,
    MegatronMIMOProvider,
    MegatronMIMORNGMode,
    get_megatron_mimo_rng_mode,
)
from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)


# MegatronMIMOProvider rejects single-module configs (language-only or no
# modality submodule spec). Tests that don't care about modality specifics
# use this dummy to satisfy the required-modules contract while keeping the
# test focused on whatever else it's asserting.
def _dummy_modality_spec_dict() -> dict:
    return {"vision": ModuleSpec(module=Mock, params={})}


def _dummy_modality_parallelism(rank_offset: int = 0, **kwargs) -> dict:
    """Module-parallelism dict containing language + a dummy 'vision' entry.

    Mirrors the language entry's geometry so the resulting layout is colocated
    by default; pass ``rank_offset`` and parallelism kwargs to override.
    """
    return {
        "vision": ModuleParallelismConfig(rank_offset=rank_offset, **kwargs),
    }


def _config(
    calculate_per_token_loss: bool = True,
    recompute_granularity=None,
    **kwargs,
) -> SimpleNamespace:
    return SimpleNamespace(
        calculate_per_token_loss=calculate_per_token_loss,
        recompute_granularity=recompute_granularity,
        **kwargs,
    )


class _DummyLanguageModel(torch.nn.Module):
    """Minimal language module for provider wiring tests."""

    def __init__(self, config, **_kwargs):
        super().__init__()
        self.config = config


class _DummyModalitySubmodule(torch.nn.Module):
    """Minimal modality module with the MCore MIMO ``from_spec`` constructor."""

    @classmethod
    def from_spec(cls, _spec, *, is_first_stage: bool, is_last_stage: bool):
        return cls()


class TestMegatronMIMOProvider:
    """Test cases for MegatronMIMOProvider."""

    def test_get_megatron_mimo_rng_mode_selects_expected_mode(self):
        """RNG mode is selected from placement plus TP symmetry."""
        non_colocated = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(
                    tensor_model_parallel_size=1,
                    data_parallel_size=2,
                    rank_offset=0,
                ),
                "vision": ModuleParallelismConfig(
                    tensor_model_parallel_size=1,
                    data_parallel_size=2,
                    rank_offset=2,
                ),
            },
        )
        non_colocated.finalize(world_size=4)
        assert get_megatron_mimo_rng_mode(non_colocated) == MegatronMIMORNGMode.SINGLETON

        symmetric_colocated = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(
                    tensor_model_parallel_size=2,
                    data_parallel_size=1,
                    rank_offset=0,
                ),
                "vision": ModuleParallelismConfig(
                    tensor_model_parallel_size=2,
                    data_parallel_size=1,
                    rank_offset=0,
                ),
            },
        )
        symmetric_colocated.finalize(world_size=2)
        assert get_megatron_mimo_rng_mode(symmetric_colocated) == MegatronMIMORNGMode.SINGLETON

        asymmetric_colocated = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(
                    tensor_model_parallel_size=2,
                    data_parallel_size=1,
                    rank_offset=0,
                ),
                "vision": ModuleParallelismConfig(
                    tensor_model_parallel_size=1,
                    data_parallel_size=2,
                    rank_offset=0,
                ),
            },
        )
        asymmetric_colocated.finalize(world_size=2)
        assert get_megatron_mimo_rng_mode(asymmetric_colocated) == MegatronMIMORNGMode.PER_MODULE

    def test_provider_initialization_minimal(self):
        """Test provider initializes with minimal required fields."""
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})

        provider = MegatronMIMOProvider(
            language_model_spec=language_spec,
        )

        assert provider.language_model_spec == language_spec
        assert provider.modality_submodules_spec == {}
        assert provider.special_token_ids == {}
        assert provider.megatron_mimo_parallelism_config is None

    def test_provider_initialization_full(self):
        """Test provider initializes with all fields."""
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        modality_spec = ModuleSpec(module=Mock, params={})
        megatron_mimo_parallelism_config = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(tensor_model_parallel_size=2),
            },
        )

        provider = MegatronMIMOProvider(
            language_model_spec=language_spec,
            modality_submodules_spec={"images": modality_spec},
            special_token_ids={"images": 32000},
            megatron_mimo_parallelism_config=megatron_mimo_parallelism_config,
            freeze_language_model=True,
            freeze_modality_encoders={"images": True},
        )

        assert provider.language_model_spec == language_spec
        assert "images" in provider.modality_submodules_spec
        assert provider.special_token_ids == {"images": 32000}
        assert provider.megatron_mimo_parallelism_config == megatron_mimo_parallelism_config
        assert provider.freeze_language_model is True
        assert provider.freeze_modality_encoders == {"images": True}

    def test_provider_has_mixin_fields(self):
        """Test provider has fields required by ModelProviderMixin."""
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        provider = MegatronMIMOProvider(language_model_spec=language_spec)

        # Check mixin-required fields exist with defaults
        assert hasattr(provider, "fp16")
        assert hasattr(provider, "bf16")
        assert hasattr(provider, "use_cpu_initialization")
        assert hasattr(provider, "init_model_with_meta_device")

        # Check defaults
        assert provider.fp16 is False
        assert provider.bf16 is True
        assert provider.use_cpu_initialization is False

    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.MimoModel")
    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.build_hypercomm_grids")
    def test_provide_returns_model_directly(self, mock_build_grids, mock_mimo_model):
        """Test provide() returns model directly, not a wrapper."""
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})

        provider = MegatronMIMOProvider(
            language_model_spec=language_spec,
            modality_submodules_spec=_dummy_modality_spec_dict(),
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

    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.MimoModel")
    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.build_hypercomm_grids")
    def test_provide_signature_matches_mixin(self, _mock_build_grids, mock_mimo_model):
        """Test provide() accepts standard mixin signature arguments."""
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        provider = MegatronMIMOProvider(
            language_model_spec=language_spec,
            modality_submodules_spec=_dummy_modality_spec_dict(),
        )

        mock_mimo_model.return_value = MagicMock()

        # Should accept pre_process, post_process, vp_stage (even if unused)
        result = provider.provide(pre_process=True, post_process=False, vp_stage=0)

        # Should still return a model
        assert result is not None

    # ── Required-spec validator (no-dist static check) ────────────────────────
    #
    # MegatronMIMOProvider must reject single-module / no-modality configs at
    # every infrastructure-build entry point — finalize() alone isn't enough
    # because build_infra() and provide() can be called without ever running
    # finalize() (e.g. in test paths or from callers that build infra directly).

    def test_build_infra_rejects_missing_modality_spec(self):
        """``build_infra`` must reject providers with empty
        ``modality_submodules_spec`` even when no parallelism config is set
        (legacy no-grid-map path) and even when ``finalize()`` is never called."""
        import pytest

        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        provider = MegatronMIMOProvider(language_model_spec=language_spec)

        with pytest.raises(ValueError, match="at least one modality submodule"):
            provider.build_infra()

    def test_provide_rejects_missing_modality_spec(self):
        """``provide`` must reject the same shape that ``build_infra`` rejects."""
        import pytest

        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        provider = MegatronMIMOProvider(language_model_spec=language_spec)

        with pytest.raises(ValueError, match="at least one modality submodule"):
            provider.provide()

    def test_build_infra_rejects_single_module_parallelism_config(self):
        """A parallelism config with only the language module must be rejected
        before ``build_hypercomm_grids`` runs.

        Pre-fix: ``provider.finalize()`` would catch this, but ``build_infra``
        and ``provide`` skipped that path under non-distributed tests and
        legacy callers, so the malformed config slipped through to the grid
        builder. ``_validate_specs_static`` now calls
        ``parallelism_config.validate_static()`` early, surfacing the right
        error regardless of dist state.
        """
        import pytest

        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        # Parallelism declares only "language" — no modality. With a valid
        # modality_submodules_spec on the provider this still fails because
        # parallelism config itself is malformed (no modality entry).
        megatron_mimo_parallelism_config = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=1),
            },
        )
        provider = MegatronMIMOProvider(
            language_model_spec=language_spec,
            modality_submodules_spec=_dummy_modality_spec_dict(),
            megatron_mimo_parallelism_config=megatron_mimo_parallelism_config,
        )

        with pytest.raises(ValueError, match="at least one modality module"):
            provider.build_infra()

    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.build_hypercomm_grids")
    def test_build_infra_without_parallelism(self, mock_build_grids):
        """Test build_infra() without parallelism config."""
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        provider = MegatronMIMOProvider(
            language_model_spec=language_spec,
            modality_submodules_spec=_dummy_modality_spec_dict(),
        )

        infra = provider.build_infra()

        # Should return infrastructure with auto-derived topology including the
        # dummy modality entry that the required-specs validator now mandates.
        assert isinstance(infra, MegatronMIMOInfra)
        assert infra.module_to_grid_map == {}
        assert infra.topology == {"vision": ["language"], "language": []}
        assert infra.pg_collections == {}
        assert infra.participating_modules == []

        # Should not build grids
        mock_build_grids.assert_not_called()

    @patch("torch.distributed.new_group")
    @patch("torch.distributed.get_process_group_ranks")
    @patch("torch.distributed.get_rank")
    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.build_hypercomm_grids")
    def test_build_infra_with_parallelism(self, mock_build_grids, mock_get_rank, mock_get_pg_ranks, mock_new_group):
        """Test build_infra() with parallelism config."""
        mock_get_rank.return_value = 0
        mock_get_pg_ranks.return_value = [0, 1, 2, 3]
        mock_new_group.return_value = MagicMock()
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})

        megatron_mimo_parallelism_config = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(
                    tensor_model_parallel_size=2,
                    data_parallel_size=2,
                ),
                "vision": ModuleParallelismConfig(
                    tensor_model_parallel_size=2,
                    data_parallel_size=2,
                ),
            },
        )

        # Mock grids — one per module since the parallelism config has both.
        language_grid = MagicMock()
        language_grid.rank_offset = 0
        language_grid.size = 4
        language_grid.get_pg.return_value = MagicMock()
        vision_grid = MagicMock()
        vision_grid.rank_offset = 0
        vision_grid.size = 4
        vision_grid.get_pg.return_value = MagicMock()
        mock_build_grids.return_value = {"language": language_grid, "vision": vision_grid}

        provider = MegatronMIMOProvider(
            language_model_spec=language_spec,
            modality_submodules_spec=_dummy_modality_spec_dict(),
            megatron_mimo_parallelism_config=megatron_mimo_parallelism_config,
        )

        infra = provider.build_infra()

        # Should build grids
        mock_build_grids.assert_called_once_with(megatron_mimo_parallelism_config)

        # Should return populated infrastructure
        assert isinstance(infra, MegatronMIMOInfra)
        assert "language" in infra.module_to_grid_map
        assert "language" in infra.pg_collections
        assert "language" in infra.participating_modules

    @patch("torch.distributed.new_group")
    @patch("torch.distributed.get_process_group_ranks")
    @patch("torch.distributed.get_rank")
    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.build_hypercomm_grids")
    def test_build_infra_is_idempotent(self, mock_build_grids, mock_get_rank, mock_get_pg_ranks, mock_new_group):
        """Test build_infra() can be called multiple times.

        ``build_infra()`` is now memoized — repeated calls return the SAME
        ``MegatronMIMOInfra`` object (identity, not just equality). This is
        load-bearing for the per-module RNG plumbing: ``setup_megatron_mimo``
        populates ``infra.cuda_rng_states_per_module`` after the first
        ``build_infra()`` call, and ``provide_distributed_model`` calls
        ``build_infra()`` again internally — the memoized return makes the
        snapshots visible to ``MimoModel.__init__``."""
        mock_get_rank.return_value = 0
        mock_get_pg_ranks.return_value = [0, 1]
        mock_new_group.return_value = MagicMock()
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})

        megatron_mimo_parallelism_config = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=1, rank_offset=0),
                "vision": ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=1, rank_offset=0),
            },
        )

        language_grid = MagicMock()
        language_grid.rank_offset = 0
        language_grid.size = 2
        language_grid.get_pg.return_value = MagicMock()
        vision_grid = MagicMock()
        vision_grid.rank_offset = 0
        vision_grid.size = 2
        vision_grid.get_pg.return_value = MagicMock()
        mock_build_grids.return_value = {"language": language_grid, "vision": vision_grid}

        provider = MegatronMIMOProvider(
            language_model_spec=language_spec,
            modality_submodules_spec=_dummy_modality_spec_dict(),
            megatron_mimo_parallelism_config=megatron_mimo_parallelism_config,
        )

        # Call multiple times
        infra1 = provider.build_infra()
        infra2 = provider.build_infra()

        # Memoized: same object identity across calls.
        assert infra1 is infra2
        # build_hypercomm_grids should only fire on the first call.
        assert mock_build_grids.call_count == 1
        # And mutations on the returned infra are visible to subsequent callers
        # — the load-bearing property for cuda_rng_states_per_module.
        infra1.cuda_rng_states_per_module["vision"] = {"sentinel": object()}
        assert "vision" in infra2.cuda_rng_states_per_module

    @patch("torch.distributed.new_group")
    @patch("torch.distributed.get_process_group_ranks")
    @patch("torch.distributed.get_rank")
    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.MimoModel")
    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.build_hypercomm_grids")
    def test_provide_with_parallelism(
        self, mock_build_grids, mock_mimo_model, mock_get_rank, mock_get_pg_ranks, mock_new_group
    ):
        """Test provide() with parallelism config."""
        mock_get_rank.return_value = 0
        mock_get_pg_ranks.return_value = [0, 1, 2, 3]
        mock_new_group.return_value = MagicMock()
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})

        megatron_mimo_parallelism_config = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(
                    tensor_model_parallel_size=2,
                    data_parallel_size=2,
                ),
                "vision": ModuleParallelismConfig(
                    tensor_model_parallel_size=2,
                    data_parallel_size=2,
                ),
            },
        )

        language_grid = MagicMock()
        language_grid.rank_offset = 0
        language_grid.size = 4
        language_grid.get_pg.return_value = MagicMock()
        vision_grid = MagicMock()
        vision_grid.rank_offset = 0
        vision_grid.size = 4
        vision_grid.get_pg.return_value = MagicMock()
        mock_build_grids.return_value = {"language": language_grid, "vision": vision_grid}

        provider = MegatronMIMOProvider(
            language_model_spec=language_spec,
            modality_submodules_spec=_dummy_modality_spec_dict(),
            megatron_mimo_parallelism_config=megatron_mimo_parallelism_config,
        )

        mock_model_instance = MagicMock()
        mock_mimo_model.return_value = mock_model_instance

        model = provider.provide()

        # Should return model directly
        assert model == mock_model_instance
        config_arg = mock_mimo_model.call_args[0][0]
        assert config_arg.module_to_grid_map == {"language": language_grid, "vision": vision_grid}

        # Infrastructure should be available via build_infra()
        infra = provider.build_infra()
        assert "language" in infra.module_to_grid_map
        assert "language" in infra.pg_collections

    @patch("torch.distributed.new_group")
    @patch("torch.distributed.get_process_group_ranks")
    @patch("torch.distributed.get_rank")
    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.MimoModel")
    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.build_hypercomm_grids")
    def test_provide_threads_cp_and_tp_groups_into_mimo_model(
        self, mock_build_grids, mock_mimo_model, mock_get_rank, mock_get_pg_ranks, mock_new_group
    ):
        """MimoModel must receive cp_group and tp_group from the language module's
        pg_collection. MimoModel's PartitionAdapter (built when language CP>1 or
        sequence_parallel=True) otherwise falls back to uninitialised globals —
        MegatronMIMO intentionally skips initialize_model_parallel()."""
        mock_get_rank.return_value = 0
        mock_get_pg_ranks.return_value = [0, 1, 2, 3]
        mock_new_group.return_value = MagicMock()
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})

        megatron_mimo_parallelism_config = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(
                    tensor_model_parallel_size=2,
                    context_parallel_size=2,
                    data_parallel_size=1,
                ),
                "vision": ModuleParallelismConfig(
                    tensor_model_parallel_size=2,
                    context_parallel_size=1,
                    data_parallel_size=2,
                ),
            },
        )

        # Distinct sentinel PGs so we can assert correct threading, not just truthiness.
        tp_pg = MagicMock(name="tp_pg")
        cp_pg = MagicMock(name="cp_pg")
        language_grid = MagicMock()
        language_grid.rank_offset = 0
        language_grid.size = 4
        language_grid.is_current_rank_in_grid.return_value = True
        language_grid.get_pg.side_effect = lambda dims: {"tp": tp_pg, "cp": cp_pg}.get(dims[0], MagicMock())
        vision_grid = MagicMock()
        vision_grid.rank_offset = 0
        vision_grid.size = 4
        vision_grid.is_current_rank_in_grid.return_value = True
        vision_grid.get_pg.return_value = MagicMock()
        mock_build_grids.return_value = {"language": language_grid, "vision": vision_grid}

        provider = MegatronMIMOProvider(
            language_model_spec=language_spec,
            modality_submodules_spec=_dummy_modality_spec_dict(),
            megatron_mimo_parallelism_config=megatron_mimo_parallelism_config,
        )

        provider.provide()

        kwargs = mock_mimo_model.call_args.kwargs
        assert kwargs["cp_group"] is cp_pg
        assert kwargs["tp_group"] is tp_pg

    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.new_group")
    @patch("torch.distributed.get_process_group_ranks")
    @patch("torch.distributed.get_rank")
    @patch("megatron.core.models.mimo.model.base.ColocatedBridgeCommunicator")
    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.build_hypercomm_grids")
    def test_provide_binds_language_cp_group_to_unwrapped_partition_adapter(
        self,
        mock_build_grids,
        mock_colocated_comm,
        mock_get_rank,
        mock_get_pg_ranks,
        mock_new_group,
        _mock_is_initialized,
    ):
        """The real MimoModel PartitionAdapter must use the language PG groups.

        Production training sees DDP/precision wrappers around the MimoModel, so
        assertions must unwrap before reading ``partition_adapter``.
        """
        from megatron.bridge.training.megatron_mimo_parallel_utils import unwrap_megatron_mimo_model

        mock_get_rank.return_value = 0
        mock_get_pg_ranks.return_value = [0]
        mock_new_group.return_value = MagicMock()
        mock_colocated_comm.return_value = MagicMock()

        language_config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=1,
            calculate_per_token_loss=True,
        )
        language_spec = ModuleSpec(
            module=_DummyLanguageModel,
            params={
                "config": language_config,
                "max_sequence_length": 16,
            },
        )
        vision_spec = ModuleSpec(module=_DummyModalitySubmodule, params={})

        language_tp_pg = MagicMock(name="language_tp_pg")
        language_cp_pg = MagicMock(name="language_cp_pg")
        language_tp_pg.size.return_value = 1
        language_tp_pg.rank.return_value = 0
        language_cp_pg.size.return_value = 2
        language_cp_pg.rank.return_value = 0
        pp_pg = MagicMock(name="pp_pg")
        pp_pg.size.return_value = 1
        pp_pg.rank.return_value = 0

        def _make_grid(module_groups: dict[str, object]):
            grid = MagicMock()
            grid.rank_offset = 0
            grid.size = 2
            grid.dim_names = ["tp", "cp", "ep", "pp", "dp"]
            grid.is_current_rank_in_grid.return_value = True
            grid.get_pg.side_effect = lambda dims: module_groups.get(
                ".".join(dims) if isinstance(dims, list) else dims
            )
            return grid

        language_grid = _make_grid(
            {
                "tp": language_tp_pg,
                "cp": language_cp_pg,
                "pp": pp_pg,
                "dp": MagicMock(name="language_dp_pg"),
                "ep": MagicMock(name="language_ep_pg"),
                "dp.cp": MagicMock(name="language_dp_cp_pg"),
                "tp.pp": MagicMock(name="language_mp_pg"),
                "tp.ep.pp": MagicMock(name="language_tp_ep_pp_pg"),
            }
        )
        vision_grid = _make_grid(
            {
                "tp": MagicMock(name="vision_tp_pg"),
                "cp": MagicMock(name="vision_cp_pg"),
                "pp": pp_pg,
                "dp": MagicMock(name="vision_dp_pg"),
                "ep": MagicMock(name="vision_ep_pg"),
                "dp.cp": MagicMock(name="vision_dp_cp_pg"),
                "tp.pp": MagicMock(name="vision_mp_pg"),
                "tp.ep.pp": MagicMock(name="vision_tp_ep_pp_pg"),
            }
        )
        mock_build_grids.return_value = {MIMO_LANGUAGE_MODULE_KEY: language_grid, "vision": vision_grid}

        parallelism_config = MegatronMIMOParallelismConfig(
            module_parallelisms={
                MIMO_LANGUAGE_MODULE_KEY: ModuleParallelismConfig(
                    tensor_model_parallel_size=1,
                    context_parallel_size=2,
                    data_parallel_size=1,
                ),
                "vision": ModuleParallelismConfig(
                    tensor_model_parallel_size=1,
                    context_parallel_size=1,
                    data_parallel_size=2,
                ),
            },
        )
        provider = MegatronMIMOProvider(
            language_model_spec=language_spec,
            modality_submodules_spec={"vision": vision_spec},
            megatron_mimo_parallelism_config=parallelism_config,
        )

        model = provider.provide()
        wrapped_model = SimpleNamespace(module=SimpleNamespace(module=model))
        inner = unwrap_megatron_mimo_model(wrapped_model)

        assert inner.partition_adapter is not None
        assert inner.partition_adapter.cfg.cp_group is language_cp_pg
        assert inner.partition_adapter.cfg.tp_group is language_tp_pg
        assert inner.partition_adapter.cfg.use_cp is True

    @patch("torch.distributed.new_group")
    @patch("torch.distributed.get_process_group_ranks")
    @patch("torch.distributed.get_rank")
    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.MimoModel")
    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.build_hypercomm_grids")
    def test_provide_threads_module_rng_scope_factories_when_snapshots_present(
        self, mock_build_grids, mock_mimo_model, mock_get_rank, mock_get_pg_ranks, mock_new_group
    ):
        """When ``infra.cuda_rng_states_per_module`` is populated (Step 3b
        seeding ran), ``provide()`` binds one factory per module and threads
        the dict into ``MimoModel(module_rng_scopes=...)``. Each factory
        captures the module name via default-arg so calling factory() yields
        a fresh context manager every time.

        Load-bearing for asymmetric TP: a regression where the kwarg gets
        dropped or the factories all close over the same loop variable would
        either skip module-scoped RNG entirely or use the wrong module's
        snapshot.
        """
        mock_get_rank.return_value = 0
        mock_get_pg_ranks.return_value = [0, 1, 2, 3]
        mock_new_group.return_value = MagicMock()
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})

        megatron_mimo_parallelism_config = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=2),
                "vision": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=4),
            },
        )
        language_grid = MagicMock()
        language_grid.rank_offset = 0
        language_grid.size = 4
        language_grid.get_pg.return_value = MagicMock()
        vision_grid = MagicMock()
        vision_grid.rank_offset = 0
        vision_grid.size = 4
        vision_grid.get_pg.return_value = MagicMock()
        mock_build_grids.return_value = {"language": language_grid, "vision": vision_grid}

        provider = MegatronMIMOProvider(
            language_model_spec=language_spec,
            modality_submodules_spec=_dummy_modality_spec_dict(),
            megatron_mimo_parallelism_config=megatron_mimo_parallelism_config,
        )

        # Simulate setup having seeded snapshots before provide() runs.
        infra = provider.build_infra()
        infra.cuda_rng_states_per_module["language"] = {"sentinel": "lang"}
        infra.cuda_rng_states_per_module["vision"] = {"sentinel": "vision"}

        provider.provide()

        scopes = mock_mimo_model.call_args.kwargs["module_rng_scopes"]
        assert scopes is not None
        assert set(scopes.keys()) == {"language", "vision"}
        # Each value is a callable returning a context manager.
        for name, factory in scopes.items():
            assert callable(factory), f"{name} factory must be callable"
            with factory():
                pass  # context-manager protocol works.

        # Default-arg capture: factories must be bound to distinct module names,
        # not all closing over the loop variable's final value. Inspect each
        # factory's __defaults__ to verify the bound name matches.
        assert scopes["language"].__defaults__ == ("language",)
        assert scopes["vision"].__defaults__ == ("vision",)

    @patch("torch.distributed.new_group")
    @patch("torch.distributed.get_process_group_ranks")
    @patch("torch.distributed.get_rank")
    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.MimoModel")
    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.build_hypercomm_grids")
    def test_provide_does_not_bind_module_rng_scopes_for_symmetric_colocated(
        self, mock_build_grids, mock_mimo_model, mock_get_rank, mock_get_pg_ranks, mock_new_group
    ):
        """Symmetric colocated layouts stay on MCore's singleton RNG tracker.

        Even if a stale test fixture populates ``cuda_rng_states_per_module``,
        ``provide()`` must not bind ``module_rng_scopes`` unless the layout
        actually requires per-module RNG.
        """
        mock_get_rank.return_value = 0
        mock_get_pg_ranks.return_value = [0, 1, 2, 3]
        mock_new_group.return_value = MagicMock()
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})

        megatron_mimo_parallelism_config = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=2),
                "vision": ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=2),
            },
        )
        language_grid = MagicMock(rank_offset=0, size=4)
        language_grid.get_pg.return_value = MagicMock()
        vision_grid = MagicMock(rank_offset=0, size=4)
        vision_grid.get_pg.return_value = MagicMock()
        mock_build_grids.return_value = {"language": language_grid, "vision": vision_grid}

        provider = MegatronMIMOProvider(
            language_model_spec=language_spec,
            modality_submodules_spec=_dummy_modality_spec_dict(),
            megatron_mimo_parallelism_config=megatron_mimo_parallelism_config,
        )
        infra = provider.build_infra()
        infra.cuda_rng_states_per_module["language"] = {"sentinel": "lang"}
        infra.cuda_rng_states_per_module["vision"] = {"sentinel": "vision"}

        provider.provide()

        assert mock_mimo_model.call_args.kwargs["module_rng_scopes"] is None

    @patch("torch.distributed.new_group")
    @patch("torch.distributed.get_process_group_ranks")
    @patch("torch.distributed.get_rank")
    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.MimoModel")
    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.build_hypercomm_grids")
    def test_provide_passes_none_module_rng_scopes_when_no_snapshots(
        self, mock_build_grids, mock_mimo_model, mock_get_rank, mock_get_pg_ranks, mock_new_group
    ):
        """Legacy / un-seeded path: ``cuda_rng_states_per_module`` empty →
        ``module_rng_scopes=None`` so MimoModel uses its prior single-tracker
        behavior. Backward-compat guard for existing non-asymmetric-TP setups."""
        mock_get_rank.return_value = 0
        mock_get_pg_ranks.return_value = [0, 1, 2, 3]
        mock_new_group.return_value = MagicMock()
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})

        megatron_mimo_parallelism_config = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=2),
                "vision": ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=2),
            },
        )
        language_grid = MagicMock()
        language_grid.rank_offset = 0
        language_grid.size = 4
        language_grid.get_pg.return_value = MagicMock()
        vision_grid = MagicMock()
        vision_grid.rank_offset = 0
        vision_grid.size = 4
        vision_grid.get_pg.return_value = MagicMock()
        mock_build_grids.return_value = {"language": language_grid, "vision": vision_grid}

        provider = MegatronMIMOProvider(
            language_model_spec=language_spec,
            modality_submodules_spec=_dummy_modality_spec_dict(),
            megatron_mimo_parallelism_config=megatron_mimo_parallelism_config,
        )

        # NO snapshot population — leaves cuda_rng_states_per_module empty.
        provider.provide()

        assert mock_mimo_model.call_args.kwargs["module_rng_scopes"] is None

    def test_inject_pg_collection_into_language_spec(self):
        """Test that pg_collection is injected into language specs."""
        language_spec = ModuleSpec(module=Mock, params={})

        provider = MegatronMIMOProvider(language_model_spec=language_spec)

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

        provider = MegatronMIMOProvider(language_model_spec=ModuleSpec(module=Mock, params={}))

        mock_pg_collection = MagicMock()
        mock_pg_collection.tp = MagicMock()

        injected_spec = provider._inject_pg_collection_into_modality_spec(modality_spec, "vision", mock_pg_collection)

        # Check modality submodule has pg_collection for checkpoint sharded-state metadata.
        assert injected_spec.params["pg_collection"] == mock_pg_collection
        # Check encoder has pg_collection
        assert injected_spec.submodules["encoders"]["clip"].params["pg_collection"] == mock_pg_collection

    def test_inject_pg_collection_sets_language_transformer_parallelism(self):
        """Language spec config must see module-local TP/PP/CP/ETP sizes."""
        language_config = SimpleNamespace()
        language_spec = ModuleSpec(module=Mock, params={"config": language_config})
        par_cfg = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(
                    tensor_model_parallel_size=2,
                    pipeline_model_parallel_size=2,
                    context_parallel_size=1,
                    expert_tensor_parallel_size=1,
                    data_parallel_size=2,
                ),
                "vision": ModuleParallelismConfig(
                    tensor_model_parallel_size=2,
                    pipeline_model_parallel_size=1,
                    context_parallel_size=1,
                    expert_tensor_parallel_size=1,
                    data_parallel_size=4,
                ),
            },
        )
        provider = MegatronMIMOProvider(
            language_model_spec=language_spec,
            modality_submodules_spec=_dummy_modality_spec_dict(),
            megatron_mimo_parallelism_config=par_cfg,
        )

        injected_spec = provider._inject_pg_collection_into_language_spec(language_spec, MagicMock())
        injected_config = injected_spec.params["config"]

        assert injected_config.tensor_model_parallel_size == 2
        assert injected_config.pipeline_model_parallel_size == 2
        assert injected_config.context_parallel_size == 1
        assert injected_config.expert_tensor_parallel_size == 1
        assert not hasattr(language_config, "pipeline_model_parallel_size")

    def test_inject_pg_collection_sets_modality_transformer_parallelism(self):
        """Modality encoder/projection configs must see module-local parallelism."""
        encoder_config = SimpleNamespace()
        projection_config = SimpleNamespace()
        encoder_spec = ModuleSpec(module=Mock, params={"transformer_config": encoder_config})
        projection_spec = ModuleSpec(module=Mock, params={"config": projection_config})
        modality_spec = ModuleSpec(
            module=Mock,
            params={},
            submodules={"encoders": {"clip": encoder_spec}, "input_projections": [projection_spec]},
        )
        par_cfg = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(
                    tensor_model_parallel_size=2,
                    pipeline_model_parallel_size=2,
                    data_parallel_size=2,
                ),
                "vision": ModuleParallelismConfig(
                    tensor_model_parallel_size=2,
                    pipeline_model_parallel_size=1,
                    data_parallel_size=4,
                ),
            },
        )
        provider = MegatronMIMOProvider(
            language_model_spec=ModuleSpec(module=Mock, params={}),
            modality_submodules_spec={"vision": modality_spec},
            megatron_mimo_parallelism_config=par_cfg,
        )
        mock_pg_collection = MagicMock()
        mock_pg_collection.tp = MagicMock()

        injected_spec = provider._inject_pg_collection_into_modality_spec(
            modality_spec,
            "vision",
            mock_pg_collection,
        )
        injected_encoder_config = injected_spec.submodules["encoders"]["clip"].params["transformer_config"]
        injected_projection_config = injected_spec.submodules["input_projections"][0].params["config"]

        for cfg in (injected_encoder_config, injected_projection_config):
            assert cfg.tensor_model_parallel_size == 2
            assert cfg.pipeline_model_parallel_size == 1
            assert cfg.context_parallel_size == 1
            assert cfg.expert_tensor_parallel_size == 1
        assert not hasattr(encoder_config, "tensor_model_parallel_size")
        assert not hasattr(projection_config, "tensor_model_parallel_size")

    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.MimoModel")
    def test_freezing_language_model(self, mock_mimo_model):
        """Test freeze_language_model works."""
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})

        # Create mock model with parameters
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.requires_grad = True
        mock_model.language_model.parameters.return_value = [mock_param]
        mock_mimo_model.return_value = mock_model

        provider = MegatronMIMOProvider(
            language_model_spec=language_spec,
            modality_submodules_spec=_dummy_modality_spec_dict(),
            freeze_language_model=True,
        )

        provider.provide()

        # Check parameter was frozen
        assert mock_param.requires_grad is False

    @patch("torch.distributed.new_group")
    @patch("torch.distributed.get_process_group_ranks")
    @patch("torch.distributed.get_rank")
    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.MimoModel")
    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.build_hypercomm_grids")
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

        megatron_mimo_parallelism_config = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(tensor_model_parallel_size=8, data_parallel_size=1),
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
            "language": llm_grid,
            "clip_encoder": clip_grid,
            "dino_encoder": dino_grid,
        }

        provider = MegatronMIMOProvider(
            language_model_spec=language_spec,
            modality_submodules_spec={
                "clip_encoder": clip_spec,
                "dino_encoder": dino_spec,
            },
            special_token_ids={
                "clip_encoder": 32000,
                "dino_encoder": 32001,
            },
            megatron_mimo_parallelism_config=megatron_mimo_parallelism_config,
        )

        mock_model_instance = MagicMock()
        mock_mimo_model.return_value = mock_model_instance

        model = provider.provide()
        infra = provider.build_infra()

        # Should build grids with all three modules
        mock_build_grids.assert_called_with(megatron_mimo_parallelism_config)

        # Should have pg_collections for all modules
        assert "language" in infra.pg_collections
        assert "clip_encoder" in infra.pg_collections
        assert "dino_encoder" in infra.pg_collections

        # Should return model directly
        assert model == mock_model_instance

    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.seed_singleton_rng_tracker")
    def test_initialize_model_parallel_builds_infra_and_seeds(self, mock_seed):
        """initialize_model_parallel is the public provider setup hook for
        MegatronMIMO: it finalizes, builds/memoizes infra, and seeds CUDA RNG
        snapshots without invoking global MPU initialization."""
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        provider = MegatronMIMOProvider(
            language_model_spec=language_spec,
            modality_submodules_spec=_dummy_modality_spec_dict(),
        )
        infra = MegatronMIMOInfra(
            module_to_grid_map={},
            topology={},
            pg_collections={},
            participating_modules=[],
        )

        with (
            patch.object(provider, "finalize") as mock_finalize,
            patch.object(provider, "build_infra", return_value=infra) as mock_build_infra,
        ):
            provider.initialize_model_parallel(seed=42, seed_kwargs={"te_rng_tracker": True})

        mock_finalize.assert_called_once()
        mock_build_infra.assert_called_once()
        mock_seed.assert_called_once_with(42, infra, seed_kwargs={"te_rng_tracker": True})
        assert provider._mimo_model_parallel_initialized is True

    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.get_model_config")
    def test_provide_distributed_model_auto_initializes_mimo_state(self, mock_get_config):
        """Provider-facing conversion paths call provide_distributed_model()
        directly. Match standard Bridge by auto-initializing MIMO provider state
        with seed=0 when the caller did not do it explicitly."""
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        provider = MegatronMIMOProvider(
            language_model_spec=language_spec,
            modality_submodules_spec=_dummy_modality_spec_dict(),
            megatron_mimo_parallelism_config=MegatronMIMOParallelismConfig(
                module_parallelisms={
                    "language": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=2),
                    "vision": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=2),
                },
            ),
            bf16=False,
            fp16=False,
        )
        infra = MegatronMIMOInfra(
            module_to_grid_map={"language": Mock(), "vision": Mock()},
            topology={"vision": ["language"], "language": []},
            pg_collections={"language": Mock(), "vision": Mock()},
            participating_modules=["language", "vision"],
        )
        model = MagicMock()
        mock_model_config = MagicMock(variable_seq_lengths=False)
        mock_get_config.return_value = mock_model_config

        with (
            patch.object(provider, "initialize_model_parallel") as mock_init,
            patch.object(provider, "build_infra", return_value=infra),
            patch.object(provider, "provide", return_value=model),
        ):
            provider.provide_distributed_model(
                wrap_with_ddp=False,
                use_cpu_initialization=True,
            )

        mock_init.assert_called_once_with(seed=0)
        assert mock_model_config.variable_seq_lengths is True

    def test_raw_provide_requires_rng_snapshots_for_colocated_asymmetric_tp(self):
        """Raw provide() is not the user-facing distributed constructor. If a
        caller bypasses initialize_model_parallel() under colocated asymmetric
        TP, fail loudly instead of constructing with the wrong singleton RNG."""
        provider = self._make_asymmetric_tp_provider()
        provider.megatron_mimo_parallelism_config.finalize(world_size=2)
        infra = MegatronMIMOInfra(
            module_to_grid_map={"language": Mock(), "vision": Mock()},
            topology={"vision": ["language"], "language": []},
            pg_collections={"language": Mock(), "vision": Mock()},
            participating_modules=["language", "vision"],
            cuda_rng_states_per_module={},
        )

        with patch.object(provider, "build_infra", return_value=infra):
            with pytest.raises(RuntimeError, match="requires per-module RNG snapshots"):
                provider.provide()

    @patch("megatron.core.transformer.module.Float16Module")
    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.get_model_config")
    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.MimoModel")
    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.build_hypercomm_grids")
    @patch("torch.distributed.is_initialized")
    def test_provide_distributed_model_sets_variable_seq_lengths(
        self, mock_is_init, mock_build_grids, mock_mimo_model, mock_get_config, mock_float16
    ):
        """Test that provide_distributed_model sets variable_seq_lengths=True."""
        mock_is_init.return_value = False
        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})

        provider = MegatronMIMOProvider(
            language_model_spec=language_spec,
            modality_submodules_spec=_dummy_modality_spec_dict(),
            bf16=False,  # Disable to simplify test
            fp16=False,
        )

        mock_model_instance = MagicMock()
        mock_model_instance.cuda = MagicMock(return_value=None)
        mock_mimo_model.return_value = mock_model_instance

        mock_config = MagicMock()
        mock_config.variable_seq_lengths = False  # Initial value
        mock_get_config.return_value = mock_config

        # No parallelism config means no DDP wrapping needed
        provider.provide_distributed_model(wrap_with_ddp=False)

        # Should have set variable_seq_lengths=True
        assert mock_config.variable_seq_lengths is True

    # ── Step 3f: asymmetric-TP unsafe-combo guards ────────────────────────────

    def _make_asymmetric_tp_provider(
        self,
        *,
        use_cpu_initialization: bool = False,
        language_recompute: Optional[str] = None,
        encoder_recompute: Optional[str] = None,
    ) -> MegatronMIMOProvider:
        """Construct a provider with the canonical fan-in asymmetric-TP shape.

        Optionally injects ``recompute_granularity`` into the language config
        and/or into the nested encoder config under
        ``vision_spec.submodules["encoders"]["clip"]``. This nesting matches
        the production modality-spec shape (vision wrapper has empty top-level
        params; the real TransformerConfig lives inside the encoder spec).
        """
        from types import SimpleNamespace

        language_config = SimpleNamespace(recompute_granularity=language_recompute)
        language_spec = ModuleSpec(module=Mock, params={"config": language_config})

        encoder_config = SimpleNamespace(recompute_granularity=encoder_recompute)
        encoder_spec = ModuleSpec(module=Mock, params={"transformer_config": encoder_config})
        # Modality wrapper has empty top-level params (matches production shape).
        vision_spec = ModuleSpec(
            module=Mock,
            params={},
            submodules={"encoders": {"clip": encoder_spec}},
        )

        return MegatronMIMOProvider(
            language_model_spec=language_spec,
            modality_submodules_spec={"vision": vision_spec},
            megatron_mimo_parallelism_config=MegatronMIMOParallelismConfig(
                module_parallelisms={
                    "language": ModuleParallelismConfig(
                        tensor_model_parallel_size=2, data_parallel_size=1, rank_offset=0
                    ),
                    "vision": ModuleParallelismConfig(
                        tensor_model_parallel_size=1, data_parallel_size=2, rank_offset=0
                    ),
                },
            ),
            use_cpu_initialization=use_cpu_initialization,
        )

    def test_asymmetric_tp_validator_accepts_clean_config(self):
        """Sanity: asymmetric-TP fan-in with no cpu-init / recompute passes."""
        provider = self._make_asymmetric_tp_provider()
        # Force-finalize the parallelism config so _is_colocated() works without dist.
        provider.megatron_mimo_parallelism_config.finalize(world_size=2)
        provider._validate_asymmetric_tp_constraints()  # must not raise

    def test_asymmetric_tp_validator_accepts_cpu_init(self):
        """``use_cpu_initialization=True`` is accepted under colocated asymmetric TP.

        CPU init's correctness mechanism (``init_method`` builds the full
        master weight on every rank from a shared ``torch.manual_seed``, then
        each layer slices its own TP shard via the module's ``tp_group``) is
        independent of the CUDA RNG tracker. The standard Bridge path uses
        the same single-``torch.manual_seed`` mechanism for CPU init across
        arbitrarily complex TP/PP layouts.

        A previous version of this validator rejected this combination —
        the rejection was over-conservative and is removed in v1. This test
        pins the accepting behavior so a future regression that re-adds the
        rejection without justification flips this test red.
        """
        provider = self._make_asymmetric_tp_provider(use_cpu_initialization=True)
        provider.megatron_mimo_parallelism_config.finalize(world_size=2)
        provider._validate_asymmetric_tp_constraints()  # must not raise

    def test_asymmetric_tp_validator_rejects_language_recompute(self):
        """Recompute on the top-level language ``TransformerConfig`` is caught."""
        provider = self._make_asymmetric_tp_provider(language_recompute="full")
        provider.megatron_mimo_parallelism_config.finalize(world_size=2)
        with pytest.raises(ValueError, match=r"recomputation is not supported.*language"):
            provider._validate_asymmetric_tp_constraints()

    def test_asymmetric_tp_validator_rejects_recompute_in_nested_encoder(self):
        """Recompute on the NESTED encoder config (inside
        ``vision_spec.submodules["encoders"]["clip"]``) must be caught.

        Critical regression guard for ``_walk_module_spec_for_recompute``: a
        non-recursive check would miss this entirely because
        ``vision_spec.params`` is empty. Production modality specs always nest
        the real TransformerConfig under ``submodules["encoders"]``.
        """
        provider = self._make_asymmetric_tp_provider(encoder_recompute="full")
        provider.megatron_mimo_parallelism_config.finalize(world_size=2)
        with pytest.raises(ValueError, match=r"vision\.encoders\.clip"):
            provider._validate_asymmetric_tp_constraints()

    def test_asymmetric_tp_validator_no_op_under_symmetric_tp(self):
        """Symmetric-TP colocated → validator is a no-op even with recompute
        set. Backward-compat for existing symmetric setups."""
        from types import SimpleNamespace

        encoder_config = SimpleNamespace(recompute_granularity="full")
        encoder_spec = ModuleSpec(module=Mock, params={"transformer_config": encoder_config})
        vision_spec = ModuleSpec(module=Mock, params={}, submodules={"encoders": {"clip": encoder_spec}})
        language_spec = ModuleSpec(module=Mock, params={"config": SimpleNamespace(recompute_granularity="full")})

        provider = MegatronMIMOProvider(
            language_model_spec=language_spec,
            modality_submodules_spec={"vision": vision_spec},
            megatron_mimo_parallelism_config=MegatronMIMOParallelismConfig(
                module_parallelisms={
                    "language": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=2),
                    "vision": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=2),
                },
            ),
        )
        provider.megatron_mimo_parallelism_config.finalize(world_size=2)
        # No raise — symmetric TP is unaffected by the asymmetric-TP guards.
        provider._validate_asymmetric_tp_constraints()

    def test_asymmetric_tp_validator_no_op_under_non_colocated(self):
        """Non-colocated (disjoint ranks) → validator is a no-op even with
        asymmetric TP and recompute. The asymmetric-TP guard is colocated-only
        because non-colocated places encoder and language on different ranks.
        """
        from types import SimpleNamespace

        encoder_config = SimpleNamespace(recompute_granularity="full")
        encoder_spec = ModuleSpec(module=Mock, params={"transformer_config": encoder_config})
        vision_spec = ModuleSpec(module=Mock, params={}, submodules={"encoders": {"clip": encoder_spec}})
        language_spec = ModuleSpec(module=Mock, params={"config": SimpleNamespace(recompute_granularity=None)})

        provider = MegatronMIMOProvider(
            language_model_spec=language_spec,
            modality_submodules_spec={"vision": vision_spec},
            megatron_mimo_parallelism_config=MegatronMIMOParallelismConfig(
                module_parallelisms={
                    # Disjoint rank ranges — non-colocated.
                    "language": ModuleParallelismConfig(
                        tensor_model_parallel_size=2, data_parallel_size=1, rank_offset=0
                    ),
                    "vision": ModuleParallelismConfig(
                        tensor_model_parallel_size=1, data_parallel_size=2, rank_offset=2
                    ),
                },
            ),
        )
        provider.megatron_mimo_parallelism_config.finalize(world_size=4)
        # No raise — non-colocated is outside the asymmetric-TP guard's scope.
        provider._validate_asymmetric_tp_constraints()

    def _make_colocated_language_pp_provider(
        self,
        *,
        language_per_token_loss: bool = True,
        encoder_per_token_loss: bool = True,
        encoder_names: tuple[str, ...] = ("clip",),
        language_pp: int = 2,
        language_cp: int = 1,
        language_tp: int = 1,
        language_config_kwargs: Optional[dict] = None,
        language_spec_kwargs: Optional[dict] = None,
    ) -> MegatronMIMOProvider:
        """Build a finalized colocated language-PP/CP provider for spec validation."""
        language_config_kwargs = language_config_kwargs or {}
        language_spec_kwargs = language_spec_kwargs or {}
        language_dp = 4 // (language_pp * language_cp * language_tp)
        language_spec = ModuleSpec(
            module=Mock,
            params={
                "config": _config(
                    calculate_per_token_loss=language_per_token_loss,
                    **language_config_kwargs,
                ),
                **language_spec_kwargs,
            },
        )
        encoder_specs = {
            name: ModuleSpec(
                module=Mock,
                params={"transformer_config": _config(calculate_per_token_loss=encoder_per_token_loss)},
            )
            for name in encoder_names
        }
        vision_spec = ModuleSpec(
            module=Mock,
            params={},
            submodules={"encoders": encoder_specs},
        )
        parallelism_config = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(
                    tensor_model_parallel_size=language_tp,
                    pipeline_model_parallel_size=language_pp,
                    context_parallel_size=language_cp,
                    data_parallel_size=language_dp,
                    rank_offset=0,
                ),
                "vision": ModuleParallelismConfig(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                    data_parallel_size=4,
                    rank_offset=0,
                ),
            },
        )
        parallelism_config.finalize(world_size=4)
        return MegatronMIMOProvider(
            language_model_spec=language_spec,
            modality_submodules_spec={"vision": vision_spec},
            megatron_mimo_parallelism_config=parallelism_config,
        )

    def test_colocated_language_pp_spec_validator_accepts_single_encoder_per_token_loss(self):
        """Language PP accepts the v1 spec shape: one modality, one encoder tower."""
        provider = self._make_colocated_language_pp_provider()
        provider._validate_colocated_language_pp_or_cp_spec_constraints()

    def test_colocated_language_pp_spec_validator_rejects_multiple_encoder_towers(self):
        """Nested multi-encoder modality inputs are rejected for v1."""
        provider = self._make_colocated_language_pp_provider(encoder_names=("clip", "dino"))
        with pytest.raises(ValueError, match="exactly one encoder tower"):
            provider._validate_colocated_language_pp_or_cp_spec_constraints()

    def test_colocated_language_pp_spec_validator_rejects_language_without_per_token_loss(self):
        """Language TransformerConfig must use calculate_per_token_loss=True."""
        provider = self._make_colocated_language_pp_provider(language_per_token_loss=False)
        with pytest.raises(ValueError, match=r"calculate_per_token_loss=True.*\['language'\]"):
            provider._validate_colocated_language_pp_or_cp_spec_constraints()

    def test_colocated_language_pp_spec_validator_rejects_encoder_without_per_token_loss(self):
        """Encoder TransformerConfig must use calculate_per_token_loss=True."""
        provider = self._make_colocated_language_pp_provider(encoder_per_token_loss=False)
        with pytest.raises(
            ValueError,
            match=r"calculate_per_token_loss=True.*\['vision\.encoders\.clip'\]",
        ):
            provider._validate_colocated_language_pp_or_cp_spec_constraints()

    def test_colocated_language_cp_spec_validator_accepts_single_encoder_per_token_loss(self):
        """Language CP shares the v1 single-modality and per-token-loss spec contract."""
        provider = self._make_colocated_language_pp_provider(language_pp=1, language_cp=2)
        provider._validate_colocated_language_pp_or_cp_spec_constraints()

    def test_colocated_language_cp_spec_validator_rejects_multiple_encoder_towers(self):
        """Language CP rejects nested multi-encoder modality specs in v1."""
        provider = self._make_colocated_language_pp_provider(
            language_pp=1,
            language_cp=2,
            encoder_names=("clip", "dino"),
        )
        with pytest.raises(ValueError, match="exactly one encoder tower"):
            provider._validate_colocated_language_pp_or_cp_spec_constraints()

    def test_colocated_language_cp_spec_validator_rejects_language_without_per_token_loss(self):
        """Language CP requires per-token loss for the DP x CP token denominator."""
        provider = self._make_colocated_language_pp_provider(
            language_pp=1,
            language_cp=2,
            language_per_token_loss=False,
        )
        with pytest.raises(ValueError, match=r"calculate_per_token_loss=True.*\['language'\]"):
            provider._validate_colocated_language_pp_or_cp_spec_constraints()

    def test_colocated_language_cp_spec_validator_rejects_hcp_cp_comm_type_string(self):
        """HCP cp_comm_type is rejected because MIMO does not expose HCP groups yet."""
        provider = self._make_colocated_language_pp_provider(
            language_pp=1,
            language_cp=2,
            language_config_kwargs={"cp_comm_type": "a2a+p2p"},
        )
        with pytest.raises(ValueError, match="cp_comm_type.*a2a\\+p2p"):
            provider._validate_colocated_language_pp_or_cp_spec_constraints()

    def test_colocated_language_cp_spec_validator_rejects_hcp_cp_comm_type_list(self):
        """List-valued cp_comm_type is rejected if any entry asks for HCP."""
        provider = self._make_colocated_language_pp_provider(
            language_pp=1,
            language_cp=2,
            language_config_kwargs={"cp_comm_type": ["p2p", "a2a+p2p"]},
        )
        with pytest.raises(ValueError, match="cp_comm_type.*a2a\\+p2p"):
            provider._validate_colocated_language_pp_or_cp_spec_constraints()

    def test_colocated_language_cp_spec_validator_rejects_hierarchical_cp_sizes(self):
        """Explicit HCP sizes are rejected until MIMO process groups expose hcp."""
        provider = self._make_colocated_language_pp_provider(
            language_pp=1,
            language_cp=2,
            language_config_kwargs={"hierarchical_context_parallel_sizes": [2, 1]},
        )
        with pytest.raises(ValueError, match="hierarchical_context_parallel_sizes=None"):
            provider._validate_colocated_language_pp_or_cp_spec_constraints()

    def test_colocated_language_cp_spec_validator_rejects_tp_comm_overlap(self):
        """TP comm overlap is kept out of the v1 CP validation surface."""
        provider = self._make_colocated_language_pp_provider(
            language_pp=1,
            language_cp=2,
            language_config_kwargs={"tp_comm_overlap": True},
        )
        with pytest.raises(ValueError, match="tp_comm_overlap=True"):
            provider._validate_colocated_language_pp_or_cp_spec_constraints()

    def test_colocated_language_cp_spec_validator_accepts_flat_cp_comm_type(self):
        """Flat CP comm types remain accepted."""
        provider = self._make_colocated_language_pp_provider(
            language_pp=1,
            language_cp=2,
            language_config_kwargs={
                "cp_comm_type": "p2p",
                "hierarchical_context_parallel_sizes": None,
                "tp_comm_overlap": False,
            },
        )
        provider._validate_colocated_language_pp_or_cp_spec_constraints()

    def test_colocated_language_cp_sp_validates_sequence_length_divisibility(self):
        """CP+SP needs max_sequence_length divisible by 2 * cp * tp."""
        provider = self._make_colocated_language_pp_provider(
            language_pp=1,
            language_cp=2,
            language_tp=2,
            language_config_kwargs={"sequence_parallel": True},
            language_spec_kwargs={"max_sequence_length": 12},
        )
        with pytest.raises(ValueError, match="divisible by 8"):
            provider._validate_colocated_language_pp_or_cp_spec_constraints()


class TestMegatronMIMOInfra:
    """Test cases for MegatronMIMOInfra dataclass."""

    def test_infra_initialization(self):
        """Test infrastructure dataclass initializes correctly."""
        grids = {"language": MagicMock()}
        topology = {"language": []}
        pg_collections = {"language": MagicMock()}
        participating = ["language"]

        infra = MegatronMIMOInfra(
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
        from megatron.bridge.models.megatron_mimo.megatron_mimo_builder import (
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
        from megatron.bridge.models.megatron_mimo.megatron_mimo_builder import (
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
        from megatron.bridge.models.megatron_mimo.megatron_mimo_builder import (
            populate_embedding_and_position_groups,
        )

        pos_embd_pg, embd_pg = populate_embedding_and_position_groups(None)

        assert pos_embd_pg is None
        assert embd_pg is None

    @patch("torch.distributed.get_process_group_ranks")
    @patch("torch.distributed.get_rank")
    def test_is_pp_first_stage_true(self, mock_get_rank, mock_get_ranks):
        """Test is_pp_first_stage returns True for first stage."""
        from megatron.bridge.models.megatron_mimo.megatron_mimo_builder import is_pp_first_stage

        mock_pp_group = MagicMock()
        mock_get_ranks.return_value = [0, 4, 8, 12]
        mock_get_rank.return_value = 0

        assert is_pp_first_stage(mock_pp_group) is True

    @patch("torch.distributed.get_process_group_ranks")
    @patch("torch.distributed.get_rank")
    def test_is_pp_first_stage_false(self, mock_get_rank, mock_get_ranks):
        """Test is_pp_first_stage returns False for non-first stage."""
        from megatron.bridge.models.megatron_mimo.megatron_mimo_builder import is_pp_first_stage

        mock_pp_group = MagicMock()
        mock_get_ranks.return_value = [0, 4, 8, 12]
        mock_get_rank.return_value = 4

        assert is_pp_first_stage(mock_pp_group) is False

    def test_is_pp_first_stage_none_group(self):
        """Test is_pp_first_stage returns True for None group (no PP)."""
        from megatron.bridge.models.megatron_mimo.megatron_mimo_builder import is_pp_first_stage

        assert is_pp_first_stage(None) is True

    @patch("torch.distributed.get_process_group_ranks")
    @patch("torch.distributed.get_rank")
    def test_is_pp_last_stage_true(self, mock_get_rank, mock_get_ranks):
        """Test is_pp_last_stage returns True for last stage."""
        from megatron.bridge.models.megatron_mimo.megatron_mimo_builder import is_pp_last_stage

        mock_pp_group = MagicMock()
        mock_get_ranks.return_value = [0, 4, 8, 12]
        mock_get_rank.return_value = 12

        assert is_pp_last_stage(mock_pp_group) is True

    @patch("torch.distributed.get_process_group_ranks")
    @patch("torch.distributed.get_rank")
    def test_is_pp_last_stage_false(self, mock_get_rank, mock_get_ranks):
        """Test is_pp_last_stage returns False for non-last stage."""
        from megatron.bridge.models.megatron_mimo.megatron_mimo_builder import is_pp_last_stage

        mock_pp_group = MagicMock()
        mock_get_ranks.return_value = [0, 4, 8, 12]
        mock_get_rank.return_value = 4

        assert is_pp_last_stage(mock_pp_group) is False

    def test_is_pp_last_stage_none_group(self):
        """Test is_pp_last_stage returns True for None group (no PP)."""
        from megatron.bridge.models.megatron_mimo.megatron_mimo_builder import is_pp_last_stage

        assert is_pp_last_stage(None) is True


class TestProcessGroupCollectionWithEmbeddingGroups:
    """Test that ProcessGroupCollection includes embedding groups."""

    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.is_pp_last_stage")
    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.is_pp_first_stage")
    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.populate_embedding_and_position_groups")
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
        megatron_mimo_parallelism_config = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=2),
            },
        )

        # Mock grid
        mock_grid = MagicMock()
        mock_grid.rank_offset = 0
        mock_grid.size = 4
        mock_grid.get_pg.return_value = MagicMock()

        provider = MegatronMIMOProvider(
            language_model_spec=language_spec,
            megatron_mimo_parallelism_config=megatron_mimo_parallelism_config,
        )

        pg_collections = provider._get_pg_collections_from_grids({"language": mock_grid})

        # First stage should have pos_embd but not embd (not last stage)
        assert pg_collections["language"].pos_embd == mock_pos_embd
        assert pg_collections["language"].embd == mock_embd  # First stage gets embd too

    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.is_pp_last_stage")
    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.is_pp_first_stage")
    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.populate_embedding_and_position_groups")
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
        megatron_mimo_parallelism_config = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=2),
            },
        )

        # Mock grid
        mock_grid = MagicMock()
        mock_grid.rank_offset = 0
        mock_grid.size = 8
        mock_grid.get_pg.return_value = MagicMock()

        provider = MegatronMIMOProvider(
            language_model_spec=language_spec,
            megatron_mimo_parallelism_config=megatron_mimo_parallelism_config,
        )

        pg_collections = provider._get_pg_collections_from_grids({"language": mock_grid})

        # Middle stage should have neither embedding group
        assert pg_collections["language"].pos_embd is None
        assert pg_collections["language"].embd is None

    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.is_pp_last_stage")
    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.is_pp_first_stage")
    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.populate_embedding_and_position_groups")
    @patch("torch.distributed.get_rank")
    def test_pg_collection_includes_composite_groups(self, mock_get_rank, mock_populate, mock_is_first, mock_is_last):
        """Test that pg_collection includes mp, tp_ep_pp, and expt_dp composite groups."""
        mock_get_rank.return_value = 0
        mock_populate.return_value = (MagicMock(), MagicMock())
        mock_is_first.return_value = True
        mock_is_last.return_value = True

        language_spec = ModuleSpec(module=Mock, params={"config": Mock()})
        megatron_mimo_parallelism_config = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=2),
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

        provider = MegatronMIMOProvider(
            language_model_spec=language_spec,
            megatron_mimo_parallelism_config=megatron_mimo_parallelism_config,
        )

        pg_collections = provider._get_pg_collections_from_grids({"language": mock_grid})

        pgc = pg_collections["language"]
        assert pgc.tp == mock_tp
        assert pgc.dp == mock_dp
        assert pgc.pp == mock_pp
        assert pgc.cp == mock_cp
        assert pgc.ep == mock_ep
        assert pgc.dp_cp == mock_dp_cp
        assert pgc.mp == mock_mp
        assert pgc.tp_ep_pp == mock_tp_ep_pp
