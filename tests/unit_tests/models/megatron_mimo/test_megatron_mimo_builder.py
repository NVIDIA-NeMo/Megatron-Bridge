# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for MegatronMIMO builder utilities."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from megatron.bridge.models.megatron_mimo import megatron_mimo_builder
from megatron.bridge.models.megatron_mimo.megatron_mimo_builder import (
    EXPERT_VIEW_NAME,
    _lazy_nccl_pg_options,
    build_hypercomm_grids,
    populate_embedding_and_position_groups,
)
from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)


class _NcclOptions:
    """Stand-in for ``ProcessGroupNCCL.Options`` on PyTorch builds that have ``lazy_init``."""

    def __init__(self):
        self.lazy_init = False


class _NcclOptionsWithoutLazyInit:
    """Stand-in for ``ProcessGroupNCCL.Options`` on PyTorch builds without ``lazy_init``."""


def _fake_dist(*, initialized=True, backend="nccl", bound_device_id="cuda:0", options_cls=_NcclOptions):
    """Minimal ``torch.distributed`` stand-in for ``_lazy_nccl_pg_options``."""
    return SimpleNamespace(
        is_initialized=lambda: initialized,
        get_backend=lambda: backend,
        Backend=SimpleNamespace(NCCL="nccl"),
        group=SimpleNamespace(WORLD=SimpleNamespace(bound_device_id=bound_device_id)),
        ProcessGroupNCCL=None if options_cls is None else SimpleNamespace(Options=options_cls),
    )


class TestLazyNcclPgOptions:
    """Test cases for _lazy_nccl_pg_options()."""

    def test_device_bound_nccl_default_group_gets_new_lazy_options_per_call(self):
        """Eager-split setups get lazy_init options, and every group gets its own object."""
        with patch.object(megatron_mimo_builder, "dist", _fake_dist()):
            first = _lazy_nccl_pg_options()
            second = _lazy_nccl_pg_options()

        assert isinstance(first, _NcclOptions)
        assert first.lazy_init is True
        assert second.lazy_init is True
        assert second is not first

    @pytest.mark.parametrize(
        "fake_dist",
        [
            _fake_dist(initialized=False),
            _fake_dist(backend="gloo"),
            _fake_dist(bound_device_id=None),
            _fake_dist(options_cls=None),
            _fake_dist(options_cls=_NcclOptionsWithoutLazyInit),
        ],
        ids=["not-initialized", "gloo-default-group", "not-device-bound", "no-nccl", "no-lazy-init-option"],
    )
    def test_default_options_otherwise(self, fake_dist):
        """Groups keep default options when they are not split eagerly or lazy_init is unavailable."""
        with patch.object(megatron_mimo_builder, "dist", fake_dist):
            assert _lazy_nccl_pg_options() is None


class TestPopulateEmbeddingAndPositionGroups:
    """Test cases for populate_embedding_and_position_groups()."""

    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.new_group")
    def test_embedding_groups_are_created_with_lazy_options(self, mock_new_group, _mock_get_rank):
        """Position and word embedding groups get their own lazy NCCL options."""
        mock_new_group.side_effect = lambda ranks, pg_options: MagicMock(name=f"group{ranks}")

        with patch.object(megatron_mimo_builder, "_lazy_nccl_pg_options", side_effect=lambda: object()):
            pos_embd_pg, embd_pg = populate_embedding_and_position_groups([[0, 1, 2, 3]])

        assert [call.kwargs["ranks"] for call in mock_new_group.call_args_list] == [[0], [0, 3]]
        pos_embd_options, embd_options = (call.kwargs["pg_options"] for call in mock_new_group.call_args_list)
        assert pos_embd_options is not None
        assert embd_options is not None
        assert pos_embd_options is not embd_options
        assert pos_embd_pg is not None
        assert embd_pg is not None


class TestBuildHypercommGrids:
    """Test cases for build_hypercomm_grids()."""

    @patch("megatron.core.hyper_comm_grid.HyperCommGrid")
    def test_build_with_single_module(self, mock_grid_class):
        """Test build_hypercomm_grids with single LLM module."""
        megatron_mimo_config = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(
                    tensor_model_parallel_size=2,
                    context_parallel_size=1,
                    expert_tensor_parallel_size=1,
                    pipeline_model_parallel_size=2,
                    data_parallel_size=2,
                ),
            }
        )

        mock_grid = MagicMock()
        mock_grid.create_pg = MagicMock(return_value=MagicMock())
        mock_grid_class.return_value = mock_grid

        grids = build_hypercomm_grids(megatron_mimo_config)

        # Should create one grid
        assert "language" in grids
        assert grids["language"] == mock_grid

        # Check grid was created with the dense (base) view shape: [tp, cp, dp, pp].
        mock_grid_class.assert_called_once()
        call_kwargs = mock_grid_class.call_args[1]
        assert call_kwargs["shape"] == [2, 1, 2, 2]  # [tp, cp, dp, pp]
        assert call_kwargs["dim_names"] == ["tp", "cp", "dp", "pp"]
        assert call_kwargs["rank_offset"] == 0
        assert call_kwargs["backend"] == "nccl"

        mock_grid.register_view.assert_called_once_with(
            EXPERT_VIEW_NAME,
            shape=[1, 1, 4, 2],
            dim_names=["expt_tp", "ep", "expt_dp", "pp"],
            shared_dims=["pp"],
        )

        # Check all process groups were created
        create_pg_calls = [(call.args[0], call.kwargs.get("view")) for call in mock_grid.create_pg.call_args_list]
        assert (["tp"], None) in create_pg_calls
        assert (["cp"], None) in create_pg_calls
        assert (["pp"], None) in create_pg_calls
        assert (["dp"], None) in create_pg_calls
        assert (["dp", "cp"], None) in create_pg_calls
        assert (["ep"], EXPERT_VIEW_NAME) in create_pg_calls
        assert (["expt_tp"], EXPERT_VIEW_NAME) in create_pg_calls
        assert (["expt_dp"], EXPERT_VIEW_NAME) in create_pg_calls
        assert (["tp", "cp", "dp", "pp"], None) in create_pg_calls
        assert (["expt_tp", "ep"], EXPERT_VIEW_NAME) in create_pg_calls
        assert (["expt_tp", "ep", "pp"], EXPERT_VIEW_NAME) in create_pg_calls

    @patch("megatron.core.hyper_comm_grid.HyperCommGrid")
    def test_build_with_multiple_modules(self, mock_grid_class):
        """Test build_hypercomm_grids with multiple modules."""
        megatron_mimo_config = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(
                    tensor_model_parallel_size=4,
                    data_parallel_size=2,
                    rank_offset=0,
                ),
                "clip_encoder": ModuleParallelismConfig(
                    tensor_model_parallel_size=2,
                    data_parallel_size=2,
                    rank_offset=8,
                ),
                "dino_encoder": ModuleParallelismConfig(
                    tensor_model_parallel_size=2,
                    data_parallel_size=2,
                    rank_offset=12,
                ),
            }
        )

        mock_grid = MagicMock()
        mock_grid.create_pg = MagicMock(return_value=MagicMock())
        mock_grid_class.return_value = mock_grid

        grids = build_hypercomm_grids(megatron_mimo_config)

        # Should create three grids
        assert "language" in grids
        assert "clip_encoder" in grids
        assert "dino_encoder" in grids
        assert len(grids) == 3

        # Verify HyperCommGrid was called 3 times
        assert mock_grid_class.call_count == 3

    @patch("megatron.core.hyper_comm_grid.HyperCommGrid")
    def test_build_with_different_parallelism_per_module(self, mock_grid_class):
        """Test grids with different parallelism configs per module."""
        megatron_mimo_config = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(
                    tensor_model_parallel_size=8,
                    pipeline_model_parallel_size=2,
                    data_parallel_size=1,
                    rank_offset=0,
                ),
                "encoder": ModuleParallelismConfig(
                    tensor_model_parallel_size=2,
                    pipeline_model_parallel_size=1,
                    data_parallel_size=2,
                    rank_offset=16,
                ),
            }
        )

        mock_grid = MagicMock()
        mock_grid.create_pg = MagicMock(return_value=MagicMock())
        mock_grid_class.return_value = mock_grid

        build_hypercomm_grids(megatron_mimo_config)

        # Check both grids created with different shapes
        assert mock_grid_class.call_count == 2

        # First call (llm): base view [tp, cp, dp, pp] == [8, 1, 1, 2].
        first_call_kwargs = mock_grid_class.call_args_list[0][1]
        assert first_call_kwargs["shape"] == [8, 1, 1, 2]
        assert first_call_kwargs["dim_names"] == ["tp", "cp", "dp", "pp"]
        assert first_call_kwargs["rank_offset"] == 0

        # Second call (encoder): base view [2, 1, 2, 1].
        second_call_kwargs = mock_grid_class.call_args_list[1][1]
        assert second_call_kwargs["shape"] == [2, 1, 2, 1]
        assert second_call_kwargs["dim_names"] == ["tp", "cp", "dp", "pp"]
        assert second_call_kwargs["rank_offset"] == 16

    @patch("megatron.core.hyper_comm_grid.HyperCommGrid")
    def test_build_creates_all_dimension_groups(self, mock_grid_class):
        """Test that all dimension process groups are created."""
        megatron_mimo_config = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(
                    tensor_model_parallel_size=2,
                    context_parallel_size=2,
                    pipeline_model_parallel_size=2,
                    data_parallel_size=2,
                ),
            }
        )

        mock_grid = MagicMock()
        create_pg_calls = []

        def track_create_pg(dims, *, view=None, pg_options=None):
            create_pg_calls.append((dims, view))
            return MagicMock()

        mock_grid.create_pg = track_create_pg
        mock_grid_class.return_value = mock_grid

        build_hypercomm_grids(megatron_mimo_config)

        # Verify all dimension groups created
        assert (["tp"], None) in create_pg_calls
        assert (["cp"], None) in create_pg_calls
        assert (["pp"], None) in create_pg_calls
        assert (["dp"], None) in create_pg_calls
        # Verify composite group created
        assert (["dp", "cp"], None) in create_pg_calls
        assert (["tp", "cp"], None) in create_pg_calls
        assert (["tp", "pp"], None) in create_pg_calls
        assert (["tp", "dp", "cp"], None) in create_pg_calls
        assert (["tp", "cp", "dp", "pp"], None) in create_pg_calls
        assert (["ep"], EXPERT_VIEW_NAME) in create_pg_calls
        assert (["expt_tp"], EXPERT_VIEW_NAME) in create_pg_calls
        assert (["expt_dp"], EXPERT_VIEW_NAME) in create_pg_calls
        assert (["expt_tp", "ep"], EXPERT_VIEW_NAME) in create_pg_calls
        assert (["expt_tp", "ep", "pp"], EXPERT_VIEW_NAME) in create_pg_calls

    @patch("megatron.core.hyper_comm_grid.HyperCommGrid")
    def test_build_creates_every_group_with_its_own_lazy_options(self, mock_grid_class):
        """Every grid group, dense and expert view, is created with a separate lazy NCCL options object."""
        megatron_mimo_config = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(
                    tensor_model_parallel_size=2,
                    pipeline_model_parallel_size=2,
                    data_parallel_size=2,
                ),
            }
        )
        mock_grid = MagicMock()
        mock_grid.create_pg = MagicMock(return_value=MagicMock())
        mock_grid_class.return_value = mock_grid

        with patch.object(megatron_mimo_builder, "_lazy_nccl_pg_options", side_effect=lambda: object()):
            build_hypercomm_grids(megatron_mimo_config)

        options = [call.kwargs["pg_options"] for call in mock_grid.create_pg.call_args_list]
        assert len(options) == 14
        assert all(option is not None for option in options)
        assert len({id(option) for option in options}) == len(options)

    @patch("megatron.core.hyper_comm_grid.HyperCommGrid")
    def test_build_registers_non_default_language_expert_view(self, mock_grid_class):
        """Non-default language EP/ETP is represented as a separate expert view."""
        megatron_mimo_config = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(
                    tensor_model_parallel_size=4,
                    expert_model_parallel_size=2,
                    expert_tensor_parallel_size=2,
                    pipeline_model_parallel_size=2,
                    data_parallel_size=2,
                ),
            }
        )

        mock_grid = MagicMock()
        mock_grid.create_pg = MagicMock(return_value=MagicMock())
        mock_grid_class.return_value = mock_grid

        build_hypercomm_grids(megatron_mimo_config)

        mock_grid_class.assert_called_once()
        assert mock_grid_class.call_args.kwargs["shape"] == [4, 1, 2, 2]
        mock_grid.register_view.assert_called_once_with(
            EXPERT_VIEW_NAME,
            shape=[2, 2, 2, 2],
            dim_names=["expt_tp", "ep", "expt_dp", "pp"],
            shared_dims=["pp"],
        )

    @patch("megatron.core.hyper_comm_grid.HyperCommGrid")
    def test_build_uses_nccl_backend(self, mock_grid_class):
        """Test that grids use nccl backend."""
        megatron_mimo_config = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=2),
            }
        )

        mock_grid = MagicMock()
        mock_grid.create_pg = MagicMock(return_value=MagicMock())
        mock_grid_class.return_value = mock_grid

        build_hypercomm_grids(megatron_mimo_config)

        # Check backend is nccl
        call_kwargs = mock_grid_class.call_args[1]
        assert call_kwargs["backend"] == "nccl"

    @patch("megatron.core.hyper_comm_grid.HyperCommGrid")
    def test_build_with_rank_offsets(self, mock_grid_class):
        """Test that rank_offset is correctly passed to grids."""
        megatron_mimo_config = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(
                    tensor_model_parallel_size=2,
                    data_parallel_size=2,
                    rank_offset=0,
                ),
                "encoder": ModuleParallelismConfig(
                    tensor_model_parallel_size=2,
                    data_parallel_size=2,
                    rank_offset=4,
                ),
            }
        )

        mock_grid = MagicMock()
        mock_grid.create_pg = MagicMock(return_value=MagicMock())
        mock_grid_class.return_value = mock_grid

        build_hypercomm_grids(megatron_mimo_config)

        # Check rank_offsets
        llm_kwargs = mock_grid_class.call_args_list[0][1]
        assert llm_kwargs["rank_offset"] == 0

        encoder_kwargs = mock_grid_class.call_args_list[1][1]
        assert encoder_kwargs["rank_offset"] == 4
