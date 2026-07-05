# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for MegatronMIMO builder utilities."""

from unittest.mock import MagicMock, patch

from megatron.bridge.models.megatron_mimo.megatron_mimo_builder import build_hypercomm_grids
from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)


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

        # Check grid was created with the dense (base) view shape: [tp, cp, dp_base, pp]
        # where dp_base == etp * dp == 1 * 2 == 2.
        mock_grid_class.assert_called_once()
        call_kwargs = mock_grid_class.call_args[1]
        assert call_kwargs["shape"] == [2, 1, 2, 2]  # [tp, cp, dp, pp]
        assert call_kwargs["dim_names"] == ["tp", "cp", "dp", "pp"]
        assert call_kwargs["rank_offset"] == 0
        assert call_kwargs["backend"] == "nccl"

        # Check the expert view was registered.
        mock_grid.register_view.assert_called_once()
        view_args, view_kwargs = mock_grid.register_view.call_args
        assert view_args[0] == "expert"
        assert view_kwargs["dim_names"] == ["expt_tp", "ep", "expt_dp", "pp"]
        assert view_kwargs["shared_dims"] == ["pp"]

        # Check dense process groups were created (base view).
        create_pg_calls = [call[0][0] for call in mock_grid.create_pg.call_args_list]
        assert ["tp"] in create_pg_calls
        assert ["cp"] in create_pg_calls
        assert ["pp"] in create_pg_calls
        assert ["dp"] in create_pg_calls
        assert ["dp", "cp"] in create_pg_calls
        assert ["tp", "cp", "dp", "pp"] in create_pg_calls
        # Check expert process groups were created (expert view).
        assert ["ep"] in create_pg_calls
        assert ["expt_tp"] in create_pg_calls
        assert ["expt_dp"] in create_pg_calls
        assert ["expt_tp", "ep", "pp"] in create_pg_calls

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

        # First call (llm): base view [tp, cp, dp_base, pp] == [8, 1, 1, 2] (dp_base == etp * dp == 1).
        first_call_kwargs = mock_grid_class.call_args_list[0][1]
        assert first_call_kwargs["shape"] == [8, 1, 1, 2]
        assert first_call_kwargs["dim_names"] == ["tp", "cp", "dp", "pp"]
        assert first_call_kwargs["rank_offset"] == 0

        # Second call (encoder): base view [2, 1, 2, 1] (dp_base == etp * dp == 2).
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
                    expert_tensor_parallel_size=2,
                    pipeline_model_parallel_size=2,
                    data_parallel_size=2,
                ),
            }
        )

        mock_grid = MagicMock()
        create_pg_calls = []

        def track_create_pg(dims, view=None):
            create_pg_calls.append(dims)
            return MagicMock()

        mock_grid.create_pg = track_create_pg
        mock_grid_class.return_value = mock_grid

        build_hypercomm_grids(megatron_mimo_config)

        # Verify dense dimension groups created (base view)
        assert ["tp"] in create_pg_calls
        assert ["cp"] in create_pg_calls
        assert ["pp"] in create_pg_calls
        assert ["dp"] in create_pg_calls
        # Verify composite group created
        assert ["dp", "cp"] in create_pg_calls
        # Verify expert dimension groups created (expert view)
        assert ["ep"] in create_pg_calls
        assert ["expt_tp"] in create_pg_calls
        assert ["expt_dp"] in create_pg_calls

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
