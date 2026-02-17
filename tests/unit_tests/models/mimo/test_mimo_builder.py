# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for MIMO builder utilities."""

from unittest.mock import MagicMock, patch

from megatron.bridge.models.mimo.mimo_builder import _default_topology, build_hypercomm_grids
from megatron.bridge.models.mimo.mimo_config import MimoParallelismConfig, ModuleParallelismConfig


class TestBuildHypercommGrids:
    """Test cases for build_hypercomm_grids()."""

    @patch("megatron.core.hyper_comm_grid.HyperCommGrid")
    def test_build_with_single_module(self, mock_grid_class):
        """Test build_hypercomm_grids with single LLM module."""
        mimo_config = MimoParallelismConfig(
            module_parallelisms={
                "llm": ModuleParallelismConfig(
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

        grids = build_hypercomm_grids(mimo_config)

        # Should create one grid
        assert "llm" in grids
        assert grids["llm"] == mock_grid

        # Check grid was created with correct shape
        mock_grid_class.assert_called_once()
        call_kwargs = mock_grid_class.call_args[1]
        assert call_kwargs["shape"] == [2, 1, 1, 2, 2]  # [tp, cp, ep, pp, dp]
        assert call_kwargs["dim_names"] == ["tp", "cp", "ep", "pp", "dp"]
        assert call_kwargs["rank_offset"] == 0
        assert call_kwargs["backend"] == "nccl"

        # Check all process groups were created
        create_pg_calls = [call[0][0] for call in mock_grid.create_pg.call_args_list]
        assert ["tp"] in create_pg_calls
        assert ["cp"] in create_pg_calls
        assert ["ep"] in create_pg_calls
        assert ["pp"] in create_pg_calls
        assert ["dp"] in create_pg_calls
        assert ["dp", "cp"] in create_pg_calls

    @patch("megatron.core.hyper_comm_grid.HyperCommGrid")
    def test_build_with_multiple_modules(self, mock_grid_class):
        """Test build_hypercomm_grids with multiple modules."""
        mimo_config = MimoParallelismConfig(
            module_parallelisms={
                "llm": ModuleParallelismConfig(
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

        grids = build_hypercomm_grids(mimo_config)

        # Should create three grids
        assert "llm" in grids
        assert "clip_encoder" in grids
        assert "dino_encoder" in grids
        assert len(grids) == 3

        # Verify HyperCommGrid was called 3 times
        assert mock_grid_class.call_count == 3

    @patch("megatron.core.hyper_comm_grid.HyperCommGrid")
    def test_build_with_different_parallelism_per_module(self, mock_grid_class):
        """Test grids with different parallelism configs per module."""
        mimo_config = MimoParallelismConfig(
            module_parallelisms={
                "llm": ModuleParallelismConfig(
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

        build_hypercomm_grids(mimo_config)

        # Check both grids created with different shapes
        assert mock_grid_class.call_count == 2

        # First call (llm): shape [8, 1, 1, 2, 1]
        first_call_kwargs = mock_grid_class.call_args_list[0][1]
        assert first_call_kwargs["shape"] == [8, 1, 1, 2, 1]
        assert first_call_kwargs["rank_offset"] == 0

        # Second call (encoder): shape [2, 1, 1, 1, 2]
        second_call_kwargs = mock_grid_class.call_args_list[1][1]
        assert second_call_kwargs["shape"] == [2, 1, 1, 1, 2]
        assert second_call_kwargs["rank_offset"] == 16

    @patch("megatron.core.hyper_comm_grid.HyperCommGrid")
    def test_build_creates_all_dimension_groups(self, mock_grid_class):
        """Test that all dimension process groups are created."""
        mimo_config = MimoParallelismConfig(
            module_parallelisms={
                "llm": ModuleParallelismConfig(
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

        def track_create_pg(dims):
            create_pg_calls.append(dims)
            return MagicMock()

        mock_grid.create_pg = track_create_pg
        mock_grid_class.return_value = mock_grid

        build_hypercomm_grids(mimo_config)

        # Verify all dimension groups created
        assert ["tp"] in create_pg_calls
        assert ["cp"] in create_pg_calls
        assert ["ep"] in create_pg_calls
        assert ["pp"] in create_pg_calls
        assert ["dp"] in create_pg_calls
        # Verify composite group created
        assert ["dp", "cp"] in create_pg_calls

    @patch("megatron.core.hyper_comm_grid.HyperCommGrid")
    def test_build_uses_nccl_backend(self, mock_grid_class):
        """Test that grids use nccl backend."""
        mimo_config = MimoParallelismConfig(
            module_parallelisms={
                "llm": ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=2),
            }
        )

        mock_grid = MagicMock()
        mock_grid.create_pg = MagicMock(return_value=MagicMock())
        mock_grid_class.return_value = mock_grid

        build_hypercomm_grids(mimo_config)

        # Check backend is nccl
        call_kwargs = mock_grid_class.call_args[1]
        assert call_kwargs["backend"] == "nccl"

    @patch("megatron.core.hyper_comm_grid.HyperCommGrid")
    def test_build_with_rank_offsets(self, mock_grid_class):
        """Test that rank_offset is correctly passed to grids."""
        mimo_config = MimoParallelismConfig(
            module_parallelisms={
                "llm": ModuleParallelismConfig(
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

        build_hypercomm_grids(mimo_config)

        # Check rank_offsets
        llm_kwargs = mock_grid_class.call_args_list[0][1]
        assert llm_kwargs["rank_offset"] == 0

        encoder_kwargs = mock_grid_class.call_args_list[1][1]
        assert encoder_kwargs["rank_offset"] == 4


class TestDefaultTopology:
    """Test cases for _default_topology()."""

    def test_topology_with_single_encoder(self):
        """Test topology with LLM and one encoder."""
        mimo_config = MimoParallelismConfig(
            module_parallelisms={
                "llm": ModuleParallelismConfig(tensor_model_parallel_size=2),
                "clip_encoder": ModuleParallelismConfig(tensor_model_parallel_size=2),
            }
        )

        topology = _default_topology(mimo_config)

        # Encoder should point to LLM
        assert topology["clip_encoder"] == ["llm"]
        # LLM should have no downstream
        assert topology["llm"] == []

    def test_topology_with_multiple_encoders(self):
        """Test topology with LLM and multiple encoders."""
        mimo_config = MimoParallelismConfig(
            module_parallelisms={
                "llm": ModuleParallelismConfig(tensor_model_parallel_size=2),
                "clip_encoder": ModuleParallelismConfig(tensor_model_parallel_size=2),
                "dino_encoder": ModuleParallelismConfig(tensor_model_parallel_size=2),
                "audio_encoder": ModuleParallelismConfig(tensor_model_parallel_size=2),
            }
        )

        topology = _default_topology(mimo_config)

        # All encoders should point to LLM
        assert topology["clip_encoder"] == ["llm"]
        assert topology["dino_encoder"] == ["llm"]
        assert topology["audio_encoder"] == ["llm"]
        # LLM should have no downstream
        assert topology["llm"] == []

    def test_topology_with_llm_only(self):
        """Test topology with only LLM module."""
        mimo_config = MimoParallelismConfig(
            module_parallelisms={
                "llm": ModuleParallelismConfig(tensor_model_parallel_size=2),
            }
        )

        topology = _default_topology(mimo_config)

        # LLM should have no downstream
        assert topology["llm"] == []
        # Should only have one entry
        assert len(topology) == 1

    def test_topology_structure(self):
        """Test that topology has correct structure (dict of lists)."""
        mimo_config = MimoParallelismConfig(
            module_parallelisms={
                "llm": ModuleParallelismConfig(tensor_model_parallel_size=2),
                "encoder": ModuleParallelismConfig(tensor_model_parallel_size=2),
            }
        )

        topology = _default_topology(mimo_config)

        # Check it's a dict
        assert isinstance(topology, dict)
        # Check values are lists
        for value in topology.values():
            assert isinstance(value, list)

    def test_topology_all_modules_present(self):
        """Test that all modules appear in topology."""
        mimo_config = MimoParallelismConfig(
            module_parallelisms={
                "llm": ModuleParallelismConfig(tensor_model_parallel_size=2),
                "encoder1": ModuleParallelismConfig(tensor_model_parallel_size=2),
                "encoder2": ModuleParallelismConfig(tensor_model_parallel_size=2),
            }
        )

        topology = _default_topology(mimo_config)

        # All modules should be present in topology
        assert "llm" in topology
        assert "encoder1" in topology
        assert "encoder2" in topology
        assert len(topology) == 3
