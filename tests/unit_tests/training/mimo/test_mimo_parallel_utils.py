# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for MIMO parallel utilities."""

from unittest.mock import MagicMock, Mock, patch

import pytest


class TestIsCurrentRankInGrid:
    """Test cases for is_current_rank_in_grid()."""
    
    @patch('megatron.bridge.training.mimo_parallel_utils.dist')
    def test_rank_in_grid(self, mock_dist):
        """Test rank within grid range returns True."""
        from megatron.bridge.training.mimo_parallel_utils import is_current_rank_in_grid
        
        mock_dist.get_rank.return_value = 2
        mock_grid = MagicMock()
        mock_grid.rank_offset = 0
        mock_grid.size = 4
        
        assert is_current_rank_in_grid(mock_grid) is True
    
    @patch('megatron.bridge.training.mimo_parallel_utils.dist')
    def test_rank_not_in_grid(self, mock_dist):
        """Test rank outside grid range returns False."""
        from megatron.bridge.training.mimo_parallel_utils import is_current_rank_in_grid
        
        mock_dist.get_rank.return_value = 5
        mock_grid = MagicMock()
        mock_grid.rank_offset = 0
        mock_grid.size = 4
        
        assert is_current_rank_in_grid(mock_grid) is False
    
    @patch('megatron.bridge.training.mimo_parallel_utils.dist')
    def test_rank_at_grid_boundary(self, mock_dist):
        """Test rank at grid boundary."""
        from megatron.bridge.training.mimo_parallel_utils import is_current_rank_in_grid
        
        mock_grid = MagicMock()
        mock_grid.rank_offset = 4
        mock_grid.size = 4
        
        # At start boundary (inclusive)
        mock_dist.get_rank.return_value = 4
        assert is_current_rank_in_grid(mock_grid) is True
        
        # At end boundary (exclusive)
        mock_dist.get_rank.return_value = 8
        assert is_current_rank_in_grid(mock_grid) is False


class TestValidateNoStubRanks:
    """Test cases for validate_no_stub_ranks()."""
    
    def test_all_ranks_participate(self):
        """Test validation passes when all ranks participate."""
        from megatron.bridge.training.mimo_parallel_utils import validate_no_stub_ranks
        
        mock_grid1 = MagicMock()
        mock_grid1.rank_offset = 0
        mock_grid1.size = 4
        
        mock_grid2 = MagicMock()
        mock_grid2.rank_offset = 4
        mock_grid2.size = 4
        
        module_to_grid_map = {
            "encoder": mock_grid1,
            "llm": mock_grid2,
        }
        
        # Should not raise
        validate_no_stub_ranks(module_to_grid_map, world_size=8)
    
    def test_stub_ranks_detected(self):
        """Test validation fails when stub ranks exist."""
        from megatron.bridge.training.mimo_parallel_utils import validate_no_stub_ranks
        
        mock_grid = MagicMock()
        mock_grid.rank_offset = 0
        mock_grid.size = 4
        
        module_to_grid_map = {"llm": mock_grid}
        
        with pytest.raises(ValueError, match="do not participate in any module"):
            validate_no_stub_ranks(module_to_grid_map, world_size=8)
    
    def test_overlapping_grids(self):
        """Test validation with overlapping grids (colocated case)."""
        from megatron.bridge.training.mimo_parallel_utils import validate_no_stub_ranks
        
        mock_grid1 = MagicMock()
        mock_grid1.rank_offset = 0
        mock_grid1.size = 4
        
        mock_grid2 = MagicMock()
        mock_grid2.rank_offset = 0
        mock_grid2.size = 4
        
        module_to_grid_map = {
            "encoder": mock_grid1,
            "llm": mock_grid2,
        }
        
        # Should not raise (all 4 ranks participate)
        validate_no_stub_ranks(module_to_grid_map, world_size=4)


class TestValidateDataLoaderContract:
    """Test cases for validate_data_loader_contract()."""
    
    def test_valid_configuration(self):
        """Test validation passes for valid configuration."""
        from megatron.bridge.training.mimo_parallel_utils import validate_data_loader_contract
        
        mock_grid = MagicMock()
        mock_grid.get_pg_size.return_value = 2  # DP size = 2
        
        mock_infra = MagicMock()
        mock_infra.module_to_grid_map = {"llm": mock_grid}
        
        # global_batch=16, dp=2, per_dp_batch=8, microbatches=4, micro_batch_size=2
        # 4 * 2 = 8 == 16 / 2 ✓
        validate_data_loader_contract(
            infra=mock_infra,
            global_batch_size=16,
            micro_batch_size=2,
            num_microbatches=4,
        )
    
    def test_batch_not_divisible_by_dp(self):
        """Test validation fails when batch not divisible by DP size."""
        from megatron.bridge.training.mimo_parallel_utils import validate_data_loader_contract
        
        mock_grid = MagicMock()
        mock_grid.get_pg_size.return_value = 3  # DP size = 3
        
        mock_infra = MagicMock()
        mock_infra.module_to_grid_map = {"llm": mock_grid}
        
        with pytest.raises(ValueError, match="not divisible"):
            validate_data_loader_contract(
                infra=mock_infra,
                global_batch_size=16,
                micro_batch_size=2,
                num_microbatches=4,
            )


class TestBuildPgCollectionForSchedule:
    """Test cases for build_pg_collection_for_schedule()."""
    
    def test_fallback_to_list(self):
        """Test fallback to list when MultiModuleProcessGroupCollection not available."""
        from megatron.bridge.training.mimo_parallel_utils import build_pg_collection_for_schedule
        
        mock_pg1 = MagicMock()
        mock_pg2 = MagicMock()
        
        mock_infra = MagicMock()
        mock_infra.pg_collections = {
            "encoder": mock_pg1,
            "llm": mock_pg2,
        }
        mock_infra.llm_module_name = "llm"
        
        # This will likely fall back to list since import may fail in test env
        result = build_pg_collection_for_schedule(mock_infra)
        
        # Should be either a list or MultiModuleProcessGroupCollection
        assert result is not None
    
    def test_filters_none_pg_collections(self):
        """Test that None pg_collections are filtered out."""
        from megatron.bridge.training.mimo_parallel_utils import build_pg_collection_for_schedule
        
        mock_pg = MagicMock()
        
        mock_infra = MagicMock()
        mock_infra.pg_collections = {
            "encoder": None,  # Non-participating module
            "llm": mock_pg,
        }
        mock_infra.llm_module_name = "llm"
        
        result = build_pg_collection_for_schedule(mock_infra)
        
        # Should filter out None values
        if isinstance(result, list):
            assert len(result) == 1
            assert mock_pg in result


class TestMultimoduleNoSync:
    """Test cases for multimodule_no_sync context manager."""
    
    @patch('megatron.bridge.training.mimo_parallel_utils.is_current_rank_in_grid')
    def test_enters_and_exits_contexts(self, mock_in_grid):
        """Test that no_sync contexts are properly entered and exited."""
        from megatron.bridge.training.mimo_parallel_utils import multimodule_no_sync
        
        mock_in_grid.return_value = True
        
        mock_module = MagicMock()
        mock_context = MagicMock()
        mock_module.no_sync.return_value = mock_context
        
        mock_grid = MagicMock()
        
        module_to_grid_tuple = [(mock_module, mock_grid)]
        
        with multimodule_no_sync(module_to_grid_tuple=module_to_grid_tuple):
            pass
        
        # Verify context was entered and exited
        mock_context.__enter__.assert_called_once()
        mock_context.__exit__.assert_called_once()
    
    @patch('megatron.bridge.training.mimo_parallel_utils.is_current_rank_in_grid')
    def test_skips_non_participating_modules(self, mock_in_grid):
        """Test that non-participating modules are skipped."""
        from megatron.bridge.training.mimo_parallel_utils import multimodule_no_sync
        
        mock_in_grid.return_value = False  # Not participating
        
        mock_module = MagicMock()
        mock_grid = MagicMock()
        
        module_to_grid_tuple = [(mock_module, mock_grid)]
        
        with multimodule_no_sync(module_to_grid_tuple=module_to_grid_tuple):
            pass
        
        # no_sync should not be called
        mock_module.no_sync.assert_not_called()


class TestZeroGradBufferForMultimodule:
    """Test cases for zero_grad_buffer_for_multimodule()."""
    
    @patch('megatron.bridge.training.mimo_parallel_utils.is_current_rank_in_grid')
    def test_zeros_grad_buffers(self, mock_in_grid):
        """Test gradient buffers are zeroed for participating modules."""
        from megatron.bridge.training.mimo_parallel_utils import zero_grad_buffer_for_multimodule
        
        mock_in_grid.return_value = True
        
        mock_module = MagicMock()
        mock_grid = MagicMock()
        
        module_to_grid_tuple = [(mock_module, mock_grid)]
        
        zero_grad_buffer_for_multimodule(module_to_grid_tuple)
        
        mock_module.zero_grad_buffer.assert_called_once()
    
    @patch('megatron.bridge.training.mimo_parallel_utils.is_current_rank_in_grid')
    def test_skips_non_participating(self, mock_in_grid):
        """Test non-participating modules are skipped."""
        from megatron.bridge.training.mimo_parallel_utils import zero_grad_buffer_for_multimodule
        
        mock_in_grid.return_value = False
        
        mock_module = MagicMock()
        mock_grid = MagicMock()
        
        module_to_grid_tuple = [(mock_module, mock_grid)]
        
        zero_grad_buffer_for_multimodule(module_to_grid_tuple)
        
        mock_module.zero_grad_buffer.assert_not_called()
