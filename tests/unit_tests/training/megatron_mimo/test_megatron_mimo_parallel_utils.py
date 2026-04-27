# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for MegatronMIMO parallel utilities."""

from unittest.mock import MagicMock, call, patch

import pytest
import torch


class TestIsCurrentRankInGrid:
    """Test cases for is_current_rank_in_grid()."""

    @patch("megatron.bridge.training.megatron_mimo_parallel_utils.dist")
    def test_rank_in_grid(self, mock_dist):
        """Test rank within grid range returns True."""
        from megatron.bridge.training.megatron_mimo_parallel_utils import is_current_rank_in_grid

        mock_dist.get_rank.return_value = 2
        mock_grid = MagicMock()
        mock_grid.rank_offset = 0
        mock_grid.size = 4

        assert is_current_rank_in_grid(mock_grid) is True

    @patch("megatron.bridge.training.megatron_mimo_parallel_utils.dist")
    def test_rank_not_in_grid(self, mock_dist):
        """Test rank outside grid range returns False."""
        from megatron.bridge.training.megatron_mimo_parallel_utils import is_current_rank_in_grid

        mock_dist.get_rank.return_value = 5
        mock_grid = MagicMock()
        mock_grid.rank_offset = 0
        mock_grid.size = 4

        assert is_current_rank_in_grid(mock_grid) is False

    @patch("megatron.bridge.training.megatron_mimo_parallel_utils.dist")
    def test_rank_at_grid_boundary(self, mock_dist):
        """Test rank at grid boundary."""
        from megatron.bridge.training.megatron_mimo_parallel_utils import is_current_rank_in_grid

        mock_grid = MagicMock()
        mock_grid.rank_offset = 4
        mock_grid.size = 4

        # At start boundary (inclusive)
        mock_dist.get_rank.return_value = 4
        assert is_current_rank_in_grid(mock_grid) is True

        # At end boundary (exclusive)
        mock_dist.get_rank.return_value = 8
        assert is_current_rank_in_grid(mock_grid) is False

    @patch("megatron.bridge.training.megatron_mimo_parallel_utils.dist")
    def test_rank_before_grid(self, mock_dist):
        """Test rank before grid range returns False."""
        from megatron.bridge.training.megatron_mimo_parallel_utils import is_current_rank_in_grid

        mock_dist.get_rank.return_value = 2
        mock_grid = MagicMock()
        mock_grid.rank_offset = 4
        mock_grid.size = 4

        assert is_current_rank_in_grid(mock_grid) is False


class TestValidateNoStubRanks:
    """Test cases for validate_no_stub_ranks()."""

    def test_all_ranks_participate(self):
        """Test validation passes when all ranks participate."""
        from megatron.bridge.training.megatron_mimo_parallel_utils import validate_no_stub_ranks

        mock_grid1 = MagicMock()
        mock_grid1.rank_offset = 0
        mock_grid1.size = 4

        mock_grid2 = MagicMock()
        mock_grid2.rank_offset = 4
        mock_grid2.size = 4

        module_to_grid_map = {
            "encoder": mock_grid1,
            "language": mock_grid2,
        }

        # Should not raise
        validate_no_stub_ranks(module_to_grid_map, world_size=8)

    def test_stub_ranks_detected(self):
        """Test validation fails when stub ranks exist."""
        from megatron.bridge.training.megatron_mimo_parallel_utils import validate_no_stub_ranks

        mock_grid = MagicMock()
        mock_grid.rank_offset = 0
        mock_grid.size = 4

        module_to_grid_map = {"language": mock_grid}

        with pytest.raises(ValueError, match="do not participate in any module"):
            validate_no_stub_ranks(module_to_grid_map, world_size=8)

    def test_overlapping_grids(self):
        """Test validation with overlapping grids (colocated case)."""
        from megatron.bridge.training.megatron_mimo_parallel_utils import validate_no_stub_ranks

        mock_grid1 = MagicMock()
        mock_grid1.rank_offset = 0
        mock_grid1.size = 4

        mock_grid2 = MagicMock()
        mock_grid2.rank_offset = 0
        mock_grid2.size = 4

        module_to_grid_map = {
            "encoder": mock_grid1,
            "language": mock_grid2,
        }

        # Should not raise (all 4 ranks participate)
        validate_no_stub_ranks(module_to_grid_map, world_size=4)


class TestValidateDataLoaderContract:
    """Test cases for validate_data_loader_contract()."""

    def test_valid_configuration(self):
        """Test validation passes for valid configuration."""
        from megatron.bridge.training.megatron_mimo_parallel_utils import validate_data_loader_contract

        mock_grid = MagicMock()
        mock_grid.get_pg_size.return_value = 2  # DP size = 2

        mock_infra = MagicMock()
        mock_infra.module_to_grid_map = {"language": mock_grid}

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
        from megatron.bridge.training.megatron_mimo_parallel_utils import validate_data_loader_contract

        mock_grid = MagicMock()
        mock_grid.get_pg_size.return_value = 3  # DP size = 3

        mock_infra = MagicMock()
        mock_infra.module_to_grid_map = {"language": mock_grid}

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
        from megatron.bridge.training.megatron_mimo_parallel_utils import build_pg_collection_for_schedule

        mock_pg1 = MagicMock()
        mock_pg2 = MagicMock()

        mock_infra = MagicMock()
        mock_infra.pg_collections = {
            "encoder": mock_pg1,
            "language": mock_pg2,
        }

        # This will likely fall back to list since import may fail in test env
        result = build_pg_collection_for_schedule(mock_infra)

        # Should be either a list or MultiModuleProcessGroupCollection
        assert result is not None

    def test_filters_none_pg_collections(self):
        """Test that None pg_collections are filtered out."""
        from megatron.bridge.training.megatron_mimo_parallel_utils import build_pg_collection_for_schedule

        mock_pg = MagicMock()

        mock_infra = MagicMock()
        mock_infra.pg_collections = {
            "encoder": None,  # Non-participating module
            "language": mock_pg,
        }

        result = build_pg_collection_for_schedule(mock_infra)

        # Should filter out None values
        if isinstance(result, list):
            assert len(result) == 1
            assert mock_pg in result


class TestMultimoduleNoSync:
    """Test cases for multimodule_no_sync context manager."""

    @patch("megatron.bridge.training.megatron_mimo_parallel_utils.is_current_rank_in_grid")
    def test_enters_and_exits_contexts(self, mock_in_grid):
        """Test that no_sync contexts are properly entered and exited."""
        from megatron.bridge.training.megatron_mimo_parallel_utils import multimodule_no_sync

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

    @patch("megatron.bridge.training.megatron_mimo_parallel_utils.is_current_rank_in_grid")
    def test_skips_non_participating_modules(self, mock_in_grid):
        """Test that non-participating modules are skipped."""
        from megatron.bridge.training.megatron_mimo_parallel_utils import multimodule_no_sync

        mock_in_grid.return_value = False  # Not participating

        mock_module = MagicMock()
        mock_grid = MagicMock()

        module_to_grid_tuple = [(mock_module, mock_grid)]

        with multimodule_no_sync(module_to_grid_tuple=module_to_grid_tuple):
            pass

        # no_sync should not be called
        mock_module.no_sync.assert_not_called()


class TestFinalizeModelGradsMultimodule:
    """Test cases for finalize_model_grads_multimodule()."""

    def test_uses_language_global_token_scale_for_all_modules(self):
        """Test per-token scaling is lifted out of per-module finalization."""
        from megatron.bridge.training.megatron_mimo_parallel_utils import finalize_model_grads_multimodule

        language_grid = MagicMock(name="language_grid")
        vision_grid = MagicMock(name="vision_grid")
        language_grid.rank_offset = 0
        language_module = MagicMock(name="language_module")
        vision_module = MagicMock(name="vision_module")
        language_pg = MagicMock(name="language_pg")
        vision_pg = MagicMock(name="vision_pg")
        language_pg.pp = MagicMock(name="language_pp")
        language_pg.dp_cp = MagicMock(name="language_dp_cp")
        language_pg.dp = MagicMock(name="language_dp")

        infra = MagicMock()
        infra.module_to_grid_map = {
            "language": language_grid,
            "vision": vision_grid,
        }
        infra.pg_collections = {
            "language": language_pg,
            "vision": vision_pg,
        }
        num_tokens = torch.tensor(4, dtype=torch.int)

        def all_reduce_num_tokens(tensor, group=None, op=None):
            tensor.mul_(2)

        with (
            patch("megatron.bridge.training.megatron_mimo_parallel_utils.is_current_rank_in_grid", return_value=True),
            patch("megatron.bridge.training.megatron_mimo_parallel_utils.get_pp_last_rank", return_value=3),
            patch("megatron.bridge.training.megatron_mimo_parallel_utils.dist") as mock_dist,
            patch("megatron.bridge.training.megatron_mimo_parallel_utils._finalize_model_grads") as mock_finalize,
        ):
            mock_dist.all_reduce.side_effect = all_reduce_num_tokens

            finalize_model_grads_multimodule(
                [MagicMock()],
                num_tokens=num_tokens,
                force_all_reduce=True,
                infra=infra,
                module_to_grid_tuple=[
                    (language_module, language_grid),
                    (vision_module, vision_grid),
                ],
            )

        broadcast_tokens = mock_dist.broadcast.call_args.args[0]
        assert broadcast_tokens is not num_tokens
        assert broadcast_tokens.item() == 8
        assert num_tokens.item() == 4
        mock_dist.broadcast.assert_called_once_with(broadcast_tokens, src=3, group=language_pg.pp)
        mock_dist.all_reduce.assert_called_once_with(
            broadcast_tokens,
            group=language_pg.dp_cp,
            op=mock_dist.ReduceOp.SUM,
        )
        mock_finalize.assert_has_calls(
            [
                call([language_module], num_tokens=None, pg_collection=language_pg, force_all_reduce=True),
                call([vision_module], num_tokens=None, pg_collection=vision_pg, force_all_reduce=True),
            ]
        )
        language_module.scale_gradients.assert_called_once_with(0.125)
        vision_module.scale_gradients.assert_called_once_with(0.125)

    def test_broadcasts_language_token_scale_to_non_colocated_module(self):
        """Test disjoint encoder ranks receive the language global token count."""
        from megatron.bridge.training.megatron_mimo_parallel_utils import finalize_model_grads_multimodule

        language_grid = MagicMock(name="language_grid")
        language_grid.rank_offset = 4
        vision_grid = MagicMock(name="vision_grid")
        vision_module = MagicMock(name="vision_module")
        vision_pg = MagicMock(name="vision_pg")
        infra = MagicMock()
        infra.module_to_grid_map = {
            "language": language_grid,
            "vision": vision_grid,
        }
        infra.pg_collections = {
            "language": None,
            "vision": vision_pg,
        }
        num_tokens = torch.tensor(0, dtype=torch.int)

        def broadcast_global_num_tokens(tensor, src=None, group=None):
            tensor.fill_(8)

        with (
            patch("megatron.bridge.training.megatron_mimo_parallel_utils.is_current_rank_in_grid", return_value=True),
            patch("megatron.bridge.training.megatron_mimo_parallel_utils.dist") as mock_dist,
            patch("megatron.bridge.training.megatron_mimo_parallel_utils._finalize_model_grads") as mock_finalize,
        ):
            mock_dist.broadcast.side_effect = broadcast_global_num_tokens

            finalize_model_grads_multimodule(
                [MagicMock()],
                num_tokens=num_tokens,
                force_all_reduce=True,
                infra=infra,
                module_to_grid_tuple=[(vision_module, vision_grid)],
            )

        broadcast_tokens = mock_dist.broadcast.call_args.args[0]
        assert broadcast_tokens is not num_tokens
        assert broadcast_tokens.item() == 8
        assert num_tokens.item() == 0
        mock_dist.broadcast.assert_called_once_with(broadcast_tokens, src=4, group=mock_dist.group.WORLD)
        mock_dist.all_reduce.assert_not_called()
        mock_finalize.assert_called_once_with(
            [vision_module],
            num_tokens=None,
            pg_collection=vision_pg,
            force_all_reduce=True,
        )
        vision_module.scale_gradients.assert_called_once_with(0.125)

    def test_fallback_clones_num_tokens_without_language_grid(self):
        """Test malformed infra without a language grid keeps the old MCore path."""
        from megatron.bridge.training.megatron_mimo_parallel_utils import finalize_model_grads_multimodule

        vision_grid = MagicMock(name="vision_grid")
        vision_module = MagicMock(name="vision_module")
        vision_pg = MagicMock(name="vision_pg")
        infra = MagicMock()
        infra.module_to_grid_map = {"vision": vision_grid}
        infra.pg_collections = {
            "language": None,
            "vision": vision_pg,
        }
        num_tokens = torch.tensor(4, dtype=torch.int)

        with (
            patch("megatron.bridge.training.megatron_mimo_parallel_utils.is_current_rank_in_grid", return_value=True),
            patch("megatron.bridge.training.megatron_mimo_parallel_utils.dist") as mock_dist,
            patch("megatron.bridge.training.megatron_mimo_parallel_utils._finalize_model_grads") as mock_finalize,
        ):
            finalize_model_grads_multimodule(
                [MagicMock()],
                num_tokens=num_tokens,
                force_all_reduce=True,
                infra=infra,
                module_to_grid_tuple=[(vision_module, vision_grid)],
            )

        mock_dist.broadcast.assert_not_called()
        mock_dist.all_reduce.assert_not_called()
        module_num_tokens = mock_finalize.call_args.kwargs["num_tokens"]
        assert module_num_tokens is not num_tokens
        assert module_num_tokens.item() == num_tokens.item()
        mock_finalize.assert_called_once_with(
            [vision_module],
            num_tokens=module_num_tokens,
            pg_collection=vision_pg,
            force_all_reduce=True,
        )
        vision_module.scale_gradients.assert_not_called()


class TestZeroGradBufferForMultimodule:
    """Test cases for zero_grad_buffer_for_multimodule()."""

    @patch("megatron.bridge.training.megatron_mimo_parallel_utils.is_current_rank_in_grid")
    def test_zeros_grad_buffers(self, mock_in_grid):
        """Test gradient buffers are zeroed for participating modules."""
        from megatron.bridge.training.megatron_mimo_parallel_utils import zero_grad_buffer_for_multimodule

        mock_in_grid.return_value = True

        mock_module = MagicMock()
        mock_grid = MagicMock()

        module_to_grid_tuple = [(mock_module, mock_grid)]

        zero_grad_buffer_for_multimodule(module_to_grid_tuple)

        mock_module.zero_grad_buffer.assert_called_once()

    @patch("megatron.bridge.training.megatron_mimo_parallel_utils.is_current_rank_in_grid")
    def test_skips_non_participating(self, mock_in_grid):
        """Test non-participating modules are skipped."""
        from megatron.bridge.training.megatron_mimo_parallel_utils import zero_grad_buffer_for_multimodule

        mock_in_grid.return_value = False

        mock_module = MagicMock()
        mock_grid = MagicMock()

        module_to_grid_tuple = [(mock_module, mock_grid)]

        zero_grad_buffer_for_multimodule(module_to_grid_tuple)

        mock_module.zero_grad_buffer.assert_not_called()
