# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for MegatronMIMO parallel utilities."""

import types
from unittest.mock import MagicMock, patch

import pytest
import torch


MODULE = "megatron.bridge.training.megatron_mimo_parallel_utils"


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

        # global_batch=8, dp=2, microbatches=2, global micro_batch_size=4.
        # Each module-local DP rank sees 4 / 2 = 2 samples per microbatch.
        validate_data_loader_contract(
            infra=mock_infra,
            global_batch_size=8,
            micro_batch_size=4,
            num_microbatches=2,
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
                global_batch_size=8,
                micro_batch_size=4,
                num_microbatches=2,
            )

    def test_microbatch_count_mismatch(self):
        """Test validation fails when accumulation does not match global batch."""
        from megatron.bridge.training.megatron_mimo_parallel_utils import validate_data_loader_contract

        mock_grid = MagicMock()
        mock_grid.get_pg_size.return_value = 2

        mock_infra = MagicMock()
        mock_infra.module_to_grid_map = {"language": mock_grid}

        with pytest.raises(ValueError, match="Microbatch mismatch"):
            validate_data_loader_contract(
                infra=mock_infra,
                global_batch_size=16,
                micro_batch_size=4,
                num_microbatches=2,
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


def _build_two_module_setup(llm_dp, encoder_dp, *, llm_rank_offset=4, llm_size=4):
    """Build an encoder + language MIMO setup for finalize_model_grads tests.

    The same grid objects are shared between ``module_to_grid_map`` and
    ``module_to_grid_tuple`` because the function matches modules to grids by
    identity (``mg is grid``).
    """
    import megatron.bridge.training.megatron_mimo_parallel_utils as mpu

    llm_key = mpu.MIMO_LANGUAGE_MODULE_KEY
    encoder_key = "encoder"

    llm_grid = MagicMock(name="llm_grid")
    llm_grid.rank_offset = llm_rank_offset
    llm_grid.size = llm_size
    encoder_grid = MagicMock(name="encoder_grid")

    llm_module = MagicMock(name="llm_module")
    encoder_module = MagicMock(name="encoder_module")
    llm_pg = MagicMock(name="llm_pg")
    encoder_pg = MagicMock(name="encoder_pg")

    infra = MagicMock()
    infra.module_to_grid_map = {encoder_key: encoder_grid, llm_key: llm_grid}
    infra.pg_collections = {encoder_key: encoder_pg, llm_key: llm_pg}

    # Encoder listed first to verify per-module routing is independent of order.
    module_to_grid_tuple = [(encoder_module, encoder_grid), (llm_module, llm_grid)]

    dp_by_grid = {id(llm_grid): llm_dp, id(encoder_grid): encoder_dp}

    return types.SimpleNamespace(
        infra=infra,
        module_to_grid_tuple=module_to_grid_tuple,
        llm_grid=llm_grid,
        encoder_grid=encoder_grid,
        llm_module=llm_module,
        encoder_module=encoder_module,
        llm_pg=llm_pg,
        encoder_pg=encoder_pg,
        dp_by_grid=dp_by_grid,
    )


class TestFinalizeModelGradsMultimodule:
    """Test cases for finalize_model_grads_multimodule().

    The function selects its gradient-normalization branch on
    ``num_tokens is not None``, which is exactly ``calculate_per_token_loss``:
    Megatron-Core's schedule passes ``total_num_tokens if
    config.calculate_per_token_loss else None`` to ``finalize_model_grads_func``.
    These tests pin that equivalence so the branch keying stays correct.
    """

    @patch(f"{MODULE}.dist")
    @patch(f"{MODULE}.is_current_rank_in_grid")
    @patch(f"{MODULE}._get_dp_size_from_grid")
    @patch(f"{MODULE}._finalize_model_grads")
    def test_per_token_loss_path(self, mock_finalize, mock_dp, mock_in_grid, mock_dist):
        """num_tokens is not None (calculate_per_token_loss=True) path.

        Only the LLM receives num_tokens (so MCore can PP-broadcast + DP-all-reduce
        the per-rank counts); encoder grads are normalized manually by 1/total with
        no DP compensation factor.
        """
        from megatron.bridge.training.megatron_mimo_parallel_utils import finalize_model_grads_multimodule

        s = _build_two_module_setup(llm_dp=2, encoder_dp=4)
        mock_in_grid.return_value = True
        mock_dp.side_effect = lambda grid: s.dp_by_grid[id(grid)]

        num_tokens = torch.tensor(100)

        finalize_model_grads_multimodule(
            [MagicMock()],  # model arg is ignored
            num_tokens,
            force_all_reduce=True,
            infra=s.infra,
            module_to_grid_tuple=s.module_to_grid_tuple,
        )

        # Phase 1: LLM finalized with num_tokens, encoder with num_tokens=None.
        finalize_by_module = {call.args[0][0]: call.kwargs for call in mock_finalize.call_args_list}
        assert finalize_by_module[s.llm_module]["num_tokens"] is num_tokens
        assert finalize_by_module[s.llm_module]["pg_collection"] is s.llm_pg
        assert finalize_by_module[s.llm_module]["force_all_reduce"] is True
        assert finalize_by_module[s.encoder_module]["num_tokens"] is None
        assert finalize_by_module[s.encoder_module]["pg_collection"] is s.encoder_pg
        assert finalize_by_module[s.encoder_module]["force_all_reduce"] is True

        # Phase 2: broadcast the global total from the LLM's last rank (4 + 4 - 1).
        mock_dist.broadcast.assert_called_once()
        assert mock_dist.broadcast.call_args.args[0] is num_tokens
        assert mock_dist.broadcast.call_args.kwargs["src"] == 7

        # Phase 3: encoder scaled by 1/total only; LLM already normalized by DDP.
        s.encoder_module.scale_gradients.assert_called_once_with(1.0 / 100)
        s.llm_module.scale_gradients.assert_not_called()

    @patch(f"{MODULE}.dist")
    @patch(f"{MODULE}.is_current_rank_in_grid")
    @patch(f"{MODULE}._get_dp_size_from_grid")
    @patch(f"{MODULE}._finalize_model_grads")
    def test_non_per_token_loss_path_applies_dp_compensation(self, mock_finalize, mock_dp, mock_in_grid, mock_dist):
        """num_tokens is None (calculate_per_token_loss=False) path.

        Every module is finalized with num_tokens=None, no broadcast happens, and
        modules whose DP differs from the LLM's are rescaled by module_dp/llm_dp.
        """
        from megatron.bridge.training.megatron_mimo_parallel_utils import finalize_model_grads_multimodule

        s = _build_two_module_setup(llm_dp=2, encoder_dp=4)
        mock_in_grid.return_value = True
        mock_dp.side_effect = lambda grid: s.dp_by_grid[id(grid)]

        finalize_model_grads_multimodule(
            [MagicMock()],
            None,
            force_all_reduce=True,
            infra=s.infra,
            module_to_grid_tuple=s.module_to_grid_tuple,
        )

        # All modules finalized without num_tokens (DDP does a plain mean).
        for call in mock_finalize.call_args_list:
            assert call.kwargs["num_tokens"] is None
            assert call.kwargs["force_all_reduce"] is True
        assert mock_dist.broadcast.call_count == 0

        # encoder_dp (4) != llm_dp (2) -> scale by 4/2; LLM matches llm_dp -> no scale.
        s.encoder_module.scale_gradients.assert_called_once_with(2.0)
        s.llm_module.scale_gradients.assert_not_called()

    @patch(f"{MODULE}.dist")
    @patch(f"{MODULE}.is_current_rank_in_grid")
    @patch(f"{MODULE}._get_dp_size_from_grid")
    @patch(f"{MODULE}._finalize_model_grads")
    def test_per_token_loss_path_skips_scaling_when_zero_tokens(self, mock_finalize, mock_dp, mock_in_grid, mock_dist):
        """Zero global tokens must not trigger a divide-by-zero in encoder scaling."""
        from megatron.bridge.training.megatron_mimo_parallel_utils import finalize_model_grads_multimodule

        s = _build_two_module_setup(llm_dp=2, encoder_dp=4)
        mock_in_grid.return_value = True
        mock_dp.side_effect = lambda grid: s.dp_by_grid[id(grid)]

        finalize_model_grads_multimodule(
            [MagicMock()],
            torch.tensor(0),
            infra=s.infra,
            module_to_grid_tuple=s.module_to_grid_tuple,
        )

        s.encoder_module.scale_gradients.assert_not_called()
        s.llm_module.scale_gradients.assert_not_called()
