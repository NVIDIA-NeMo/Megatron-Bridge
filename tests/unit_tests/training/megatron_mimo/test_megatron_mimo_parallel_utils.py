# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for MegatronMIMO parallel utilities."""

from types import SimpleNamespace
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

    def test_language_cp_local_token_count_reduces_over_language_dp_cp(self):
        """CP-local token counts are reduced over language DP x CP before scaling."""
        from megatron.bridge.training.megatron_mimo_parallel_utils import finalize_model_grads_multimodule
        from megatron.bridge.training.megatron_mimo_step import loss_func

        full_sequence_loss_mask = torch.tensor([[1, 1, 0, 1, 0, 1, 1, 1]], dtype=torch.float)
        cp_local_loss_mask = full_sequence_loss_mask[:, :4]
        cp_local_losses = torch.ones_like(cp_local_loss_mask)
        _loss, cp_local_num_tokens, _metrics = loss_func(cp_local_loss_mask, cp_local_losses)
        assert cp_local_num_tokens.item() == 3

        language_dp_size = 2
        expected_global_tokens = int(full_sequence_loss_mask.sum().item()) * language_dp_size

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
        observed_pre_reduce_tokens = []

        def all_reduce_language_dp_cp(tensor, group=None, op=None):
            observed_pre_reduce_tokens.append(tensor.item())
            tensor.fill_(expected_global_tokens)

        with (
            patch("megatron.bridge.training.megatron_mimo_parallel_utils.is_current_rank_in_grid", return_value=True),
            patch("megatron.bridge.training.megatron_mimo_parallel_utils.get_pp_last_rank", return_value=0),
            patch("megatron.bridge.training.megatron_mimo_parallel_utils.dist") as mock_dist,
            patch("megatron.bridge.training.megatron_mimo_parallel_utils._finalize_model_grads") as mock_finalize,
            patch(
                "megatron.bridge.training.megatron_mimo_parallel_utils.get_model_config",
                return_value=SimpleNamespace(calculate_per_token_loss=True),
            ),
        ):
            mock_dist.all_reduce.side_effect = all_reduce_language_dp_cp

            finalize_model_grads_multimodule(
                [MagicMock()],
                num_tokens=cp_local_num_tokens,
                force_all_reduce=True,
                infra=infra,
                module_to_grid_tuple=[
                    (language_module, language_grid),
                    (vision_module, vision_grid),
                ],
            )

        assert observed_pre_reduce_tokens == [3]
        assert cp_local_num_tokens.item() == 3
        broadcast_tokens = mock_dist.broadcast.call_args.args[0]
        assert broadcast_tokens is not cp_local_num_tokens
        assert broadcast_tokens.item() == expected_global_tokens
        mock_dist.broadcast.assert_called_once_with(broadcast_tokens, src=0, group=language_pg.pp)
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
        expected_grad_scale = 1.0 / expected_global_tokens
        language_module.scale_gradients.assert_called_once_with(expected_grad_scale)
        vision_module.scale_gradients.assert_called_once_with(expected_grad_scale)

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


class TestPerTokenLossPrecondition:
    """Per-token-loss precondition guard inside ``finalize_model_grads_multimodule``.

    With ``num_tokens != None`` the finalizer applies a uniform
    ``1/global_num_tokens`` scale to every module. That math only matches
    mcore's DDP behavior when each module's TransformerConfig has
    ``calculate_per_token_loss=True`` (which pins DDP's
    ``gradient_scaling_factor`` to 1.0). With the flag off, mcore's DDP
    applies its own per-microbatch averaging, and our SUM denominator on top
    mis-scales gradients. The precondition fails fast with a clear error.
    """

    @staticmethod
    def _make_module_with_config(*, calculate_per_token_loss: bool) -> MagicMock:
        """Mock module whose ``get_model_config()`` returns a config with the flag."""
        module = MagicMock()
        cfg = MagicMock()
        cfg.calculate_per_token_loss = calculate_per_token_loss
        module._test_cfg = cfg  # so the patched get_model_config can route per module
        return module

    @staticmethod
    def _patched_get_model_config(module):
        return module._test_cfg

    def _make_infra(self, language_grid, vision_grid):
        infra = MagicMock()
        infra.module_to_grid_map = {"language": language_grid, "vision": vision_grid}
        infra.pg_collections = {"language": MagicMock(), "vision": MagicMock()}
        infra.pg_collections["language"].pp = MagicMock()
        infra.pg_collections["language"].dp_cp = MagicMock()
        infra.pg_collections["language"].dp = MagicMock()
        return infra

    def test_no_op_when_num_tokens_is_none(self):
        """Per-token-loss check skipped when num_tokens isn't provided —
        configs without the flag are still valid for the SUM-only path."""
        from megatron.bridge.training.megatron_mimo_parallel_utils import finalize_model_grads_multimodule

        language_grid = MagicMock(rank_offset=0)
        vision_grid = MagicMock(rank_offset=0)
        infra = self._make_infra(language_grid, vision_grid)
        # Module has flag OFF — would trigger the guard if num_tokens were set.
        language_module = self._make_module_with_config(calculate_per_token_loss=False)

        with (
            patch("megatron.bridge.training.megatron_mimo_parallel_utils.is_current_rank_in_grid", return_value=True),
            patch("megatron.bridge.training.megatron_mimo_parallel_utils.dist"),
            patch("megatron.bridge.training.megatron_mimo_parallel_utils._finalize_model_grads"),
            patch(
                "megatron.bridge.training.megatron_mimo_parallel_utils.get_model_config",
                side_effect=self._patched_get_model_config,
            ),
        ):
            finalize_model_grads_multimodule(
                [MagicMock()],
                num_tokens=None,
                infra=infra,
                module_to_grid_tuple=[(language_module, language_grid)],
            )
        # No exception raised — None bypasses the precondition.

    def test_raises_when_language_module_lacks_per_token_loss(self):
        """Language module with calculate_per_token_loss=False → raise naming language."""
        from megatron.bridge.training.megatron_mimo_parallel_utils import finalize_model_grads_multimodule

        language_grid = MagicMock(rank_offset=0)
        vision_grid = MagicMock(rank_offset=0)
        infra = self._make_infra(language_grid, vision_grid)
        language_module = self._make_module_with_config(calculate_per_token_loss=False)
        vision_module = self._make_module_with_config(calculate_per_token_loss=True)

        with (
            patch("megatron.bridge.training.megatron_mimo_parallel_utils.is_current_rank_in_grid", return_value=True),
            patch("megatron.bridge.training.megatron_mimo_parallel_utils.dist"),
            patch("megatron.bridge.training.megatron_mimo_parallel_utils._finalize_model_grads"),
            patch(
                "megatron.bridge.training.megatron_mimo_parallel_utils.get_model_config",
                side_effect=self._patched_get_model_config,
            ),
        ):
            with pytest.raises(ValueError, match=r"calculate_per_token_loss=True.*\['language'\]"):
                finalize_model_grads_multimodule(
                    [MagicMock()],
                    num_tokens=torch.tensor(4, dtype=torch.int),
                    infra=infra,
                    module_to_grid_tuple=[
                        (language_module, language_grid),
                        (vision_module, vision_grid),
                    ],
                )

    def test_raises_when_modality_module_lacks_per_token_loss(self):
        """Encoder module with calculate_per_token_loss=False → raise naming the encoder.

        Asymmetric DP between language and a modality is correct only when both
        modules use SUM-DDP — naming the encoder explicitly tells the user
        which TransformerConfig to fix.
        """
        from megatron.bridge.training.megatron_mimo_parallel_utils import finalize_model_grads_multimodule

        language_grid = MagicMock(rank_offset=0)
        vision_grid = MagicMock(rank_offset=0)
        infra = self._make_infra(language_grid, vision_grid)
        language_module = self._make_module_with_config(calculate_per_token_loss=True)
        vision_module = self._make_module_with_config(calculate_per_token_loss=False)

        with (
            patch("megatron.bridge.training.megatron_mimo_parallel_utils.is_current_rank_in_grid", return_value=True),
            patch("megatron.bridge.training.megatron_mimo_parallel_utils.dist"),
            patch("megatron.bridge.training.megatron_mimo_parallel_utils._finalize_model_grads"),
            patch(
                "megatron.bridge.training.megatron_mimo_parallel_utils.get_model_config",
                side_effect=self._patched_get_model_config,
            ),
        ):
            with pytest.raises(ValueError, match=r"calculate_per_token_loss=True.*\['vision'\]"):
                finalize_model_grads_multimodule(
                    [MagicMock()],
                    num_tokens=torch.tensor(4, dtype=torch.int),
                    infra=infra,
                    module_to_grid_tuple=[
                        (language_module, language_grid),
                        (vision_module, vision_grid),
                    ],
                )

    def test_lists_all_offending_modules(self):
        """Multiple offenders → message lists all of them, not just the first."""
        from megatron.bridge.training.megatron_mimo_parallel_utils import finalize_model_grads_multimodule

        language_grid = MagicMock(rank_offset=0)
        vision_grid = MagicMock(rank_offset=0)
        infra = self._make_infra(language_grid, vision_grid)
        language_module = self._make_module_with_config(calculate_per_token_loss=False)
        vision_module = self._make_module_with_config(calculate_per_token_loss=False)

        with (
            patch("megatron.bridge.training.megatron_mimo_parallel_utils.is_current_rank_in_grid", return_value=True),
            patch("megatron.bridge.training.megatron_mimo_parallel_utils.dist"),
            patch("megatron.bridge.training.megatron_mimo_parallel_utils._finalize_model_grads"),
            patch(
                "megatron.bridge.training.megatron_mimo_parallel_utils.get_model_config",
                side_effect=self._patched_get_model_config,
            ),
        ):
            with pytest.raises(ValueError) as exc_info:
                finalize_model_grads_multimodule(
                    [MagicMock()],
                    num_tokens=torch.tensor(4, dtype=torch.int),
                    infra=infra,
                    module_to_grid_tuple=[
                        (language_module, language_grid),
                        (vision_module, vision_grid),
                    ],
                )
        msg = str(exc_info.value)
        assert "language" in msg
        assert "vision" in msg

    def test_skips_non_participating_modules(self):
        """A module the rank doesn't participate in (is_current_rank_in_grid → False)
        is excluded from the precondition check even if its config is malformed."""
        from megatron.bridge.training.megatron_mimo_parallel_utils import finalize_model_grads_multimodule

        language_grid = MagicMock(rank_offset=0)
        vision_grid = MagicMock(rank_offset=4)  # not active on this rank
        infra = self._make_infra(language_grid, vision_grid)
        language_module = self._make_module_with_config(calculate_per_token_loss=True)
        vision_module = self._make_module_with_config(calculate_per_token_loss=False)

        # Only the language grid is "active" here.
        def _in_grid(grid):
            return grid is language_grid

        with (
            patch(
                "megatron.bridge.training.megatron_mimo_parallel_utils.is_current_rank_in_grid",
                side_effect=_in_grid,
            ),
            patch("megatron.bridge.training.megatron_mimo_parallel_utils.dist"),
            patch("megatron.bridge.training.megatron_mimo_parallel_utils.get_pp_last_rank", return_value=0),
            patch("megatron.bridge.training.megatron_mimo_parallel_utils._finalize_model_grads"),
            patch(
                "megatron.bridge.training.megatron_mimo_parallel_utils.get_model_config",
                side_effect=self._patched_get_model_config,
            ),
        ):
            # Vision config has flag off, but vision isn't on this rank — no raise.
            finalize_model_grads_multimodule(
                [MagicMock()],
                num_tokens=torch.tensor(4, dtype=torch.int),
                infra=infra,
                module_to_grid_tuple=[
                    (language_module, language_grid),
                    (vision_module, vision_grid),
                ],
            )


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
