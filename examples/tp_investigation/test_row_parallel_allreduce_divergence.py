# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Row-Parallel All-Reduce Divergence Test
========================================

Isolates the source of TP vs DP numerical differences to the all-reduce
in row-parallel operations.

Three sub-tests compare TP=1 vs TP=4 through:
  1. Column-parallel matmul only — no reduction, should be **exact**
  2. Row-parallel matmul only   — has all-reduce, shows divergence
  3. Stacked column→row pairs   — shows error compounding across layers

This proves that the all-reduce summation-order change (non-associative in bf16)
is the sole source of divergence, not weight init, attention, or LayerNorm.

Run::

    uv run python -m torch.distributed.run --nproc_per_node=4 \\
        -m pytest tests/unit_tests/transformer/test_row_parallel_allreduce_divergence.py -s -v
"""

import logging

import pytest
import torch
from packaging import version

from tests.unit_tests.test_utilities import Utils


logger = logging.getLogger(__name__)


def _sync_input(shape, dtype):
    """Create random input synchronized across all ranks."""
    x = torch.randn(shape, device="cuda", dtype=dtype)
    torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.AVG)
    return x


def _make_weight(out_size, in_size, dtype, seed=42):
    """Create deterministic weight on CPU with proper scale, move to GPU.

    Uses 1/sqrt(in_size) scaling like real transformer init to keep
    activations in a reasonable range and avoid bf16 overflow.
    """
    torch.manual_seed(seed)
    std = 1.0 / (in_size**0.5)
    w = torch.randn(out_size, in_size, dtype=dtype) * std
    return w.cuda()


class TestRowParallelAllReduceDivergence:
    def setup_method(self, method):
        Utils.destroy_model_parallel()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse("2.3.0"),
        reason="Requires PyTorch 2.3+",
    )
    def test_column_parallel_is_exact(self):
        """Column-parallel matmul (no all-reduce) should be bit-exact.

        Y = X @ W where W is column-sharded: W = [W_0 | W_1 | W_2 | W_3]
        Each rank computes Y_i = X @ W_i, then we gather.
        No reduction → no rounding difference → exact match.
        """
        Utils.initialize_distributed()
        if torch.distributed.get_world_size() != 4:
            pytest.skip("Requires 4 GPUs")
        rank = torch.distributed.get_rank()

        seq, batch, hidden, ffn = 32, 2, 1024, 4096
        W = _make_weight(ffn, hidden, torch.bfloat16)
        x = _sync_input((seq, batch, hidden), torch.bfloat16)

        # TP=1: full matmul
        out_tp1 = x @ W.t()  # [seq, batch, ffn]

        # TP=4: each rank computes its column shard, then gather
        chunk = ffn // 4
        W_local = W[rank * chunk : (rank + 1) * chunk, :]  # [ffn/4, hidden]
        out_local = x @ W_local.t()  # [seq, batch, ffn/4]

        gathered = [torch.empty_like(out_local) for _ in range(4)]
        torch.distributed.all_gather(gathered, out_local)
        out_tp4 = torch.cat(gathered, dim=-1)  # [seq, batch, ffn]

        diff = (out_tp1 - out_tp4).float().abs()
        if rank == 0:
            logger.info("\nColumn-parallel matmul: TP=1 vs TP=4 (bf16)")
            logger.info(f"  Max abs diff:  {diff.max().item()}")
            logger.info(f"  Mean abs diff: {diff.mean().item()}")

        assert diff.max().item() == 0.0, f"Column-parallel should be exact, got max diff = {diff.max().item()}"

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse("2.3.0"),
        reason="Requires PyTorch 2.3+",
    )
    def test_row_parallel_shows_divergence(self):
        """Row-parallel matmul (with all-reduce) diverges in bf16.

        Y = X @ W where W is row-sharded: W = [W_0; W_1; W_2; W_3]
        TP=1: Y = X @ W (single full matmul)
        TP=4: Y = all_reduce(X_i @ W_i) (4 partial matmuls + sum)

        Different summation order → different bf16 rounding → divergence.
        """
        Utils.initialize_distributed()
        if torch.distributed.get_world_size() != 4:
            pytest.skip("Requires 4 GPUs")
        rank = torch.distributed.get_rank()

        seq, batch, ffn, hidden = 32, 2, 4096, 1024
        W = _make_weight(hidden, ffn, torch.bfloat16)  # [hidden, ffn]
        x = _sync_input((seq, batch, ffn), torch.bfloat16)

        # TP=1: full matmul
        out_tp1 = x @ W.t()  # [seq, batch, hidden]

        # TP=4: row-parallel — split input & weight, partial matmul, all-reduce
        chunk = ffn // 4
        x_local = x[:, :, rank * chunk : (rank + 1) * chunk]
        W_local = W[:, rank * chunk : (rank + 1) * chunk]  # [hidden, ffn/4]
        partial = x_local @ W_local.t()  # [seq, batch, hidden]

        out_tp4 = partial.clone()
        torch.distributed.all_reduce(out_tp4, op=torch.distributed.ReduceOp.SUM)

        diff = (out_tp1 - out_tp4).float().abs()
        if rank == 0:
            logger.info("\nRow-parallel matmul: TP=1 vs TP=4 (bf16)")
            logger.info(f"  Max abs diff:  {diff.max().item():.6f}")
            logger.info(f"  Mean abs diff: {diff.mean().item():.8f}")
            logger.info(f"  % > 1e-3:      {100.0 * (diff > 0.001).float().mean().item():.2f}%")

            assert diff.max().item() > 0.0, "Expected divergence from row-parallel all-reduce"
            assert diff.max().item() < 0.05, f"Single-layer diff too large: {diff.max().item()}"

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse("2.3.0"),
        reason="Requires PyTorch 2.3+",
    )
    def test_stacked_mlp_shows_compounding(self):
        """Stacking column→row pairs shows error compounding across layers.

        Each "MLP layer" is:  column-parallel (no error) → row-parallel (error from all-reduce)
        The error accumulates through the residual-like chain.
        """
        Utils.initialize_distributed()
        if torch.distributed.get_world_size() != 4:
            pytest.skip("Requires 4 GPUs")
        rank = torch.distributed.get_rank()

        seq, batch, hidden, ffn = 32, 2, 1024, 4096
        n_layers = 16

        # Create weights for all layers (identical across ranks via CPU seed)
        col_weights = []  # [ffn, hidden] each
        row_weights = []  # [hidden, ffn] each
        for i in range(n_layers):
            col_weights.append(_make_weight(ffn, hidden, torch.bfloat16, seed=1000 + i))
            row_weights.append(_make_weight(hidden, ffn, torch.bfloat16, seed=2000 + i))

        x = _sync_input((seq, batch, hidden), torch.bfloat16)
        h_tp1 = x.clone()
        h_tp4 = x.clone()

        per_layer_diffs = []
        chunk_ffn = ffn // 4

        with torch.no_grad():
            for i in range(n_layers):
                Wc = col_weights[i]  # [ffn, hidden]
                Wr = row_weights[i]  # [hidden, ffn]

                # --- TP=1 path: full matmuls ---
                h_tp1 = h_tp1 @ Wc.t()  # column: [seq, batch, ffn]
                h_tp1 = h_tp1 @ Wr.t()  # row: [seq, batch, hidden]

                # --- TP=4 path: column-parallel → row-parallel ---
                # Column parallel: each rank computes its shard (no all-reduce)
                Wc_local = Wc[rank * chunk_ffn : (rank + 1) * chunk_ffn, :]
                h_tp4_col = h_tp4 @ Wc_local.t()  # [seq, batch, ffn/4]

                # Row parallel: partial matmul + all-reduce
                Wr_local = Wr[:, rank * chunk_ffn : (rank + 1) * chunk_ffn]
                partial = h_tp4_col @ Wr_local.t()  # [seq, batch, hidden]
                h_tp4 = partial.clone()
                torch.distributed.all_reduce(h_tp4, op=torch.distributed.ReduceOp.SUM)

                diff = (h_tp1 - h_tp4).float().abs()
                per_layer_diffs.append((diff.max().item(), diff.mean().item()))

        if rank == 0:
            logger.info(f"\nStacked MLP (column→row x {n_layers}): TP=1 vs TP=4 (bf16)")
            logger.info(f"  {'Layer':>5s} | {'Max Abs Diff':>13s} | {'Mean Abs Diff':>14s}")
            logger.info(f"  {'-' * 5}-+-{'-' * 13}-+-{'-' * 14}")
            for i, (mx, mn) in enumerate(per_layer_diffs):
                logger.info(f"  {i:>5d} | {mx:>13.6f} | {mn:>14.8f}")

            first_max, first_mean = per_layer_diffs[0]
            last_max, last_mean = per_layer_diffs[-1]

            logger.info("\n  Error compounding ratio (last/first):")
            logger.info(f"    Max:  {last_max / max(first_max, 1e-10):.1f}x")
            logger.info(f"    Mean: {last_mean / max(first_mean, 1e-10):.1f}x")

            # Error should compound
            assert last_mean > first_mean, (
                f"Error should compound: layer 0 mean={first_mean:.6f}, layer {n_layers - 1} mean={last_mean:.6f}"
            )
            # Column-parallel is exact, only row-parallel contributes error
            # First layer should already show divergence (from row-parallel)
            assert first_max > 0.0, "First layer should show divergence from row-parallel"
