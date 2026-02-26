# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
TP vs DP Output Equivalence Test (bf16)
=======================================

Verifies that tensor-parallel (TP) sharding produces numerically close outputs
to data-parallel (DP) for the same model weights and inputs.

Weight initialization
---------------------
With ``use_cpu_initialization=True`` and the same ``torch.manual_seed``, every
TP configuration gets bit-identical weights.  ``_initialize_affine_weight_cpu``
always creates the **full** master weight on CPU, calls ``init_method`` on it,
then slices the relevant partition for each TP rank.  LayerNorm params are
deterministic (gamma=1, beta=0).  This has been verified empirically: after
all-gathering TP-sharded weights, max diff across all 26 parameters is 0.0.

Expected bf16 numerical differences
------------------------------------
Even with identical weights, TP introduces differences because the
row-parallel linear layers (attention output projection, MLP fc2) split the
reduction dimension across ranks and use ``all_reduce`` to sum partial results.
This changes the floating-point summation order, which is non-associative in
bf16.  The error compounds through the residual stream across layers:

    Layer 0:  max_abs_diff ~0.03,  mean_abs_diff ~0.0001
    Layer 9:  max_abs_diff ~0.06,  mean_abs_diff ~0.002
    Layer 27: max_abs_diff ~0.11,  mean_abs_diff ~0.005

Higher TP degree produces slightly larger error (more all-reduce splits).
These magnitudes are consistent with bf16 precision (~3 significant decimal
digits) accumulated over 28 layers.

The fp32 version of this test (``test_tp_vs_dp_fp32_equivalence.py``) passes
with ``atol=1e-3`` because fp32 has ~7 significant digits.

Environment setup
-----------------
* ``NVTE_FLASH_ATTN=0``, ``NVTE_FUSED_ATTN=0``, ``NVTE_UNFUSED_ATTN=1``:
  Force TE's unfused attention backend for reproducibility.
  Must be set **after** ``Utils.initialize_distributed()`` which pops them.
* ``NVTE_ALLOW_NONDETERMINISTIC_ALGO=0``: Deterministic cuDNN attention path.
* ``CUBLAS_WORKSPACE_CONFIG=:4096:8``: Deterministic cuBLAS.
* ``deterministic_mode=True`` in ``TransformerConfig``.

Run command::

    uv run python -m torch.distributed.run --nproc_per_node=4 \\
        -m pytest tests/unit_tests/transformer/test_tp_vs_dp_equivalence.py
"""

import logging
import os

import pytest
import torch
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from packaging import version

from tests.unit_tests.test_utilities import Utils


logger = logging.getLogger(__name__)


class TestTensorParallelVsDataParallel:
    def setup_method(self, method):
        Utils.destroy_model_parallel()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    @staticmethod
    def _set_deterministic_env():
        """Set TE env vars for deterministic unfused attention.

        Must be called AFTER Utils.initialize_distributed() which pops
        NVTE_FLASH_ATTN, NVTE_FUSED_ATTN, and NVTE_UNFUSED_ATTN.
        """
        os.environ["NVTE_FLASH_ATTN"] = "0"
        os.environ["NVTE_FUSED_ATTN"] = "0"
        os.environ["NVTE_UNFUSED_ATTN"] = "1"
        os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse("2.3.0"),
        reason="Device mesh feature requires PyTorch 2.3 or later",
    )
    def test_tp_vs_dp_output_equivalence(self):
        """
        Test that transformer block outputs match between:
        - Scenario A: Data Parallel (DP=4, TP=1) with full model on all GPUs
        - Scenario B: Hybrid (TP=2, DP=2) with model sharded across 2 GPUs, replicated across 2
        - Scenario C: Tensor Parallel (TP=4, DP=1) with model sharded across all GPUs

        This verifies that tensor parallel correctly shards model weights and
        produces numerically close results to data parallel with the full model.
        Differences are expected due to bf16 non-associativity in row-parallel
        all-reduce operations, compounded across layers.
        """
        # Initialize distributed backend
        Utils.initialize_distributed()
        self._set_deterministic_env()

        # Check world size after initialization
        world_size = 4
        actual_world_size = torch.distributed.get_world_size()
        if actual_world_size != world_size:
            pytest.skip(f"Test requires world_size={world_size}, but got {actual_world_size}")

        # Create transformer configuration
        sequence_length = 4096
        micro_batch_size = 2

        # Shared config parameters
        base_config_kwargs = dict(
            num_layers=28,
            hidden_size=1024,
            num_attention_heads=16,
            use_cpu_initialization=True,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            bf16=True,
            deterministic_mode=True,
            no_rope_freq=[1] * 28,
        )

        # ===== SCENARIO A: Data Parallel (DP=4, TP=1) =====
        # Create process groups for DP=4
        dp_grid = HyperCommGrid([1, 1, 1, 1, 4], ["tp", "cp", "ep", "pp", "dp"])

        dp_tp_group = dp_grid.create_pg("tp")
        dp_cp_group = dp_grid.create_pg("cp")
        dp_ep_group = dp_grid.create_pg("ep")
        dp_pp_group = dp_grid.create_pg("pp")
        dp_group = dp_grid.create_pg("dp")

        dp_pg_collection = ProcessGroupCollection(
            tp=dp_tp_group, cp=dp_cp_group, ep=dp_ep_group, pp=dp_pp_group, dp=dp_group
        )
        dp_cp_combined = dp_grid.create_pg(["dp", "cp"])
        dp_pg_collection.dp_cp = dp_cp_combined

        # Create DP transformer block
        dp_transformer_config = TransformerConfig(
            **base_config_kwargs,
            tensor_model_parallel_size=1,
            context_parallel_size=1,
        )

        # Set CPU RNG seed before model creation for deterministic weight initialization
        torch.manual_seed(12345)
        dp_block = (
            TransformerBlock(
                dp_transformer_config,
                get_gpt_layer_with_transformer_engine_spec(),
                pg_collection=dp_pg_collection,
            )
            .cuda()
            .bfloat16()
        )

        # ===== SCENARIO B: Hybrid (TP=2, DP=2) =====
        # Create process groups for TP=2, DP=2
        hybrid_grid = HyperCommGrid([2, 1, 1, 1, 2], ["tp", "cp", "ep", "pp", "dp"])

        hybrid_tp_group = hybrid_grid.create_pg("tp")
        hybrid_cp_group = hybrid_grid.create_pg("cp")
        hybrid_ep_group = hybrid_grid.create_pg("ep")
        hybrid_pp_group = hybrid_grid.create_pg("pp")
        hybrid_dp_group = hybrid_grid.create_pg("dp")

        hybrid_pg_collection = ProcessGroupCollection(
            tp=hybrid_tp_group, cp=hybrid_cp_group, ep=hybrid_ep_group, pp=hybrid_pp_group, dp=hybrid_dp_group
        )
        hybrid_dp_cp_combined = hybrid_grid.create_pg(["dp", "cp"])
        hybrid_pg_collection.dp_cp = hybrid_dp_cp_combined

        # Create hybrid transformer block
        hybrid_transformer_config = TransformerConfig(
            **base_config_kwargs,
            tensor_model_parallel_size=2,
            context_parallel_size=1,
        )

        # Reset CPU RNG seed so master weights are identical to DP model;
        # _initialize_affine_weight_cpu creates the full master weight then slices for TP rank
        torch.manual_seed(12345)
        hybrid_block = (
            TransformerBlock(
                hybrid_transformer_config,
                get_gpt_layer_with_transformer_engine_spec(),
                pg_collection=hybrid_pg_collection,
            )
            .cuda()
            .bfloat16()
        )

        # ===== SCENARIO C: Tensor Parallel (TP=4, DP=1) =====
        # Create process groups for TP=4
        tp_grid = HyperCommGrid([4, 1, 1, 1, 1], ["tp", "cp", "ep", "pp", "dp"])

        tp_tp_group = tp_grid.create_pg("tp")
        tp_cp_group = tp_grid.create_pg("cp")
        tp_ep_group = tp_grid.create_pg("ep")
        tp_pp_group = tp_grid.create_pg("pp")
        tp_dp_group = tp_grid.create_pg("dp")

        tp_pg_collection = ProcessGroupCollection(
            tp=tp_tp_group, cp=tp_cp_group, ep=tp_ep_group, pp=tp_pp_group, dp=tp_dp_group
        )
        tp_dp_cp_combined = tp_grid.create_pg(["dp", "cp"])
        tp_pg_collection.dp_cp = tp_dp_cp_combined

        # Create TP=4 transformer block
        tp_transformer_config = TransformerConfig(
            **base_config_kwargs,
            tensor_model_parallel_size=4,
            context_parallel_size=1,
        )

        # Reset CPU RNG seed so master weights are identical to DP model;
        # _initialize_affine_weight_cpu creates the full master weight then slices for TP rank
        torch.manual_seed(12345)
        tp_block = (
            TransformerBlock(
                tp_transformer_config,
                get_gpt_layer_with_transformer_engine_spec(),
                pg_collection=tp_pg_collection,
            )
            .cuda()
            .bfloat16()
        )

        # ===== Create test input =====
        # Create the same input on all ranks
        hidden_states = torch.randn(
            (sequence_length, micro_batch_size, dp_transformer_config.hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
        )

        # Synchronize input across all ranks
        torch.distributed.all_reduce(hidden_states, op=torch.distributed.ReduceOp.AVG)

        logger.info(f"\n[Rank {torch.distributed.get_rank()}] Input hidden_states shape: {hidden_states.shape}")
        logger.info(
            f"[Rank {torch.distributed.get_rank()}] Input sum: {hidden_states.sum():.6f}, mean: {hidden_states.float().mean():.6f}"
        )

        # Clone for each scenario
        dp_hidden_states = hidden_states.clone().detach()
        hybrid_hidden_states = hidden_states.clone().detach()
        tp_hidden_states = hidden_states.clone().detach()

        # ===== SCENARIO A: Forward pass with DP (full model, full sequence) =====
        with torch.no_grad():
            dp_output = dp_block(hidden_states=dp_hidden_states, attention_mask=None)
            logger.info(
                f"[Rank {torch.distributed.get_rank()}] DP output shape: {dp_output.shape}, sum: {dp_output.sum():.6f}"
            )

        # ===== SCENARIO B: Forward pass with Hybrid TP=2,DP=2 =====
        with torch.no_grad():
            hybrid_output = hybrid_block(hidden_states=hybrid_hidden_states, attention_mask=None)
            logger.info(
                f"[Rank {torch.distributed.get_rank()}] Hybrid output shape: {hybrid_output.shape}, sum: {hybrid_output.sum():.6f}"
            )

        # ===== SCENARIO C: Forward pass with TP=4 (sharded model, full sequence) =====
        with torch.no_grad():
            tp_output = tp_block(hidden_states=tp_hidden_states, attention_mask=None)
            logger.info(
                f"[Rank {torch.distributed.get_rank()}] TP output shape: {tp_output.shape}, sum: {tp_output.sum():.6f}"
            )

        # ===== Compare outputs =====
        # bf16 tolerances: row-parallel all-reduce changes summation order, causing
        # non-associative rounding that compounds across 28 layers.
        # Empirically measured on H100 (sm90) with unfused attention:
        #   Layer  0: max_abs ~0.03, mean_abs ~0.0001
        #   Layer 27: max_abs ~0.11, mean_abs ~0.005
        # atol=0.15 provides headroom above the ~0.11 max observed.
        bf16_atol = 0.15
        bf16_rtol = 0.0  # absolute tolerance is sufficient for bf16

        rank = torch.distributed.get_rank()
        comparisons = [
            ("A (DP=4)", "B (TP=2,DP=2)", dp_output, hybrid_output),
            ("A (DP=4)", "C (TP=4)", dp_output, tp_output),
            ("B (TP=2,DP=2)", "C (TP=4)", hybrid_output, tp_output),
        ]

        for name_a, name_b, output_a, output_b in comparisons:
            logger.info(f"\n{'=' * 80}")
            logger.info(f"[Rank {rank}] OUTPUT COMPARISON: {name_a} vs {name_b}")
            logger.info(f"{'=' * 80}")
            logger.info(f"[Rank {rank}] {name_a} output shape: {output_a.shape}")
            logger.info(f"[Rank {rank}] {name_b} output shape: {output_b.shape}")
            logger.info(
                f"[Rank {rank}] {name_a} output mean: {output_a.float().mean():.6f}, std: {output_a.float().std():.6f}"
            )
            logger.info(
                f"[Rank {rank}] {name_b} output mean: {output_b.float().mean():.6f}, std: {output_b.float().std():.6f}"
            )

            # Compute differences
            diff = (output_a - output_b).float()
            abs_diff = diff.abs()

            if rank == 0:
                logger.info(f"\n[Rank {rank}] DIFFERENCE STATISTICS ({name_a} vs {name_b}):")
                logger.info(f"  Max absolute difference: {abs_diff.max():.6f}")
                logger.info(f"  Mean absolute difference: {abs_diff.mean():.6f}")
                logger.info(f"  Median absolute difference: {abs_diff.median():.6f}")
                logger.info(f"  Min absolute difference: {abs_diff.min():.6f}")
                logger.info(f"  Std of absolute difference: {abs_diff.std():.6f}")

                # Find locations of largest differences
                flat_abs_diff = abs_diff.view(-1)
                max_diff_idx = flat_abs_diff.argmax()
                seq_idx = max_diff_idx // (micro_batch_size * dp_transformer_config.hidden_size)
                batch_idx = (
                    max_diff_idx % (micro_batch_size * dp_transformer_config.hidden_size)
                ) // dp_transformer_config.hidden_size
                hidden_idx = max_diff_idx % dp_transformer_config.hidden_size

                logger.info(f"\n[Rank {rank}] LOCATION OF MAX DIFFERENCE ({name_a} vs {name_b}):")
                logger.info(f"  Sequence position: {seq_idx}/{sequence_length}")
                logger.info(f"  Batch index: {batch_idx}/{micro_batch_size}")
                logger.info(f"  Hidden dimension: {hidden_idx}/{dp_transformer_config.hidden_size}")
                logger.info(f"  {name_a} value: {output_a[seq_idx, batch_idx, hidden_idx]:.6f}")
                logger.info(f"  {name_b} value: {output_b[seq_idx, batch_idx, hidden_idx]:.6f}")
                logger.info(f"  Difference: {diff[seq_idx, batch_idx, hidden_idx]:.6f}")

                # Sample values from different sequence positions
                logger.info(f"\n[Rank {rank}] SAMPLE VALUES AT DIFFERENT POSITIONS ({name_a} vs {name_b}):")
                sample_positions = [
                    0,
                    sequence_length // 4,
                    sequence_length // 2,
                    3 * sequence_length // 4,
                    sequence_length - 1,
                ]
                for pos in sample_positions:
                    logger.info(f"\n  Position {pos} (batch 0, first 5 hidden dims):")
                    logger.info(f"    {name_a}: {[f'{output_a[pos, 0, i]:.4f}' for i in range(5)]}")
                    logger.info(f"    {name_b}: {[f'{output_b[pos, 0, i]:.4f}' for i in range(5)]}")
                    logger.info(f"    Diff:     {[f'{diff[pos, 0, i]:.4f}' for i in range(5)]}")

                # Check percentage of elements exceeding different thresholds
                thresholds = [0.001, 0.005, 0.01, 0.02, 0.03]
                logger.info(f"\n[Rank {rank}] PERCENTAGE OF ELEMENTS EXCEEDING THRESHOLDS ({name_a} vs {name_b}):")
                total_elements = abs_diff.numel()
                for thresh in thresholds:
                    count = (abs_diff > thresh).sum().item()
                    percentage = 100.0 * count / total_elements
                    logger.info(f"  > {thresh:.3f}: {count}/{total_elements} ({percentage:.2f}%)")

                # Relative error
                rel_error = abs_diff / (output_a.float().abs() + 1e-8)
                logger.info(f"\n[Rank {rank}] RELATIVE ERROR STATISTICS ({name_a} vs {name_b}):")
                logger.info(f"  Max relative error: {rel_error.max():.6f}")
                logger.info(f"  Mean relative error: {rel_error.mean():.6f}")
                logger.info(f"  Median relative error: {rel_error.median():.6f}")

                logger.info(f"{'=' * 80}\n")

                torch.testing.assert_close(
                    output_a,
                    output_b,
                    rtol=bf16_rtol,
                    atol=bf16_atol,
                    msg=f"Outputs don't match between {name_a} and {name_b}",
                )

                logger.info(f"[Rank {rank}] {name_a} vs {name_b} passed!")

        if rank == 0:
            logger.info(f"\n[Rank {rank}] All comparisons passed! DP, Hybrid, and TP outputs match.")
