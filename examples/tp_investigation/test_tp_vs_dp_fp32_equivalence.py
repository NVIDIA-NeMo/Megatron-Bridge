# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging

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

# to run the test
# uv run python -m torch.distributed.run --nproc_per_node=2  -m pytest tests/unit_tests/transformer/test_tp_vs_dp_equivalence.py


class TestTensorParallelVsDataParallel:
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
        reason="Device mesh feature requires PyTorch 2.3 or later",
    )
    def test_tp_vs_dp_output_equivalence(self):
        """
        Test that transformer block outputs match between:
        - Data Parallel (DP=2) with full model on both GPUs
        - Tensor Parallel (TP=2) with sharded model across GPUs

        This verifies that tensor parallel correctly shards model weights and
        produces the same result as data parallel with the full model.
        """
        # Initialize distributed backend
        Utils.initialize_distributed()

        # Check world size after initialization
        world_size = 2
        actual_world_size = torch.distributed.get_world_size()
        if actual_world_size != world_size:
            pytest.skip(f"Test requires world_size={world_size}, but got {actual_world_size}")

        # Create transformer configuration
        sequence_length = 4096
        micro_batch_size = 2

        num_layers = 1
        # Shared config parameters
        base_config_kwargs = dict(
            num_layers=num_layers,
            hidden_size=1024,
            num_attention_heads=16,
            use_cpu_initialization=True,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            bf16=False,
            no_rope_freq=[1] * num_layers,
        )

        # ===== SCENARIO A: Data Parallel (DP=2, TP=1) =====
        # Create process groups for DP=2
        dp_grid = HyperCommGrid([1, 1, 1, 1, 2], ["tp", "cp", "ep", "pp", "dp"])

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
        dp_block = TransformerBlock(
            dp_transformer_config,
            get_gpt_layer_with_transformer_engine_spec(),
            pg_collection=dp_pg_collection,
        ).cuda()

        # ===== SCENARIO B: Tensor Parallel (TP=2, DP=1) =====
        # Create process groups for TP=2
        tp_grid = HyperCommGrid([2, 1, 1, 1, 1], ["tp", "cp", "ep", "pp", "dp"])

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

        # Create TP transformer block
        tp_transformer_config = TransformerConfig(
            **base_config_kwargs,
            tensor_model_parallel_size=2,
            context_parallel_size=1,
            sequence_parallel=False,
        )

        # Reset CPU RNG seed so master weights are identical to DP model;
        # _initialize_affine_weight_cpu creates the full master weight then slices for TP rank
        torch.manual_seed(12345)
        tp_block = TransformerBlock(
            tp_transformer_config,
            get_gpt_layer_with_transformer_engine_spec(),
            pg_collection=tp_pg_collection,
        ).cuda()

        # ===== Create test input =====
        # Create the same input on all ranks
        hidden_states = torch.randn(
            (sequence_length, micro_batch_size, dp_transformer_config.hidden_size),
            device="cuda",
            dtype=torch.float32,
        )

        # Synchronize input across all ranks
        torch.distributed.all_reduce(hidden_states, op=torch.distributed.ReduceOp.AVG)

        logger.info(f"\n[Rank {torch.distributed.get_rank()}] Input hidden_states shape: {hidden_states.shape}")
        logger.info(
            f"[Rank {torch.distributed.get_rank()}] Input sum: {hidden_states.sum():.6f}, mean: {hidden_states.float().mean():.6f}"
        )

        # Clone for each scenario
        dp_hidden_states = hidden_states.clone().detach()
        tp_hidden_states = hidden_states.clone().detach()

        # ===== SCENARIO A: Forward pass with DP (full model, full sequence) =====
        with torch.no_grad():
            dp_output = dp_block(hidden_states=dp_hidden_states, attention_mask=None)
            logger.info(
                f"[Rank {torch.distributed.get_rank()}] DP output shape: {dp_output.shape}, sum: {dp_output.sum():.6f}"
            )

        # ===== SCENARIO B: Forward pass with TP (sharded model, full sequence) =====
        with torch.no_grad():
            tp_output = tp_block(hidden_states=tp_hidden_states, attention_mask=None)
            logger.info(
                f"[Rank {torch.distributed.get_rank()}] TP output shape: {tp_output.shape}, sum: {tp_output.sum():.6f}"
            )

        # ===== Compare outputs =====
        logger.info(f"\n{'=' * 80}")
        logger.info(f"[Rank {torch.distributed.get_rank()}] OUTPUT COMPARISON")
        logger.info(f"{'=' * 80}")
        logger.info(f"[Rank {torch.distributed.get_rank()}] DP output shape: {dp_output.shape}")
        logger.info(f"[Rank {torch.distributed.get_rank()}] TP output shape: {tp_output.shape}")
        logger.info(
            f"[Rank {torch.distributed.get_rank()}] DP output mean: {dp_output.float().mean():.6f}, std: {dp_output.float().std():.6f}"
        )
        logger.info(
            f"[Rank {torch.distributed.get_rank()}] TP output mean: {tp_output.float().mean():.6f}, std: {tp_output.float().std():.6f}"
        )

        # Compute differences
        diff = (dp_output - tp_output).float()
        abs_diff = diff.abs()

        if torch.distributed.get_rank() == 0:
            logger.info(f"\n[Rank {torch.distributed.get_rank()}] DIFFERENCE STATISTICS:")
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

            logger.info(f"\n[Rank {torch.distributed.get_rank()}] LOCATION OF MAX DIFFERENCE:")
            logger.info(f"  Sequence position: {seq_idx}/{sequence_length}")
            logger.info(f"  Batch index: {batch_idx}/{micro_batch_size}")
            logger.info(f"  Hidden dimension: {hidden_idx}/{dp_transformer_config.hidden_size}")
            logger.info(f"  DP value: {dp_output[seq_idx, batch_idx, hidden_idx]:.6f}")
            logger.info(f"  TP value: {tp_output[seq_idx, batch_idx, hidden_idx]:.6f}")
            logger.info(f"  Difference: {diff[seq_idx, batch_idx, hidden_idx]:.6f}")

            # Sample values from different sequence positions
            logger.info(f"\n[Rank {torch.distributed.get_rank()}] SAMPLE VALUES AT DIFFERENT POSITIONS:")
            sample_positions = [
                0,
                sequence_length // 4,
                sequence_length // 2,
                3 * sequence_length // 4,
                sequence_length - 1,
            ]
            for pos in sample_positions:
                logger.info(f"\n  Position {pos} (batch 0, first 5 hidden dims):")
                logger.info(f"    DP:   {[f'{dp_output[pos, 0, i]:.4f}' for i in range(5)]}")
                logger.info(f"    TP:   {[f'{tp_output[pos, 0, i]:.4f}' for i in range(5)]}")
                logger.info(f"    Diff: {[f'{diff[pos, 0, i]:.4f}' for i in range(5)]}")

            # Check percentage of elements exceeding different thresholds
            thresholds = [0.001, 0.005, 0.01, 0.02, 0.03]
            logger.info(f"\n[Rank {torch.distributed.get_rank()}] PERCENTAGE OF ELEMENTS EXCEEDING THRESHOLDS:")
            total_elements = abs_diff.numel()
            for thresh in thresholds:
                count = (abs_diff > thresh).sum().item()
                percentage = 100.0 * count / total_elements
                logger.info(f"  > {thresh:.3f}: {count}/{total_elements} ({percentage:.2f}%)")

            # Relative error
            rel_error = abs_diff / (dp_output.float().abs() + 1e-8)
            logger.info(f"\n[Rank {torch.distributed.get_rank()}] RELATIVE ERROR STATISTICS:")
            logger.info(f"  Max relative error: {rel_error.max():.6f}")
            logger.info(f"  Mean relative error: {rel_error.mean():.6f}")
            logger.info(f"  Median relative error: {rel_error.median():.6f}")

            logger.info(f"{'=' * 80}\n")

            torch.testing.assert_close(
                dp_output,
                tp_output,
                rtol=1e-3,
                atol=1e-3,
                msg="Outputs don't match between Data Parallel (full model) and Tensor Parallel (sharded model)",
            )

            logger.info(f"[Rank {torch.distributed.get_rank()}] Test passed! DP and TP outputs match.")
