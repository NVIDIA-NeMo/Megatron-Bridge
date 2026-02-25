#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
==============================================================================
Example: Qwen3 Pretraining with Decentralized Process Groups (Advanced/Manual)
==============================================================================

This example demonstrates how to MANUALLY create process groups using
HyperCommGrid and ProcessGroupCollection for distributed training.

Instead of relying on the automatic setup in `pretrain()`, this example shows
the explicit steps to:
  1. Initialize torch.distributed
  2. Create HyperCommGrid with desired topology
  3. Create all required process groups from the grid
  4. Build ProcessGroupCollection
  5. Pass pg_collection explicitly to model, optimizer, and training loop

This gives you full control over the parallelism topology.

For a simpler approach that uses a recipe with automatic pg_collection creation,
see `pretrain_qwen3_simple.py`.

How to Run
----------
# 8 GPUs: TP2 x PP2 x DP2
uv run python -m torch.distributed.run --nproc_per_node=8 examples/decentralized_pg/pretrain_qwen3_with_decentralized_pg.py

# 4 GPUs: TP2 x PP2 x DP1
uv run python -m torch.distributed.run --nproc_per_node=4 examples/decentralized_pg/pretrain_qwen3_with_decentralized_pg.py \
    --tp-size 2 --pp-size 2

# 2 GPUs: TP2 x PP1 x DP1
uv run python -m torch.distributed.run --nproc_per_node=2 examples/decentralized_pg/pretrain_qwen3_with_decentralized_pg.py \
    --tp-size 2 --pp-size 1
"""

import argparse
import os
import tempfile

import torch
import torch.distributed

# ==============================================================================
# Core Megatron imports for manual process group creation
# ==============================================================================
from megatron.core import parallel_state, tensor_parallel
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.num_microbatches_calculator import init_num_microbatches_calculator
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_pg_rank

# ==============================================================================
# Megatron-Bridge imports
# ==============================================================================
from megatron.bridge.data.loaders import setup_data_iterators
from megatron.bridge.data.utils import get_dataset_provider
from megatron.bridge.models.qwen import Qwen3ModelProvider4B
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    DistributedInitConfig,
    LoggerConfig,
    MockGPTDatasetConfig,
    OptimizerConfig,
    RNGConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.optim import setup_optimizer
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from megatron.bridge.training.train import train
from megatron.bridge.utils.common_utils import get_rank_safe, print_rank_0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Qwen3 Pretraining with Manual Decentralized Process Groups")

    # Parallelism settings
    parser.add_argument("--tp-size", type=int, default=2, help="Tensor parallel size (default: 2)")
    parser.add_argument("--pp-size", type=int, default=2, help="Pipeline parallel size (default: 2)")
    parser.add_argument("--cp-size", type=int, default=1, help="Context parallel size (default: 1)")

    # Training settings
    parser.add_argument("--num-layers", type=int, default=4, help="Number of layers (default: 4)")
    parser.add_argument("--seq-length", type=int, default=1024, help="Sequence length (default: 1024)")
    parser.add_argument("--train-iters", type=int, default=100, help="Training iterations (default: 100)")
    parser.add_argument("--global-batch-size", type=int, default=32, help="Global batch size (default: 32)")
    parser.add_argument("--micro-batch-size", type=int, default=1, help="Micro batch size (default: 1)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")

    return parser.parse_args()


# ==============================================================================
# STEP 1: Initialize torch.distributed
# ==============================================================================
def initialize_torch_distributed() -> None:
    """
    Initialize torch.distributed process group.

    This must be called before creating any process groups.
    In production, this is typically handled by torchrun.
    """
    if torch.distributed.is_initialized():
        print_rank_0("torch.distributed already initialized, skipping...")
        return

    # Get rank/world_size from environment (set by torchrun)
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Set CUDA device before distributed init
    torch.cuda.set_device(local_rank)

    print_rank_0(f"> Initializing torch.distributed with world_size={world_size}")

    torch.distributed.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
    )

    # Barrier to ensure all ranks are ready
    torch.distributed.barrier()


# ==============================================================================
# STEP 2: Create ProcessGroupCollection using HyperCommGrid
# ==============================================================================
def create_process_group_collection(
    tp_size: int,
    pp_size: int,
    cp_size: int = 1,
) -> ProcessGroupCollection:
    """
    Manually create all process groups using HyperCommGrid.

    This is the CORE of this example - showing explicit process group creation
    instead of relying on mpu's global state.

    Args:
        tp_size: Tensor parallel size
        pp_size: Pipeline parallel size
        cp_size: Context parallel size (default: 1)

    Returns:
        ProcessGroupCollection containing all required process groups

    The HyperCommGrid creates a multi-dimensional grid of ranks:
        shape = [TP, CP, DP, PP]

    From this grid, we create various process groups by selecting dimensions:
        - tp_pg:     select ["tp"]           -> ranks within same TP group
        - pp_pg:     select ["pp"]           -> ranks within same PP group
        - dp_pg:     select ["dp"]           -> ranks within same DP group
        - mp_pg:     select ["tp", "pp"]     -> model parallel (TP + PP)
        - tp_cp_pg:  select ["tp", "cp"]     -> tensor + context parallel
        - etc.
    """
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    # ===========================================================================
    # Calculate data parallel size from available world size
    # ===========================================================================
    model_parallel_size = tp_size * pp_size * cp_size
    if world_size % model_parallel_size != 0:
        raise RuntimeError(f"world_size ({world_size}) must be divisible by TP*PP*CP ({model_parallel_size})")
    dp_size = world_size // model_parallel_size

    if rank == 0:
        print(f"\n{'=' * 60}")
        print("Creating ProcessGroupCollection with HyperCommGrid")
        print(f"{'=' * 60}")
        print(f"  World Size:           {world_size}")
        print(f"  Tensor Parallel (TP): {tp_size}")
        print(f"  Pipeline Parallel (PP): {pp_size}")
        print(f"  Context Parallel (CP): {cp_size}")
        print(f"  Data Parallel (DP):   {dp_size}")
        print(f"  Grid Shape [TP, CP, DP, PP]: [{tp_size}, {cp_size}, {dp_size}, {pp_size}]")
        print(f"{'=' * 60}\n")

    # ===========================================================================
    # Create HyperCommGrid with the parallelism topology
    # ===========================================================================
    # The grid arranges all ranks in a multi-dimensional structure.
    # Dimension order: [TP, CP, DP, PP]
    grid = HyperCommGrid(
        shape=[tp_size, cp_size, dp_size, pp_size],
        dim_names=["tp", "cp", "dp", "pp"],
        rank_offset=0,  # Start from global rank 0
        backend="nccl",  # Use NCCL for GPU communication
    )

    # ===========================================================================
    # Create CORE process groups from the grid
    # ===========================================================================
    # Each create_pg() call creates a process group containing ranks that share
    # the SAME coordinates on all dimensions NOT listed in the argument.

    # Tensor Parallel group: ranks that differ only in TP dimension
    # Used for: column/row parallel linear layers, all-reduce in attention
    tp_pg = grid.create_pg(["tp"])

    # Context Parallel group: ranks that differ only in CP dimension
    # Used for: ring attention, sequence splitting
    cp_pg = grid.create_pg(["cp"])

    # Pipeline Parallel group: ranks that differ only in PP dimension
    # Used for: send/recv between pipeline stages
    pp_pg = grid.create_pg(["pp"])

    # Data Parallel group: ranks that differ only in DP dimension
    # Used for: gradient all-reduce, optimizer state sharding
    dp_pg = grid.create_pg(["dp"])

    # ===========================================================================
    # Create COMPOUND process groups
    # ===========================================================================
    # Model Parallel: combines TP and PP (all ranks in same model replica)
    mp_pg = grid.create_pg(["tp", "pp"])

    # Tensor + Context Parallel: used for some attention computations
    tp_cp_pg = grid.create_pg(["tp", "cp"])

    # TP + DP + CP: used for distributed optimizer across non-PP dimensions
    tp_dp_cp_pg = grid.create_pg(["tp", "dp", "cp"])

    # DP + CP: data parallel including context parallel
    dp_cp_pg = grid.create_pg(["dp", "cp"])

    # ===========================================================================
    # Create embedding/position embedding groups
    # ===========================================================================
    # Embedding group connects first and last PP stages (for tied embeddings)
    # Position embedding group is just the first PP stage
    pp_rank_lists = grid._gen_rank_enum(["pp"])

    embedding_rank_lists = []
    pos_embedding_rank_lists = []
    for ranks in pp_rank_lists:
        if not ranks:
            continue
        # Embedding: first and last stage (or just first if pp_size==1)
        embedding_rank_lists.append([ranks[0]] if len(ranks) == 1 else [ranks[0], ranks[-1]])
        # Position embedding: only first stage
        pos_embedding_rank_lists.append([ranks[0]])

    embd_pg, _ = torch.distributed.new_subgroups_by_enumeration(embedding_rank_lists, backend="nccl")
    pos_embd_pg, _ = torch.distributed.new_subgroups_by_enumeration(pos_embedding_rank_lists, backend="nccl")

    # ===========================================================================
    # Create Expert/MoE groups (simplified - no expert parallelism here)
    # ===========================================================================
    # For MoE models, you would create additional expert-specific groups.
    # Here we reuse TP groups since we're not using expert parallelism.
    ep_pg = None  # No expert parallelism in this example
    expt_tp_pg = tp_pg  # Expert TP same as regular TP
    tp_ep_pg = tp_pg  # TP + EP = just TP when EP=1
    tp_ep_pp_pg = mp_pg  # TP + EP + PP = MP when EP=1
    expt_dp_pg = dp_pg  # Expert DP same as regular DP

    # ===========================================================================
    # Initialize global memory buffer (required by Megatron-Core)
    # ===========================================================================
    parallel_state._set_global_memory_buffer()

    # ===========================================================================
    # Build the ProcessGroupCollection
    # ===========================================================================
    # This is the single object that contains ALL process groups and gets
    # passed through function calls in decentralized process groups mode.
    pg_collection = ProcessGroupCollection(
        # Core parallelism groups
        tp=tp_pg,
        pp=pp_pg,
        mp=mp_pg,
        cp=cp_pg,
        dp=dp_pg,
        dp_cp=dp_cp_pg,
        tp_cp=tp_cp_pg,
        tp_dp_cp=tp_dp_cp_pg,
        # Embedding groups
        embd=embd_pg,
        pos_embd=pos_embd_pg,
        # Expert/MoE groups (simplified)
        ep=ep_pg,
        expt_tp=expt_tp_pg,
        tp_ep=tp_ep_pg,
        tp_ep_pp=tp_ep_pp_pg,
        expt_dp=expt_dp_pg,
        intra_dp_cp=dp_cp_pg,
        intra_expt_dp=expt_dp_pg,
        # Hierarchical context parallel (not used)
        hcp=None,
        # Distributed optimizer groups (not using partial optimizer here)
        inter_dist_opt=None,
        intra_dist_opt=None,
    )

    if rank == 0:
        print("ProcessGroupCollection created successfully!")
        print(f"  tp_pg world size: {torch.distributed.get_world_size(tp_pg)}")
        print(f"  pp_pg world size: {torch.distributed.get_world_size(pp_pg)}")
        print(f"  dp_pg world size: {torch.distributed.get_world_size(dp_pg)}")
        print()

    return pg_collection


# ==============================================================================
# STEP 3: Set random seeds (required for model initialization)
# ==============================================================================
def set_random_seeds(
    seed: int,
    pg_collection: ProcessGroupCollection,
    data_parallel_random_init: bool = False,
) -> None:
    """
    Set random seeds for reproducibility.

    This is REQUIRED before creating the model because Megatron-Core's
    tensor parallel layers use a CUDA RNG tracker for weight initialization.

    The RNG tracker ensures that:
    - Different TP ranks initialize different weight partitions correctly
    - Different PP stages get different seeds for reproducibility
    - (Optionally) different DP ranks can have different initialization

    Args:
        seed: Base random seed
        pg_collection: ProcessGroupCollection containing all process groups
        data_parallel_random_init: If True, vary seed by DP rank
    """
    import random

    import numpy as np

    current_rank = torch.distributed.get_rank()

    # Different PP stages get different seeds (for reproducibility across stages)
    pp_rank = torch.distributed.get_group_rank(pg_collection.pp, current_rank)
    adjusted_seed = seed + (100 * pp_rank)

    # Optionally vary by DP rank (for different random init per replica)
    if data_parallel_random_init:
        dp_rank = torch.distributed.get_group_rank(pg_collection.dp, current_rank)
        adjusted_seed = adjusted_seed + (10 * dp_rank)

    # Set seeds for Python, NumPy, PyTorch
    random.seed(adjusted_seed)
    np.random.seed(adjusted_seed)
    torch.manual_seed(adjusted_seed)

    # ===========================================================================
    # CRITICAL: Initialize CUDA RNG tracker for tensor parallelism
    # ===========================================================================
    # This sets up the "model-parallel-rng" state used by ColumnParallelLinear,
    # RowParallelLinear, and other TP layers during weight initialization.
    if torch.cuda.device_count() > 0:
        # Get TP rank from our process group
        tp_rank = get_pg_rank(pg_collection.tp)
        # EP rank (no expert parallelism in this example)
        ep_rank = get_pg_rank(pg_collection.ep) if pg_collection.ep is not None else 0
        # Expert TP rank
        etp_rank = get_pg_rank(pg_collection.expt_tp)

        # This function creates the CUDA RNG tracker with "model-parallel-rng" state
        tensor_parallel.model_parallel_cuda_manual_seed(
            adjusted_seed,
            te_rng_tracker=False,  # Transformer Engine RNG tracker
            inference_rng_tracker=False,  # Inference-specific RNG
            use_cudagraphable_rng=False,  # CUDA graph compatible RNG
            tp_rank=tp_rank,
            ep_rank=ep_rank,
            etp_rank=etp_rank,
        )

    print_rank_0(f"Random seeds set (base={seed}, adjusted={adjusted_seed})")


# ==============================================================================
# STEP 4: Create model, optimizer, and run training
# ==============================================================================
def run_training(args: argparse.Namespace, pg_collection: ProcessGroupCollection) -> None:
    """
    Create model, optimizer, dataloaders, and run the training loop.

    This shows how to pass pg_collection explicitly to all components.
    """
    rank = get_rank_safe()
    world_size = torch.distributed.get_world_size()

    # Calculate DP size
    dp_size = world_size // (args.tp_size * args.pp_size * args.cp_size)

    # ===========================================================================
    # Create output directories
    # ===========================================================================
    base_dir = tempfile.mkdtemp(prefix="mbridge_decentralized_pg_")
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    tensorboard_dir = os.path.join(base_dir, "tensorboard")

    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
        print(f"Output directory: {base_dir}\n")

    torch.distributed.barrier()

    # ===========================================================================
    # Create ConfigContainer with use_decentralized_pg=True
    # ===========================================================================
    # IMPORTANT: When use_decentralized_pg=True, the setup functions
    # expect pg_collection to be passed explicitly rather than reading from mpu.

    model_cfg = Qwen3ModelProvider4B(
        # Parallelism - must match what we used to create pg_collection
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        context_parallel_size=args.cp_size,
        sequence_parallel=(args.tp_size > 1),
        # Model architecture (scaled down for demo)
        num_layers=args.num_layers,
        seq_length=args.seq_length,
        share_embeddings_and_output_weights=False,
        # Precision
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        attention_softmax_in_fp32=True,
        make_vocab_size_divisible_by=128,
        vocab_size=None,
    )

    train_cfg = TrainingConfig(
        train_iters=args.train_iters,
        eval_interval=args.train_iters,
        eval_iters=0,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        exit_signal_handler=True,
    )

    optimizer_cfg = OptimizerConfig(
        optimizer="adam",
        bf16=True,
        use_distributed_optimizer=True,
        clip_grad=1.0,
        lr=args.lr,
        weight_decay=0.01,
        min_lr=args.lr / 10,
    )

    scheduler_cfg = SchedulerConfig(
        lr_decay_style="cosine",
        lr_warmup_iters=10,
        lr_warmup_init=0.0,
        lr_decay_iters=args.train_iters,
        override_opt_param_scheduler=True,
        start_weight_decay=0.01,
        end_weight_decay=0.01,
        weight_decay_incr_style="constant",
    )

    ddp_cfg = DistributedDataParallelConfig(
        check_for_nan_in_grad=True,
        grad_reduce_in_fp32=True,
        # Disable overlap features for simplicity in this manual setup example
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        use_distributed_optimizer=True,
    )

    # KEY: use_decentralized_pg=True tells Megatron-Bridge that we're
    # managing process groups ourselves via pg_collection
    dist_cfg = DistributedInitConfig(
        use_decentralized_pg=True,
        use_gloo_process_groups=False,  # Gloo not supported with decentralized PG
    )

    dataset_cfg = MockGPTDatasetConfig(
        random_seed=1234,
        seq_length=args.seq_length,
        dataloader_type="single",
        num_workers=1,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
    )

    tokenizer_cfg = TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=10000)
    logger_cfg = LoggerConfig(log_interval=10, tensorboard_dir=tensorboard_dir)
    checkpoint_cfg = CheckpointConfig(save_interval=args.train_iters, save=checkpoint_dir)
    rng_cfg = RNGConfig(seed=1234)

    cfg = ConfigContainer(
        model=model_cfg,
        train=train_cfg,
        optimizer=optimizer_cfg,
        scheduler=scheduler_cfg,
        ddp=ddp_cfg,
        dist=dist_cfg,
        dataset=dataset_cfg,
        logger=logger_cfg,
        tokenizer=tokenizer_cfg,
        checkpoint=checkpoint_cfg,
        rng=rng_cfg,
    )

    # ===========================================================================
    # Initialize microbatch calculator
    # ===========================================================================
    init_num_microbatches_calculator(
        rank=rank,
        rampup_batch_size=None,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        data_parallel_size=dp_size,
    )

    # ===========================================================================
    # Build tokenizer and set vocab size
    # ===========================================================================
    tokenizer = build_tokenizer(tokenizer_cfg)
    cfg.model.vocab_size = tokenizer.vocab_size
    cfg.dataset.tokenizer = tokenizer
    cfg.validate()

    # ===========================================================================
    # Create model - PASS pg_collection explicitly
    # ===========================================================================
    print_rank_0("Creating model with pg_collection...")

    model = cfg.model.provide_distributed_model(
        ddp_config=ddp_cfg,
        use_megatron_fsdp=False,
        use_torch_fsdp2=False,
        overlap_param_gather_with_optimizer_step=False,
        data_parallel_random_init=False,
        pg_collection=pg_collection,  # <-- Explicitly pass our pg_collection!
    )

    print_rank_0(f"Model created: {len(model)} chunks")

    # ===========================================================================
    # Create optimizer - PASS pg_collection explicitly
    # ===========================================================================
    print_rank_0("Creating optimizer with pg_collection...")

    optimizer, scheduler = setup_optimizer(
        optimizer_config=optimizer_cfg,
        scheduler_config=scheduler_cfg,
        model=model,
        use_gloo_process_groups=False,
        pg_collection=pg_collection,  # <-- Explicitly pass our pg_collection!
    )

    print_rank_0("Optimizer created")

    # ===========================================================================
    # Create GlobalState (singleton pattern - no args, then set cfg)
    # ===========================================================================
    state = GlobalState()
    state.cfg = cfg

    # ===========================================================================
    # Create data iterators - use dp_group from pg_collection
    # ===========================================================================
    print_rank_0("Creating data iterators...")

    # Get the dataset provider based on the dataset config type
    # MockGPTDatasetConfig will create mock datasets for testing/demo
    dataset_provider = get_dataset_provider(cfg.dataset)

    # The data iterators need the DP group for sharding data across DP ranks
    train_data_iterator, valid_data_iterator, test_data_iterator = setup_data_iterators(
        cfg=cfg,
        train_state=state.train_state,
        model_length=len(model),
        train_valid_test_datasets_provider=dataset_provider,
        dp_group=pg_collection.dp,  # <-- Use DP group from our pg_collection!
    )

    print_rank_0("Data iterators created\n")

    print_rank_0("=" * 60)
    print_rank_0("Starting training with manually created process groups")
    print_rank_0("=" * 60)
    print_rank_0(f"  pg_collection.tp world size: {torch.distributed.get_world_size(pg_collection.tp)}")
    print_rank_0(f"  pg_collection.pp world size: {torch.distributed.get_world_size(pg_collection.pp)}")
    print_rank_0(f"  pg_collection.dp world size: {torch.distributed.get_world_size(pg_collection.dp)}")
    print_rank_0("")

    # Run the training loop
    train(
        forward_step_func=forward_step,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_data_iterator=train_data_iterator,
        valid_data_iterator=valid_data_iterator,
        global_state=state,
        checkpointing_context={},
        pg_collection=pg_collection,  # <-- Pass to training loop!
    )

    print_rank_0("\nTraining complete!")


def main() -> None:
    """Main entry point demonstrating manual process group creation."""
    args = parse_args()

    print_rank_0("=" * 70)
    print_rank_0("Qwen3 Pretraining with MANUALLY Created Decentralized Process Groups")
    print_rank_0("=" * 70)
    print_rank_0("")
    print_rank_0("This example shows how to:")
    print_rank_0("  1. Initialize torch.distributed")
    print_rank_0("  2. Create HyperCommGrid with your parallelism topology")
    print_rank_0("  3. Create ProcessGroupCollection from the grid")
    print_rank_0("  4. Set random seeds (required for model weight initialization)")
    print_rank_0("  5. Pass pg_collection explicitly to model, optimizer, training")
    print_rank_0("")

    # =========================================================================
    # STEP 1: Initialize torch.distributed
    # =========================================================================
    print_rank_0("STEP 1: Initializing torch.distributed...")
    initialize_torch_distributed()

    # Validate parallelism settings
    world_size = torch.distributed.get_world_size()
    required = args.tp_size * args.pp_size * args.cp_size
    if world_size < required:
        raise RuntimeError(
            f"Need at least {required} GPUs for TP={args.tp_size}, PP={args.pp_size}, CP={args.cp_size}"
        )
    if args.num_layers % args.pp_size != 0:
        raise RuntimeError(f"num_layers ({args.num_layers}) must be divisible by PP ({args.pp_size})")

    # =========================================================================
    # STEP 2: Create ProcessGroupCollection manually
    # =========================================================================
    print_rank_0("\nSTEP 2: Creating ProcessGroupCollection with HyperCommGrid...")
    pg_collection = create_process_group_collection(
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        cp_size=args.cp_size,
    )

    # =========================================================================
    # STEP 3: Set random seeds (REQUIRED before model creation)
    # =========================================================================
    print_rank_0("STEP 3: Setting random seeds for CUDA RNG tracker...")
    set_random_seeds(seed=1234, pg_collection=pg_collection)

    # =========================================================================
    # STEP 4: Run training with our pg_collection
    # =========================================================================
    print_rank_0("\nSTEP 4: Creating model/optimizer and running training...")
    run_training(args, pg_collection)

    # =========================================================================
    # Cleanup
    # =========================================================================
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    print_rank_0("\nDone!")


if __name__ == "__main__":
    main()
