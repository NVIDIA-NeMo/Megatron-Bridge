#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
Example: Eval-time Context Parallelism with Decentralized Process Groups
==============================================================================

This example demonstrates running validation with a HIGHER context-parallel
degree than training — without modifying Megatron-Core.

Motivation
----------
For training runs with large DP and a small validation set, the training
CP/DP layout forces wasted work at eval time.  Concrete (8192-GPU) scenario:

    World: 8192 GPUs, TP=2, PP=4, CP_train=1  ->  DP_train=1024
    Validation set: 1024 samples
    To keep the PP pipeline full (GA >= PP=4) you need GA=4 -> 4096 sample-steps
    That is 4x redundant work vs processing each sample exactly once.

    With CP_eval=4:  DP_eval=256, GA=4 covers 1024 samples in 1 iter —
    no redundancy, ~2.3x wall-time speedup, and CP_eval-x lower activation memory.

This 8-GPU run demonstrates the same mechanism at a smaller scale:

    TP=2, PP=2, CP_train=1, CP_eval=2
    world=8 -> DP_train=2, DP_eval=1

How it works
------------
Two ProcessGroupCollections are created from the same world at startup:

    train_pgs:  TP=2, PP=2, CP=1  (DP=2)
    eval_pgs:   TP=2, PP=2, CP=2  (DP=1)

The model is built once, with train_pgs.  At eval time, eval_cp_context()
rebinds every module's cached CP-group references to eval_pgs, runs evaluation,
then restores train_pgs — all without touching Megatron-Core or reloading weights.

How to Run
----------
# 8 GPUs: TP2 x PP2 x CP_train=1 x CP_eval=2
uv run python -m torch.distributed.run --nproc_per_node=8 \\
    examples/decentralized_pg/pretrain_qwen3_eval_cp.py

# 4 GPUs: TP2 x PP1 x CP_train=1 x CP_eval=2
uv run python -m torch.distributed.run --nproc_per_node=4 \\
    examples/decentralized_pg/pretrain_qwen3_eval_cp.py \\
    --tp-size 2 --pp-size 1 --cp-train 1 --cp-eval 2

Prerequisites
-------------
  use_decentralized_pg=True  (always set by this example)
  No FSDP, no CUDA graphs, no hierarchical CP
  seq_length % (2 * cp_eval) == 0
"""

import argparse
import os
import tempfile

import torch
import torch.distributed
from megatron.core import parallel_state, tensor_parallel
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.num_microbatches_calculator import init_num_microbatches_calculator
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_model_config, get_pg_rank

from megatron.bridge.data.loaders import setup_data_iterators
from megatron.bridge.data.utils import get_dataset_provider
from megatron.bridge.models import AutoBridge
from megatron.bridge.training.checkpointing import DefaultCheckpointManager
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
from megatron.bridge.training.eval import evaluate_and_print_results
from megatron.bridge.training.eval_cp import eval_cp_context
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.optim import setup_optimizer
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from megatron.bridge.training.train import train
from megatron.bridge.utils.common_utils import get_rank_safe, print_rank_0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Eval-time Context Parallelism Demo")
    parser.add_argument("--tp-size", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--pp-size", type=int, default=2, help="Pipeline parallel size")
    parser.add_argument("--cp-train", type=int, default=1, help="Training CP degree")
    parser.add_argument("--cp-eval", type=int, default=2, help="Eval CP degree (must be > cp-train)")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--seq-length", type=int, default=1024, help="Sequence length")
    parser.add_argument("--train-iters", type=int, default=5, help="Training iterations before eval")
    parser.add_argument("--global-batch-size", type=int, default=4, help="Training global batch size")
    parser.add_argument("--micro-batch-size", type=int, default=1, help="Micro batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    return parser.parse_args()


def initialize_torch_distributed() -> None:
    """Initialize torch.distributed if not already done."""
    if torch.distributed.is_initialized():
        return
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    torch.distributed.barrier()


def create_pg_collection(
    tp_size: int,
    pp_size: int,
    *,
    cp_size: int = 1,
    label: str = "",
) -> ProcessGroupCollection:
    """Create a ProcessGroupCollection via HyperCommGrid.

    Grid shape: [TP, CP, DP, PP] where DP = world_size / (TP * CP * PP).

    Args:
        tp_size: Tensor parallel size.
        pp_size: Pipeline parallel size.
        cp_size: Context parallel size.
        label: Descriptive label for logging.

    Returns:
        Fully-constructed ProcessGroupCollection.
    """
    world_size = torch.distributed.get_world_size()
    model_parallel_size = tp_size * pp_size * cp_size
    if world_size % model_parallel_size != 0:
        raise RuntimeError(
            f"world_size ({world_size}) must be divisible by TP*PP*CP "
            f"({tp_size}*{pp_size}*{cp_size}={model_parallel_size})"
        )
    dp_size = world_size // model_parallel_size

    if get_rank_safe() == 0:
        tag = f" ({label})" if label else ""
        print(f"\nCreating ProcessGroupCollection{tag}: TP={tp_size} CP={cp_size} DP={dp_size} PP={pp_size}")

    grid = HyperCommGrid(
        shape=[tp_size, cp_size, dp_size, pp_size],
        dim_names=["tp", "cp", "dp", "pp"],
        rank_offset=0,
        backend="nccl",
    )

    tp_pg = grid.create_pg(["tp"])
    cp_pg = grid.create_pg(["cp"])
    pp_pg = grid.create_pg(["pp"])
    dp_pg = grid.create_pg(["dp"])
    mp_pg = grid.create_pg(["tp", "pp"])
    tp_cp_pg = grid.create_pg(["tp", "cp"])
    tp_dp_cp_pg = grid.create_pg(["tp", "dp", "cp"])
    dp_cp_pg = grid.create_pg(["dp", "cp"])

    # Embedding groups: first and last PP stage (or just first when pp_size==1).
    pp_rank_lists = grid._gen_rank_enum(["pp"])
    embedding_rank_lists, pos_embedding_rank_lists = [], []
    for ranks in pp_rank_lists:
        if not ranks:
            continue
        embedding_rank_lists.append([ranks[0]] if len(ranks) == 1 else [ranks[0], ranks[-1]])
        pos_embedding_rank_lists.append([ranks[0]])
    embd_pg, _ = torch.distributed.new_subgroups_by_enumeration(embedding_rank_lists, backend="nccl")
    pos_embd_pg, _ = torch.distributed.new_subgroups_by_enumeration(pos_embedding_rank_lists, backend="nccl")

    return ProcessGroupCollection(
        tp=tp_pg,
        pp=pp_pg,
        mp=mp_pg,
        cp=cp_pg,
        dp=dp_pg,
        dp_cp=dp_cp_pg,
        tp_cp=tp_cp_pg,
        tp_dp_cp=tp_dp_cp_pg,
        embd=embd_pg,
        pos_embd=pos_embd_pg,
        ep=None,
        expt_tp=tp_pg,
        tp_ep=tp_pg,
        tp_ep_pp=mp_pg,
        expt_dp=dp_pg,
        intra_dp_cp=dp_cp_pg,
        intra_expt_dp=dp_pg,
        hcp=None,
        inter_dist_opt=None,
        intra_dist_opt=None,
    )


def set_random_seeds(seed: int, pg_collection: ProcessGroupCollection) -> None:
    """Set random seeds for reproducibility (required before model creation)."""
    import random

    import numpy as np

    current_rank = torch.distributed.get_rank()
    pp_rank = torch.distributed.get_group_rank(pg_collection.pp, current_rank)
    adjusted_seed = seed + (100 * pp_rank)

    random.seed(adjusted_seed)
    np.random.seed(adjusted_seed)
    torch.manual_seed(adjusted_seed)

    if torch.cuda.device_count() > 0:
        tp_rank = get_pg_rank(pg_collection.tp)
        ep_rank = get_pg_rank(pg_collection.ep) if pg_collection.ep is not None else 0
        etp_rank = get_pg_rank(pg_collection.expt_tp)
        tensor_parallel.model_parallel_cuda_manual_seed(
            adjusted_seed,
            te_rng_tracker=False,
            inference_rng_tracker=False,
            use_cudagraphable_rng=False,
            tp_rank=tp_rank,
            ep_rank=ep_rank,
            etp_rank=etp_rank,
        )


def main() -> None:
    """Entry point for the eval-time CP demo."""
    args = parse_args()

    print_rank_0("=" * 70)
    print_rank_0("Eval-time Context Parallelism Demo")
    print_rank_0("=" * 70)

    # =========================================================================
    # Step 1: Initialize torch.distributed
    # =========================================================================
    initialize_torch_distributed()
    world_size = torch.distributed.get_world_size()

    if args.cp_eval <= args.cp_train:
        raise ValueError(f"cp-eval ({args.cp_eval}) must be > cp-train ({args.cp_train})")
    if args.seq_length % (2 * args.cp_eval) != 0:
        raise ValueError(f"seq_length ({args.seq_length}) must be divisible by 2*cp_eval ({2 * args.cp_eval})")
    if args.num_layers % args.pp_size != 0:
        raise RuntimeError(f"num_layers ({args.num_layers}) must be divisible by pp_size ({args.pp_size})")

    dp_train = world_size // (args.tp_size * args.pp_size * args.cp_train)
    dp_eval = world_size // (args.tp_size * args.pp_size * args.cp_eval)

    print_rank_0(f"\n  World: {world_size} GPUs")
    print_rank_0(f"  TP={args.tp_size}  PP={args.pp_size}  CP_train={args.cp_train}  CP_eval={args.cp_eval}")
    print_rank_0(f"  DP_train={dp_train}  DP_eval={dp_eval}  (only DP changes between modes)")
    print_rank_0(f"  seq_length={args.seq_length}")

    # =========================================================================
    # Step 2: Create TWO ProcessGroupCollections — one for train, one for eval
    # =========================================================================
    # Both span all world_size ranks; they differ only in the CP (and derived DP) size.
    print_rank_0("\n--- Step 2: Create train and eval ProcessGroupCollections ---")

    # _set_global_memory_buffer is required before ProcessGroupCollection is used.
    parallel_state._set_global_memory_buffer()

    train_pgs = create_pg_collection(args.tp_size, args.pp_size, cp_size=args.cp_train, label="train_pgs")
    eval_pgs = create_pg_collection(args.tp_size, args.pp_size, cp_size=args.cp_eval, label="eval_pgs")

    print_rank_0(
        f"\n  train_pgs: dp={torch.distributed.get_world_size(train_pgs.dp)} cp={torch.distributed.get_world_size(train_pgs.cp)}"
    )
    print_rank_0(
        f"  eval_pgs:  dp={torch.distributed.get_world_size(eval_pgs.dp)} cp={torch.distributed.get_world_size(eval_pgs.cp)}"
    )

    # =========================================================================
    # Step 3: Set random seeds (required before model weight initialization)
    # =========================================================================
    print_rank_0("\n--- Step 3: Setting random seeds ---")
    set_random_seeds(seed=1234, pg_collection=train_pgs)

    # =========================================================================
    # Step 4: Build config
    # =========================================================================
    print_rank_0("\n--- Step 4: Building config ---")

    rank = get_rank_safe()
    base_dir = tempfile.mkdtemp(prefix="mbridge_eval_cp_")
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    tensorboard_dir = os.path.join(base_dir, "tensorboard")
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
    torch.distributed.barrier()

    bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen3-4B")
    # Randomly-initialised weights (skip HF weight loading): we train a small
    # downsized Qwen3-4B on mock data, so the real checkpoint is irrelevant.
    model_cfg = bridge.to_megatron_provider(load_weights=False)

    # Parallelism: model is always built with the TRAINING CP degree.
    # eval_cp_context() rebinds to eval_pgs at eval time.
    model_cfg.tensor_model_parallel_size = args.tp_size
    model_cfg.pipeline_model_parallel_size = args.pp_size
    model_cfg.context_parallel_size = args.cp_train
    model_cfg.sequence_parallel = args.tp_size > 1
    model_cfg.num_layers = args.num_layers
    model_cfg.seq_length = args.seq_length
    model_cfg.pipeline_dtype = torch.bfloat16
    model_cfg.attention_softmax_in_fp32 = True
    model_cfg.make_vocab_size_divisible_by = 128
    model_cfg.vocab_size = None
    # Tied embeddings rely on Megatron-Core globals not populated in decentralized-PG mode.
    model_cfg.share_embeddings_and_output_weights = False

    train_cfg = TrainingConfig(
        train_iters=args.train_iters,
        # Disable auto-eval inside train(); we call evaluate manually after.
        eval_interval=args.train_iters + 1,
        eval_iters=1,
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
        lr_warmup_iters=1,
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
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        use_distributed_optimizer=True,
    )

    dist_cfg = DistributedInitConfig(
        use_decentralized_pg=True,
        use_gloo_process_groups=False,
        eval_context_parallel_size=args.cp_eval,
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
    logger_cfg = LoggerConfig(log_interval=1, tensorboard_dir=tensorboard_dir)
    checkpoint_cfg = CheckpointConfig(save_interval=args.train_iters + 1, save=checkpoint_dir)
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

    # Microbatch calculator uses DP_train size (only training DP matters here).
    init_num_microbatches_calculator(
        rank=rank,
        rampup_batch_size=None,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        data_parallel_size=dp_train,
    )

    tokenizer = build_tokenizer(tokenizer_cfg)
    cfg.model.vocab_size = tokenizer.vocab_size
    cfg.dataset.tokenizer = tokenizer
    cfg.validate()

    # =========================================================================
    # Step 5: Build model with train_pgs
    # =========================================================================
    print_rank_0("\n--- Step 5: Creating model (with train_pgs, CP_train) ---")
    model = cfg.model.provide_distributed_model(
        ddp_config=ddp_cfg,
        use_megatron_fsdp=False,
        use_torch_fsdp2=False,
        overlap_param_gather_with_optimizer_step=False,
        data_parallel_random_init=False,
        pg_collection=train_pgs,
    )
    print_rank_0(f"  Model created: {len(model)} virtual PP chunk(s)")

    # model_config (TransformerConfig) is what evaluate() expects as its `config` argument.
    model_config = get_model_config(model[0])

    optimizer, scheduler = setup_optimizer(
        optimizer_config=optimizer_cfg,
        scheduler_config=scheduler_cfg,
        model=model,
        use_gloo_process_groups=False,
        pg_collection=train_pgs,
    )

    # =========================================================================
    # Step 6: Build GlobalState and data iterators
    # =========================================================================
    print_rank_0("\n--- Step 6: Building data iterators ---")
    state = GlobalState()
    state.cfg = cfg
    # Store both pg_collections on state for downstream access if needed.
    state.train_pgs = train_pgs
    state.eval_pgs = eval_pgs

    dataset_provider = get_dataset_provider(cfg.dataset)
    train_data_iterator, valid_data_iterator, _ = setup_data_iterators(
        cfg=cfg,
        train_state=state.train_state,
        model_length=len(model),
        train_valid_test_datasets_provider=dataset_provider,
        dp_group=train_pgs.dp,  # train iterator sharded by train DP
    )

    # =========================================================================
    # Step 7: Train (auto-eval is disabled; eval_interval > train_iters)
    # =========================================================================
    print_rank_0(f"\n--- Step 7: Training for {args.train_iters} iterations (CP_train={args.cp_train}) ---")
    checkpoint_manager = DefaultCheckpointManager(checkpoint_config=cfg.checkpoint)
    train(
        forward_step_func=forward_step,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_data_iterator=train_data_iterator,
        valid_data_iterator=valid_data_iterator,
        global_state=state,
        checkpoint_manager=checkpoint_manager,
        pg_collection=train_pgs,
    )

    torch.distributed.barrier()
    print_rank_0("\nTraining complete.")

    # =========================================================================
    # Step 8: Baseline eval with train_pgs (CP_train)
    # =========================================================================
    print_rank_0("\n" + "=" * 60)
    print_rank_0(f"Step 8: Baseline eval  (CP={args.cp_train}, train_pgs)")
    print_rank_0("=" * 60)
    evaluate_and_print_results(
        state,
        prefix=f"baseline CP={args.cp_train}",
        forward_step_func=forward_step,
        data_iterator=valid_data_iterator,
        model=model,
        config=model_config,  # TransformerConfig, same pattern as train.py
        verbose=True,
        write_to_tensorboard=False,
        pg_collection=train_pgs,
    )

    # =========================================================================
    # Step 9: Eval with eval_pgs (CP_eval) via eval_cp_context
    # =========================================================================
    print_rank_0("\n" + "=" * 60)
    print_rank_0(f"Step 9: Eval-CP demo  (CP={args.cp_eval}, eval_pgs)")
    print_rank_0("=" * 60)
    print_rank_0(
        "  eval_cp_context() rebinds all cached CP-group references on every\n"
        "  module to eval_pgs, runs evaluation, then restores train_pgs.\n"
    )
    with eval_cp_context(model, eval_pgs, train_pgs):
        evaluate_and_print_results(
            state,
            prefix=f"eval-CP CP={args.cp_eval}",
            forward_step_func=forward_step,
            data_iterator=valid_data_iterator,
            model=model,
            config=model_config,
            verbose=True,
            write_to_tensorboard=False,
            pg_collection=eval_pgs,
        )

    # =========================================================================
    # Step 10: Post-restore eval to verify train_pgs was correctly reinstalled
    # =========================================================================
    print_rank_0("\n" + "=" * 60)
    print_rank_0(f"Step 10: Verify restore  (CP={args.cp_train}, train_pgs)")
    print_rank_0("=" * 60)
    evaluate_and_print_results(
        state,
        prefix=f"post-restore CP={args.cp_train}",
        forward_step_func=forward_step,
        data_iterator=valid_data_iterator,
        model=model,
        config=model_config,
        verbose=True,
        write_to_tensorboard=False,
        pg_collection=train_pgs,
    )

    print_rank_0("\n" + "=" * 70)
    print_rank_0("SUCCESS: Eval-time CP demo completed without errors.")
    print_rank_0(
        f"  Trained CP={args.cp_train}, evaluated CP={args.cp_eval}, confirmed restore with CP={args.cp_train}."
    )
    print_rank_0("=" * 70)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    print_rank_0("\nDone!")


if __name__ == "__main__":
    main()
