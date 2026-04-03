# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""MIMO Training Loop for heterogeneous multi-module training.

This module provides the dedicated training loop for MIMO models with
heterogeneous parallelism. It uses MultiModulePipelineCommunicator for
cross-module communication and supports per-module gradient handling.

Key differences from standard train():
- Creates MultiModulePipelineCommunicator for cross-module communication
- Creates MultiModuleProcessGroupCollection for the schedule
- Uses forward_backward_pipelining_without_interleaving with multimodule support
- Uses zero_grad_buffer_for_multimodule() for gradient clearing
- Supports per-module optimizers

Note: Stub ranks are disallowed - validated at setup time.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel.schedules import forward_backward_pipelining_without_interleaving
from megatron.core.utils import get_model_config

from megatron.bridge.training.checkpointing import maybe_finalize_async_save, save_checkpoint
from megatron.bridge.training.eval import evaluate_and_print_results
from megatron.bridge.training.mimo_parallel_utils import (
    build_pg_collection_for_schedule,
    finalize_model_grads_multimodule,
    get_module_to_grid_tuple,
    multimodule_no_sync,
    unwrap_mimo_model,
    zero_grad_buffer_for_multimodule,
)
from megatron.bridge.training.profiling import (
    handle_profiling_step,
    handle_profiling_stop,
    initialize_pytorch_profiler,
    should_profile_rank,
)
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.train_utils import (
    prepare_forward_step_func,
    training_log,
)


if TYPE_CHECKING:
    from megatron.core.models.mimo import MimoModel
    from megatron.core.models.mimo.optimizer import MimoOptimizer
    from megatron.core.optimizer.optimizer_param_scheduler import OptimizerParamScheduler
    from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator

    from megatron.bridge.models.mimo.mimo_provider import MimoModelInfra


logger = logging.getLogger(__name__)


def train_step_mimo(
    forward_step_func: Callable,
    data_iterator: Iterator,
    model: "MimoModel",
    optimizer: "MimoOptimizer",
    schedulers: Dict[str, "OptimizerParamScheduler"],
    global_state: GlobalState,
    multimodule_communicator: "MultiModulePipelineCommunicator",
    multimodule_pg_collection,
    infra: "MimoModelInfra",
    module_to_grid_tuple: List,
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
) -> Tuple[Dict[str, torch.Tensor], Optional[float], Optional[int]]:
    """
    Execute a single MIMO training iteration: run forward/backward pipelined across modules, perform the coordinated optimizer update, advance per-module schedulers on success, reduce per-stage losses on the pipeline's final stage, and broadcast the final loss dictionary to the logging rank.
    
    Parameters:
        forward_step_func (Callable): Wrapped forward step function used by the pipeline.
        data_iterator (Iterator): Iterator yielding training microbatches.
        model (MimoModel): The MIMO-wrapped model participating in the pipeline.
        optimizer (MimoOptimizer): Optimizer that manages per-module updates and returns update status and gradient statistics.
        schedulers (Dict[str, OptimizerParamScheduler]): Per-module learning-rate schedulers; any None entries are skipped.
        global_state (GlobalState): Global training state and timers (provides cfg.data_parallel_size and timers).
        multimodule_communicator (MultiModulePipelineCommunicator): P2P communicator used by the multimodule pipeline schedule.
        multimodule_pg_collection: Process-group collection used by the pipeline schedule.
        infra (MimoModelInfra): MIMO infra containing module grids and pg_collections (used for final-stage reductions and optional all-reduce).
        module_to_grid_tuple (List): Mapping of modules to their device/grid tuples used for gradient clearing.
        num_microbatches (int): Number of microbatches per training iteration.
        seq_length (int): Sequence length for the forward step.
        micro_batch_size (int): Micro-batch size.
    
    Returns:
        Tuple containing:
          - loss_dict (Dict[str, torch.Tensor]): Reduced loss metrics produced by the pipeline's last stage (empty if no losses produced on this rank). All tensors are on the local CUDA device for the logging rank.
          - skipped_iter (int): `0` if the optimizer update succeeded and schedulers were stepped, `1` if the update was skipped.
          - grad_norm (Optional[float]): Global gradient norm computed by the optimizer, or `None` if unavailable.
          - num_zeros_in_grad (Optional[int]): Number of zero gradients reported by the optimizer, or `None` if unavailable.
    """
    timers = global_state.timers

    # Zero gradients for all modules
    zero_grad_buffer_for_multimodule(module_to_grid_tuple)

    # Run forward-backward schedule
    timers("forward-backward", log_level=1).start(barrier=False)

    losses_reduced = forward_backward_pipelining_without_interleaving(
        forward_step_func=forward_step_func,
        data_iterator=data_iterator,
        model=[model],
        num_microbatches=num_microbatches,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        forward_only=False,
        p2p_communicator=multimodule_communicator,
        pg_collection=multimodule_pg_collection,
    )

    timers("forward-backward").stop()

    # Optimizer step - MimoOptimizer handles all modules and computes global grad norm
    timers("optimizer", log_level=1).start(barrier=False)

    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()

    timers("optimizer").stop()

    # Step learning rate schedulers
    if update_successful:
        increment = num_microbatches * micro_batch_size * global_state.cfg.data_parallel_size
        for module_name, scheduler in schedulers.items():
            if scheduler is not None:
                scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

    loss_dict = {}
    if losses_reduced:
        is_last_stage = False
        # Access role from unwrapped model (handles Float16Module wrapper)
        mimo_model = unwrap_mimo_model(model)
        if mimo_model.role is None:
            is_last_stage = True
        elif mimo_model.role.has_language_module:
            is_last_stage = mimo_model.role.is_last_stage(MIMO_LANGUAGE_MODULE_KEY)

        if is_last_stage:
            llm_pg = infra.pg_collections.get(MIMO_LANGUAGE_MODULE_KEY) if infra.pg_collections else None
            for key in losses_reduced[0].keys():
                val = [x[key].view(-1) for x in losses_reduced]
                if val[0].numel() == 2:
                    val = torch.vstack(val).sum(dim=0)
                    if llm_pg is not None and llm_pg.dp_cp is not None:
                        torch.distributed.all_reduce(val, group=llm_pg.dp_cp)
                    loss_dict[key] = val[0] / val[1]
                elif val[0].numel() == 1:
                    loss_dict[key] = torch.cat(val).mean()
                else:
                    raise ValueError(f"Invalid value shape: {val[0].shape} for key {key}")

    # Broadcast loss_dict to all ranks (the last rank is the logging rank for
    # W&B/TensorBoard). Use broadcast_object_list from the source rank so every
    # rank ends up with the same dict — no fragile P2P or GPU-side pickle needed.
    last_rank = dist.get_world_size() - 1
    my_rank = dist.get_rank()

    # All ranks agree on which rank holds the loss (pick highest rank with data).
    has_loss = 1 if loss_dict else 0
    source_tensor = torch.tensor([my_rank if has_loss else -1], dtype=torch.int32, device="cuda")
    torch.distributed.all_reduce(source_tensor, op=torch.distributed.ReduceOp.MAX)
    source_rank = int(source_tensor.item())

    # Only broadcast if the source and logging rank differ and a valid source exists.
    if source_rank >= 0 and source_rank != last_rank:
        obj = [loss_dict if my_rank == source_rank else None]
        torch.distributed.broadcast_object_list(obj, src=source_rank)
        if my_rank == last_rank:
            received = obj[0] or {}
            # Tensors inside the received dict carry the source rank's CUDA device;
            # move them to this rank's device so training_log arithmetic works.
            loss_dict = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in received.items()}

    return loss_dict, skipped_iter, grad_norm, num_zeros_in_grad


def train_mimo(
    forward_step_func: Callable,
    model: "MimoModel",
    optimizer: "MimoOptimizer",
    schedulers: Dict[str, "OptimizerParamScheduler"],
    train_data_iterator: Iterator,
    valid_data_iterator: Optional[Iterator],
    global_state: GlobalState,
    mimo_infra: "MimoModelInfra",
    multimodule_communicator: "MultiModulePipelineCommunicator",
) -> None:
    """
    Run the full MIMO training loop coordinating multimodule pipelining, optimization, scheduling, logging, evaluation, and checkpointing.
    
    This function:
    - Builds multimodule process-group collection and validates MIMO-specific requirements.
    - Configures model gradient hooks for multimodule training and prepares the forward step.
    - Iterates until the configured number of training iterations, calling train_step_mimo each iteration, updating training state, stepping per-module schedulers on successful updates, logging metrics, running evaluation, and saving checkpoints (including async finalize behavior).
    - Manages optional profiling and ensures graceful shutdown of profilers and async checkpoint saves.
    
    Parameters:
        forward_step_func (Callable): Per-microbatch forward function; will be wrapped with GlobalState.
        model (MimoModel): The distributed MIMO model to train.
        optimizer (MimoOptimizer): Optimizer coordinating per-module optimizer behavior and gradient scaling.
        schedulers (Dict[str, OptimizerParamScheduler]): Mapping of module names to their learning-rate schedulers.
        train_data_iterator (Iterator): Iterator yielding training microbatches.
        valid_data_iterator (Optional[Iterator]): Optional iterator for validation/evaluation.
        global_state (GlobalState): Global training state and configuration (timers, train_state, loggers).
        mimo_infra (MimoModelInfra): MIMO infrastructure describing module grids, topology, and process-group collections.
        multimodule_communicator (MultiModulePipelineCommunicator): Communicator for cross-module point-to-point pipeline transfers.
    """
    timers = global_state.timers
    train_state = global_state.train_state
    cfg = global_state.cfg

    # Get training config
    train_config = cfg.train
    num_microbatches = get_num_microbatches()
    seq_length = cfg.dataset.seq_length
    micro_batch_size = train_config.micro_batch_size

    # Prepare forward step function with GlobalState injection
    wrapped_forward_step_func = prepare_forward_step_func(forward_step_func, global_state)

    # Build module-to-grid mapping for gradient operations
    module_to_grid_tuple = get_module_to_grid_tuple(model, mimo_infra)

    # Build pg_collection for schedule
    multimodule_pg_collection = build_pg_collection_for_schedule(mimo_infra)

    # Guard against list fallback - MIMO training requires MultiModuleProcessGroupCollection
    if isinstance(multimodule_pg_collection, list):
        raise RuntimeError(
            "MultiModuleProcessGroupCollection is required for MIMO training. "
            "The list-based fallback is not supported. Ensure Megatron-LM PR 3212 is available."
        )

    # Use rank-local module PG for logging reductions to avoid global MPU fallback.
    # NOTE: In non-colocated MIMO each rank participates in exactly one module, so
    # "first non-None" unambiguously selects that module's PG. For colocated MIMO
    # (where a rank participates in multiple modules), this selection must be
    # replaced with per-module logging or an explicit module-aware reduction strategy.
    local_pg_collection = next((pg for pg in mimo_infra.pg_collections.values() if pg is not None), None)
    if local_pg_collection is None:
        raise RuntimeError(
            "No local ProcessGroupCollection found for this rank. "
            "Ensure rank participation is correctly configured in MIMO infrastructure."
        )

    # Configure gradient hooks on model config
    model_config = get_model_config(model)

    # Bind custom parameters via partial(), leaving schedule-provided args unbound
    model_config.no_sync_func = partial(multimodule_no_sync, module_to_grid_tuple=module_to_grid_tuple)

    model_config.finalize_model_grads_func = partial(
        finalize_model_grads_multimodule,
        infra=mimo_infra,
        module_to_grid_tuple=module_to_grid_tuple,
    )

    # Optional: Set grad_scale_func from MimoOptimizer
    if optimizer is not None and hasattr(optimizer, "scale_loss"):
        model_config.grad_scale_func = optimizer.scale_loss

    # Validation: variable_seq_lengths should already be True (set by MimoModelProvider)
    assert model_config.variable_seq_lengths, (
        "variable_seq_lengths must be True for MIMO training. "
        "This should be set by MimoModelProvider.provide_distributed_model()."
    )

    # Initialize tracking variables
    total_loss_dict = {}
    history_wct = []
    report_memory_flag = True

    # Get first scheduler for checkpoint saving.
    # All modules share the same LR schedule, so first scheduler state is representative.
    first_scheduler = next(iter(schedulers.values()), None) if schedulers else None

    # Profiler setup (mirrors train.py behavior)
    prof = None
    nsys_nvtx_context = None
    prof_config = cfg.profiling
    if prof_config and should_profile_rank(prof_config, dist.get_rank()):
        if prof_config.use_pytorch_profiler:
            prof = initialize_pytorch_profiler(prof_config, cfg.logger.tensorboard_dir)
            prof.start()

    logger.info(f"Rank {dist.get_rank()}: Starting MIMO training loop")

    # Main training loop
    timers("interval-time", log_level=0).start(barrier=True)

    while train_state.step < train_config.train_iters:
        # Handle profiling
        nsys_ctx = handle_profiling_step(
            prof_config,
            train_state.step,
            dist.get_rank(),
            prof,
        )
        if nsys_ctx is not None:
            nsys_nvtx_context = nsys_ctx

        # Start iteration timer
        timers("iteration-time", log_level=0).start(barrier=False)

        # Run single training step
        loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = train_step_mimo(
            forward_step_func=wrapped_forward_step_func,
            data_iterator=train_data_iterator,
            model=model,
            optimizer=optimizer,
            schedulers=schedulers,
            global_state=global_state,
            multimodule_communicator=multimodule_communicator,
            multimodule_pg_collection=multimodule_pg_collection,
            infra=mimo_infra,
            module_to_grid_tuple=module_to_grid_tuple,
            num_microbatches=num_microbatches,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
        )

        # Stop iteration timer
        timers("iteration-time").stop(barrier=False)
        iteration_time = timers("iteration-time").elapsed(reset=True, barrier=False)
        history_wct.append(iteration_time)

        # Update training state
        train_state.step += 1
        train_state.consumed_train_samples += micro_batch_size * num_microbatches * cfg.data_parallel_size

        # Get learning rate from first scheduler
        learning_rate = None
        if schedulers:
            sched = next(iter(schedulers.values()))
            if sched is not None:
                learning_rate = sched.get_lr(sched.optimizer.param_groups[0])

        # Log training metrics
        if not cfg.logger.skip_train_metrics_log:
            # Get loss scale from MimoOptimizer
            if optimizer is not None and hasattr(optimizer, "get_loss_scale"):
                loss_scale = optimizer.get_loss_scale()
                if hasattr(loss_scale, "item"):
                    loss_scale = loss_scale.item()
            else:
                loss_scale = 1.0

            report_memory_flag = training_log(
                loss_dict=loss_dict,
                total_loss_dict=total_loss_dict,
                learning_rate=learning_rate,
                decoupled_learning_rate=None,
                loss_scale=loss_scale,
                report_memory_flag=report_memory_flag,
                skipped_iter=skipped_iter,
                grad_norm=grad_norm,
                params_norm=None,
                num_zeros_in_grad=num_zeros_in_grad,
                config=cfg,
                global_state=global_state,
                history_wct=history_wct,
                model=[model],
                pg_collection=local_pg_collection,
            )

            # Log iteration-time directly for MIMO models.
            # training_log only logs this inside a hasattr(config.model, "kv_channels")
            # block which MIMO models don't satisfy, so we log it here as a workaround.
            if cfg.logger.log_timers_to_tensorboard and train_state.step % cfg.logger.log_interval == 0:
                writer = global_state.tensorboard_logger
                if writer:
                    writer.add_scalar("iteration-time", iteration_time, train_state.step)
                wandb_writer = global_state.wandb_logger
                if wandb_writer:
                    wandb_writer.log({"iteration-time": iteration_time}, train_state.step)

        # Evaluation at specified intervals
        if (
            train_config.eval_interval is not None
            and train_state.step % train_config.eval_interval == 0
            and valid_data_iterator is not None
        ):
            timers("evaluate", log_level=0).start(barrier=True)
            evaluate_and_print_results(
                state=global_state,
                prefix=f"iteration {train_state.step}",
                forward_step_func=forward_step_func,
                data_iterator=valid_data_iterator,
                model=[model],
                config=cfg,
                verbose=False,
                write_to_tensorboard=True,
                p2p_communicator=multimodule_communicator,
                pg_collection=multimodule_pg_collection,
            )
            timers("evaluate").stop()

        # Checkpointing at specified intervals
        if cfg.checkpoint.save_interval is not None and train_state.step % cfg.checkpoint.save_interval == 0:
            timers("save-checkpoint", log_level=0).start(barrier=True)
            save_checkpoint(
                state=global_state,
                model=[model],
                optimizer=optimizer,
                opt_param_scheduler=first_scheduler,
                num_floating_point_operations_so_far=0,  # TODO: Add proper FLOPs tracking
            )
            timers("save-checkpoint").stop()

        # Finalize any pending async saves (non-blocking during training)
        maybe_finalize_async_save(
            global_state=global_state,
            ckpt_cfg=cfg.checkpoint,
            blocking=False,
        )

    # Stop profiling
    handle_profiling_stop(
        prof_config,
        train_state.step,
        dist.get_rank(),
        prof,
        nsys_nvtx_context,
    )

    # Finalize any remaining async saves before exit
    maybe_finalize_async_save(
        global_state=global_state,
        ckpt_cfg=cfg.checkpoint,
        blocking=True,
        terminate=True,
    )

    timers("interval-time").stop()

    logger.info(f"Rank {dist.get_rank()}: MIMO training completed")
