# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""MegatronMIMO Training Loop for heterogeneous multi-module training.

This module provides the dedicated training loop for MegatronMIMO models with
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
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY, ModuleLayout
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel.schedules import forward_backward_pipelining_without_interleaving

from megatron.bridge.training.checkpointing import CheckpointManager, DefaultCheckpointManager
from megatron.bridge.training.eval import evaluate_and_print_results
from megatron.bridge.training.megatron_mimo_parallel_utils import (
    build_pg_collection_for_schedule,
    get_module_to_grid_tuple,
    unwrap_megatron_mimo_model,
    zero_grad_buffer_for_multimodule,
)
from megatron.bridge.training.profiling import (
    handle_profiling_step,
    handle_profiling_stop,
    initialize_pytorch_profiler,
    should_profile_rank,
)
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.train import checkpoint_and_decide_exit
from megatron.bridge.training.utils.train_utils import (
    prepare_forward_step_func,
    training_log,
)


if TYPE_CHECKING:
    from megatron.core.models.mimo import MimoModel
    from megatron.core.models.mimo.optimizer import MimoOptimizer
    from megatron.core.optimizer.optimizer_param_scheduler import OptimizerParamScheduler
    from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
    from megatron.core.process_groups_config import MultiModuleProcessGroupCollection, ProcessGroupCollection

    from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import MegatronMIMOInfra


logger = logging.getLogger(__name__)


def train_step_megatron_mimo(
    forward_step_func: Callable,
    data_iterator: Iterator,
    model: "MimoModel",
    optimizer: "MimoOptimizer",
    schedulers: Dict[str, "OptimizerParamScheduler"],
    global_state: GlobalState,
    multimodule_communicator: "MultiModulePipelineCommunicator",
    multimodule_pg_collection,
    infra: "MegatronMIMOInfra",
    module_to_grid_tuple: List,
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
) -> Tuple[Dict[str, torch.Tensor], Optional[float], Optional[int]]:
    """Single MegatronMIMO training step.

    Args:
        forward_step_func: Forward step function (wrapped with GlobalState).
        data_iterator: Iterator over the dataset.
        model: MimoModel instance.
        optimizer: MimoOptimizer managing per-module optimizers.
        schedulers: Per-module learning rate schedulers {module_name: scheduler}.
        global_state: GlobalState containing timers, config, train_state.
        multimodule_communicator: MultiModulePipelineCommunicator for P2P.
        multimodule_pg_collection: PG collection for schedule.
        infra: MegatronMIMOInfra with grids, topology, pg_collections.
        module_to_grid_tuple: List of (module, grid) tuples.
        num_microbatches: Number of microbatches per iteration.
        seq_length: Sequence length.
        micro_batch_size: Micro batch size.

    Returns:
        Tuple of (loss_dict, skipped_iter, grad_norm, num_zeros_in_grad).
    """
    timers = global_state.timers

    # Zero gradients for all modules
    zero_grad_buffer_for_multimodule(module_to_grid_tuple)

    # Schedule dispatch: the colocated path with LLM PP>1 needs the three-phase
    # schedule (encoder full-batch forward → LLM 1F1B pipeline → encoder
    # backward) to avoid deadlocking encoder collectives inside the pipeline
    # staggering. Other cases — non-colocated (any PP layout) and colocated
    # with LLM PP=1 — use the standard schedule; colocated PP=1 works because
    # MimoModel._forward_all_modules runs encoder+communicate+LLM in a single
    # forward on every rank, which is what the standard schedule expects.
    megatron_mimo_model = unwrap_megatron_mimo_model(model)
    is_colocated = megatron_mimo_model.role.mode is ModuleLayout.COLOCATED
    needs_three_phase = is_colocated and megatron_mimo_model.lm_has_pp
    if needs_three_phase:
        # Three-phase schedule integration is deferred. See
        # colocated_forward_backward_with_pp in the submodule and the
        # "Known limitations" section of playground/colocated_mode_plan.md
        # for the scope of work still needed (iterator shape, loss func,
        # p2p_communicator plumbing).
        raise NotImplementedError(
            "Colocated MegatronMIMO with LLM PP>1 requires the three-phase schedule "
            "(colocated_forward_backward_with_pp) which is not yet wired into MB's "
            "train_step. Use LLM PP=1 in colocated mode, or non-colocated mode for PP>1, "
            "until the three-phase integration lands."
        )

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
        megatron_mimo_model = unwrap_megatron_mimo_model(model)
        if megatron_mimo_model.role is None:
            is_last_stage = True
        elif megatron_mimo_model.role.has_language_module:
            is_last_stage = megatron_mimo_model.role.is_last_stage(MIMO_LANGUAGE_MODULE_KEY)

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


def train_megatron_mimo(
    forward_step_func: Callable,
    model: "MimoModel",
    optimizer: "MimoOptimizer",
    schedulers: Dict[str, "OptimizerParamScheduler"],
    train_data_iterator: Iterator,
    valid_data_iterator: Optional[Iterator],
    global_state: GlobalState,
    megatron_mimo_infra: "MegatronMIMOInfra",
    multimodule_communicator: "MultiModulePipelineCommunicator",
    active_module_name: str,
    local_pg_collection: "ProcessGroupCollection",
    checkpoint_manager: Optional[CheckpointManager] = None,
    multimodule_pg_collection: Optional["MultiModuleProcessGroupCollection"] = None,
    module_to_grid_tuple: Optional[List] = None,
) -> None:
    """Main MegatronMIMO training loop.

    Key differences from standard train():
    - Uses MultiModuleProcessGroupCollection for the schedule
    - Uses forward_backward_pipelining_without_interleaving with multimodule support
    - Uses zero_grad_buffer_for_multimodule() for gradient clearing
    - Uses MimoOptimizer for coordinated gradient clipping with global norm

    Reuses from existing Bridge training:
    - GlobalState for timers, config, train_state
    - training_log() for metrics reporting
    - handle_profiling_step() and handle_profiling_stop() for profiler lifecycle
    - save_checkpoint_and_time() / checkpoint_and_decide_exit() for checkpointing
    - evaluate_and_print_results() for validation with multimodule support
    - maybe_finalize_async_save() for async checkpoint finalization

    Args:
        forward_step_func: Forward step function.
        model: MimoModel instance.
        optimizer: MimoOptimizer managing per-module optimizers.
        schedulers: Per-module learning rate schedulers {module_name: scheduler}.
        train_data_iterator: Training data iterator.
        valid_data_iterator: Validation data iterator (optional).
        global_state: GlobalState containing timers, config, train_state.
        megatron_mimo_infra: MegatronMIMOInfra with grids, topology, pg_collections.
        multimodule_communicator: MultiModulePipelineCommunicator for P2P.
        active_module_name: Canonical module name for this rank (from setup). In
            non-colocated mode this is the single active module; in colocated
            mode it defaults to the language module. Used for logging reductions
            and legacy consumers that require a single module identity.
        local_pg_collection: Canonical per-rank ProcessGroupCollection matching
            ``active_module_name`` (from setup). Per-module operations should
            still iterate ``megatron_mimo_infra.pg_collections`` directly.
        checkpoint_manager: CheckpointManager for save operations. Created by
            setup_megatron_mimo(). If None, a DefaultCheckpointManager is created.
        multimodule_pg_collection: Pre-built PG collection for the pipeline schedule.
            If None, built from megatron_mimo_infra.
        module_to_grid_tuple: Pre-built (module, grid) pairs for gradient ops.
            If None, built from model and megatron_mimo_infra.
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

    # Use pre-built objects from setup_megatron_mimo if provided, otherwise build them.
    if module_to_grid_tuple is None:
        module_to_grid_tuple = get_module_to_grid_tuple(model, megatron_mimo_infra)
    if multimodule_pg_collection is None:
        multimodule_pg_collection = build_pg_collection_for_schedule(megatron_mimo_infra)

    # Guard against list fallback - MegatronMIMO training requires MultiModuleProcessGroupCollection
    if isinstance(multimodule_pg_collection, list):
        raise RuntimeError(
            "MultiModuleProcessGroupCollection is required for MegatronMIMO training. "
            "The list-based fallback is not supported. Ensure Megatron-LM PR 3212 is available."
        )

    if checkpoint_manager is None:
        checkpoint_manager = DefaultCheckpointManager(cfg.checkpoint)

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

    logger.info(f"Rank {dist.get_rank()}: Starting MegatronMIMO training loop")

    # Main training loop
    timers("interval-time", log_level=0).start(barrier=True)

    while train_state.step < train_config.train_iters:
        # Finalize any pending async saves (non-blocking). Placed at the top
        # of the loop so async saves get a full iteration to complete.
        checkpoint_manager.finalize_async_saves(
            state=global_state,
            blocking=False,
        )

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
        loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = train_step_megatron_mimo(
            forward_step_func=wrapped_forward_step_func,
            data_iterator=train_data_iterator,
            model=model,
            optimizer=optimizer,
            schedulers=schedulers,
            global_state=global_state,
            multimodule_communicator=multimodule_communicator,
            multimodule_pg_collection=multimodule_pg_collection,
            infra=megatron_mimo_infra,
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

            # Log iteration-time directly for MegatronMIMO models.
            # training_log only logs this inside a hasattr(config.model, "kv_channels")
            # block which MegatronMIMO models don't satisfy, so we log it here as a workaround.
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

        # Checkpointing (interval, signal, duration, exit-interval) and exit decision.
        # TODO: MegatronMIMO FLOPs estimation is non-trivial (heterogeneous modules); pass 0 for now.
        should_exit = checkpoint_and_decide_exit(
            state=global_state,
            model=[model],
            optimizer=optimizer,
            opt_param_scheduler=first_scheduler,
            num_floating_point_operations_so_far=0,
            checkpoint_manager=checkpoint_manager,
            train_data_iterator=train_data_iterator,
            pg_collection=local_pg_collection,
            module_name=active_module_name,
        )
        if should_exit:
            break

    # Stop profiling
    handle_profiling_stop(
        prof_config,
        train_state.step,
        dist.get_rank(),
        prof,
        nsys_nvtx_context,
    )

    timers("interval-time").stop()

    logger.info(f"Rank {dist.get_rank()}: MegatronMIMO training completed")
