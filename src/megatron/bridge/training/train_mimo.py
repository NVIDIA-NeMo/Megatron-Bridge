# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from megatron.core.pipeline_parallel.schedules import forward_backward_pipelining_without_interleaving
from megatron.core.utils import get_model_config

from megatron.bridge.training.checkpointing import maybe_finalize_async_save, save_checkpoint
from megatron.bridge.training.eval import evaluate_and_print_results
from megatron.bridge.training.mimo_parallel_utils import (
    build_pg_collection_for_schedule,
    get_module_to_grid_tuple,
    multimodule_no_sync,
    finalize_model_grads_multimodule,
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
from megatron.bridge.training.utils.flop_utils import num_floating_point_operations

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
    """Single MIMO training step.
    
    Args:
        forward_step_func: Forward step function (wrapped with GlobalState).
        data_iterator: Iterator over the dataset.
        model: MimoModel instance.
        optimizer: MimoOptimizer managing per-module optimizers.
        schedulers: Per-module learning rate schedulers {module_name: scheduler}.
        global_state: GlobalState containing timers, config, train_state.
        multimodule_communicator: MultiModulePipelineCommunicator for P2P.
        multimodule_pg_collection: PG collection for schedule.
        infra: MimoModelInfra with grids, topology, pg_collections.
        module_to_grid_tuple: List of (module, grid) tuples.
        num_microbatches: Number of microbatches per iteration.
        seq_length: Sequence length.
        micro_batch_size: Micro batch size.
        
    Returns:
        Tuple of (loss_dict, grad_norm, num_zeros_in_grad).
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
    for module_name, scheduler in schedulers.items():
        if scheduler is not None and update_successful:
            scheduler.step()
    
    loss_dict = {}
    if losses_reduced:
        is_last_stage = False
        if model.role is None:
            is_last_stage = True
        elif model.role.has_language_module:
            is_last_stage = model.role.is_last_stage(model.role.language_module_name)

        if is_last_stage:
            llm_pg = infra.pg_collections.get("llm") if infra.pg_collections else None
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

    return loss_dict, grad_norm, num_zeros_in_grad


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
    """Main MIMO training loop.
    
    Key differences from standard train():
    - Creates MultiModuleProcessGroupCollection for the schedule
    - Uses forward_backward_pipelining_without_interleaving with multimodule support
    - Uses zero_grad_buffer_for_multimodule() for gradient clearing
    - Uses MimoOptimizer for coordinated gradient clipping with global norm
    
    Reuses from existing Bridge training:
    - GlobalState for timers, config, train_state
    - training_log() for metrics reporting
    - handle_profiling_step() and handle_profiling_stop() for profiler lifecycle
    - save_checkpoint() with MimoOptimizer for checkpointing
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
        mimo_infra: MimoModelInfra with grids, topology, pg_collections.
        multimodule_communicator: MultiModulePipelineCommunicator for P2P.
    """
    timers = global_state.timers
    train_state = global_state.train_state
    cfg = global_state.cfg
    
    # Get training config
    train_config = cfg.train
    num_microbatches = train_config.num_microbatches
    seq_length = cfg.dataset.seq_length
    micro_batch_size = train_config.micro_batch_size
    
    # Prepare forward step function with GlobalState injection
    wrapped_forward_step_func = prepare_forward_step_func(forward_step_func, global_state)
    
    # Build module-to-grid mapping for gradient operations
    module_to_grid_tuple = get_module_to_grid_tuple(model, mimo_infra)
    
    # Build pg_collection for schedule
    multimodule_pg_collection = build_pg_collection_for_schedule(mimo_infra)
    
    # Configure gradient hooks on model config
    model_config = get_model_config(model)
    
    # Bind custom parameters via partial(), leaving schedule-provided args unbound
    model_config.no_sync_func = partial(
        multimodule_no_sync,
        module_to_grid_tuple=module_to_grid_tuple
    )
    
    model_config.finalize_model_grads_func = partial(
        finalize_model_grads_multimodule,
        infra=mimo_infra,
        module_to_grid_tuple=module_to_grid_tuple,
    )
    
    # Optional: Set grad_scale_func from MimoOptimizer
    if optimizer is not None and hasattr(optimizer, 'scale_loss'):
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
    
    # Get first scheduler for checkpoint saving (schedulers remain per-module)
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
        loss_dict, grad_norm, num_zeros_in_grad = train_step_mimo(
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
        train_state.consumed_train_samples += (
            micro_batch_size * num_microbatches * cfg.data_parallel_size
        )
        
        # Get learning rate from first scheduler
        learning_rate = None
        if schedulers:
            sched = next(iter(schedulers.values()))
            if sched is not None:
                learning_rate = sched.get_lr()
        
        # Get loss scale from MimoOptimizer
        loss_scale = 1.0
        if optimizer is not None:
            loss_scale = optimizer.get_loss_scale()
        
        # Log training metrics
        report_memory_flag = training_log(
            loss_dict=loss_dict,
            total_loss_dict=total_loss_dict,
            learning_rate=learning_rate,
            decoupled_learning_rate=None,
            loss_scale=loss_scale,
            report_memory_flag=report_memory_flag,
            skipped_iter=0,
            grad_norm=grad_norm,
            params_norm=None,
            num_zeros_in_grad=num_zeros_in_grad,
            config=cfg,
            global_state=global_state,
            history_wct=history_wct,
            model=[model],
        )
        
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
        if (
            cfg.checkpoint.save_interval is not None
            and train_state.step % cfg.checkpoint.save_interval == 0
        ):
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
