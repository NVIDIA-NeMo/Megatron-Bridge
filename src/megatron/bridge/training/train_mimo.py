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
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Optional

import torch
import torch.distributed as dist
from megatron.core.pipeline_parallel.schedules import forward_backward_pipelining_without_interleaving
from megatron.core.utils import get_model_config

from megatron.bridge.training.mimo_parallel_utils import (
    build_pg_collection_for_schedule,
    finalize_model_grads_multimodule,
    get_module_to_grid_tuple,
    multimodule_no_sync,
    zero_grad_buffer_for_multimodule,
)
from megatron.bridge.training.profiling import handle_profiling_step, handle_profiling_stop
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.train_utils import (
    prepare_forward_step_func,
)


if TYPE_CHECKING:
    from megatron.core.models.mimo import MimoModel
    from megatron.core.optimizer import MegatronOptimizer
    from megatron.core.optimizer.optimizer_param_scheduler import OptimizerParamScheduler
    from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator

    from megatron.bridge.models.mimo.mimo_provider import MimoModelInfra


logger = logging.getLogger(__name__)


def train_step_mimo(
    forward_step_func: Callable,
    data_iterator: Iterator,
    model: "MimoModel",
    optimizer: "MegatronOptimizer",
    schedulers: Dict[str, "OptimizerParamScheduler"],
    global_state: GlobalState,
    multimodule_communicator: "MultiModulePipelineCommunicator",
    multimodule_pg_collection,
    infra: "MimoModelInfra",
    module_to_grid_tuple: List,
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
) -> Dict[str, torch.Tensor]:
    """Single MIMO training step.

    Args:
        forward_step_func: Forward step function (wrapped with GlobalState).
        data_iterator: Iterator over the dataset.
        model: MimoModel instance.
        optimizer: MimoOptimizer (handles per-module dispatch internally).
        schedulers: Learning rate schedulers dict (can be empty).
        global_state: GlobalState containing timers, config, train_state.
        multimodule_communicator: MultiModulePipelineCommunicator for P2P.
        multimodule_pg_collection: PG collection for schedule.
        infra: MimoModelInfra with grids, topology, pg_collections.
        module_to_grid_tuple: List of (module, grid) tuples.
        num_microbatches: Number of microbatches per iteration.
        seq_length: Sequence length.
        micro_batch_size: Micro batch size.

    Returns:
        Dictionary of reduced losses.
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

    # Optimizer step (MimoOptimizer handles per-module dispatch)
    timers("optimizer", log_level=1).start(barrier=False)

    # MimoOptimizer.step() -> Tuple[bool, Optional[float], Optional[int]]
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()

    timers("optimizer").stop()

    # Step learning rate schedulers
    for module_name, scheduler in schedulers.items():
        if scheduler is not None and update_successful:
            scheduler.step()

    # losses_reduced is a list of dicts (one per microbatch). Merge into single dict.
    if not losses_reduced:
        return {}
    merged = {}
    for d in losses_reduced:
        if isinstance(d, dict):
            merged.update(d)
    return merged


def train_mimo(
    forward_step_func: Callable,
    model: "MimoModel",
    optimizer: "MegatronOptimizer",
    schedulers: Dict[str, "OptimizerParamScheduler"],
    train_data_iterator: Iterator,
    valid_data_iterator: Optional[Iterator],
    global_state: GlobalState,
    mimo_infra: "MimoModelInfra",
    multimodule_communicator: "MultiModulePipelineCommunicator",
    checkpointing_context: Optional[dict] = None,
) -> None:
    """Main MIMO training loop.

    Key differences from standard train():
    - Creates MultiModuleProcessGroupCollection for the schedule
    - Uses forward_backward_pipelining_without_interleaving with multimodule support
    - Uses zero_grad_buffer_for_multimodule() for gradient clearing
    - Single MimoOptimizer handles per-module dispatch internally

    Note: Stub ranks are disallowed - validated at setup time.

    Reuses from existing Bridge training:
    - GlobalState for timers, config, train_state
    - training_log() for metrics reporting
    - handle_profiling_step() AND handle_profiling_stop() for full profiler lifecycle
    - num_floating_point_operations() for throughput calculations
    - prepare_forward_step_func() for GlobalState injection into forward_step

    Args:
        forward_step_func: Forward step function.
        model: MimoModel instance.
        optimizer: MimoOptimizer (handles per-module dispatch internally).
        schedulers: Learning rate schedulers dict (can be empty).
        train_data_iterator: Training data iterator.
        valid_data_iterator: Validation data iterator (optional).
        global_state: GlobalState containing timers, config, train_state.
        mimo_infra: MimoModelInfra with grids, topology, pg_collections.
        multimodule_communicator: MultiModulePipelineCommunicator for P2P.
        checkpointing_context: Checkpointing context (optional, for Phase 5).
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

    # Extract language module key from model
    _model = model.module if hasattr(model, "module") else model
    lang_key = _model.mimo_config.language_module_key

    # Compute dp_size from the language module's grid shape (works on all ranks,
    # including those that don't participate in the language module, because
    # grid.shape is a plain list—not a process group handle).
    lang_grid = mimo_infra.module_to_grid_map.get(lang_key)
    dp_size = lang_grid.shape[lang_grid.dim_names.index("dp")] if lang_grid else 1

    # Build module-to-grid mapping for gradient operations
    module_to_grid_tuple = get_module_to_grid_tuple(model, mimo_infra)

    # Build pg_collection for schedule
    multimodule_pg_collection = build_pg_collection_for_schedule(
        mimo_infra,
        language_module_key=lang_key,
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

    # Optional: Set grad_scale_func from optimizer
    if optimizer is not None and hasattr(optimizer, "scale_loss"):
        model_config.grad_scale_func = optimizer.scale_loss

    # Validation: variable_seq_lengths should already be True (set by MimoModelProvider)
    assert model_config.variable_seq_lengths, (
        "variable_seq_lengths must be True for MIMO training. "
        "This should be set by MimoModelProvider.provide_distributed_model()."
    )

    logger.info(f"Rank {dist.get_rank()}: Starting MIMO training loop")

    # Main training loop
    timers("interval-time", log_level=0).start(barrier=True)

    while train_state.step < train_config.train_iters:
        # Handle profiling
        handle_profiling_step(
            config=global_state.cfg.profiling if hasattr(global_state.cfg, "profiling") else None,
            iteration=train_state.step,
            rank=dist.get_rank(),
            pytorch_prof=None,
        )

        # Start iteration timer
        timers("iteration-time", log_level=0).start(barrier=False)

        # Run single training step
        loss_dict = train_step_mimo(
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
        timers("iteration-time").stop()
        iteration_time = timers("iteration-time").elapsed(barrier=False)

        # Update training state
        train_state.step += 1
        train_state.consumed_train_samples += micro_batch_size * num_microbatches * dp_size

        # Get learning rate from first scheduler
        learning_rate = None
        if schedulers:
            first_scheduler = next(iter(schedulers.values()))
            if first_scheduler is not None:
                learning_rate = first_scheduler.get_lr()

        # Get loss scale from optimizer
        loss_scale = 1.0
        if optimizer is not None and hasattr(optimizer, "get_loss_scale"):
            loss_scale = optimizer.get_loss_scale()

        # Log training metrics (simple MIMO-specific logging;
        # training_log() assumes config.model is a TransformerConfig which
        # doesn't hold for MimoModelProvider)
        loss_str = ", ".join(f"{k}: {v}" for k, v in loss_dict.items()) if loss_dict else "n/a"
        logger.info(
            f"Rank {dist.get_rank()} | step {train_state.step} | "
            f"loss: {loss_str} | iter_time: {iteration_time:.3f}s | "
            f"lr: {learning_rate} | loss_scale: {loss_scale}"
        )

        # TODO: Add checkpointing logic (Phase 5)
        # TODO: Add evaluation logic

    # Stop profiling
    handle_profiling_stop(
        config=global_state.cfg.profiling if hasattr(global_state.cfg, "profiling") else None,
        iteration=train_state.step,
        rank=dist.get_rank(),
        pytorch_prof=None,
    )

    timers("interval-time").stop()

    logger.info(f"Rank {dist.get_rank()}: MIMO training completed")
