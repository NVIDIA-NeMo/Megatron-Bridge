# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Entry point for MIMO pretraining.

This module provides the entry point for MIMO pretraining with heterogeneous
parallelism support. It uses a setup_mimo() helper that composes with existing
setup logic rather than duplicating pretrain.py.

Key components:
- setup_mimo(): MIMO-specific setup helper
- pretrain_mimo(): Entry point for MIMO pretraining
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Optional

import torch.distributed as dist
from megatron.core.models.mimo import MimoModel
from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
from megatron.core.utils import get_model_config

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mimo_parallel_utils import (
    build_pg_collection_for_schedule,
    get_module_to_grid_tuple,
    is_current_rank_in_grid,
    unwrap_mimo_model,
    validate_no_stub_ranks,
)
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.train_mimo import train_mimo


if TYPE_CHECKING:
    from megatron.core.models.mimo.optimizer import MimoOptimizer
    from megatron.core.optimizer.optimizer_param_scheduler import OptimizerParamScheduler
    from megatron.core.process_groups_config import MultiModuleProcessGroupCollection

    from megatron.bridge.models.mimo.mimo_provider import MimoModelInfra, MimoModelProvider


logger = logging.getLogger(__name__)


def _set_mimo_random_seeds(
    cfg: ConfigContainer,
    mimo_infra: "MimoModelInfra",
) -> None:
    """Initialize random seeds with per-module TP/PP awareness.

    Mirrors the standard path's ``_set_random_seed()`` but derives TP/PP ranks
    from the per-module HyperCommGrids instead of global MPU state.

    Must be called **after** ``build_infra()`` (grids exist) and **before**
    ``provide_distributed_model()`` (weight init needs the CUDA RNG tracker).
    """
    import random

    import numpy as np
    import torch
    from megatron.core import tensor_parallel

    seed = cfg.rng.seed

    current_rank = dist.get_rank()

    # Find which module this rank belongs to and get its TP/PP ranks.
    tp_rank = 0
    pp_rank = 0
    for module_name, grid in mimo_infra.module_to_grid_map.items():
        if is_current_rank_in_grid(grid):
            tp_rank = dist.get_group_rank(grid.get_pg(["tp"]), current_rank)
            pp_rank = dist.get_group_rank(grid.get_pg(["pp"]), current_rank)
            break

    # Different PP stages get different seeds (consistent with standard path).
    seed = seed + (100 * pp_rank)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.device_count() > 0:
        tensor_parallel.model_parallel_cuda_manual_seed(seed, tp_rank=tp_rank, ep_rank=0, etp_rank=0)

    logger.info(
        f"Rank {current_rank}: Initialized MIMO random seeds (base_seed={seed}, tp_rank={tp_rank}, pp_rank={pp_rank})"
    )


@dataclass
class MimoSetupOutput:
    """Output from setup_mimo() containing all components needed for training.

    Attributes:
        model: MimoModel (distributed, DDP-wrapped).
        mimo_infra: MimoModelInfra (grids, topology, pg_collections).
        multimodule_pg_collection: PG collection for schedule.
        multimodule_communicator: MultiModulePipelineCommunicator for P2P.
        module_to_grid_tuple: List of (module, grid) tuples for gradient handling.
        optimizer: MimoOptimizer (None when ``build_optimizer=False``).
        schedulers: Per-module LR schedulers (empty when ``build_optimizer=False``).
        train_data_iterator: Training data iterator.
        valid_data_iterator: Validation data iterator (optional).
        global_state: GlobalState containing timers, config, train_state.
    """

    model: "MimoModel"
    mimo_infra: "MimoModelInfra"
    multimodule_pg_collection: "MultiModuleProcessGroupCollection"
    multimodule_communicator: MultiModulePipelineCommunicator
    module_to_grid_tuple: List
    optimizer: Optional["MimoOptimizer"]
    schedulers: Dict[str, "OptimizerParamScheduler"]
    train_data_iterator: Iterator
    valid_data_iterator: Optional[Iterator]
    global_state: GlobalState


def _update_mimo_model_config_funcs(
    model: "MimoModel",
    optimizer: Optional["MimoOptimizer"],
    mimo_infra: "MimoModelInfra",
    module_to_grid_tuple: List,
) -> None:
    """Set model config hooks for MIMO training.

    Mirrors the standard path's ``_update_model_config_funcs`` (in ``setup.py``)
    but uses per-module gradient operations instead of global ones.

    Sets:
    - ``no_sync_func``: per-module ``no_sync`` via ``multimodule_no_sync``
    - ``finalize_model_grads_func``: per-module grad all-reduce via
      ``finalize_model_grads_multimodule``
    - ``grad_scale_func``: loss scaling from ``MimoOptimizer`` (if present)
    """
    from functools import partial

    from megatron.bridge.training.mimo_parallel_utils import (
        finalize_model_grads_multimodule,
        multimodule_no_sync,
    )

    model_config = get_model_config(model)

    model_config.no_sync_func = partial(multimodule_no_sync, module_to_grid_tuple=module_to_grid_tuple)

    model_config.finalize_model_grads_func = partial(
        finalize_model_grads_multimodule,
        infra=mimo_infra,
        module_to_grid_tuple=module_to_grid_tuple,
    )

    if optimizer is not None and hasattr(optimizer, "scale_loss"):
        model_config.grad_scale_func = optimizer.scale_loss

    assert model_config.variable_seq_lengths, (
        "variable_seq_lengths must be True for MIMO training. "
        "This should be set by MimoModelProvider.provide_distributed_model()."
    )


def setup_mimo(
    cfg: ConfigContainer,
    mimo_provider: "MimoModelProvider",
    build_data_iterators_fn: Optional[Callable] = None,
    build_optimizer: bool = True,
    global_state: Optional[GlobalState] = None,
) -> MimoSetupOutput:
    """MIMO-specific setup helper.

    This function sets up all components needed for MIMO training:
    - Builds distributed model via MimoModelProvider
    - Builds MIMO infrastructure (grids, topology, pg_collections)
    - Creates MultiModulePipelineCommunicator
    - Creates MimoOptimizer and per-module LR schedulers (when ``build_optimizer=True``)
    - Builds data iterators (if function provided)
    - Validates configuration

    Args:
        cfg: ConfigContainer with training configuration.  ``cfg.optimizer``
            is used to create the optimizer when ``build_optimizer=True``.
        mimo_provider: MimoModelProvider for building model and infrastructure.
        build_data_iterators_fn: Optional function to build data iterators.
            Should have signature: (cfg, mimo_infra) -> (train_iter, valid_iter)
        build_optimizer: Whether to create optimizer and schedulers.  Set to
            ``False`` for inference or evaluation-only callers.
        global_state: Optional GlobalState. If not provided, creates a new one.

    Returns:
        MimoSetupOutput containing all components for training.

    Reuses from setup.py:
        - Logging setup (via global_state)
        - Timer infrastructure (via global_state)
    """
    # Create GlobalState if not provided
    if global_state is None:
        from megatron.core.timers import Timers

        from megatron.bridge.training.state import GlobalState, TrainState

        timers = Timers(
            log_level=cfg.logger.timing_log_level,
            log_option=cfg.logger.timing_log_option,
        )
        train_state = TrainState()
        global_state = GlobalState()
        global_state.cfg = cfg
        global_state._timers = timers
        global_state.train_state = train_state

    logger.info(f"Rank {dist.get_rank()}: Setting up MIMO training")

    # Finalize and build infrastructure
    mimo_provider.finalize()
    mimo_infra = mimo_provider.build_infra()

    # Validate no stub ranks
    world_size = dist.get_world_size()
    validate_no_stub_ranks(mimo_infra.module_to_grid_map, world_size)

    # Initialize per-module random seeds before model construction.
    # MIMO bypasses initialize_megatron() (to avoid global MPU corruption), which
    # also skips model_parallel_cuda_manual_seed(). Without it, GPU weight init and
    # TP-region dropout crash because CudaRNGStatesTracker is empty. We look up the
    # per-module TP/PP ranks from HyperCommGrids and pass them explicitly.
    _set_mimo_random_seeds(cfg, mimo_infra)

    logger.info(f"Rank {dist.get_rank()}: Building distributed model")

    # Build distributed model
    # Use DDP config from cfg if available
    from megatron.core.distributed import DistributedDataParallelConfig

    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=getattr(cfg.train, "grad_reduce_in_fp32", False),
        overlap_grad_reduce=getattr(cfg.train, "overlap_grad_reduce", True),
        use_distributed_optimizer=getattr(cfg.train, "use_distributed_optimizer", False),
        check_for_nan_in_grad=getattr(cfg.train, "check_for_nan_in_grad", False),
    )

    model_list = mimo_provider.provide_distributed_model(
        ddp_config=ddp_config,
        fp16=cfg.model.fp16 if hasattr(cfg.model, "fp16") else False,
        bf16=cfg.model.bf16 if hasattr(cfg.model, "bf16") else True,
    )
    model = model_list[0]

    logger.info(f"Rank {dist.get_rank()}: Creating multimodule communicator")

    # Create MultiModulePipelineCommunicator
    # IMPORTANT: MimoModel produces SBH tensors (seq, batch, hidden), NOT BSH
    # See MimoModel.align_embeddings_by_token_positions() which returns [s, b, h]
    model_config = get_model_config(model)

    # Ensure pipeline_dtype is set for P2P communication (required when any module uses PP > 1)
    # The model config may not have this set if individual modules don't use PP
    import torch

    if model_config.pipeline_dtype is None:
        if getattr(model_config, "bf16", False):
            model_config.pipeline_dtype = torch.bfloat16
        elif getattr(model_config, "fp16", False):
            model_config.pipeline_dtype = torch.float16
        else:
            model_config.pipeline_dtype = torch.float32

    multimodule_communicator = MultiModulePipelineCommunicator(
        mimo_infra.module_to_grid_map,
        mimo_infra.topology,
        model_config,
        dim_mapping={"s": 0, "b": 1, "h": 2},  # SBH mapping - matches MimoModel output
        module_output_ndim=mimo_infra.module_output_ndim,
    )

    # Build pg_collection for schedule
    multimodule_pg_collection = build_pg_collection_for_schedule(mimo_infra)

    # Build module-to-grid tuple for gradient operations
    module_to_grid_tuple = get_module_to_grid_tuple(model, mimo_infra)

    # Build optimizer and per-module LR schedulers
    optimizer = None
    schedulers: Dict[str, "OptimizerParamScheduler"] = {}
    if build_optimizer:
        unwrapped_model = unwrap_mimo_model(model)
        if mimo_infra.module_to_grid_map:
            assert unwrapped_model.mimo_config.module_to_grid_map is not None, (
                "MimoModelConfig.module_to_grid_map must be set at model construction time. "
                "Ensure MimoModelProvider.provide() passes module_to_grid_map for MIMO parallelism."
            )

        logger.info(f"Rank {dist.get_rank()}: Creating MimoOptimizer")
        from megatron.core.models.mimo.optimizer import get_mimo_optimizer

        opt_config = cfg.optimizer
        if hasattr(opt_config, "finalize"):
            opt_config.finalize()

        optimizer = get_mimo_optimizer(unwrapped_model, opt_config)

        # Auto-create per-module LR schedulers
        cfg._calculate_scheduler_steps()
        from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

        for name, info in optimizer.module_infos.items():
            if info.is_active and info.optimizer is not None:
                schedulers[name] = OptimizerParamScheduler(
                    info.optimizer,
                    init_lr=cfg.scheduler.lr_warmup_init,
                    max_lr=opt_config.lr,
                    min_lr=opt_config.min_lr,
                    lr_warmup_steps=cfg.scheduler.lr_warmup_steps,
                    lr_decay_steps=cfg.scheduler.lr_decay_steps,
                    lr_decay_style=cfg.scheduler.lr_decay_style,
                    start_wd=cfg.scheduler.start_weight_decay,
                    end_wd=cfg.scheduler.end_weight_decay,
                    wd_incr_steps=cfg.scheduler.wd_incr_steps,
                    wd_incr_style=cfg.scheduler.weight_decay_incr_style,
                    use_checkpoint_opt_param_scheduler=cfg.scheduler.use_checkpoint_opt_param_scheduler,
                    override_opt_param_scheduler=cfg.scheduler.override_opt_param_scheduler,
                    wsd_decay_steps=cfg.scheduler.wsd_decay_steps,
                    lr_wsd_decay_style=cfg.scheduler.lr_wsd_decay_style,
                )
        logger.info(f"Rank {dist.get_rank()}: Auto-created schedulers for modules: {list(schedulers.keys())}")

    _update_mimo_model_config_funcs(model, optimizer, mimo_infra, module_to_grid_tuple)

    # Build data iterators if function provided
    train_data_iterator = None
    valid_data_iterator = None
    if build_data_iterators_fn is not None:
        logger.info(f"Rank {dist.get_rank()}: Building data iterators")
        train_data_iterator, valid_data_iterator = build_data_iterators_fn(cfg, mimo_infra)

    logger.info(f"Rank {dist.get_rank()}: MIMO setup complete")

    return MimoSetupOutput(
        model=model,
        mimo_infra=mimo_infra,
        multimodule_pg_collection=multimodule_pg_collection,
        multimodule_communicator=multimodule_communicator,
        module_to_grid_tuple=module_to_grid_tuple,
        optimizer=optimizer,
        schedulers=schedulers,
        train_data_iterator=train_data_iterator,
        valid_data_iterator=valid_data_iterator,
        global_state=global_state,
    )


def pretrain_mimo(
    cfg: ConfigContainer,
    mimo_provider: "MimoModelProvider",
    forward_step_func: Callable,
    build_data_iterators_fn: Callable,
    schedulers: Optional[Dict[str, "OptimizerParamScheduler"]] = None,
    global_state: Optional[GlobalState] = None,
) -> None:
    """Entry point for MIMO pretraining.

    Steps:
    1. Call setup_mimo() to get model, optimizer, schedulers, infra, communicators
    2. Call train_mimo() with all components

    Args:
        cfg: ConfigContainer with training configuration.  ``cfg.optimizer``
            (a ``BridgeOptimizerConfig``, which inherits from MCore's
            ``OptimizerConfig``) is used to create the ``MimoOptimizer``
            and per-module LR schedulers.
        mimo_provider: MimoModelProvider for building model and infrastructure.
        forward_step_func: Forward step function for training.
        build_data_iterators_fn: Function to build data iterators.
            Signature: (cfg, mimo_infra) -> (train_iter, valid_iter)
        schedulers: Per-module learning rate schedulers {module_name: scheduler}.
            If not provided, auto-created from ``cfg.optimizer`` and ``cfg.scheduler``.
        global_state: Optional GlobalState. If not provided, creates a new one.
    """
    logger.info("Starting MIMO pretraining")

    # MIMO: data_parallel_size is always 1 from the training loop's perspective.
    # All ranks load the same global micro-batch; per-module DP sharding is handled
    # by slice_batch_for_mimo() in the forward step, not by the data loader or
    # training loop.  Hard-coding this avoids requiring callers to set it manually
    # and prevents incorrect consumed-sample / scheduler-increment calculations.
    cfg.data_parallel_size = 1

    # Initialize num-microbatches calculator if not already set.
    from megatron.core import num_microbatches_calculator as nmc

    if nmc._GLOBAL_NUM_MICROBATCHES_CALCULATOR is None:
        nmc.init_num_microbatches_calculator(
            dist.get_rank(),
            getattr(cfg.train, "rampup_batch_size", None),
            cfg.train.global_batch_size,
            cfg.train.micro_batch_size,
            cfg.data_parallel_size,
            getattr(cfg.train, "decrease_batch_size_if_needed", False),
        )

    # Setup all MIMO components (model, optimizer, schedulers, data, communicators)
    setup_output = setup_mimo(
        cfg=cfg,
        mimo_provider=mimo_provider,
        build_data_iterators_fn=build_data_iterators_fn,
        build_optimizer=True,
        global_state=global_state,
    )

    # Allow caller-provided schedulers to override auto-created ones
    final_schedulers = schedulers if schedulers else setup_output.schedulers

    logger.info(f"Rank {dist.get_rank()}: Starting training loop")

    # Run training loop
    train_mimo(
        forward_step_func=forward_step_func,
        model=setup_output.model,
        optimizer=setup_output.optimizer,
        schedulers=final_schedulers,
        train_data_iterator=setup_output.train_data_iterator,
        valid_data_iterator=setup_output.valid_data_iterator,
        global_state=setup_output.global_state,
        mimo_infra=setup_output.mimo_infra,
        multimodule_communicator=setup_output.multimodule_communicator,
        multimodule_pg_collection=setup_output.multimodule_pg_collection,
        module_to_grid_tuple=setup_output.module_to_grid_tuple,
    )

    logger.info("MIMO pretraining completed")
