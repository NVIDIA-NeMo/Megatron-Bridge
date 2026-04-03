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
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional

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
    from megatron.core.optimizer.optimizer_config import OptimizerConfig
    from megatron.core.optimizer.optimizer_param_scheduler import OptimizerParamScheduler

    from megatron.bridge.models.mimo.mimo_provider import MimoModelInfra, MimoModelProvider


logger = logging.getLogger(__name__)


def _set_mimo_random_seeds(
    cfg: ConfigContainer,
    mimo_infra: "MimoModelInfra",
) -> None:
    """
    Set random seeds for Python, NumPy, PyTorch, and (if available) CUDA using the
    tensor-parallel (TP) and pipeline-parallel (PP) ranks derived from the MIMO
    module grids.
    
    This function reads the base seed from `cfg.seed` or `cfg.rng.seed` (default 1234),
    determines the TP and PP ranks for the current process by inspecting
    `mimo_infra.module_to_grid_map`, offsets the base seed by `100 * pp_rank`, and
    applies the resulting seed to Python `random`, `numpy.random`, `torch.manual_seed`,
    and the Megatron tensor-parallel CUDA RNG initializer.
    
    Parameters:
        cfg: Configuration container exposing `seed` or `rng.seed`.
        mimo_infra: MIMO infrastructure containing `module_to_grid_map` used to
            determine per-module TP/PP ranks.
    
    """
    import random

    import numpy as np
    import torch
    from megatron.core import tensor_parallel

    seed = getattr(cfg, "seed", None) or getattr(getattr(cfg, "rng", None), "seed", None) or 1234

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
        train_data_iterator: Training data iterator.
        valid_data_iterator: Validation data iterator (optional).
        global_state: GlobalState containing timers, config, train_state.
    """

    model: "MimoModel"
    mimo_infra: "MimoModelInfra"
    multimodule_pg_collection: Any
    multimodule_communicator: MultiModulePipelineCommunicator
    module_to_grid_tuple: List
    train_data_iterator: Iterator
    valid_data_iterator: Optional[Iterator]
    global_state: GlobalState


def setup_mimo(
    cfg: ConfigContainer,
    mimo_provider: "MimoModelProvider",
    build_data_iterators_fn: Optional[Callable] = None,
    global_state: Optional[GlobalState] = None,
) -> MimoSetupOutput:
    """
    Set up all components required for MIMO pretraining and return them as a MimoSetupOutput.
    
    This initializes GlobalState if absent, finalizes and builds MIMO infrastructure, seeds RNGs per module grid, constructs the distributed model and the MultiModulePipelineCommunicator, prepares scheduling/gradient helper structures, and optionally builds train/validation data iterators.
    
    Parameters:
    	cfg (ConfigContainer): Global configuration container.
    	mimo_provider (MimoModelProvider): Provider responsible for finalizing and constructing MIMO model and infra.
    	build_data_iterators_fn (Optional[Callable]): Optional callable with signature (cfg, mimo_infra) -> (train_iter, valid_iter) to create data iterators.
    	global_state (Optional[GlobalState]): Pre-existing GlobalState to reuse; if omitted a new GlobalState is created.
    
    Returns:
    	MimoSetupOutput: Container with the constructed model, mimo_infra, multimodule_pg_collection, multimodule_communicator, module_to_grid_tuple, train/valid iterators (may be None), and the GlobalState.
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
        train_data_iterator=train_data_iterator,
        valid_data_iterator=valid_data_iterator,
        global_state=global_state,
    )


def pretrain_mimo(
    cfg: ConfigContainer,
    mimo_provider: "MimoModelProvider",
    forward_step_func: Callable,
    build_data_iterators_fn: Callable,
    opt_config: "OptimizerConfig",
    schedulers: Optional[Dict[str, "OptimizerParamScheduler"]] = None,
    global_state: Optional[GlobalState] = None,
) -> None:
    """
    Orchestrate MIMO pretraining: prepare infrastructure and model, create optimizer and schedulers, and run the training loop.
    
    Sets up MIMO infrastructure and distributed model via the provided provider, constructs a MIMO optimizer from the unwrapped model and the given optimizer configuration, optionally auto-creates per-module learning-rate schedulers when `schedulers` is empty, and executes the training loop using the supplied forward step and data iterators.
    
    Parameters:
        cfg: Configuration container with training, model, scheduler, and logging settings.
        mimo_provider: Provider responsible for finalizing and constructing the MIMO model and infra.
        forward_step_func: Callable that performs a single forward/backward step for training.
        build_data_iterators_fn: Callable used to build data iterators. Signature: (cfg, mimo_infra) -> (train_iter, valid_iter).
        opt_config: Optimizer configuration used to construct the MIMO optimizer; if it exposes `finalize()`, that will be invoked.
        schedulers: Optional mapping of module name to OptimizerParamScheduler; if empty or None, per-module schedulers are created automatically.
        global_state: Optional GlobalState to use for timers and training state; if omitted, a new GlobalState is created by setup.
    """
    if schedulers is None:
        schedulers = {}

    logger.info("Starting MIMO pretraining")

    # Ensure optimizer config computes derived fields expected by core optimizers.
    if hasattr(opt_config, "finalize"):
        opt_config.finalize()

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

    # Setup MIMO components
    setup_output = setup_mimo(
        cfg=cfg,
        mimo_provider=mimo_provider,
        build_data_iterators_fn=build_data_iterators_fn,
        global_state=global_state,
    )

    # Unwrap Float16Module/DDP wrapper to access mimo_config on the underlying MimoModel
    unwrapped_model = unwrap_mimo_model(setup_output.model)
    if setup_output.mimo_infra.module_to_grid_map:
        # Role/materialization decisions happen in MimoModel.__init__. Provider wiring
        # must pass these fields at construction time, not by mutating afterwards.
        assert unwrapped_model.mimo_config.module_to_grid_map is not None, (
            "MimoModelConfig.module_to_grid_map must be set at model construction time. "
            "Ensure MimoModelProvider.provide() passes module_to_grid_map for MIMO parallelism."
        )
    logger.info(f"Rank {dist.get_rank()}: Creating MimoOptimizer")

    # Create MimoOptimizer using the factory function
    # Note: get_mimo_optimizer needs the unwrapped MimoModel to access mimo_config and submodules
    from megatron.core.models.mimo.optimizer import get_mimo_optimizer

    optimizer = get_mimo_optimizer(unwrapped_model, opt_config)

    # Auto-create per-module LR schedulers when none are provided
    if not schedulers:
        cfg._calculate_scheduler_steps()

        from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

        schedulers = {}
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

    logger.info(f"Rank {dist.get_rank()}: Starting training loop")

    # Run training loop
    train_mimo(
        forward_step_func=forward_step_func,
        model=setup_output.model,
        optimizer=optimizer,
        schedulers=schedulers,
        train_data_iterator=setup_output.train_data_iterator,
        valid_data_iterator=setup_output.valid_data_iterator,
        global_state=setup_output.global_state,
        mimo_infra=setup_output.mimo_infra,
        multimodule_communicator=setup_output.multimodule_communicator,
    )

    logger.info("MIMO pretraining completed")
