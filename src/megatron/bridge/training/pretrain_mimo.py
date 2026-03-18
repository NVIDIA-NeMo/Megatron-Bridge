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

import torch
import torch.distributed as dist
from megatron.core.models.mimo import MimoModel
from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
from megatron.core.utils import get_model_config

from megatron.bridge.training.checkpointing import init_checkpointing_context, load_checkpoint
from megatron.bridge.training.utils.checkpoint_utils import checkpoint_exists
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mimo_parallel_utils import (
    build_pg_collection_for_schedule,
    get_module_to_grid_tuple,
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
        checkpointing_context: Dictionary holding checkpoint-related state
            (save strategy cache, LocalCheckpointManager for local saves).
    """

    model: "MimoModel"
    mimo_infra: "MimoModelInfra"
    multimodule_pg_collection: Any
    multimodule_communicator: MultiModulePipelineCommunicator
    module_to_grid_tuple: List
    train_data_iterator: Iterator
    valid_data_iterator: Optional[Iterator]
    global_state: GlobalState
    checkpointing_context: Dict[str, Any]


def setup_mimo(
    cfg: ConfigContainer,
    mimo_provider: "MimoModelProvider",
    build_data_iterators_fn: Optional[Callable] = None,
    global_state: Optional[GlobalState] = None,
) -> MimoSetupOutput:
    """MIMO-specific setup helper.

    This function sets up all components needed for MIMO training:
    - Builds distributed model via MimoModelProvider
    - Builds MIMO infrastructure (grids, topology, pg_collections)
    - Creates MultiModulePipelineCommunicator
    - Builds data iterators (if function provided)
    - Validates configuration

    Args:
        cfg: ConfigContainer with training configuration.
        mimo_provider: MimoModelProvider for building model and infrastructure.
        build_data_iterators_fn: Optional function to build data iterators.
            Should have signature: (cfg, mimo_infra) -> (train_iter, valid_iter)
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
        global_state._timers = timers
        global_state.train_state = train_state

    global_state.cfg = cfg

    logger.info(f"Rank {dist.get_rank()}: Setting up MIMO training")

    # Finalize and build infrastructure
    mimo_provider.finalize()
    mimo_infra = mimo_provider.build_infra()

    # Validate no stub ranks
    world_size = dist.get_world_size()
    validate_no_stub_ranks(mimo_infra.module_to_grid_map, world_size)

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

    # Initialize async checkpoint worker (idempotent if already initialized).
    global_state.initialize_async_checkpoint_worker()

    # Initialize checkpointing context (save strategy cache + LocalCheckpointManager).
    checkpointing_context = init_checkpointing_context(cfg.checkpoint)

    # Align start_time across ranks so duration-based exit is consistent.
    start_time_tensor = torch.tensor([global_state.start_time], dtype=torch.double, device="cuda")
    dist.all_reduce(start_time_tensor, op=dist.ReduceOp.MIN)
    global_state.start_time = start_time_tensor.item()

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
        checkpointing_context=checkpointing_context,
    )


def pretrain_mimo(
    cfg: ConfigContainer,
    mimo_provider: "MimoModelProvider",
    forward_step_func: Callable,
    build_data_iterators_fn: Callable,
    opt_config: "OptimizerConfig",
    schedulers: Optional[Dict[str, "OptimizerParamScheduler"]] = None,
    global_state: Optional[GlobalState] = None,
    save_initial_checkpoint: bool = False,
) -> None:
    """Entry point for MIMO pretraining.

    Steps:
    1. Call setup_mimo() to get model, infra, communicators
    2. Validate constructor-time MIMO config wiring
    3. Create MimoOptimizer using get_mimo_optimizer()
    4. Call train_mimo() with all components

    Args:
        cfg: ConfigContainer with training configuration.
        mimo_provider: MimoModelProvider for building model and infrastructure.
        forward_step_func: Forward step function for training.
        build_data_iterators_fn: Function to build data iterators.
            Signature: (cfg, mimo_infra) -> (train_iter, valid_iter)
        opt_config: OptimizerConfig for creating MimoOptimizer.
        schedulers: Per-module learning rate schedulers {module_name: scheduler}.
        global_state: Optional GlobalState. If not provided, creates a new one.
        save_initial_checkpoint: If True, save a checkpoint at step 0 before training.
    """
    if schedulers is None:
        schedulers = {}

    logger.info("Starting MIMO pretraining")

    # Ensure optimizer config computes derived fields expected by core optimizers.
    if hasattr(opt_config, "finalize"):
        opt_config.finalize()

    # Initialize num-microbatches calculator if not already set.
    from megatron.core import num_microbatches_calculator as nmc

    rampup_batch_size = getattr(cfg.train, "rampup_batch_size", None)
    assert rampup_batch_size is None, (
        "Microbatch rampup is not supported in MiMo training. "
        "Set rampup_batch_size to None."
    )

    if nmc._GLOBAL_NUM_MICROBATCHES_CALCULATOR is None:
        nmc.init_num_microbatches_calculator(
            dist.get_rank(),
            rampup_batch_size,
            cfg.train.global_batch_size,
            cfg.train.micro_batch_size,
            cfg.data_parallel_size,
            getattr(cfg.train, "decrease_batch_size_if_needed", False),
        )

    # Setup MIMO components (iterators deferred until after checkpoint load)
    setup_output = setup_mimo(
        cfg=cfg,
        mimo_provider=mimo_provider,
        build_data_iterators_fn=None,
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
        assert unwrapped_model.mimo_config.language_module_key is not None, (
            "MimoModelConfig.language_module_key must be set at model construction time. "
            "Ensure MimoModelProvider.provide() sets language_module_key for MIMO parallelism."
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

    # Select rank-local PG collection for non-colocated MiMo.
    # Each rank participates in exactly one module, so "first non-None" is unambiguous.
    active_pgs = [pg for pg in setup_output.mimo_infra.pg_collections.values() if pg is not None]
    assert len(active_pgs) == 1, (
        f"Non-colocated MiMo requires exactly one active ProcessGroupCollection per rank, "
        f"got {len(active_pgs)}. Colocated MiMo is not supported by this code path."
    )
    local_pg_collection = active_pgs[0]

    # Bridge MiMo's per-module process groups into Megatron's global parallel
    # state.  MiMo intentionally skips global MPU init (see
    # MimoModelProvider.initialize_model_parallel), but checkpoint save/load
    # paths (sharded_state_dict, ensure_metadata_has_dp_cp_group) rely on the
    # globals.  For non-colocated MiMo every rank is active in exactly one
    # module, so we can safely set the globals from that module's collection.
    from megatron.core import parallel_state as mpu

    mpu._TENSOR_MODEL_PARALLEL_GROUP = local_pg_collection.tp
    mpu._DATA_PARALLEL_GROUP = local_pg_collection.dp
    mpu._DATA_PARALLEL_GROUP_WITH_CP = getattr(local_pg_collection, "dp_cp", local_pg_collection.dp)
    if hasattr(local_pg_collection, "pp"):
        mpu._PIPELINE_MODEL_PARALLEL_GROUP = local_pg_collection.pp

    first_scheduler = next(iter(schedulers.values()), None) if schedulers else None

    # Broadened load-intent gating: includes non-persistent resume intent
    has_persistent = cfg.checkpoint.load is not None and checkpoint_exists(cfg.checkpoint.load)
    has_pretrained = (
        cfg.checkpoint.pretrained_checkpoint is not None
        and checkpoint_exists(cfg.checkpoint.pretrained_checkpoint)
    )
    wants_non_persistent = cfg.checkpoint.non_persistent_ckpt_type is not None
    should_load = has_persistent or has_pretrained or wants_non_persistent

    if should_load:
        timers = setup_output.global_state.timers
        timers("load-checkpoint", log_level=0).start(barrier=True)
        load_checkpoint(
            setup_output.global_state,
            model=[setup_output.model],
            optimizer=optimizer,
            opt_param_scheduler=first_scheduler,
            checkpointing_context=setup_output.checkpointing_context,
            pg_collection=local_pg_collection,
        )
        timers("load-checkpoint").stop(barrier=True)
        timers.log(["load-checkpoint"])

        # Fan out loaded scheduler state to all active module schedulers.
        # v1: checkpoints contain a single scheduler blob (first_scheduler).
        if first_scheduler is not None and len(schedulers) > 1:
            loaded_state = first_scheduler.state_dict()
            for sched in schedulers.values():
                if sched is not first_scheduler:
                    sched.load_state_dict(loaded_state)

    # Build data iterators after load decision (resume-safe ordering).
    # When resuming, train_state has restored consumed-sample offsets that
    # the iterator builder must honor to avoid replaying data from sample 0.
    train_state = setup_output.global_state.train_state
    is_resuming = train_state.step > 0

    if is_resuming:
        import inspect

        sig = inspect.signature(build_data_iterators_fn)
        if "train_state" in sig.parameters:
            train_data_iterator, valid_data_iterator = build_data_iterators_fn(
                cfg, setup_output.mimo_infra, train_state=train_state,
            )
        else:
            raise RuntimeError(
                "Resuming from checkpoint but build_data_iterators_fn does not accept "
                "'train_state' argument. The iterator builder must support a train_state "
                "keyword argument to honor restored consumed-sample offsets during resume."
            )
    else:
        train_data_iterator, valid_data_iterator = build_data_iterators_fn(cfg, setup_output.mimo_infra)

    # Load checkpoint if configured
    from megatron.bridge.training.checkpointing import load_checkpoint
    from megatron.bridge.training.utils.checkpoint_utils import checkpoint_exists

    should_load_checkpoint = cfg.checkpoint.load is not None and checkpoint_exists(cfg.checkpoint.load)
    if should_load_checkpoint:
        timers = setup_output.global_state._timers
        # Use first scheduler (all modules share the same LR schedule)
        first_scheduler = next(iter(schedulers.values()), None) if schedulers else None
        timers("load-checkpoint", log_level=0).start(barrier=True)
        load_checkpoint(
            setup_output.global_state,
            [setup_output.model],
            optimizer,
            first_scheduler,
        )
        timers("load-checkpoint").stop(barrier=True)
        timers.log(["load-checkpoint"])
        logger.info(f"Rank {dist.get_rank()}: Checkpoint loaded from {cfg.checkpoint.load}")

    logger.info(f"Rank {dist.get_rank()}: Starting training loop")

    # Run training loop
    train_mimo(
        forward_step_func=forward_step_func,
        model=setup_output.model,
        optimizer=optimizer,
        schedulers=schedulers,
        train_data_iterator=train_data_iterator,
        valid_data_iterator=valid_data_iterator,
        global_state=setup_output.global_state,
        mimo_infra=setup_output.mimo_infra,
        multimodule_communicator=setup_output.multimodule_communicator,
        checkpointing_context=setup_output.checkpointing_context,
        save_initial_checkpoint=save_initial_checkpoint,
    )

    logger.info("MIMO pretraining completed")
