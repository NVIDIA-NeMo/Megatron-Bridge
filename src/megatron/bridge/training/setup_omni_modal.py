# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""OmniModal-specific setup for heterogeneous multi-module training.

This module provides the setup logic for OmniModal training, mirroring the standard
``setup.py`` but adapted for per-module parallelism.

Key components:
- setup_omni_modal(): OmniModal-specific setup helper (analogous to setup())
- _set_omni_modal_random_seeds(): Per-module TP/PP seed initialization
- _update_omni_modal_model_config_funcs(): Model config hooks (analogous to _update_model_config_funcs)
- OmniModalSetupOutput: Dataclass containing all setup outputs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Optional

import torch.distributed as dist
from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
from megatron.core.utils import get_model_config

from megatron.bridge.training.checkpointing import CheckpointManager, create_checkpoint_manager, load_checkpoint
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.omni_modal_parallel_utils import (
    build_pg_collection_for_schedule,
    get_active_module_pg,
    get_module_to_grid_tuple,
    is_current_rank_in_grid,
    unwrap_omni_modal_model,
    validate_no_stub_ranks,
)
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.checkpoint_utils import checkpoint_exists


if TYPE_CHECKING:
    from megatron.core.models.mimo import MimoModel
    from megatron.core.models.mimo.optimizer import MimoOptimizer
    from megatron.core.optimizer.optimizer_param_scheduler import OptimizerParamScheduler
    from megatron.core.process_groups_config import MultiModuleProcessGroupCollection, ProcessGroupCollection

    from megatron.bridge.models.omni_modal.omni_modal_provider import OmniModalInfra


logger = logging.getLogger(__name__)


def _set_omni_modal_random_seeds(
    cfg: ConfigContainer,
    omni_modal_infra: "OmniModalInfra",
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
    for module_name, grid in omni_modal_infra.module_to_grid_map.items():
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
        f"Rank {current_rank}: Initialized OmniModal random seeds (base_seed={seed}, tp_rank={tp_rank}, pp_rank={pp_rank})"
    )


@dataclass
class OmniModalSetupOutput:
    """Output from setup_omni_modal() containing all components needed for training.

    Attributes:
        model: MimoModel (distributed, DDP-wrapped).
        omni_modal_infra: OmniModalInfra (grids, topology, pg_collections).
        multimodule_pg_collection: PG collection for schedule.
        multimodule_communicator: MultiModulePipelineCommunicator for P2P.
        module_to_grid_tuple: List of (module, grid) tuples for gradient handling.
        optimizer: MimoOptimizer.
        schedulers: Per-module LR schedulers.
        train_data_iterator: Training data iterator.
        valid_data_iterator: Validation data iterator (optional).
        global_state: GlobalState containing timers, config, train_state.
    """

    model: "MimoModel"
    omni_modal_infra: "OmniModalInfra"
    multimodule_pg_collection: "MultiModuleProcessGroupCollection"
    multimodule_communicator: "MultiModulePipelineCommunicator"
    module_to_grid_tuple: List
    optimizer: "MimoOptimizer"
    schedulers: Dict[str, "OptimizerParamScheduler"]
    train_data_iterator: Iterator
    valid_data_iterator: Optional[Iterator]
    global_state: GlobalState
    checkpoint_manager: CheckpointManager
    active_module_name: str
    local_pg_collection: "ProcessGroupCollection"


def _update_omni_modal_model_config_funcs(
    model: "MimoModel",
    optimizer: "MimoOptimizer",
    omni_modal_infra: "OmniModalInfra",
    module_to_grid_tuple: List,
) -> None:
    """Set model config hooks for OmniModal training.

    Mirrors the standard path's ``_update_model_config_funcs`` (in ``setup.py``)
    but uses per-module gradient operations instead of global ones.

    Sets:
    - ``no_sync_func``: per-module ``no_sync`` via ``multimodule_no_sync``
    - ``finalize_model_grads_func``: per-module grad all-reduce via
      ``finalize_model_grads_multimodule``
    - ``grad_scale_func``: loss scaling from ``MimoOptimizer``
    """
    from functools import partial

    from megatron.bridge.training.omni_modal_parallel_utils import (
        finalize_model_grads_multimodule,
        multimodule_no_sync,
    )

    model_config = get_model_config(model)

    model_config.no_sync_func = partial(multimodule_no_sync, module_to_grid_tuple=module_to_grid_tuple)

    model_config.finalize_model_grads_func = partial(
        finalize_model_grads_multimodule,
        infra=omni_modal_infra,
        module_to_grid_tuple=module_to_grid_tuple,
    )

    if hasattr(optimizer, "scale_loss"):
        model_config.grad_scale_func = optimizer.scale_loss

    assert model_config.variable_seq_lengths, (
        "variable_seq_lengths must be True for OmniModal training. "
        "This should be set by OmniModalProvider.provide_distributed_model()."
    )


def setup_omni_modal(
    state: GlobalState,
    build_data_iterators_fn: Optional[Callable] = None,
) -> OmniModalSetupOutput:
    """OmniModal-specific setup helper.

    This function sets up all components needed for OmniModal training:
    - Builds distributed model via ``cfg.model`` (an ``OmniModalProvider``)
    - Builds OmniModal infrastructure (grids, topology, pg_collections)
    - Creates MultiModulePipelineCommunicator
    - Creates MimoOptimizer and per-module LR schedulers
    - Loads checkpoint (if one exists)
    - Builds data iterators (if function provided, after checkpoint load)
    - Validates configuration

    Args:
        state: GlobalState with ``state.cfg`` already set.  ``state.cfg.model``
            must be an ``OmniModalProvider``.  ``state.cfg.optimizer`` is used to
            create the optimizer.
        build_data_iterators_fn: Optional function to build data iterators.
            Should have signature: (cfg, omni_modal_infra) -> (train_iter, valid_iter)

    Returns:
        OmniModalSetupOutput containing all components for training.
    """
    cfg = state.cfg
    global_state = state

    logger.info(f"Rank {dist.get_rank()}: Setting up OmniModal training")

    # Initialize num-microbatches calculator (standard path does this in initialize_megatron).
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

    # Finalize provider and build infrastructure.
    # cfg.model.finalize() is called here (not in omni_modal_runtime_config_update)
    # because it validates parallelism config which depends on distributed state.
    omni_modal_provider = cfg.model
    omni_modal_provider.finalize()
    omni_modal_infra = omni_modal_provider.build_infra()

    # Validate no stub ranks
    world_size = dist.get_world_size()
    validate_no_stub_ranks(omni_modal_infra.module_to_grid_map, world_size)

    # Initialize per-module random seeds before model construction.
    # OmniModal bypasses initialize_megatron() (to avoid global MPU corruption), which
    # also skips model_parallel_cuda_manual_seed(). Without it, GPU weight init and
    # TP-region dropout crash because CudaRNGStatesTracker is empty. We look up the
    # per-module TP/PP ranks from HyperCommGrids and pass them explicitly.
    _set_omni_modal_random_seeds(cfg, omni_modal_infra)

    logger.info(f"Rank {dist.get_rank()}: Building distributed model")

    # Use cfg.ddp directly (already finalized by omni_modal_runtime_config_update).
    # Matches the standard path which passes cfg.ddp to provide_distributed_model.
    model_list = omni_modal_provider.provide_distributed_model(
        ddp_config=cfg.ddp,
        fp16=getattr(cfg.model, "fp16", False),
        bf16=getattr(cfg.model, "bf16", True),
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
        omni_modal_infra.module_to_grid_map,
        omni_modal_infra.topology,
        model_config,
        dim_mapping={"s": 0, "b": 1, "h": 2},  # SBH mapping - matches MimoModel output
        module_output_ndim=omni_modal_infra.module_output_ndim,
    )

    # Build pg_collection for schedule
    multimodule_pg_collection = build_pg_collection_for_schedule(omni_modal_infra)

    # Build module-to-grid tuple for gradient operations
    module_to_grid_tuple = get_module_to_grid_tuple(model, omni_modal_infra)

    # Build optimizer and per-module LR schedulers
    unwrapped_model = unwrap_omni_modal_model(model)
    if omni_modal_infra.module_to_grid_map:
        assert unwrapped_model.mimo_config.module_to_grid_map is not None, (
            "MimoModelConfig.module_to_grid_map must be set at model construction time. "
            "Ensure OmniModalProvider.provide() passes module_to_grid_map for OmniModal parallelism."
        )

    logger.info(f"Rank {dist.get_rank()}: Creating MimoOptimizer")
    from megatron.core.models.mimo.optimizer import get_mimo_optimizer

    # cfg.optimizer already finalized by omni_modal_runtime_config_update().
    optimizer = get_mimo_optimizer(unwrapped_model, cfg.optimizer)

    # Auto-create per-module LR schedulers
    cfg._calculate_scheduler_steps()
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

    schedulers: Dict[str, "OptimizerParamScheduler"] = {}
    for name, info in optimizer.module_infos.items():
        if info.is_active and info.optimizer is not None:
            schedulers[name] = OptimizerParamScheduler(
                info.optimizer,
                init_lr=cfg.scheduler.lr_warmup_init,
                max_lr=cfg.optimizer.lr,
                min_lr=cfg.optimizer.min_lr,
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

    # Configure model config hooks (mirrors standard path's _update_model_config_funcs in setup.py).
    _update_omni_modal_model_config_funcs(model, optimizer, omni_modal_infra, module_to_grid_tuple)

    # Select rank-local PG collection for non-colocated OmniModal.
    active_module_name, local_pg_collection = get_active_module_pg(omni_modal_infra)

    # Initialize checkpoint manager (owns checkpointing_context internally).
    checkpoint_manager = create_checkpoint_manager(cfg.checkpoint)

    # Bridge OmniModal's per-module process groups into Megatron's global parallel
    # state.  OmniModal intentionally skips global MPU init (see
    # OmniModalProvider.initialize_model_parallel), but checkpoint save/load
    # paths (sharded_state_dict, ensure_metadata_has_dp_cp_group) rely on the
    # globals.  For non-colocated OmniModal every rank is active in exactly one
    # module, so we can safely set the globals from that module's collection.
    from megatron.core import parallel_state as mpu

    mpu._TENSOR_MODEL_PARALLEL_GROUP = local_pg_collection.tp
    mpu._DATA_PARALLEL_GROUP = local_pg_collection.dp
    mpu._DATA_PARALLEL_GROUP_WITH_CP = getattr(local_pg_collection, "dp_cp", local_pg_collection.dp)
    if hasattr(local_pg_collection, "pp"):
        mpu._PIPELINE_MODEL_PARALLEL_GROUP = local_pg_collection.pp

    # Load checkpoint if one exists (persistent, pretrained, or non-persistent).
    first_scheduler = next(iter(schedulers.values()), None) if schedulers else None

    has_persistent = cfg.checkpoint.load is not None and checkpoint_exists(cfg.checkpoint.load)
    has_pretrained = cfg.checkpoint.pretrained_checkpoint is not None and checkpoint_exists(
        cfg.checkpoint.pretrained_checkpoint
    )
    wants_non_persistent = cfg.checkpoint.non_persistent_ckpt_type is not None
    should_load = has_persistent or has_pretrained or wants_non_persistent

    if should_load:
        timers = global_state.timers
        timers("load-checkpoint", log_level=0).start(barrier=True)
        load_checkpoint(
            global_state,
            model=[model],
            optimizer=optimizer,
            opt_param_scheduler=first_scheduler,
            checkpointing_context=checkpoint_manager.checkpointing_context,
            pg_collection=local_pg_collection,
            module_name=active_module_name,
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

    # Initialize async checkpoint worker (idempotent if already initialized).
    global_state.initialize_async_checkpoint_worker()

    # Align start_time across ranks so duration-based exit is consistent.
    import torch

    start_time_tensor = torch.tensor([global_state.start_time], dtype=torch.double, device="cuda")
    dist.all_reduce(start_time_tensor, op=dist.ReduceOp.MIN)
    global_state.start_time = start_time_tensor.item()

    # Build data iterators after checkpoint load (resume-safe ordering).
    # When resuming, train_state has restored consumed-sample offsets that
    # the iterator builder must honor to avoid replaying data from sample 0.
    train_data_iterator = None
    valid_data_iterator = None
    if build_data_iterators_fn is not None:
        logger.info(f"Rank {dist.get_rank()}: Building data iterators")
        train_state = global_state.train_state
        is_resuming = train_state.step > 0

        if is_resuming:
            import inspect

            sig = inspect.signature(build_data_iterators_fn)
            accepts_train_state = "train_state" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            if accepts_train_state:
                train_data_iterator, valid_data_iterator = build_data_iterators_fn(
                    cfg,
                    omni_modal_infra,
                    train_state=train_state,
                )
            else:
                raise RuntimeError(
                    "Resuming from checkpoint but build_data_iterators_fn does not accept "
                    "'train_state' argument. The iterator builder must support a train_state "
                    "keyword argument to honor restored consumed-sample offsets during resume."
                )
        else:
            train_data_iterator, valid_data_iterator = build_data_iterators_fn(cfg, omni_modal_infra)

    logger.info(f"Rank {dist.get_rank()}: OmniModal setup complete")

    return OmniModalSetupOutput(
        model=model,
        omni_modal_infra=omni_modal_infra,
        multimodule_pg_collection=multimodule_pg_collection,
        multimodule_communicator=multimodule_communicator,
        module_to_grid_tuple=module_to_grid_tuple,
        optimizer=optimizer,
        schedulers=schedulers,
        train_data_iterator=train_data_iterator,
        valid_data_iterator=valid_data_iterator,
        global_state=global_state,
        checkpoint_manager=checkpoint_manager,
        active_module_name=active_module_name,
        local_pg_collection=local_pg_collection,
    )
