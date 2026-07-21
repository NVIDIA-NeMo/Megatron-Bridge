# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""MegatronMIMO-specific setup for heterogeneous multi-module training.

This module provides the setup logic for MegatronMIMO training, mirroring the standard
``setup.py`` but adapted for per-module parallelism.

Key components:
- setup_megatron_mimo(): MegatronMIMO-specific setup helper (analogous to setup())
- _update_megatron_mimo_model_config_funcs(): Model config hooks (analogous to _update_model_config_funcs)
- MegatronMIMOSetupOutput: Dataclass containing all setup outputs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Optional

import torch.distributed as dist
from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
from megatron.core.utils import get_model_config

from megatron.bridge.training.checkpointing import CheckpointLoadContext, CheckpointManager, create_checkpoint_manager
from megatron.bridge.training.megatron_mimo_parallel_utils import (
    build_pg_collection_for_schedule,
    get_active_module_pg,
    get_module_to_grid_tuple,
    unwrap_megatron_mimo_model,
)
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.checkpoint_utils import checkpoint_exists, is_hf_checkpoint_dir


if TYPE_CHECKING:
    from megatron.core.models.mimo import MimoModel
    from megatron.core.models.mimo.optimizer import MimoOptimizer
    from megatron.core.optimizer.optimizer_param_scheduler import OptimizerParamScheduler
    from megatron.core.process_groups_config import MultiModuleProcessGroupCollection, ProcessGroupCollection

    from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import MegatronMIMOInfra


logger = logging.getLogger(__name__)


@dataclass
class MegatronMIMOSetupOutput:
    """Output from setup_megatron_mimo() containing all components needed for training.

    Attributes:
        model: MimoModel (distributed, DDP-wrapped).
        megatron_mimo_infra: MegatronMIMOInfra (grids, topology, pg_collections).
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
    megatron_mimo_infra: "MegatronMIMOInfra"
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


def _update_megatron_mimo_model_config_funcs(
    model: "MimoModel",
    optimizer: "MimoOptimizer",
    megatron_mimo_infra: "MegatronMIMOInfra",
    module_to_grid_tuple: List,
) -> None:
    """Set model config hooks for MegatronMIMO training.

    Mirrors the standard path's ``_update_model_config_funcs`` (in ``setup.py``)
    but uses per-module gradient operations instead of global ones.

    Sets:
    - ``no_sync_func``: per-module ``no_sync`` via ``multimodule_no_sync``
    - ``finalize_model_grads_func``: per-module grad all-reduce via
      ``finalize_model_grads_multimodule``
    - ``grad_scale_func``: loss scaling from ``MimoOptimizer``
    """
    from functools import partial

    from megatron.bridge.training.megatron_mimo_parallel_utils import (
        finalize_model_grads_multimodule,
        multimodule_no_sync,
    )

    model_config = get_model_config(model)

    model_config.no_sync_func = partial(multimodule_no_sync, module_to_grid_tuple=module_to_grid_tuple)

    model_config.finalize_model_grads_func = partial(
        finalize_model_grads_multimodule,
        infra=megatron_mimo_infra,
        module_to_grid_tuple=module_to_grid_tuple,
    )

    if hasattr(optimizer, "scale_loss"):
        model_config.grad_scale_func = optimizer.scale_loss

    assert model_config.variable_seq_lengths, (
        "variable_seq_lengths must be True for MegatronMIMO training. "
        "This should be set by MegatronMIMOProvider.provide_distributed_model()."
    )


def setup_megatron_mimo(
    state: GlobalState,
    build_data_iterators_fn: Optional[Callable] = None,
) -> MegatronMIMOSetupOutput:
    """MegatronMIMO-specific setup helper.

    This function sets up all components needed for MegatronMIMO training:
    - Builds distributed model via ``cfg.model`` (an ``MegatronMIMOProvider``)
    - Builds MegatronMIMO infrastructure (grids, topology, pg_collections)
    - Creates MultiModulePipelineCommunicator
    - Creates MimoOptimizer and per-module LR schedulers
    - Loads checkpoint (if one exists)
    - Builds data iterators (if function provided, after checkpoint load)
    - Validates configuration

    Args:
        state: GlobalState with ``state.cfg`` already set.  ``state.cfg.model``
            must be an ``MegatronMIMOProvider``.  ``state.cfg.optimizer`` is used to
            create the optimizer.
        build_data_iterators_fn: Optional function to build data iterators.
            Should have signature: (cfg, megatron_mimo_infra) -> (train_iter, valid_iter)

    Returns:
        MegatronMIMOSetupOutput containing all components for training.
    """
    cfg = state.cfg
    global_state = state

    logger.info(f"Rank {dist.get_rank()}: Setting up MegatronMIMO training")

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

    # Build the distributed MIMO model + infra. This single call replaces
    # the previously-inlined finalize / build_infra / validate-no-stub /
    # set-per-module-seeds / provide_distributed_model / parallel_state-bridge
    # sequence. The same helper is the entry point for the conversion CLI,
    # which is why it lives under ``models/megatron_mimo`` rather than
    # inside this training-setup module.
    from megatron.bridge.models.megatron_mimo import build_megatron_mimo_model

    megatron_mimo_provider = cfg.model
    model, megatron_mimo_infra = build_megatron_mimo_model(
        megatron_mimo_provider,
        ddp_config=cfg.ddp,
        fp16=getattr(cfg.model, "fp16", False),
        bf16=getattr(cfg.model, "bf16", True),
        seed=cfg.rng.seed,
    )

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
        megatron_mimo_infra.module_to_grid_map,
        megatron_mimo_infra.topology,
        model_config,
        dim_mapping={"s": 0, "b": 1, "h": 2},  # SBH mapping - matches MimoModel output
        module_output_ndim=megatron_mimo_infra.module_output_ndim,
    )

    # Build pg_collection for schedule
    multimodule_pg_collection = build_pg_collection_for_schedule(megatron_mimo_infra)

    # Build module-to-grid tuple for gradient operations
    module_to_grid_tuple = get_module_to_grid_tuple(model, megatron_mimo_infra)

    # Build optimizer and per-module LR schedulers
    unwrapped_model = unwrap_megatron_mimo_model(model)
    if megatron_mimo_infra.module_to_grid_map:
        assert unwrapped_model.mimo_config.module_to_grid_map is not None, (
            "MimoModelConfig.module_to_grid_map must be set at model construction time. "
            "Ensure MegatronMIMOProvider.provide() passes module_to_grid_map for MegatronMIMO parallelism."
        )

    logger.info(f"Rank {dist.get_rank()}: Creating MimoOptimizer")
    from megatron.core.models.mimo.optimizer import get_mimo_optimizer

    # cfg.optimizer already finalized by megatron_mimo_runtime_config_update().
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
    _update_megatron_mimo_model_config_funcs(model, optimizer, megatron_mimo_infra, module_to_grid_tuple)

    # Select rank-local PG collection for non-colocated MegatronMIMO.
    # ``build_megatron_mimo_model`` has already bridged
    # ``parallel_state`` globals from this rank's pg_collection.
    active_module_name, local_pg_collection = get_active_module_pg(megatron_mimo_infra)

    # Initialize checkpoint manager.
    checkpoint_manager = create_checkpoint_manager(cfg.checkpoint)

    # Load checkpoint if one exists (persistent, pretrained, or non-persistent).
    first_scheduler = next(iter(schedulers.values()), None) if schedulers else None

    # HF directories are included in load detection only to route to the
    # targeted checkpoint.load error in checkpointing.
    has_persistent = cfg.checkpoint.load is not None and (
        checkpoint_exists(cfg.checkpoint.load) or is_hf_checkpoint_dir(cfg.checkpoint.load)
    )
    has_pretrained = cfg.checkpoint.pretrained_checkpoint is not None and (
        checkpoint_exists(cfg.checkpoint.pretrained_checkpoint)
        or is_hf_checkpoint_dir(cfg.checkpoint.pretrained_checkpoint)
    )
    wants_non_persistent = cfg.checkpoint.non_persistent_ckpt_type is not None
    should_load = has_persistent or has_pretrained or wants_non_persistent

    if should_load:
        timers = global_state.timers
        timers("load-checkpoint", log_level=0).start(barrier=True)
        checkpoint_manager.load(
            CheckpointLoadContext(
                state=global_state,
                model=[model],
                optimizer=optimizer,
                opt_param_scheduler=first_scheduler,
                pg_collection=local_pg_collection,
                module_name=active_module_name,
            )
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
                    megatron_mimo_infra,
                    train_state=train_state,
                )
            else:
                raise RuntimeError(
                    "Resuming from checkpoint but build_data_iterators_fn does not accept "
                    "'train_state' argument. The iterator builder must support a train_state "
                    "keyword argument to honor restored consumed-sample offsets during resume."
                )
        else:
            train_data_iterator, valid_data_iterator = build_data_iterators_fn(cfg, megatron_mimo_infra)

    logger.info(f"Rank {dist.get_rank()}: MegatronMIMO setup complete")

    return MegatronMIMOSetupOutput(
        model=model,
        megatron_mimo_infra=megatron_mimo_infra,
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
