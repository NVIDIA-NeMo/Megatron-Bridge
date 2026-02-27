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
from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
from megatron.core.utils import get_model_config

from megatron.bridge.training.config import ConfigContainer, OptimizerConfig
from megatron.bridge.training.mimo_parallel_utils import (
    build_pg_collection_for_schedule,
    get_module_to_grid_tuple,
    validate_no_stub_ranks,
)
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.train_mimo import train_mimo


if TYPE_CHECKING:
    from megatron.core.models.mimo import MimoModel
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
    """

    model: "MimoModel"
    mimo_infra: "MimoModelInfra"
    multimodule_pg_collection: any
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
    logger.info(
        f"Rank {dist.get_rank()}: Creating communicator with pipeline_dtype={model_config.pipeline_dtype}, "
        f"bf16={model_config.bf16}, variable_seq_lengths={model_config.variable_seq_lengths}"
    )
    multimodule_communicator = MultiModulePipelineCommunicator(
        mimo_infra.module_to_grid_map,
        mimo_infra.topology,
        model_config,
        dim_mapping={"s": 0, "b": 1, "h": 2},  # SBH mapping - matches MimoModel output
    )

    # Build pg_collection for schedule
    multimodule_pg_collection = build_pg_collection_for_schedule(
        mimo_infra,
        language_module_key=mimo_provider.LANGUAGE_MODULE_KEY,
    )

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
    opt_config: OptimizerConfig,
    schedulers: Optional[Dict[str, "OptimizerParamScheduler"]] = None,
    global_state: Optional[GlobalState] = None,
) -> None:
    """Entry point for MIMO pretraining.

    Steps:
    1. Call setup_mimo() to get model, infra, communicators
    2. Build MimoOptimizer internally from opt_config
    3. Compute runtime fields (dp_size, num_microbatches)
    4. Configure gradient functions (no_sync_func, finalize_model_grads_func)
    5. Call train_mimo() with all components

    Note: Checkpointing and evaluation are deferred to Phase 5.

    Args:
        cfg: ConfigContainer with training configuration.
        mimo_provider: MimoModelProvider for building model and infrastructure.
        forward_step_func: Forward step function for training.
        build_data_iterators_fn: Function to build data iterators.
            Signature: (cfg, mimo_infra) -> (train_iter, valid_iter)
        opt_config: OptimizerConfig used to build MimoOptimizer internally.
        schedulers: Pre-built scheduler dict (can be empty {} or None).
        global_state: Optional GlobalState. If not provided, creates a new one.
    """
    if schedulers is None:
        schedulers = {}

    logger.info("Starting MIMO pretraining")

    # Setup MIMO components
    setup_output = setup_mimo(
        cfg=cfg,
        mimo_provider=mimo_provider,
        build_data_iterators_fn=build_data_iterators_fn,
        global_state=global_state,
    )

    logger.info(f"Rank {dist.get_rank()}: Building optimizer")

    # Unwrap Float16Module to access MimoModel directly
    _model = setup_output.model.module if hasattr(setup_output.model, "module") else setup_output.model

    # Build MimoOptimizer internally (needs unwrapped MimoModel for grid_map access)
    from megatron.core.models.mimo.optimizer import get_mimo_optimizer

    optimizer = get_mimo_optimizer(_model, opt_config)

    # Compute runtime fields from LLM grid's DP size (use grid shape, not PG handle,
    # so this works on all ranks including those outside the language module's grid).
    lang_key = _model.mimo_config.language_module_key
    lang_grid = setup_output.mimo_infra.module_to_grid_map.get(lang_key)
    dp_size = lang_grid.shape[lang_grid.dim_names.index("dp")] if lang_grid else 1

    if not hasattr(cfg.train, "num_microbatches") or cfg.train.num_microbatches is None:
        cfg.train.num_microbatches = cfg.train.global_batch_size // (cfg.train.micro_batch_size * dp_size)

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
        checkpointing_context=None,  # Deferred to Phase 5
    )

    logger.info("MIMO pretraining completed")
