# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Entry point for MIMO pretraining.

Thin entry point that orchestrates runtime config updates, setup, and training.
Mirrors the standard ``pretrain.py`` → ``setup()`` → ``train()`` pattern.

See also:
- ``setup_mimo.py``: MIMO-specific setup logic (analogous to ``setup.py``)
- ``train_mimo.py``: MIMO training loop (analogous to ``train.py``)
- ``config.py``: ``mimo_runtime_config_update()`` (analogous to ``runtime_config_update()``)
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch.distributed as dist

from megatron.bridge.training.checkpointing import load_checkpoint
from megatron.bridge.training.config import ConfigContainer, mimo_runtime_config_update
from megatron.bridge.training.setup_mimo import setup_mimo
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.train_mimo import train_mimo
from megatron.bridge.training.utils.checkpoint_utils import checkpoint_exists


logger = logging.getLogger(__name__)


def pretrain_mimo(
    cfg: ConfigContainer,
    forward_step_func: Callable,
    build_data_iterators_fn: Callable,
    global_state: Optional[GlobalState] = None,
) -> None:
    """Entry point for MIMO pretraining.

    Steps:
    1. Apply MIMO runtime config updates (finalize sub-configs, set data_parallel_size=1)
    2. Call setup_mimo() to get model, optimizer, schedulers, infra, communicators
    3. Call train_mimo() with all components

    Args:
        cfg: ConfigContainer with training configuration.  ``cfg.model`` must be
            a ``MimoModelProvider``.  ``cfg.optimizer`` (a ``BridgeOptimizerConfig``)
            is used to create the ``MimoOptimizer`` and per-module LR schedulers.
        forward_step_func: Forward step function for training.
        build_data_iterators_fn: Function to build data iterators.
            Signature: (cfg, mimo_infra) -> (train_iter, valid_iter)
        global_state: Optional GlobalState. If not provided, creates a new one.

    TODO(liding): check if build_data_iterators_fn and global_state are needed (deferred to phase 5 review)
    """
    logger.info("Starting MIMO pretraining")

    # Apply runtime config updates (MIMO-equivalent of runtime_config_update).
    mimo_runtime_config_update(cfg)

    # Setup MIMO components (iterators deferred until after checkpoint load)
    setup_output = setup_mimo(
        cfg=cfg,
        build_data_iterators_fn=None,
        build_optimizer=True,
        global_state=global_state,
    )

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

    first_scheduler = next(iter(setup_output.schedulers.values()), None) if setup_output.schedulers else None

    # Broadened load-intent gating: includes non-persistent resume intent
    has_persistent = cfg.checkpoint.load is not None and checkpoint_exists(cfg.checkpoint.load)
    has_pretrained = cfg.checkpoint.pretrained_checkpoint is not None and checkpoint_exists(
        cfg.checkpoint.pretrained_checkpoint
    )
    wants_non_persistent = cfg.checkpoint.non_persistent_ckpt_type is not None
    should_load = has_persistent or has_pretrained or wants_non_persistent

    if should_load:
        timers = setup_output.global_state.timers
        timers("load-checkpoint", log_level=0).start(barrier=True)
        load_checkpoint(
            setup_output.global_state,
            model=[setup_output.model],
            optimizer=setup_output.optimizer,
            opt_param_scheduler=first_scheduler,
            checkpointing_context=setup_output.checkpoint_manager.checkpointing_context,
            pg_collection=local_pg_collection,
        )
        timers("load-checkpoint").stop(barrier=True)
        timers.log(["load-checkpoint"])

        # Fan out loaded scheduler state to all active module schedulers.
        # v1: checkpoints contain a single scheduler blob (first_scheduler).
        if first_scheduler is not None and len(setup_output.schedulers) > 1:
            loaded_state = first_scheduler.state_dict()
            for sched in setup_output.schedulers.values():
                if sched is not first_scheduler:
                    sched.load_state_dict(loaded_state)

    # Build data iterators after checkpoint load (resume-safe ordering).
    # When resuming, train_state has restored consumed-sample offsets that
    # the iterator builder must honor to avoid replaying data from sample 0.
    train_state = setup_output.global_state.train_state
    is_resuming = train_state.step > 0

    if is_resuming:
        import inspect

        sig = inspect.signature(build_data_iterators_fn)
        if "train_state" in sig.parameters:
            train_data_iterator, valid_data_iterator = build_data_iterators_fn(
                cfg,
                setup_output.mimo_infra,
                train_state=train_state,
            )
        else:
            raise RuntimeError(
                "Resuming from checkpoint but build_data_iterators_fn does not accept "
                "'train_state' argument. The iterator builder must support a train_state "
                "keyword argument to honor restored consumed-sample offsets during resume."
            )
    else:
        train_data_iterator, valid_data_iterator = build_data_iterators_fn(cfg, setup_output.mimo_infra)

    logger.info(f"Rank {dist.get_rank()}: Starting training loop")

    # Run training loop
    train_mimo(
        forward_step_func=forward_step_func,
        model=setup_output.model,
        optimizer=setup_output.optimizer,
        schedulers=setup_output.schedulers,
        train_data_iterator=train_data_iterator,
        valid_data_iterator=valid_data_iterator,
        global_state=setup_output.global_state,
        mimo_infra=setup_output.mimo_infra,
        multimodule_communicator=setup_output.multimodule_communicator,
        checkpoint_manager=setup_output.checkpoint_manager,
        multimodule_pg_collection=setup_output.multimodule_pg_collection,
        module_to_grid_tuple=setup_output.module_to_grid_tuple,
    )

    logger.info("MIMO pretraining completed")
