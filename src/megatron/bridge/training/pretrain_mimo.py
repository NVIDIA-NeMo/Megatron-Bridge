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

from megatron.bridge.training.config import ConfigContainer, mimo_runtime_config_update
from megatron.bridge.training.setup_mimo import setup_mimo
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.train_mimo import train_mimo


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

    # Setup all MIMO components (model, optimizer, schedulers, data, communicators)
    setup_output = setup_mimo(
        cfg=cfg,
        build_data_iterators_fn=build_data_iterators_fn,
        build_optimizer=True,
        global_state=global_state,
    )

    logger.info(f"Rank {dist.get_rank()}: Starting training loop")

    # Run training loop
    train_mimo(
        forward_step_func=forward_step_func,
        model=setup_output.model,
        optimizer=setup_output.optimizer,
        schedulers=setup_output.schedulers,
        train_data_iterator=setup_output.train_data_iterator,
        valid_data_iterator=setup_output.valid_data_iterator,
        global_state=setup_output.global_state,
        mimo_infra=setup_output.mimo_infra,
        multimodule_communicator=setup_output.multimodule_communicator,
        multimodule_pg_collection=setup_output.multimodule_pg_collection,
        module_to_grid_tuple=setup_output.module_to_grid_tuple,
    )

    logger.info("MIMO pretraining completed")
