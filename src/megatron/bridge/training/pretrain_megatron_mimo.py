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
"""Entry point for MegatronMIMO pretraining.

Thin entry point that orchestrates runtime config updates, setup, and training.
Mirrors the standard ``pretrain.py`` → ``setup()`` → ``train()`` pattern.

See also:
- ``setup_megatron_mimo.py``: MegatronMIMO-specific setup logic (analogous to ``setup.py``)
- ``train_megatron_mimo.py``: MegatronMIMO training loop (analogous to ``train.py``)
- ``config.py``: ``megatron_mimo_runtime_config_update()`` (analogous to ``runtime_config_update()``)
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch.distributed as dist

from megatron.bridge.training.config import ConfigContainer, megatron_mimo_runtime_config_update
from megatron.bridge.training.pretrain import _maybe_destroy_process_group
from megatron.bridge.training.setup_megatron_mimo import setup_megatron_mimo
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.train import _finish_train
from megatron.bridge.training.train_megatron_mimo import train_megatron_mimo


logger = logging.getLogger(__name__)


def pretrain_megatron_mimo(
    cfg: ConfigContainer,
    forward_step_func: Callable,
    build_data_iterators_fn: Callable,
    global_state: Optional[GlobalState] = None,
) -> None:
    """Entry point for MegatronMIMO pretraining.

    Steps:
    1. Apply MegatronMIMO runtime config updates (finalize sub-configs, set data_parallel_size=1)
    2. Call setup_megatron_mimo() to get model, optimizer, schedulers, infra, communicators
    3. Call train_megatron_mimo() with all components

    Args:
        cfg: ConfigContainer with training configuration.  ``cfg.model`` must be
            an ``MegatronMIMOProvider``.  ``cfg.optimizer`` (a ``BridgeOptimizerConfig``)
            is used to create the ``MimoOptimizer`` and per-module LR schedulers.
        forward_step_func: Forward step function for training.
        build_data_iterators_fn: Function to build data iterators.
            Signature: (cfg, megatron_mimo_infra) -> (train_iter, valid_iter)
        global_state: Optional GlobalState for testing.  If not provided,
            creates a new one.  Production callers should not pass this.
    """
    logger.info("Starting MegatronMIMO pretraining")

    # If the caller already initialized distributed, we should not destroy it on exit.
    should_destroy_process_group = not dist.is_initialized()

    # Apply runtime config updates (MegatronMIMO-equivalent of runtime_config_update).
    megatron_mimo_runtime_config_update(cfg)

    # Create GlobalState (mirrors standard pretrain path).
    state = global_state if global_state is not None else GlobalState()
    state.cfg = cfg

    # Setup: model, optimizer, schedulers, MPU bridging, checkpoint load, data iterators.
    setup_output = setup_megatron_mimo(
        state=state,
        build_data_iterators_fn=build_data_iterators_fn,
    )

    logger.info(f"Rank {dist.get_rank()}: Starting training loop")

    # Run training loop
    train_megatron_mimo(
        forward_step_func=forward_step_func,
        model=setup_output.model,
        optimizer=setup_output.optimizer,
        schedulers=setup_output.schedulers,
        train_data_iterator=setup_output.train_data_iterator,
        valid_data_iterator=setup_output.valid_data_iterator,
        global_state=setup_output.global_state,
        megatron_mimo_infra=setup_output.megatron_mimo_infra,
        multimodule_communicator=setup_output.multimodule_communicator,
        checkpoint_manager=setup_output.checkpoint_manager,
        multimodule_pg_collection=setup_output.multimodule_pg_collection,
        module_to_grid_tuple=setup_output.module_to_grid_tuple,
    )

    # Post-training cleanup: finalize async saves, shut down NVRx/FT, flush
    # loggers, destroy GlobalState (which calls destroy_model_parallel internally).
    _finish_train(setup_output.global_state, setup_output.checkpoint_manager)

    _maybe_destroy_process_group(should_destroy_process_group)

    logger.info("MegatronMIMO pretraining completed")
