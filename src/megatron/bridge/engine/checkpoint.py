# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Checkpoint I/O over Bridge's distributed checkpointing.

Bridge's ``load_checkpoint`` / ``save_checkpoint`` read a handful of trainer-config sections through
``GlobalState.cfg``. This module is the only place that knows about that: it presents the recipe
plus an engine-owned ``CheckpointConfig`` to those functions, so the rest of the engine never
touches ``ConfigContainer``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from megatron.core.transformer import MegatronModule

from megatron.bridge.engine.datum import CheckpointHandle
from megatron.bridge.engine.recipes.base import TrainingRecipe
from megatron.bridge.training.checkpointing import CheckpointLoadContext, create_checkpoint_manager
from megatron.bridge.training.config import (
    CheckpointConfig,
    DistributedDataParallelConfig,
    DistributedInitConfig,
    LoggerConfig,
    OptimizerConfig,
    RNGConfig,
)
from megatron.bridge.training.optim import sync_hybrid_device_optimizer_fp32_master_copies
from megatron.bridge.training.state import GlobalState


@dataclass
class _CheckpointConfigView:
    """The subset of ``ConfigContainer`` that Bridge's checkpoint functions read via ``state.cfg``."""

    model: Any
    optimizer: OptimizerConfig
    ddp: DistributedDataParallelConfig
    dist: DistributedInitConfig
    rng: RNGConfig
    checkpoint: CheckpointConfig
    peft: None = None
    train: None = None  # GlobalState installs a signal handler only when train.exit_signal_handler is set
    # The load-success hook reads state.wandb_logger, which consults logger.wandb_project (None: no logger).
    logger: LoggerConfig = field(default_factory=LoggerConfig)


def _engine_checkpoint_config(pretrained_checkpoint: str | None) -> CheckpointConfig:
    return CheckpointConfig(
        save=None,
        load=None,
        pretrained_checkpoint=pretrained_checkpoint,
        load_optim=False,
        load_rng=False,
        ckpt_format="torch_dist",
        fully_parallel_save=True,
    )


def _checkpoint_state(recipe: TrainingRecipe, checkpoint_config: CheckpointConfig) -> GlobalState:
    state = GlobalState()
    state.cfg = _CheckpointConfigView(
        model=recipe.model,
        optimizer=recipe.optimizer,
        ddp=recipe.ddp,
        dist=recipe.dist,
        rng=recipe.rng,
        checkpoint=checkpoint_config,
    )
    return state


def load_pretrained_weights(
    recipe: TrainingRecipe, model: list[MegatronModule], optimizer: Any | None, checkpoint_path: str
) -> None:
    """Load model weights from a Megatron (torch_dist) or HF checkpoint; optimizer and RNG state are not loaded."""
    checkpoint_config = _engine_checkpoint_config(checkpoint_path)
    checkpoint_config.finalize()
    manager = create_checkpoint_manager(checkpoint_config)
    manager.load(
        CheckpointLoadContext(
            state=_checkpoint_state(recipe, checkpoint_config),
            model=model,
            optimizer=None,
            opt_param_scheduler=None,
        )
    )
    sync_hybrid_device_optimizer_fp32_master_copies(optimizer)


def save_engine_state(engine: Any, path: str, *, user_state: Any, async_save: bool) -> CheckpointHandle:
    """Save model, optimizer, progress and user state as one checkpoint version (phase B3)."""
    raise NotImplementedError("save_engine_state is implemented in phase B3.")


def load_engine_state(engine: Any, path: str) -> Any | None:
    """Restore a checkpoint saved by ``save_engine_state`` and return the local rank's user state (phase B3)."""
    raise NotImplementedError("load_engine_state is implemented in phase B3.")
