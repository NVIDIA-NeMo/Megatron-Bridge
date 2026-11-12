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

Layout of ``save_state(root)``: Bridge writes ``<root>/iter_<step>`` (model, optimizer, RNG,
``train_state.pt``, ``run_config.yaml``) and then the tracker files in ``<root>`` that mark the step
complete. Per-rank user state goes to ``<root>/user_state_<step>/rank_<r>.pt`` before the Bridge save
starts, so the tracker also vouches for it. ``load_state`` accepts ``<root>`` (latest step) or an
``iter_<step>`` directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch
import yaml
from megatron.core.dist_checkpointing.strategies.fully_parallel import FullyParallelSaveStrategyWrapper
from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule

from megatron.bridge.engine.datum import CheckpointHandle
from megatron.bridge.engine.recipes.base import TrainingRecipe
from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.training.checkpointing import (
    _DIRECT_ITERATION_DIR_SENTINEL,
    CheckpointLoadContext,
    CheckpointSaveContext,
    _checkpoint_dp_cp_group,
    _resolve_checkpoint_iteration,
    create_checkpoint_manager,
    maybe_finalize_async_save,
)
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
from megatron.bridge.training.utils.checkpoint_utils import (
    get_checkpoint_name,
    get_checkpoint_run_config_filename,
    is_checkpoint_iteration_directory,
    read_train_state,
)


_RUN_CONFIG_MODEL_KEYS = (
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "context_parallel_size",
    "expert_model_parallel_size",
    "expert_tensor_parallel_size",
    "tensor_parallel_num_weight_shards",
    "expert_tensor_parallel_num_weight_shards",
    "gtp_weight_remat_size",
    "expert_gtp_weight_remat_size",
)
_RUN_CONFIG_CHECKPOINT_KEYS = (
    "save_optim",
    "save_rng",
    "save_rng_state_per_dp_rank",
    "fully_parallel_save",
    "ckpt_format",
)


@dataclass
class _SchedulerView:
    """Bridge's optimizer-state load reads this flag; the engine has no scheduler."""

    override_opt_param_scheduler: bool = False


@dataclass
class _CheckpointConfigView:
    """The subset of ``ConfigContainer`` that Bridge's checkpoint functions read via ``state.cfg``."""

    model: Any
    optimizer: OptimizerConfig
    ddp: DistributedDataParallelConfig
    dist: DistributedInitConfig
    rng: RNGConfig
    checkpoint: CheckpointConfig
    # A LoRA engine's LoRA: Bridge then saves only the adapters (and MoE router expert biases) and resumes
    # them non-strictly into a model that already holds the base weights. None for full-weight engines.
    peft: PEFT | None = None
    train: None = None  # GlobalState installs a signal handler only when train.exit_signal_handler is set
    ft: None = None  # fault-tolerance hooks around save/load are no-ops when unset
    dataset: None = None  # no dataloader state to save
    scheduler: _SchedulerView = field(default_factory=_SchedulerView)
    # The load-success hook reads state.wandb_logger, which consults logger.wandb_project (None: no logger).
    logger: LoggerConfig = field(default_factory=LoggerConfig)

    def to_yaml(self, path: str) -> None:
        """Write the ``run_config.yaml`` the loader reads back: weight topology, the save flags and the
        LoRA settings (``lora``, checked by ``load_engine_state``; ``null`` without LoRA)."""
        run_config = {
            "model": {key: getattr(self.model, key) for key in _RUN_CONFIG_MODEL_KEYS if hasattr(self.model, key)},
            "checkpoint": {key: getattr(self.checkpoint, key) for key in _RUN_CONFIG_CHECKPOINT_KEYS},
            "lora": _lora_settings(self.peft),
        }
        with open(path, "w") as handle:
            yaml.safe_dump(run_config, handle)


def _lora_settings(peft: PEFT | None) -> dict[str, Any] | None:
    """The LoRA settings a checkpoint records, named as in ``LoraConfig``; ``None`` without LoRA.

    Rank and targets also show in the adapter tensors, but alpha and the dropout do not, and Bridge's PEFT
    resume ignores checkpoint adapters the engine has no target for.
    """
    if peft is None:
        return None
    if not isinstance(peft, LoRA):
        raise TypeError(f"TrainingEngine checkpoints record LoRA settings only, got {type(peft).__name__}.")
    return {
        "rank": int(peft.dim),
        "alpha": float(peft.alpha),
        "dropout": float(peft.dropout),
        "target_modules": [str(name) for name in peft.target_modules],
    }


def _check_lora_settings(path: str, peft: PEFT | None) -> None:
    """Raise unless the step ``path`` resolves to was saved with ``peft``'s LoRA settings (or both have none).

    The step is resolved as Bridge's loader resolves it. A checkpoint saved before the settings were
    recorded reads as having no LoRA.
    """
    iteration, release = _resolve_checkpoint_iteration(path, None)
    step_dir = path if iteration == _DIRECT_ITERATION_DIR_SENTINEL else get_checkpoint_name(path, iteration, release)
    run_config_file = get_checkpoint_run_config_filename(step_dir)
    saved = None
    if os.path.exists(run_config_file):
        with open(run_config_file) as handle:
            saved = (yaml.safe_load(handle) or {}).get("lora")
    expected = _lora_settings(peft)
    if saved != expected:
        raise ValueError(
            f"{step_dir} was saved with lora={saved} but this engine has lora={expected} (None: no LoRA, or saved "
            "before checkpoints recorded LoRA settings). Adapter tensors carry neither alpha nor dropout, so the "
            "engine has to be built with the saved settings."
        )


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


def _checkpoint_state(
    recipe: TrainingRecipe, checkpoint_config: CheckpointConfig, *, peft: PEFT | None = None
) -> GlobalState:
    state = GlobalState()
    state.cfg = _CheckpointConfigView(
        model=recipe.model,
        optimizer=recipe.optimizer,
        ddp=recipe.ddp,
        dist=recipe.dist,
        rng=recipe.rng,
        checkpoint=checkpoint_config,
        peft=peft,
    )
    return state


def load_pretrained_weights(
    recipe: TrainingRecipe,
    model: list[MegatronModule],
    optimizer: Any | None,
    checkpoint_path: str,
) -> None:
    """Load model weights from a Megatron (torch_dist) or HF checkpoint; optimizer and RNG state are not loaded.

    The optimizer is passed to the loader even though its state is not loaded (``load_optim=False``):
    Bridge refreshes the optimizer's FP32 master parameters from the loaded weights only when it is
    given the optimizer. Without it the masters keep their initialization values and the first
    ``optim_step`` would overwrite the model from them. Loading into a bare model before the optimizer
    exists (``optimizer=None``) is fine too: the masters are then built from the loaded weights.
    """
    checkpoint_config = _engine_checkpoint_config(checkpoint_path)
    checkpoint_config.finalize()
    manager = create_checkpoint_manager(checkpoint_config)
    manager.load(
        CheckpointLoadContext(
            state=_checkpoint_state(recipe, checkpoint_config),
            model=model,
            optimizer=optimizer,
            opt_param_scheduler=None,
        )
    )
    sync_hybrid_device_optimizer_fp32_master_copies(optimizer)


def user_state_file(root: str, step: int, rank: int) -> str:
    """Per-rank user-state file for one saved step."""
    return os.path.join(root, f"user_state_{step:07d}", f"rank_{rank:05d}.pt")


def write_user_state(root: str, step: int, rank: int, user_state: Any) -> None:
    """Write this rank's user state for ``step``. ``None`` is written too, so saving the same step
    again replaces an older sidecar instead of leaving it to be restored."""
    path = user_state_file(root, step, rank)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(user_state, path)


def read_user_state(root: str, step: int, rank: int) -> Any | None:
    """This rank's user state for ``step``. A missing file reads as ``None`` so checkpoints written by
    Bridge's own trainer (no sidecar) load as well."""
    path = user_state_file(root, step, rank)
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def checkpoint_root(path: str) -> str:
    """The save root for ``path``, which is either the root itself or one of its ``iter_<step>`` directories."""
    normalized = os.path.normpath(path)
    if os.path.basename(normalized).startswith("iter_"):
        return os.path.dirname(normalized)
    return normalized


def start_async_checkpoint_worker(recipe: TrainingRecipe) -> GlobalState:
    """Start Bridge's persistent asynchronous-save worker and return the state that owns it.

    Call this before the model is built. The worker is a spawned process that initializes its own CUDA
    context at start and later stages each checkpoint device-to-host by mapping the model's tensors
    over CUDA IPC; started on an already full GPU, that mapping fails with out-of-memory.
    """
    checkpoint_config = CheckpointConfig(
        save="<engine-async-worker>",
        load=None,
        async_save=True,
        ckpt_format="torch_dist",
        fully_parallel_save=True,
    )
    checkpoint_config.finalize()
    state = _checkpoint_state(recipe, checkpoint_config)
    state.initialize_async_checkpoint_worker()
    return state


def stop_async_checkpoint_worker(state: GlobalState) -> None:
    """Finish any pending asynchronous save on every rank and shut the worker down."""
    maybe_finalize_async_save(state, state.cfg.checkpoint, blocking=True, terminate=True)


def _async_save_strategy(checkpoint_config: CheckpointConfig, pg_collection: ProcessGroupCollection) -> Any:
    """Bridge's torch_dist save strategy with ``cpu_shm_mode``: GPU tensors are copied to CPU shared memory
    in this process before the hand-off, instead of being mapped by the worker over CUDA IPC (which fails
    with out-of-memory on a full GPU).

    TODO(checkpoint-config): these settings (torch_dist, fully parallel save, persistent worker, CPU
    shared-memory staging, one storage writer per rank) are what worked for the Qwen3.5 POC, not a tuned
    choice. Verify the best configuration: measure the CUDA IPC hand-off with the default allocator,
    the staging cost against ``storage_writers_per_rank`` and pinned memory, and whether the worker can
    skip its CUDA context (upstream passes ``cpu_shm_mode`` to the worker too; Bridge does not), ideally
    through a Bridge ``CheckpointConfig`` knob instead of seeding the strategy here.
    """
    strategy = TorchDistSaveShardedStrategy(
        "torch_dist", 1, thread_count=checkpoint_config.storage_writers_per_rank, cpu_shm_mode=True
    )
    if checkpoint_config.fully_parallel_save:
        strategy = FullyParallelSaveStrategyWrapper(
            strategy, _checkpoint_dp_cp_group(pg_collection), checkpoint_config.ckpt_assume_constant_structure
        )
    return strategy


def save_engine_state(
    recipe: TrainingRecipe,
    model: list[MegatronModule],
    optimizer: Any | None,
    *,
    step: int,
    root: str,
    user_state: Any,
    async_save: bool,
    async_state: GlobalState | None,
    pg_collection: ProcessGroupCollection,
    peft: PEFT | None = None,
) -> CheckpointHandle:
    """Save model, optimizer (when present), RNG, progress and user state as ``<root>/iter_<step>``.

    Collective: every rank calls it. User state is written first, then Bridge's save runs; its tracker
    files are written last (inline for a synchronous save, by the async worker's finalize steps
    otherwise), so a step marked complete has everything. ``async_save`` needs the worker from
    ``start_async_checkpoint_worker`` (``async_state``; ``None`` when the engine was built with
    ``spawn_async_save_worker=False``) and stages through CPU shared memory; the returned handle's
    ``wait()`` then blocks until the finalize steps ran on every rank. With ``peft`` (a LoRA engine)
    the model section holds only the adapters and the MoE router expert biases, and ``run_config.yaml``
    records the LoRA settings.
    """
    if async_save and async_state is None:
        raise ValueError(
            "async_save=True needs the save worker; this engine was built with spawn_async_save_worker=False."
        )
    checkpoint_config = CheckpointConfig(
        save=root,
        load=None,
        save_optim=optimizer is not None,
        save_rng=True,
        async_save=async_save,
        ckpt_format="torch_dist",
        fully_parallel_save=True,
    )
    checkpoint_config.finalize()
    if async_save:
        state = async_state
        state.cfg = _checkpoint_state(recipe, checkpoint_config, peft=peft).cfg
    else:
        state = _checkpoint_state(recipe, checkpoint_config, peft=peft)
    state.train_state.step = step
    write_user_state(root, step, torch.distributed.get_rank(), user_state)
    manager = create_checkpoint_manager(checkpoint_config)
    if async_save:
        manager.checkpointing_context["save_strategy"] = _async_save_strategy(checkpoint_config, pg_collection)
    manager.save(
        CheckpointSaveContext(
            state=state,
            model=model,
            optimizer=optimizer,
            opt_param_scheduler=None,
            num_floating_point_operations_so_far=0,
            pg_collection=pg_collection,
        ),
        None,
    )
    waiter = None
    if async_save:

        def waiter() -> None:
            manager.finalize_async_saves(state, blocking=True)

    return CheckpointHandle(path=os.path.join(root, f"iter_{step:07d}"), _waiter=waiter)


def load_engine_state(
    recipe: TrainingRecipe,
    model: list[MegatronModule],
    optimizer: Any | None,
    *,
    path: str,
    peft: PEFT | None = None,
) -> tuple[int, Any | None]:
    """Restore a checkpoint written by ``save_engine_state``; returns ``(step, this rank's user state)``.

    ``path`` is the save root (latest complete step) or one ``iter_<step>`` directory. Optimizer state
    is restored when this engine has an optimizer and the checkpoint holds one; RNG state when the
    parallel layout matches. With ``peft`` (a LoRA engine) only the adapters and router expert biases
    are read, non-strictly (Bridge's PEFT resume): the base weights are the ones the engine loaded at
    construction, so it must have been built from the same base checkpoint. The checkpoint's LoRA
    settings must equal ``peft``'s (or both have none); a mismatch raises before anything is read.
    """
    root = checkpoint_root(path)
    if not (os.path.exists(os.path.join(root, "latest_train_state.pt")) or is_checkpoint_iteration_directory(path)):
        raise FileNotFoundError(f"{path} holds no complete engine checkpoint.")
    # Bridge caches the root's tracker per process, but every save to the root rewrites it; without the
    # clear, a load after a newer save in the same process would resolve (and load) the older step.
    read_train_state.cache_clear()
    _check_lora_settings(path, peft)
    checkpoint_config = CheckpointConfig(
        save=None,
        load=path,
        pretrained_checkpoint=None,
        load_optim=optimizer is not None,
        load_rng=True,
        ckpt_format="torch_dist",
        fully_parallel_save=True,
    )
    checkpoint_config.finalize()
    state = _checkpoint_state(recipe, checkpoint_config, peft=peft)
    manager = create_checkpoint_manager(checkpoint_config)
    step, _ = manager.load(
        CheckpointLoadContext(state=state, model=model, optimizer=optimizer, opt_param_scheduler=None)
    )
    sync_hybrid_device_optimizer_fp32_master_copies(optimizer)
    return int(step), read_user_state(root, int(step), torch.distributed.get_rank())
