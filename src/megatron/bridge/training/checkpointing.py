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

"""Input/output checkpointing."""

import contextlib
import os
import random
import shutil
import sys
import threading
from enum import Enum, auto
from logging import getLogger
from pathlib import Path
from time import time
from typing import Any, Callable, Literal

import numpy as np
import torch
import torch.nn.functional as F
from megatron.core import dist_checkpointing, mpu, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedStateDict
from megatron.core.dist_checkpointing.serialization import (
    StateDict,
    get_default_load_sharded_strategy,
    get_default_save_sharded_strategy,
)
from megatron.core.dist_checkpointing.strategies.async_utils import AsyncRequest
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)
from megatron.core.msc_utils import MultiStorageClientFeature
from megatron.core.num_microbatches_calculator import update_num_microbatches
from megatron.core.optimizer import MegatronOptimizer
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.transformer import MegatronModule
from megatron.core.utils import unwrap_model

from megatron.bridge.peft.base import PEFT
from megatron.bridge.training import fault_tolerance
from megatron.bridge.training.config import CheckpointConfig
from megatron.bridge.training.state import GlobalState, TrainState
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.tokenizers.tokenizer import MegatronTokenizer
from megatron.bridge.training.utils import wandb_utils
from megatron.bridge.training.utils.checkpoint_utils import (
    checkpoint_exists,
    ensure_directory_exists,
    file_exists,
    get_checkpoint_name,
    get_checkpoint_run_config_filename,
    get_checkpoint_tracker_filename,
    get_checkpoint_train_state_filename,
    read_run_config,
    read_train_state,
)
from megatron.bridge.training.utils.log_utils import append_to_progress_log
from megatron.bridge.utils.common_utils import (
    get_rank_safe,
    is_last_rank,
    print_rank_0,
)
from megatron.bridge.utils.import_utils import safe_import


_, HAVE_RESIL = safe_import("nvidia_resiliency_ext.checkpointing")

try:
    from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
        preprocess_state_dict_for_uneven_dtensor,
    )
    from megatron.core.transformer.fsdp_dtensor_checkpoint import (
        handle_fp8_extra_state_case,
        handle_swiglu_in_state_dict,
        print_diff_in_state_dicts,
    )

    HAVE_MEGATRON_FSDP = True
except ImportError:
    HAVE_MEGATRON_FSDP = False


# [ModelOpt]: Import
try:
    from modelopt.torch.opt.plugins import (
        restore_modelopt_state,
        restore_sharded_modelopt_state,
        save_modelopt_state,
        save_sharded_modelopt_state,
    )

    has_nvidia_modelopt = True
except Exception:
    has_nvidia_modelopt = False

TRACKER_PREFIX = "latest"
_CHECKPOINT_VERSION = None

logger = getLogger(__name__)
_NON_PERSISTENT_CKPT_SUBDIR = "non_persistent"


# ============================================================================
# Checkpoint version and utilities
# ============================================================================


def set_checkpoint_version(value: float) -> None:
    """Set the global checkpoint version number.

    Args:
        value: The checkpoint version number (e.g., 3.0).
    """
    global _CHECKPOINT_VERSION
    if _CHECKPOINT_VERSION is not None:
        assert _CHECKPOINT_VERSION == value, "checkpoint versions do not match"
    _CHECKPOINT_VERSION = value


def get_checkpoint_version() -> float | None:
    """Get the global checkpoint version number.

    Returns:
        The checkpoint version number, or None if not set.
    """
    global _CHECKPOINT_VERSION
    return _CHECKPOINT_VERSION


def delete_extra_state(state_dict):
    """Delete all extra state keys from the model state dictionary.

    This function removes all keys containing '_extra_state' from the model
    portion of the state dictionary. This is useful for cleaning up corrupted
    or problematic extra state that can cause issues during model loading.

    Args:
        state_dict: The state dictionary. Can be either:
                   - A full checkpoint dict with a "model" key, or
                   - A model state dict directly

    Returns:
        The modified state dictionary with extra state keys removed.
    """
    # Handle both cases: full checkpoint dict with "model" key or direct model state dict
    if isinstance(state_dict, dict) and "model" in state_dict:
        # Full checkpoint dict case
        target_dict = state_dict["model"]
    else:
        # Direct model state dict case
        target_dict = state_dict

    # If target is not a mapping-like object, nothing to clean
    if not hasattr(target_dict, "keys"):
        return state_dict

    # Some objects may implement keys() but not be directly iterable into a list (e.g., mocks)
    try:
        keys = list(target_dict.keys())
    except Exception:
        return state_dict

    for key in keys:
        if isinstance(key, str) and "_extra_state" in key:
            del target_dict[key]
    return state_dict


def _get_checkpoint_format(checkpoint_path: str) -> str:
    """Determine the checkpoint format by examining the checkpoint directory.

    Args:
        checkpoint_path: Path to the checkpoint directory.

    Returns:
        The checkpoint format string.
    """
    # Check for Megatron Core distributed checkpoint first
    if dist_checkpointing.check_is_distributed_checkpoint(checkpoint_path):
        return "torch_dist"

    # Check for PyTorch DCP format (.metadata file exists)
    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        checkpoint_dir = msc.Path(checkpoint_path)
        is_torch_dcp = checkpoint_dir.joinpath(".metadata").exists()
    else:
        is_torch_dcp = os.path.exists(os.path.join(checkpoint_path, ".metadata"))

    if is_torch_dcp:
        # Assume fsdp_dtensor for PyTorch DCP format
        return "fsdp_dtensor"
    else:
        raise NotImplementedError(f"Unknown checkpoint format in {checkpoint_path}")


def find_checkpoint_rank_0(checkpoints_path: str, iteration: int, release: bool = False) -> str | None:
    """Find the checkpoint directory for a given iteration, assuming distributed checkpoints.

    Args:
        checkpoints_path: Base directory where checkpoints are stored.
        iteration: The training iteration number.
        release: If True, searches within the 'release' directory.

    Returns:
        The full path to the checkpoint directory if it's a valid distributed checkpoint, else None.
    """
    # Get the base directory for the iteration using the simplified get_checkpoint_name
    checkpoint_dir = get_checkpoint_name(checkpoints_path, iteration, release=release)

    # Check if this directory is a valid distributed checkpoint
    if dist_checkpointing.check_is_distributed_checkpoint(checkpoint_dir):
        return checkpoint_dir

    return None


def read_metadata(tracker_filename: str) -> tuple[int, bool]:
    """Read the metadata from the Megatron-LM tracker file.

    Args:
        tracker_filename: Path to the tracker file.

    Returns:
        A tuple containing the iteration number and a boolean indicating if it's a release checkpoint.
    """
    iteration = 0
    release = False

    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        open_file = msc.open
    else:
        open_file = open

    with open_file(tracker_filename, "r") as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            release = metastring == "release"
            if not release:
                print_rank_0("ERROR: Invalid metadata file {}. Exiting".format(tracker_filename))
                sys.exit()
    assert iteration > 0 or release, "error parsing metadata file {}".format(tracker_filename)

    # Get the max iteration retrieved across the ranks.
    if torch.distributed.is_initialized():
        iters_cuda = torch.tensor([iteration], dtype=torch.long, device="cuda")
        torch.distributed.all_reduce(iters_cuda, op=torch.distributed.ReduceOp.MAX)
        max_iter = iters_cuda[0].item()

        # We should now have all the same iteration.
        # If not, print a warning and chose the maximum
        # iteration across all ranks.
        if iteration != max_iter:
            rank = torch.distributed.get_rank()
            print_rank_0(
                "WARNING: on rank {} found iteration {} in the "
                "metadata while max iteration across the ranks "
                "is {}, replacing it with max iteration.".format(rank, iteration, max_iter),
                flush=True,
            )
    else:
        # When loading a checkpoint outside of training (for example,
        # when editing it), we might not have torch distributed
        # initialized, in this case, just assume we have the latest
        max_iter = iteration
    return max_iter, release


def _extract_megatron_lm_args_from_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract and convert legacy Megatron-LM args from checkpoint state_dict to Megatron-Bridge config format.

    Args:
        state_dict: The loaded checkpoint state dictionary.

    Returns:
        A dictionary in Megatron-Bridge config format with the essential fields.

    Raises:
        RuntimeError: If args are not found in the state_dict.
    """
    if "args" not in state_dict:
        raise RuntimeError("Legacy checkpoint missing 'args' field in state_dict")

    args = state_dict["args"]

    # Convert args to minimal config container dict format
    config = {
        "model": {
            "tensor_model_parallel_size": getattr(args, "tensor_model_parallel_size", 1),
            "pipeline_model_parallel_size": getattr(args, "pipeline_model_parallel_size", 1),
            "encoder_tensor_model_parallel_size": getattr(args, "encoder_tensor_model_parallel_size", 0),
            "encoder_pipeline_model_parallel_size": getattr(args, "encoder_pipeline_model_parallel_size", 0),
        },
        "checkpoint": {
            "save_optim": not getattr(args, "no_save_optim", False),  # Invert no_save_optim
            "save_rng": not getattr(args, "no_save_rng", False),  # Invert no_save_rng
            "fully_parallel_save": getattr(args, "ckpt_fully_parallel_save", False),
        },
    }

    return config


# ============================================================================
# Async checkpoint utilities
# ============================================================================


def schedule_async_save(global_state: GlobalState, async_request: AsyncRequest) -> None:
    """Schedule the async save request.

    Args:
        global_state: The global training state containing the async calls queue.
        async_request: the async save request.
    """
    async_queue = global_state.async_calls_queue
    if async_queue is not None:
        async_queue.schedule_async_request(async_request)


def maybe_finalize_async_save(
    global_state: GlobalState, ckpt_cfg: CheckpointConfig, blocking: bool = False, terminate: bool = False
) -> None:
    """Finalizes active async save calls.

    Args:
        global_state: The global training state containing the async calls queue.
        ckpt_cfg (CheckpointConfig): The checkpoint configuration.
        blocking (bool, optional): if True, will wait until all active requests
            are done. Otherwise, finalizes only the async request that already
            finished. Defaults to False.
        terminate (bool, optional): if True, the asynchronous queue will
                be closed as the last action of this function.
    """
    if not ckpt_cfg.async_save:
        return

    async_queue = global_state.async_calls_queue
    if async_queue is None:
        return

    if blocking and not is_empty_async_queue(global_state):
        print_rank_0("Unfinalized async checkpoint saves. Finalizing them synchronously now.")

    async_queue.maybe_finalize_async_calls(blocking)

    if terminate:
        async_queue.close()


def is_empty_async_queue(global_state: GlobalState) -> bool:
    """Check if async calls queue is empty. This result is consistent across ranks.

    Args:
        global_state: The global training state containing the async calls queue.

    Returns:
        bool: True if there is any ongoing async call.
    """
    async_queue = global_state.async_calls_queue
    if async_queue is None:
        return True
    return async_queue.get_num_unfinalized_calls() == 0


def get_rng_state(data_parallel_random_init: bool, ckpt_format: str = "torch_dist") -> ShardedObject | dict:
    """Get the random number generator states for all necessary libraries.

    Collects states from random, numpy, torch, cuda, and the Megatron RNG tracker.
    Optionally gathers states across data parallel ranks.
    Returns format depends on checkpoint format.

    Args:
        data_parallel_random_init: If True, gathers RNG states across data parallel ranks.
        ckpt_format: The checkpoint format being used.

    Returns:
        For torch_dist: A ShardedObject containing the RNG states.
        For fsdp_dtensor: A dict mapping (pp_rank, tp_rank) to RNG state lists.
    """
    rng_state = {
        "random_rng_state": random.getstate(),
        "np_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
        "rng_tracker_states": tensor_parallel.get_cuda_rng_tracker().get_states(),
    }

    rng_state_list = None
    if torch.distributed.is_initialized() and mpu.get_data_parallel_world_size() > 1 and data_parallel_random_init:
        rng_state_list = [None for i in range(mpu.get_data_parallel_world_size())]
        torch.distributed.all_gather_object(rng_state_list, rng_state, group=mpu.get_data_parallel_group())
    else:
        rng_state_list = [rng_state]

    if ckpt_format == "torch_dist":
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        rng_state_list = ShardedObject(
            "rng_state",
            rng_state_list,
            (pp_size, tp_size),
            (pp_rank, tp_rank),
            replica_id=mpu.get_data_parallel_rank(with_context_parallel=True),
        )
    elif ckpt_format == "fsdp_dtensor":
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        rng_state_list = {f"({pp_rank}, {tp_rank})": rng_state_list}

    return rng_state_list


class CheckpointType(Enum):
    """Types of checkpoints to save."""

    LOCAL = auto()
    GLOBAL = auto()
    FSDP_DTENSOR = auto()


def save_checkpoint(
    state: GlobalState,
    model: list[MegatronModule],
    optimizer: MegatronOptimizer | None,
    opt_param_scheduler: Any | None,
    num_floating_point_operations_so_far: int,
    checkpointing_context: dict[str, Any] | None = None,
    pipeline_rank: int | None = None,
    tensor_rank: int | None = None,
    non_persistent_ckpt: bool = False,
    train_data_iterator: Any | None = None,
    preprocess_common_state_dict_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> None:
    """Save a model checkpoint.

    Handles saving the model state, optimizer state, scheduler state, RNG state,
    and other metadata based on the configuration and checkpoint type (global or local).
    Supports synchronous and asynchronous saving.

    Args:
        state: The GlobalState object.
        model: The model module(s) to save.
        optimizer: The optimizer instance.
        opt_param_scheduler: The optimizer parameter scheduler instance.
        num_floating_point_operations_so_far: Total FLOPs computed so far.
        checkpointing_context: Dictionary to store context across saves (e.g., strategies).
        pipeline_rank: Pipeline parallel rank (defaults to current rank).
        tensor_rank: Tensor parallel rank (defaults to current rank).
        non_persistent_ckpt: If True, saves as a non-persistent checkpoint.
        train_data_iterator: The training data iterator (for saving state if supported).
        preprocess_common_state_dict_fn: Optional function to preprocess the common state dict
                                         before consistency checks in distributed checkpointing.
    """

    train_state = state.train_state
    start_ckpt = time()
    cfg = state.cfg
    ckpt_cfg = cfg.checkpoint

    if ckpt_cfg.async_save and not is_empty_async_queue(state):
        print_rank_0(
            "WARNING: Starting a checkpoint save before previous has finished. "
            "Consider increasing the checkpoint interval."
        )

    # Monitor for the checkpointing timeout (no-op if FT is not enabled)
    fault_tolerance.on_checkpointing_start(state)

    # Only rank zero of the data parallel writes to the disk.
    model = unwrap_model(model)

    # Determine checkpoint type and save directory
    save_dir = ckpt_cfg.save
    if non_persistent_ckpt and ckpt_cfg.non_persistent_ckpt_type == "local":
        ckpt_type = CheckpointType.LOCAL
        save_dir = checkpointing_context["local_checkpoint_manager"].local_ckpt_dir
    elif non_persistent_ckpt and ckpt_cfg.non_persistent_ckpt_type == "global":
        ckpt_type = CheckpointType.GLOBAL
        save_dir = (
            ckpt_cfg.non_persistent_global_ckpt_dir
            if ckpt_cfg.non_persistent_global_ckpt_dir
            else os.path.join(save_dir, _NON_PERSISTENT_CKPT_SUBDIR)
        )
        # TODO Can we ensure the previous checkpoint is saved? We don't want to allow two saves in parallel.
        cleanup_old_non_persistent_checkpoint(save_dir, leave_ckpt_num=1, do_async=ckpt_cfg.async_save)
    elif non_persistent_ckpt:
        # Invalid non_persistent_ckpt_type value
        raise ValueError(
            f"Invalid non_persistent_ckpt_type: {ckpt_cfg.non_persistent_ckpt_type}. Must be 'local' or 'global'."
        )
    else:
        # Regular persistent checkpoint - always GLOBAL
        ckpt_type = CheckpointType.GLOBAL

    ckpt_format = ckpt_cfg.ckpt_format if ckpt_type == CheckpointType.GLOBAL else "torch"  # torch for local
    print_rank_0(f"saving checkpoint at iteration {train_state.step:7d} to {save_dir} in {ckpt_format} format")

    # Collect rng state across data parallel ranks.
    rng_state = get_rng_state(
        data_parallel_random_init=cfg.rng.data_parallel_random_init, ckpt_format=ckpt_cfg.ckpt_format
    )

    # Collect rerun state across all ranks
    rerun_state_machine = get_rerun_state_machine()
    rerun_state = rerun_state_machine.state_dict(
        data_iterator=train_data_iterator,
        ckpt_format=ckpt_cfg.ckpt_format,
    )

    # Checkpoint name.
    checkpoint_name = get_checkpoint_name(save_dir, train_state.step, release=False)

    # Save dataloader state if the dataloader supports it (currently only Megatron Energon).
    maybe_save_dataloader_state(train_data_iterator, train_state.step, getattr(cfg.dataset, "dataloader_save", None))

    async_save_request = None
    if ckpt_cfg.async_save:
        if ckpt_type == CheckpointType.GLOBAL and ckpt_cfg.ckpt_format != "torch_dist":
            raise NotImplementedError(
                f"Async checkpoint save not implemented for {ckpt_cfg.ckpt_format} distributed checkpoint format"
            )

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    # Collect cfg, model, RNG.
    sharded_sd_metadata = _build_sharded_state_dict_metadata(
        cfg.optimizer.use_distributed_optimizer, ckpt_cfg.fully_parallel_save, ckpt_cfg.ckpt_format
    )
    if cfg.optimizer.use_distributed_optimizer:
        print_rank_0(
            f"Storing distributed optimizer sharded state of type {sharded_sd_metadata['distrib_optim_sharding_type']}"
        )

    state_dict = generate_state_dict(
        ckpt_cfg,
        model,
        optimizer,
        opt_param_scheduler,
        rng_state,
        iteration=train_state.step,
        optim_sd_kwargs=dict(metadata=sharded_sd_metadata),
        model_sd_kwargs=dict(metadata=sharded_sd_metadata),
        rerun_state=rerun_state,
    )

    # Apply PEFT filtering to save adapter-only checkpoints
    if cfg.peft is not None:
        state_dict = apply_peft_adapter_filter_to_state_dict(state_dict, cfg.peft)

    if ckpt_type == CheckpointType.GLOBAL:
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            # TODO Handle non-empty directories (e.g., after a crash during saving).
            ensure_directory_exists(checkpoint_name, check_parent=False)

        if ckpt_cfg.ckpt_format == "fsdp_dtensor":
            # FSDP DTensor checkpoint save path using PyTorch Distributed Checkpointing
            fs_storage_writer = torch.distributed.checkpoint.FileSystemWriter(checkpoint_name)
            torch.distributed.checkpoint.save(
                state_dict=state_dict,
                storage_writer=fs_storage_writer,
            )
        else:
            # torch_dist and other formats using MCore distributed checkpointing
            if checkpointing_context is not None and "save_strategy" in checkpointing_context:
                save_strategy = checkpointing_context["save_strategy"]
                # Already saved once before - don't need to rerun sharding validation
                validate_sharding_integrity = not ckpt_cfg.ckpt_assume_constant_structure
            else:
                validate_sharding_integrity = True
                save_strategy = get_default_save_sharded_strategy(ckpt_cfg.ckpt_format)
                if ckpt_cfg.ckpt_assume_constant_structure and ckpt_cfg.ckpt_format == "torch_dist":
                    save_strategy.use_cached_ckpt_structure = ckpt_cfg.ckpt_assume_constant_structure
                    if checkpointing_context is not None and "load_strategy" in checkpointing_context:
                        cached_global_metadata = getattr(
                            checkpointing_context["load_strategy"], "cached_global_metadata", None
                        )
                        if cached_global_metadata is not None:
                            logger.debug("Plugging in the read metadata from the load strategy...")
                            save_strategy.cached_global_metadata = cached_global_metadata
                        else:
                            logger.debug("Failed to plug in the read metadata from the load strategy...")

                if ckpt_cfg.fully_parallel_save:
                    save_strategy = FullyParallelSaveStrategyWrapper(
                        save_strategy,
                        mpu.get_data_parallel_group(with_context_parallel=True),
                        ckpt_cfg.ckpt_assume_constant_structure,
                    )
            # Store save strategy for future checkpoint saves
            if checkpointing_context is not None:
                checkpointing_context["save_strategy"] = save_strategy
            end_ckpt = time()
            logger.debug(f"rank: {rank}, takes {end_ckpt - start_ckpt} to prepare state dict for ckpt ")
            async_save_request = dist_checkpointing.save(
                state_dict,
                checkpoint_name,
                save_strategy,
                async_sharded_save=ckpt_cfg.async_save,
                validate_access_integrity=validate_sharding_integrity,
                preprocess_common_before_consistancy_check=preprocess_common_state_dict_fn,
                content_metadata=sharded_sd_metadata,
            )
            # [ModelOpt]: save sharded modelopt_state
            if has_nvidia_modelopt:
                save_sharded_modelopt_state(model, checkpoint_name, (ckpt_cfg.ckpt_format, 1))
    else:
        # [ModelOpt]: Inject modelopt_state into state_dict
        if has_nvidia_modelopt:
            if ckpt_type == CheckpointType.LOCAL:
                print_rank_0("WARNING: Local checkpointing does not support nvidia_modelopt.")
            else:  # GLOBAL checkpoint type
                save_modelopt_state(model, state_dict)

        end_ckpt = time()
        logger.debug(f"rank: {rank}, takes {end_ckpt - start_ckpt} to prepare state dict for ckpt ")
        if ckpt_type == CheckpointType.LOCAL:
            try:
                from megatron.core.dist_checkpointing.tensor_aware_state_dict import MCoreTensorAwareStateDict
            except ModuleNotFoundError:
                raise RuntimeError(
                    "The 'nvidia_resiliency_ext' module is required for local "
                    "checkpointing but was not found. Please ensure it is installed."
                )
            algo = ckpt_cfg.non_persistent_local_ckpt_algo
            cached_metadata = None
            if ckpt_cfg.ckpt_assume_constant_structure and "local_checkpoint_cache" in checkpointing_context:
                cached_metadata = checkpointing_context["local_checkpoint_cache"]
            state_dict_for_save, cacheable_metadata = MCoreTensorAwareStateDict.from_state_dict(
                state_dict,
                algo=algo,
                cached_metadata=cached_metadata,
                parallelization_group=mpu.get_data_parallel_group(with_context_parallel=True),
            )
            async_save_request = checkpointing_context["local_checkpoint_manager"].save(
                state_dict_for_save, train_state.step, is_async=bool(ckpt_cfg.async_save)
            )
            checkpointing_context["local_checkpoint_cache"] = cacheable_metadata

    start_misc = time()
    if ckpt_type != CheckpointType.LOCAL:
        if not ckpt_cfg.async_save:
            assert async_save_request is None
            # Wait so everyone is done (necessary)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

    # And update the latest train state
    if get_rank_safe() == 0:
        train_state_local_filename = get_checkpoint_train_state_filename(checkpoint_name)
        train_state_global_filename = get_checkpoint_train_state_filename(save_dir, prefix=TRACKER_PREFIX)
        config_filename = get_checkpoint_run_config_filename(checkpoint_name)
        tracker_filename = get_checkpoint_tracker_filename(save_dir)
        if ckpt_type == CheckpointType.LOCAL:

            def train_state_finalize_fn():
                print_rank_0(f"  successfully saved local checkpoint from iteration {train_state.step:7d}")
                if cfg.logger.log_progress and ckpt_cfg.async_save:
                    append_to_progress_log(
                        ckpt_cfg.save, f"Saved async local checkpoint\tIteration: {train_state.step}", barrier=False
                    )

        else:
            train_state_dict = train_state.state_dict()

            def train_state_finalize_fn() -> None:
                train_state_dict["floating_point_operations_so_far"] = torch.tensor(
                    num_floating_point_operations_so_far, dtype=torch.float32
                )
                if MultiStorageClientFeature.is_enabled():
                    msc = MultiStorageClientFeature.import_package()
                    msc.torch.save(train_state_dict, train_state_local_filename)
                    msc.torch.save(train_state_dict, train_state_global_filename)
                    # Write Megatron-LM tracker file for compatibility
                    with msc.open(tracker_filename, "w") as f:
                        f.write(str(train_state.step))
                else:
                    torch.save(train_state_dict, train_state_local_filename)
                    shutil.copy(train_state_local_filename, train_state_global_filename)
                    # Write Megatron-LM tracker file for compatibility
                    with open(tracker_filename, "w") as f:
                        f.write(str(train_state.step))

                cfg.to_yaml(config_filename)

                # Save tokenizer files for self-contained checkpoints (if enabled)
                if ckpt_cfg.save_tokenizer_assets:
                    tokenizer_instance = getattr(cfg.dataset, "tokenizer", None) if cfg.dataset else None
                    if tokenizer_instance is not None:
                        save_tokenizer_assets(tokenizer_instance, cfg.tokenizer, checkpoint_name)

                tp_rank = (tensor_rank if tensor_rank is not None else mpu.get_tensor_model_parallel_rank()) + 1
                tp_world_size = mpu.get_tensor_model_parallel_world_size()
                pp_rank = (pipeline_rank if pipeline_rank is not None else mpu.get_pipeline_model_parallel_rank()) + 1
                pp_world_size = mpu.get_pipeline_model_parallel_world_size()
                print_rank_0(
                    f"  successfully saved checkpoint from iteration {train_state_dict['step'].item():7d} "
                    f"to {ckpt_cfg.save} [ t {tp_rank}/{tp_world_size}, p {pp_rank}/{pp_world_size} ]"
                )

                if cfg.logger.log_progress and ckpt_cfg.async_save:
                    append_to_progress_log(
                        ckpt_cfg.save,
                        f"Saved async checkpoint\tIteration: {train_state_dict['step'].item()}",
                        barrier=False,
                    )

        if ckpt_cfg.async_save:
            assert async_save_request is not None
            async_save_request.add_finalize_fn(train_state_finalize_fn)
        else:
            train_state_finalize_fn()

    # Ensure all ranks see a fully written checkpoint (e.g., run_config.yaml) before W&B scans.
    def _post_save_global_barrier() -> None:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    if ckpt_cfg.async_save:
        assert async_save_request is not None
        async_save_request.add_finalize_fn(_post_save_global_barrier)
    else:
        _post_save_global_barrier()

    # Additional callback for wandb (last rank)
    if not torch.distributed.is_initialized() or is_last_rank():

        def wandb_finalize_fn() -> None:
            wandb_utils.on_save_checkpoint_success(
                checkpoint_name,
                save_dir,
                train_state.step,
                wandb_writer=state.wandb_logger,
            )

        if ckpt_cfg.async_save:
            assert async_save_request is not None
            async_save_request.add_finalize_fn(wandb_finalize_fn)
        else:
            wandb_finalize_fn()

    if ckpt_cfg.async_save:
        schedule_async_save(state, async_save_request)
        print_rank_0(f"  scheduled an async checkpoint save at iteration {train_state.step:7d} to {save_dir}")

    end_misc = time()
    logger.debug(f"rank: {rank}, takes {end_misc - start_misc} to finalize ckpt save ")

    fault_tolerance.on_checkpointing_end(global_state=state, is_async_finalization=False)

    # keep only last k checkpoints
    if ckpt_cfg.most_recent_k > -1:
        cleanup_old_non_persistent_checkpoint(
            save_dir, leave_ckpt_num=ckpt_cfg.most_recent_k, do_async=ckpt_cfg.async_save
        )

    # Wait so everyone is done (not necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def cleanup_old_non_persistent_checkpoint(
    save_dir: str,
    leave_ckpt_num: int = 1,
    do_async: bool = False,
) -> None:
    """Clean up old non-persistent checkpoints in a directory.

    Keeps the specified number of latest checkpoints and removes older ones.
    Currently only cleans up directories matching "iter_*".

    Args:
        save_dir: The directory containing non-persistent checkpoints.
        leave_ckpt_num: The number of latest checkpoints to keep.
        do_async: If True, performs cleanup in a background thread.
    """
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return
    save_dir = Path(save_dir)

    iter_prefix = "iter_"
    iter_ckpts = save_dir.rglob(f"{iter_prefix}*")
    sorted_iter_ckpts = sorted(iter_ckpts, key=lambda ckpt_name: int(ckpt_name.name[len(iter_prefix) :]))
    if not sorted_iter_ckpts:
        return
    rm_iter_ckpts = sorted_iter_ckpts[:-leave_ckpt_num]
    print_rank_0(f"Non-persistent checkpoints scheduled for removal: {rm_iter_ckpts}")
    print_rank_0(f"Non-persistent checkpoints to be kept: {sorted_iter_ckpts[-leave_ckpt_num:]}")

    def remove_iter_ckpts(_iter_ckpts):
        for ckpt in _iter_ckpts:
            shutil.rmtree(ckpt)

    if do_async:
        threading.Thread(target=remove_iter_ckpts, args=(rm_iter_ckpts,)).start()
    else:
        remove_iter_ckpts(rm_iter_ckpts)


def maybe_save_dataloader_state(train_iterator: Any, iteration: int, dataloader_save_path: str | None) -> None:
    """Save the dataloader state if the iterator supports it.

    Checks if the train_iterator has a `save_state` method and calls it.

    Args:
        train_iterator: The training data iterator.
        iteration: The current training iteration.
        dataloader_save_path: The path where the dataloader state should be saved.
    """
    # If no dataloader or saving path is provided, exit early, otherwise, raise an error.
    if train_iterator is None or dataloader_save_path is None or dataloader_save_path == "":
        return

    # If dataloader doesn't support saving state, raise an error.
    if not hasattr(train_iterator.iterable, "save_state"):
        raise RuntimeError(f"Could not find a save_state for the train_iterator of type {type(train_iterator)}")

    # Save dataloader state for each data parallel rank only once.
    first_rank = mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0
    if not first_rank:
        return

    dp_rank = mpu.get_data_parallel_rank()
    print_rank_0(f"saving dataloader checkpoint at iteration {iteration} to {dataloader_save_path}")
    train_dataloader_state_dict = train_iterator.iterable.save_state()
    # Get the base directory for the current iteration
    iter_dir = get_checkpoint_name(dataloader_save_path, iteration)
    # Construct the specific filename within that iteration directory
    data_state_save_path = os.path.join(iter_dir, f"train_dataloader_dprank{dp_rank:03d}.pt")

    torch.distributed.barrier(group=mpu.get_data_parallel_group())

    if mpu.get_data_parallel_rank() == 0:
        ensure_directory_exists(data_state_save_path)

    torch.distributed.barrier(group=mpu.get_data_parallel_group())

    dataloader_save_dict = {}
    dataloader_save_dict["dataloader_state_dict"] = train_dataloader_state_dict
    torch.save(dataloader_save_dict, data_state_save_path)


def save_tokenizer_assets(
    tokenizer: MegatronTokenizer,
    tokenizer_config: TokenizerConfig,
    checkpoint_path: str,
) -> None:
    """Save tokenizer files to the checkpoint directory.

    Always saves tokenizer files to ensure checkpoints are self-contained
    and portable. Handles both HuggingFace tokenizers and file-based tokenizers.
    Compatible with MultiStorageClient for cloud storage support.

    Args:
        tokenizer: The tokenizer instance to save.
        tokenizer_config: The tokenizer configuration (used for file-based tokenizers).
        checkpoint_path: The checkpoint directory path.
    """
    if tokenizer is None:
        return

    # Only rank 0 saves tokenizer files
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank != 0:
        return

    def resolve_path(path_str: str) -> str:
        """Resolve relative paths to absolute paths."""
        if not path_str:
            return path_str
        path_obj = Path(path_str)
        if path_obj.is_absolute():
            return path_str
        # Resolve relative to current working directory
        return str(path_obj.resolve())

    try:
        # Check if MultiStorageClient is enabled
        if MultiStorageClientFeature.is_enabled():
            msc = MultiStorageClientFeature.import_package()
            checkpoint_path_obj = msc.Path(checkpoint_path)
            tokenizer_dir = checkpoint_path_obj / "tokenizer"
            tokenizer_dir.mkdir(parents=True, exist_ok=True)
            use_msc = True
        else:
            tokenizer_dir = os.path.join(checkpoint_path, "tokenizer")
            os.makedirs(tokenizer_dir, exist_ok=True)
            use_msc = False

        tokenizer_type = tokenizer_config.tokenizer_type

        # Handle HuggingFace and Multimodal tokenizers
        if tokenizer_type in ("HuggingFaceTokenizer", "MultimodalTokenizer"):
            if use_msc:
                import tempfile

                with tempfile.TemporaryDirectory() as tmp_dir:
                    if hasattr(tokenizer, "save_pretrained"):
                        tokenizer.save_pretrained(tmp_dir)
                    elif hasattr(tokenizer, "_tokenizer") and hasattr(tokenizer._tokenizer, "save_pretrained"):
                        tokenizer._tokenizer.save_pretrained(tmp_dir)
                    else:
                        logger.debug(f"{tokenizer_type} does not support save_pretrained(), skipping tokenizer save")
                        return

                    logger.debug(f"Saving {tokenizer_type} files to {tokenizer_dir}")
                    for filename in os.listdir(tmp_dir):
                        src_path = os.path.join(tmp_dir, filename)
                        if os.path.isfile(src_path):
                            dest_path = tokenizer_dir / filename
                            with open(src_path, "rb") as src_f:
                                with msc.open(str(dest_path), "wb") as dest_f:
                                    dest_f.write(src_f.read())
            else:
                logger.debug(f"Saving {tokenizer_type} files to {tokenizer_dir}")
                if hasattr(tokenizer, "save_pretrained"):
                    tokenizer.save_pretrained(tokenizer_dir)
                elif hasattr(tokenizer, "_tokenizer") and hasattr(tokenizer._tokenizer, "save_pretrained"):
                    tokenizer._tokenizer.save_pretrained(tokenizer_dir)
            return

        # Handle file-based tokenizers - resolve all paths
        files_to_copy = []

        if tokenizer_type in ("BertWordPieceLowerCase", "BertWordPieceCase"):
            if tokenizer_config.vocab_file:
                resolved_path = resolve_path(tokenizer_config.vocab_file)
                files_to_copy.append(("vocab_file", resolved_path, "vocab.txt"))

        elif tokenizer_type == "GPT2BPETokenizer":
            if tokenizer_config.vocab_file:
                resolved_path = resolve_path(tokenizer_config.vocab_file)
                files_to_copy.append(("vocab_file", resolved_path, "vocab.json"))
            if tokenizer_config.merge_file:
                resolved_path = resolve_path(tokenizer_config.merge_file)
                files_to_copy.append(("merge_file", resolved_path, "merges.txt"))

        elif tokenizer_type in ("SentencePieceTokenizer", "GPTSentencePieceTokenizer", "Llama2Tokenizer"):
            if tokenizer_config.tokenizer_model:
                resolved_path = resolve_path(tokenizer_config.tokenizer_model)
                files_to_copy.append(("tokenizer_model", resolved_path, "tokenizer.model"))

        elif tokenizer_type == "TikTokenizer":
            if tokenizer_config.tokenizer_model:
                resolved_path = resolve_path(tokenizer_config.tokenizer_model)
                files_to_copy.append(("tokenizer_model", resolved_path, "tokenizer.json"))

        elif tokenizer_type == "NullTokenizer":
            logger.debug(f"{tokenizer_type} requires no file artifacts")
            return

        # Copy the files
        if files_to_copy:
            logger.debug(f"Saving {tokenizer_type} files to {tokenizer_dir}")
            for config_attr, source_path, dest_filename in files_to_copy:
                if source_path and os.path.exists(source_path):
                    if use_msc:
                        dest_path = tokenizer_dir / dest_filename
                        with open(source_path, "rb") as src_f:
                            with msc.open(str(dest_path), "wb") as dest_f:
                                dest_f.write(src_f.read())
                        logger.debug(f"Copied {config_attr}: {source_path} -> {dest_path}")
                    else:
                        dest_path = os.path.join(tokenizer_dir, dest_filename)
                        shutil.copy2(source_path, dest_path)
                        logger.debug(f"Copied {config_attr}: {source_path} -> {dest_path}")
                else:
                    logger.debug(f"{config_attr} not found at resolved path: {source_path}")

    except Exception as e:
        if get_rank_safe() == 0:
            logger.error(f"Failed to save tokenizer files: {e}")
            import traceback

            logger.error(traceback.format_exc())


def _generate_model_state_dict(
    model: list[MegatronModule],
    model_sd_kwargs: dict[str, Any] | None = None,
    ckpt_format: str = "torch_dist",
) -> dict[str, ShardedStateDict]:
    """Generate the model subset of the state dictionary to be saved in a checkpoint.

    Can be added to the full checkpoint state dictionary with dict.update().

    Args:
        model: The model module(s).
        model_sd_kwargs: Metadata for model state dict generation.
        ckpt_format: The checkpoint format being used.

    Returns:
        A dictionary containing the model state to be saved.
    """
    state_dict = {}

    if len(model) == 1:
        if ckpt_format == "torch_dist":
            state_dict["model"] = model[0].sharded_state_dict(**(model_sd_kwargs or {}))
        else:  # fsdp_dtensor and other formats
            state_dict["model"] = model[0].state_dict_for_save_checkpoint()
    else:
        for i in range(len(model)):
            if ckpt_format == "torch_dist":
                state_dict["model%d" % i] = model[i].sharded_state_dict(**(model_sd_kwargs or {}))
            else:  # fsdp_dtensor and other formats
                state_dict["model%d" % i] = model[i].state_dict_for_save_checkpoint()

    return state_dict


def generate_state_dict(
    ckpt_cfg: CheckpointConfig,
    model: list[MegatronModule],
    optimizer: MegatronOptimizer | None,
    opt_param_scheduler: Any | None,
    rng_state: ShardedObject | None,
    iteration: int | None = None,
    optim_sd_kwargs: dict[str, Any] | None = None,
    model_sd_kwargs: dict[str, Any] | None = None,
    rerun_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate the state dictionary to be saved in a checkpoint.

    Args:
        cfg: The configuration container.
        model: The model module(s).
        optimizer: The optimizer instance.
        opt_param_scheduler: The optimizer parameter scheduler instance.
        rng_state: Collected RNG states as a ShardedObject.
        iteration: The current training iteration.
        optim_sd_kwargs: Additional keyword arguments for optimizer state dict generation.
        model_sd_kwargs: Metadata for model state dict generation.
        rerun_state: State dictionary from the rerun state machine.

    Returns:
        A dictionary containing the complete state to be saved.
    """
    # Arguments, iteration, and model.
    state_dict = {}
    state_dict["checkpoint_version"] = 3.0
    if iteration is not None:
        state_dict["iteration"] = iteration

    state_dict.update(_generate_model_state_dict(model, model_sd_kwargs, ckpt_cfg.ckpt_format))

    # Optimizer stuff.
    if ckpt_cfg.save_optim:
        if optimizer is not None and not getattr(optimizer, "is_stub_optimizer", False):
            if ckpt_cfg.ckpt_format == "torch_dist":
                state_dict["optimizer"] = optimizer.sharded_state_dict(state_dict, **(optim_sd_kwargs or {}))
            elif ckpt_cfg.ckpt_format == "fsdp_dtensor":
                if optim_sd_kwargs is None:
                    optim_sd_kwargs = {}
                if "metadata" not in optim_sd_kwargs:
                    optim_sd_kwargs["metadata"] = {}
                # Use the metadata that was passed in (should include FSDP-specific metadata)
                state_dict["optimizer"] = optimizer.sharded_state_dict(state_dict, **optim_sd_kwargs)
            else:
                state_dict["optimizer"] = optimizer.state_dict()
        if opt_param_scheduler is not None:
            state_dict["opt_param_scheduler"] = opt_param_scheduler.state_dict()

    # Rerun state
    if rerun_state is not None:
        state_dict["rerun_state_machine"] = rerun_state

    # RNG states.
    if ckpt_cfg.save_rng:
        state_dict["rng_state"] = rng_state

    # FSDP DTensor checkpoint specific state dict preprocessing
    if ckpt_cfg.ckpt_format == "fsdp_dtensor":
        if not HAVE_MEGATRON_FSDP:
            raise RuntimeError("Megatron FSDP is enabled but Megatron-FSDP is not available.")
        if len(model) != 1:
            raise RuntimeError("FSDP DTensor checkpoints are not supported for multiple models.")

        # Apply FSDP-specific preprocessing for SwiGLU
        from megatron.core.utils import get_model_config

        model_config = get_model_config(model[0])
        # SWiGLU is enabled when activation is SiLU and GLU gating is on
        is_swiglu = (
            getattr(model_config, "gated_linear_unit", False)
            and getattr(model_config, "activation_func", None) is F.silu
        )
        if is_swiglu:
            state_dict = state_dict.copy()
            handle_swiglu_in_state_dict(model[0], state_dict["model"], state_dict.get("optimizer"))
        handle_fp8_extra_state_case(state_dict["model"])
        preprocess_state_dict_for_uneven_dtensor(state_dict)

    return state_dict


def _load_model_weights_from_checkpoint(
    checkpoint_path: str,
    model: list[MegatronModule],
    fully_parallel_load: bool = False,
    return_state_dict: bool = False,
    dist_ckpt_strictness: Literal[
        "assume_ok_unexpected",
        "log_unexpected",
        "log_all",
        "raise_unexpected",
        "raise_all",
        "return_unexpected",
        "return_all",
        "ignore_all",
    ] = "assume_ok_unexpected",
    strict: bool = True,
) -> StateDict | tuple[StateDict, set[str], set[str]] | None:
    """Load model weights from a checkpoint.

    MCore distributed checkpoints from both Megatron Bridge and MegatronLM are supported.
    This function duplicates some logic from load_checkpoint() to simplify model
    loading for inference.

    Args:
        checkpoint_path: path to a distributed checkpoint.
        model: The model module(s) to load weights into.
        fully_parallel_load: Apply full load parallelization across DP.
        return_state_dict: Skips loading state dict into model and returns model state dict
            itself. Default False.
        dist_ckpt_strictness: Determine handling of key mismatch during checkpoint load.
        strict: Whether to enforce strict loading (see torch.nn.Module.load_state_dict).
    """

    state_dict = dist_checkpointing.load_common_state_dict(checkpoint_path)
    assert state_dict is not None

    sharded_sd_metadata = dist_checkpointing.load_content_metadata(preloaded_state_dict=state_dict)
    print_rank_0(f"sharded_state_dict metadata loaded from the checkpoint: {sharded_sd_metadata}")
    model_sd_kwargs = dict(metadata=sharded_sd_metadata)

    if has_nvidia_modelopt:
        restore_modelopt_state(model, state_dict)

    model = unwrap_model(model)
    sharded_state_dict = _generate_model_state_dict(model, model_sd_kwargs)

    load_strategy = get_default_load_sharded_strategy(checkpoint_path)
    if fully_parallel_load:
        load_strategy = FullyParallelLoadStrategyWrapper(
            load_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
        )
    state_dict = dist_checkpointing.load(
        sharded_state_dict, checkpoint_path, load_strategy, strict=dist_ckpt_strictness
    )
    # we keep weights only for bridge use, remove extra state
    # because they are not needed and could cause unexpected issues.
    delete_extra_state(state_dict)
    if return_state_dict:
        return state_dict

    if len(model) == 1:
        _load_model_state_dict(model[0], state_dict["model"], strict)
    else:
        for i in range(len(model)):
            # If there is no corresponding model in the state_dict, it will be ignored.
            # It means that this is an empty stage.
            model_key = "model%d" % i
            if model_key not in state_dict:
                continue
            _load_model_state_dict(model[i], state_dict[model_key], strict)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def load_checkpoint(
    state: GlobalState,
    model: list[MegatronModule],
    optimizer: MegatronOptimizer | None,
    opt_param_scheduler: Any | None,
    strict: bool = True,
    checkpointing_context: dict[str, Any] | None = None,
    skip_load_to_model_and_opt: bool = False,
) -> tuple[int, int]:
    """Load a model checkpoint.

    Handles loading model state, optimizer state, scheduler state, RNG state,
    and other metadata based on the configuration and checkpoint type.
    Supports loading global distributed and local non-persistent checkpoints.

    Args:
        state: The GlobalState object.
        model: The model module(s) to load state into.
        optimizer: The optimizer instance to load state into.
        opt_param_scheduler: The scheduler instance to load state into.
        strict: Whether to enforce strict loading (see torch.nn.Module.load_state_dict).
        checkpointing_context: Dictionary to store context across loads (e.g., strategies).
        skip_load_to_model_and_opt: If True, only loads metadata (iteration, rng) but
                                      skips loading state into model and optimizer modules.

    Returns:
        A tuple containing:
        - iteration: The training iteration number.
        - num_floating_point_operations_so_far: The total FLOPs computed so far.
    """
    cfg = state.cfg
    load_dir = cfg.checkpoint.load

    # Finetuning directories
    pretrained_dir = cfg.checkpoint.pretrained_checkpoint
    if pretrained_dir is not None and not checkpoint_exists(load_dir):
        print_rank_0(
            f"Checkpoint file not found in load directory {load_dir}. "
            f"Attempting to finetune with checkpoint in {pretrained_dir}"
        )
        load_dir = pretrained_dir
        if not checkpoint_exists(load_dir):
            raise FileNotFoundError("No checkpoint found in load directory or pretrained directory")
        cfg.checkpoint.finetune = True

    return _load_checkpoint_from_path(
        load_dir, state, model, optimizer, opt_param_scheduler, strict, checkpointing_context
    )


def _load_model_state_dict(module: torch.nn.Module, state_dict: dict[str, Any], strict: bool):
    """Helper function to load state dict with fallback for missing extra states."""
    try:
        module.load_state_dict(state_dict, strict=strict)
    except Exception as e:
        if strict:
            # Fallback support for backward compatibility breaking changes in TransformerEngine
            print_rank_0(f"Warning: Exception during strict loading: {e}")
            load_return = module.load_state_dict(state_dict, strict=False)
            print_rank_0(f"load_return: {load_return}")
        else:
            # Re-raise if we were already in non-strict mode
            raise


def _load_checkpoint_from_path(
    load_dir: str,
    state: GlobalState,
    model: list[MegatronModule],
    optimizer: MegatronOptimizer | None,
    opt_param_scheduler: Any | None,
    strict: bool = True,
    checkpointing_context: dict[str, Any] | None = None,
    skip_load_to_model_and_opt: bool = False,
) -> tuple[int, int]:
    """Load a checkpoint from a given path.

    Args:
        load_dir: The directory containing the checkpoint.
        state: The GlobalState object.
        model: The model module(s) to load state into.
        optimizer: The optimizer instance to load state into.
        opt_param_scheduler: The scheduler instance to load state into.
        strict: Whether to enforce strict loading (see torch.nn.Module.load_state_dict).
        checkpointing_context: Dictionary to store context across loads (e.g., strategies).
        skip_load_to_model_and_opt: If True, only loads metadata (iteration, rng) but
                                      skips loading state into model and optimizer modules.

    Returns:
        A tuple containing:
        - iteration: The training iteration number.
        - num_floating_point_operations_so_far: The total FLOPs computed so far.
    """
    cfg = state.cfg
    model = unwrap_model(model)
    ckpt_format = cfg.checkpoint.ckpt_format

    # Step 1: Load base checkpoint with rank0=True (torch_dist only)
    if ckpt_format == "torch_dist":
        state_dict, checkpoint_name, release, ckpt_type = _load_base_checkpoint(
            load_dir, cfg.checkpoint, rank0=True, checkpointing_context=checkpointing_context
        )

    # Step 2: Initialize scaffolding
    load_kwargs = {}
    ignore_rng_state = False
    ignore_rerun_state = True

    # Step 3: Format-specific preparation
    if ckpt_format == "torch_dist":
        if state_dict is None:
            return 0, 0

        # Read run_config for TP/PP compatibility checks
        run_config_filename = get_checkpoint_run_config_filename(checkpoint_name)
        if file_exists(run_config_filename):
            run_config = read_run_config(run_config_filename)
        else:
            print_rank_0("run_config.yaml not found, extracting config from legacy Megatron-LM checkpoint")
            run_config = _extract_megatron_lm_args_from_state_dict(state_dict)

        ckpt_tp_pp = (
            run_config["model"]["tensor_model_parallel_size"],
            run_config["model"]["pipeline_model_parallel_size"],
        )
        run_tp_pp = (
            cfg.model.tensor_model_parallel_size,
            cfg.model.pipeline_model_parallel_size,
        )
        mismatch_msg = "(TP, PP) mismatch after resume ({} vs {} from checkpoint)".format(run_tp_pp, ckpt_tp_pp)

        # Determine if RNG state will be loaded
        if (
            ckpt_tp_pp == run_tp_pp
            and not release
            and not cfg.checkpoint.finetune
            and cfg.checkpoint.load_rng
            and run_config["checkpoint"]["save_rng"]
        ):
            gen_sd_rng_state = get_rng_state(cfg.rng.data_parallel_random_init, ckpt_format)
        else:
            ignore_rng_state = True
            gen_sd_rng_state = None
            if ckpt_tp_pp != run_tp_pp:
                print_rank_0("{}: RNG state will be ignored".format(mismatch_msg))

        sharded_sd_metadata = dist_checkpointing.load_content_metadata(preloaded_state_dict=state_dict)
        print_rank_0(f"sharded_state_dict metadata loaded from the checkpoint: {sharded_sd_metadata}")

        # Determine if optimizer state will be loaded
        if (
            not release
            and not cfg.checkpoint.finetune
            and cfg.checkpoint.load_optim
            and run_config["checkpoint"]["save_optim"]
        ):
            gen_sd_optim = optimizer
            gen_sd_opt_param_scheduler = opt_param_scheduler

            if cfg.optimizer.use_distributed_optimizer:
                if sharded_sd_metadata is None:
                    sharded_sd_metadata = {
                        "distrib_optim_sharding_type": (
                            "fully_sharded_model_space"
                            if run_config["checkpoint"]["fully_parallel_save"]
                            else "dp_zero_gather_scatter"
                        ),
                    }
                if (
                    ckpt_tp_pp != run_tp_pp
                    and sharded_sd_metadata["distrib_optim_sharding_type"] != "fully_sharded_model_space"
                ):
                    raise RuntimeError(
                        f"{mismatch_msg}: not supported for DistributedOptimizer with sharding type"
                        f" {sharded_sd_metadata['distrib_optim_sharding_type']}."
                        f" Please use `checkpoint_config.fully_parallel_save=True` for checkpoint saving."
                    )
        else:
            gen_sd_optim = None
            gen_sd_opt_param_scheduler = None

        # Determine if rerun state will be loaded
        if (
            ckpt_tp_pp == run_tp_pp
            and not release
            and not cfg.checkpoint.finetune
            and "rerun_state_machine" in state_dict
        ):
            rerun_state_machine = get_rerun_state_machine()
            gen_sd_rerun_state = rerun_state_machine.state_dict(
                data_iterator=None, ckpt_format=ckpt_format, force=True
            )
            ignore_rerun_state = False
        else:
            gen_sd_rerun_state = None
            if ckpt_tp_pp != run_tp_pp:
                print_rank_0("{}: Rerun state will be ignored".format(mismatch_msg))

        optim_sd_kwargs = dict(metadata=sharded_sd_metadata, is_loading=True)
        model_sd_kwargs = dict(metadata=sharded_sd_metadata)

        # ModelOpt restoration
        if has_nvidia_modelopt:
            if ckpt_type == CheckpointType.LOCAL:
                print_rank_0("WARNING: Local checkpointing does not support nvidia_modelopt.")
            elif ckpt_type == CheckpointType.GLOBAL:
                restore_modelopt_state(model, state_dict)
            else:
                restore_sharded_modelopt_state(model, checkpoint_name)

        # Build sharded state dict for loading
        with contextlib.ExitStack() as stack:
            if cfg.checkpoint.finetune and hasattr(model[0], "hide_loss_modules"):
                for m in model:
                    stack.enter_context(m.hide_loss_modules())
            load_kwargs["sharded_state_dict"] = generate_state_dict(
                cfg.checkpoint,
                model,
                gen_sd_optim,
                gen_sd_opt_param_scheduler,
                gen_sd_rng_state,
                optim_sd_kwargs=optim_sd_kwargs,
                model_sd_kwargs=model_sd_kwargs,
                rerun_state=gen_sd_rerun_state,
            )

    elif ckpt_format == "fsdp_dtensor":
        # Handle fsdp_dtensor format
        from torch.distributed.checkpoint import FileSystemReader

        # Get checkpoint path using Bridge's utilities
        tracker_filename = get_checkpoint_train_state_filename(load_dir, prefix=TRACKER_PREFIX)
        if file_exists(tracker_filename):
            train_state = read_train_state(tracker_filename)
            iteration = train_state.step
            release = False
        else:
            # Fallback to legacy Megatron-LM tracker
            legacy_tracker_filename = get_checkpoint_tracker_filename(load_dir)
            if file_exists(legacy_tracker_filename):
                iteration, release = read_metadata(legacy_tracker_filename)
            else:
                print_rank_0(f"WARNING: could not find metadata file in {load_dir}")
                return 0, 0

        checkpoint_name = get_checkpoint_name(load_dir, iteration, release)
        reader = FileSystemReader(checkpoint_name)
        try:
            state_dict_metadata = reader.read_metadata().state_dict_metadata
        except FileNotFoundError:
            state_dict_metadata = {}

        # Decide what sections to load based on metadata and config
        gen_sd_rerun_state = None
        gen_sd_opt_param_scheduler = None
        gen_sd_rng_state = None
        gen_sd_optim = None
        if not cfg.checkpoint.finetune:
            if "rerun_state_machine" in state_dict_metadata:
                gen_sd_rerun_state = get_rerun_state_machine().state_dict(
                    data_iterator=None, ckpt_format=ckpt_format, force=True
                )
            if cfg.checkpoint.load_rng:
                gen_sd_rng_state = get_rng_state(cfg.rng.data_parallel_random_init, ckpt_format)
            if cfg.checkpoint.load_optim:
                gen_sd_optim = optimizer
                gen_sd_opt_param_scheduler = opt_param_scheduler

        optim_sd_kwargs = dict(
            metadata=_build_sharded_state_dict_metadata(
                cfg.optimizer.use_distributed_optimizer, cfg.checkpoint.fully_parallel_save, ckpt_format
            ),
            is_loading=True,
        )

        load_kwargs["sharded_state_dict"] = generate_state_dict(
            cfg.checkpoint,
            model=model,
            optimizer=gen_sd_optim,
            opt_param_scheduler=gen_sd_opt_param_scheduler,
            rng_state=gen_sd_rng_state,
            optim_sd_kwargs=optim_sd_kwargs,
            rerun_state=gen_sd_rerun_state,
            iteration=1,
        )

    # Apply PEFT resume filtering (common across all checkpoint formats)
    is_peft_resume = (
        cfg.peft is not None
        and cfg.checkpoint.load is not None
        and load_dir == cfg.checkpoint.load
        and load_dir != cfg.checkpoint.pretrained_checkpoint
        and not cfg.checkpoint.finetune
    )
    if is_peft_resume and "sharded_state_dict" in load_kwargs:
        load_kwargs["sharded_state_dict"] = apply_peft_adapter_filter_to_state_dict(
            load_kwargs["sharded_state_dict"], cfg.peft
        )

    # Load the checkpoint
    state_dict, checkpoint_name, release, ckpt_type = _load_base_checkpoint(
        load_dir, cfg.checkpoint, rank0=False, checkpointing_context=checkpointing_context, **load_kwargs
    )

    # Checkpoint not loaded
    if state_dict is None:
        return 0, 0

    # Common finalization
    set_checkpoint_version(state_dict.get("checkpoint_version", 0))

    # Handle train state
    if not cfg.checkpoint.finetune:
        train_state_filename = get_checkpoint_train_state_filename(checkpoint_name)
        if file_exists(train_state_filename):
            state.train_state = read_train_state(train_state_filename)
        else:
            print_rank_0(f"{train_state_filename} not found, creating TrainState from checkpoint state dict")
            state.train_state = _get_train_state_from_state_dict(state_dict)

    if cfg.checkpoint.finetune or release:
        state.train_state.step = 0

    if not cfg.checkpoint.finetune:
        update_num_microbatches(consumed_samples=state.train_state.consumed_train_samples, verbose=True)

    # Load model weights
    if not skip_load_to_model_and_opt:
        # Handle PEFT resume for strict loading
        load_strict = strict
        is_peft_resume = (
            cfg.peft is not None
            and cfg.checkpoint.load is not None
            and load_dir == cfg.checkpoint.load
            and load_dir != cfg.checkpoint.pretrained_checkpoint
            and not cfg.checkpoint.finetune
        )
        load_strict = False if is_peft_resume else strict

        if len(model) == 1:
            _load_model_state_dict(model[0], state_dict["model"], load_strict)
        else:
            for i in range(len(model)):
                model_key = "model%d" % i
                if model_key not in state_dict:
                    continue
                _load_model_state_dict(model[i], state_dict[model_key], load_strict)

    checkpoint_version = get_checkpoint_version()
    print_rank_0(f" checkpoint version {checkpoint_version}")

    # Load optimizer and scheduler
    if not release and not cfg.checkpoint.finetune and cfg.checkpoint.load_optim:
        try:
            if (
                not skip_load_to_model_and_opt
                and optimizer is not None
                and not getattr(optimizer, "is_stub_optimizer", False)
            ):
                optimizer.load_state_dict(state_dict["optimizer"])

            if opt_param_scheduler is not None:
                if "lr_scheduler" in state_dict:
                    opt_param_scheduler.load_state_dict(state_dict["lr_scheduler"])
                else:
                    opt_param_scheduler.load_state_dict(state_dict["opt_param_scheduler"])
        except KeyError as e:
            print_rank_0(
                "Unable to load optimizer from checkpoint {}. "
                "Specify load_optim=False or finetune=True to prevent "
                "attempting to load the optimizer state.".format(checkpoint_name)
            )
            raise e
    else:
        if (cfg.model.fp16 or cfg.model.bf16) and optimizer is not None:
            if cfg.checkpoint.load_main_params_from_ckpt:
                optimizer.reload_model_params(state_dict=state_dict)
            else:
                optimizer.reload_model_params()

    # Load rerun state
    if not ignore_rerun_state:
        try:
            if "rerun_state_machine" in state_dict:
                get_rerun_state_machine().load_state_dict(state_dict["rerun_state_machine"])
        except Exception as e:
            print_rank_0(f"Unable to restore RerunMachine from checkpoint: {e}. Skipping.")
            sys.exit()

    # Load RNG states
    if not release and not cfg.checkpoint.finetune and cfg.checkpoint.load_rng and not ignore_rng_state:
        try:
            if "rng_state" in state_dict:
                if ckpt_format == "fsdp_dtensor":
                    # FSDP DTensor format: {(pp_rank, tp_rank): rng_state_list}
                    tp_rank = mpu.get_tensor_model_parallel_rank()
                    pp_rank = mpu.get_pipeline_model_parallel_rank()
                    key = f"({pp_rank}, {tp_rank})"
                    if key in state_dict["rng_state"]:
                        rng_state_list = state_dict["rng_state"][key]
                    else:
                        print_rank_0("WARNING: RNG state not found for current TP/PP rank")
                        rng_state_list = next(iter(state_dict["rng_state"].values()))
                    rng_state = (
                        rng_state_list[mpu.get_data_parallel_rank()]
                        if cfg.rng.data_parallel_random_init
                        else rng_state_list[0]
                    )
                else:
                    # torch_dist format: ShardedObject
                    rng_state = (
                        state_dict["rng_state"][mpu.get_data_parallel_rank()]
                        if cfg.rng.data_parallel_random_init
                        else state_dict["rng_state"][0]
                    )

                random.setstate(rng_state["random_rng_state"])
                np.random.set_state(rng_state["np_rng_state"])
                torch.set_rng_state(rng_state["torch_rng_state"])
                torch.cuda.set_rng_state(rng_state["cuda_rng_state"])
                if not rng_state["rng_tracker_states"]:
                    raise KeyError
                tensor_parallel.get_cuda_rng_tracker().set_states(rng_state["rng_tracker_states"])
            else:  # backward compatibility
                random.setstate(state_dict["random_rng_state"])
                np.random.set_state(state_dict["np_rng_state"])
                torch.set_rng_state(state_dict["torch_rng_state"])
                torch.cuda.set_rng_state(state_dict["cuda_rng_state"])
                if not state_dict["rng_tracker_states"]:
                    raise KeyError
                tensor_parallel.get_cuda_rng_tracker().set_states(state_dict["rng_tracker_states"])
        except KeyError:
            print_rank_0(
                "Unable to load rng state from checkpoint {}. "
                "Specify load_rng=False or finetune=True to prevent "
                "attempting to load the rng state.".format(checkpoint_name)
            )
            sys.exit()

    # Final synchronization and logging
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0(
        f"  successfully loaded checkpoint from {load_dir} "
        f"[ t {mpu.get_tensor_model_parallel_rank() + 1}/{mpu.get_tensor_model_parallel_world_size()}, "
        f"p {mpu.get_pipeline_model_parallel_rank() + 1}/{mpu.get_pipeline_model_parallel_world_size()} ] "
        f"at iteration {state.train_state.step}"
    )

    if not torch.distributed.is_initialized() or is_last_rank():
        wandb_utils.on_load_checkpoint_success(checkpoint_name, load_dir, state.wandb_logger)

    torch.cuda.empty_cache()

    if state.train_state.step > 0:
        is_local_chkpt = ckpt_type == CheckpointType.LOCAL
        fault_tolerance.on_checkpoint_loaded(is_local_chkpt=is_local_chkpt, global_state=state)

    return state.train_state.step, state.train_state.floating_point_operations_so_far


def init_checkpointing_context(checkpoint_config: CheckpointConfig) -> dict[str, Any]:
    """Initialize the checkpointing context, primarily for local checkpointing support.

    If `non_persistent_ckpt_type` is set to "local", this function sets up
    the `LocalCheckpointManager` and replication strategy based on the provided
    `checkpoint_config`.

    Args:
        checkpoint_config: The checkpoint configuration object.

    Returns:
        A dictionary containing the checkpointing context. This will include
        a `local_checkpoint_manager` if local checkpointing is enabled,
        otherwise it will be an empty dictionary.

    Raises:
        RuntimeError: If local checkpointing is configured but the
                      `nvidia_resiliency_ext` module is not found.
    """
    if checkpoint_config.non_persistent_ckpt_type != "local":
        return {}

    if not HAVE_RESIL:
        raise RuntimeError(
            "The 'nvidia_resiliency_ext' module is required for local "
            "checkpointing but was not found. Please ensure it is installed."
        )

    from nvidia_resiliency_ext.checkpointing.local.ckpt_managers.local_manager import LocalCheckpointManager
    from nvidia_resiliency_ext.checkpointing.local.replication.strategies import CliqueReplicationStrategy

    if checkpoint_config.replication:
        repl_strategy = CliqueReplicationStrategy.from_replication_params(
            checkpoint_config.replication_jump,
            checkpoint_config.replication_factor,
        )
    else:
        repl_strategy = None

    checkpointing_context = {
        "local_checkpoint_manager": LocalCheckpointManager(
            checkpoint_config.non_persistent_local_ckpt_dir,
            repl_strategy=repl_strategy,
        )
    }
    return checkpointing_context


def apply_peft_adapter_filter_to_state_dict(state_dict: dict[str, Any], peft_config: PEFT) -> dict[str, Any]:
    """Filter state dict to contain only PEFT adapter parameters in model sections.

    This function takes a complete state dict (generated by generate_state_dict) and
    filters it to retain only PEFT adapter parameters for checkpoint saving.
    Follows the same key logic pattern as generate_state_dict for consistency.

    Args:
        state_dict: Complete state dict from generate_state_dict()
        peft_config: PEFT configuration for filtering logic

    Returns:
        Filtered state dict containing only adapter parameters in model weights,
        while preserving all non-model metadata (checkpoint_version, iteration, etc.)
    """
    return {
        checkpoint_section_key: (
            # Filter model parameters to only include adapter weights
            {
                parameter_name: parameter_value
                for parameter_name, parameter_value in checkpoint_section_value.items()
                if peft_config.adapter_key_filter(parameter_name)
            }
            if _is_model_section(checkpoint_section_key)
            else checkpoint_section_value
        )
        for checkpoint_section_key, checkpoint_section_value in state_dict.items()
    }


def _is_model_section(section_key: str) -> bool:
    """Check if a checkpoint section contains model parameters.

    Model sections are named:
    - "model" (single model)
    - "model0", "model1", etc. (pipeline parallel models)

    Non-model sections include: "optimizer", "iteration", "checkpoint_version", etc.
    """
    is_single_model = section_key == "model"
    is_pipeline_model = (
        section_key.startswith("model")
        and section_key != "model"
        and section_key[5:].isdigit()  # to match virtual pipeline state dict handling
    )
    return is_single_model or is_pipeline_model


def _transpose_first_dim(
    t: torch.Tensor, num_splits: int, num_splits_first: bool, model: torch.nn.Module
) -> torch.Tensor:
    """Helper function to transpose first dimension of tensor t."""
    input_shape = t.size()
    # We use a self_attention module but the values extracted aren't
    # specific to self attention so should work for cross attention as well
    while hasattr(model, "module"):
        model = model.module
    attention_module = model.language_model.encoder.layers[0].self_attention
    hidden_size_per_attention_head = attention_module.hidden_size_per_attention_head
    num_attention_heads_per_partition = attention_module.num_attention_heads_per_partition
    if num_splits_first:
        """[num_splits * np * hn, h]
        -->(view) [num_splits, np, hn, h]
        -->(tranpose) [np, num_splits, hn, h]
        -->(view) [np * num_splits * hn, h]"""

        intermediate_shape = (
            num_splits,
            num_attention_heads_per_partition,
            hidden_size_per_attention_head,
        ) + input_shape[1:]

        t = t.view(*intermediate_shape)
        t = t.transpose(0, 1).contiguous()
    else:
        """[np * hn * num_splits, h]
        -->(view) [np, hn, num_splits, h]
        -->(tranpose) [np, num_splits, hn, h]
        -->(view) [np * num_splits * hn, h]"""

        intermediate_shape = (
            num_attention_heads_per_partition,
            hidden_size_per_attention_head,
            num_splits,
        ) + input_shape[1:]

        t = t.view(*intermediate_shape)
        t = t.transpose(1, 2).contiguous()
    t = t.view(*input_shape)

    return t


def _get_non_persistent_iteration(
    non_persistent_global_dir: str,
    non_persistent_ckpt_type: Literal["global", "local"] | None = None,
    checkpointing_context: dict[str, Any] | None = None,
) -> int:
    """Get iteration number from non-persistent checkpoint."""
    if non_persistent_ckpt_type is None:
        return -1
    elif non_persistent_ckpt_type == "global":
        train_state_filename = get_checkpoint_train_state_filename(non_persistent_global_dir, prefix=TRACKER_PREFIX)
        if file_exists(train_state_filename):
            train_state = read_train_state(train_state_filename)
            iteration = train_state.step
            # if train_state.release:
            #     raise RuntimeError("Non-persistent checkpoint can't be a release checkpoint")
        else:
            iteration = -1
            print_rank_0("WARNING: could not find the metadata file {}".format(train_state_filename))
            print_rank_0("    will not load any non-persistent checkpoint")
        return iteration
    elif non_persistent_ckpt_type == "local":
        return checkpointing_context["local_checkpoint_manager"].find_latest()
    else:
        raise ValueError(f"Please use local or global non-persistent checkpoints. Got: {non_persistent_ckpt_type})")


def _load_non_persistent_base_checkpoint(
    non_persistent_global_dir: str,
    ckpt_cfg: CheckpointConfig,
    rank0: bool,
    sharded_state_dict: dict[str, Any] | None,
    non_persistent_iteration: int,
    checkpointing_context: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str, bool, CheckpointType]:
    """Load the base state_dict from a non-persistent distributed checkpoint."""
    assert ckpt_cfg.non_persistent_ckpt_type is not None
    if ckpt_cfg.non_persistent_ckpt_type == "global":
        if not rank0:
            print_rank_0(f"Loading from a non-persistent checkpoint (non-persistent iter {non_persistent_iteration})")
        return _load_global_dist_base_checkpoint(
            non_persistent_global_dir,
            ckpt_cfg,
            rank0,
            sharded_state_dict,
            non_persistent_iteration,
            False,
            checkpointing_context=checkpointing_context,
        )
    elif ckpt_cfg.non_persistent_ckpt_type == "local":
        intermediate_state_dict, checkpoint_name = checkpointing_context["local_checkpoint_manager"].load()
        state_dict = intermediate_state_dict.to_state_dict(
            sharded_state_dict,
            algo=ckpt_cfg.non_persistent_local_ckpt_algo,
            parallelization_group=mpu.get_data_parallel_group(with_context_parallel=True),
        )
        return state_dict, checkpoint_name, False, CheckpointType.LOCAL
    else:
        raise ValueError(
            f"Please use local or global non-persistent checkpoints. Got: {ckpt_cfg.non_persistent_ckpt_type})"
        )


def _load_global_dist_base_checkpoint(
    load_dir: str,
    ckpt_cfg: CheckpointConfig,
    rank0: bool,
    sharded_state_dict: dict[str, Any] | None,
    iteration: int,
    release: bool,
    checkpointing_context: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str, bool, CheckpointType]:
    """Load the base state_dict from the given directory containing the global distributed checkpoint."""
    if rank0:
        checkpoint_name = find_checkpoint_rank_0(load_dir, iteration, release)
        state_dict = dist_checkpointing.load_common_state_dict(checkpoint_name)
        return state_dict, checkpoint_name, release, CheckpointType.GLOBAL

    if sharded_state_dict is None:
        raise RuntimeError("Detected load from a distributed checkpoint, but sharded state dict is not provided.")

    checkpoint_name = get_checkpoint_name(load_dir, iteration, release)
    load_strategy = get_default_load_sharded_strategy(checkpoint_name)
    if ckpt_cfg.fully_parallel_load:
        load_strategy = FullyParallelLoadStrategyWrapper(
            load_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
        )
    if checkpointing_context is not None:
        checkpointing_context["load_strategy"] = load_strategy
    state_dict = dist_checkpointing.load(
        sharded_state_dict, checkpoint_name, load_strategy, strict=ckpt_cfg.dist_ckpt_strictness
    )
    return state_dict, checkpoint_name, release, CheckpointType.GLOBAL


def _load_base_checkpoint(
    load_dir: str | None,
    ckpt_cfg: CheckpointConfig,
    rank0: bool = False,
    sharded_state_dict: dict[str, Any] | None = None,
    checkpointing_context: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str, bool, CheckpointType] | None:
    """Load the base state_dict from the given directory."""
    # Try to load non-persistent checkpoint first
    non_persistent_global_dir = (
        ckpt_cfg.non_persistent_global_ckpt_dir
        if ckpt_cfg.non_persistent_global_ckpt_dir or load_dir is None
        else os.path.join(load_dir, _NON_PERSISTENT_CKPT_SUBDIR)
    )
    non_persistent_iteration = _get_non_persistent_iteration(
        non_persistent_global_dir, ckpt_cfg.non_persistent_ckpt_type, checkpointing_context
    )
    iteration, release = -1, False
    tracker_filename = "because load directory is not defined"
    if load_dir is not None:
        tracker_filename = get_checkpoint_train_state_filename(load_dir, prefix=TRACKER_PREFIX)
        if file_exists(tracker_filename):
            train_state = read_train_state(tracker_filename)
            iteration = train_state.step
            # release = train_state.release
        else:
            # Fallback to legacy Megatron-LM tracker file format
            legacy_tracker_filename = get_checkpoint_tracker_filename(load_dir)
            if file_exists(legacy_tracker_filename):
                print_rank_0(f"Loading from legacy Megatron-LM checkpoint format: {legacy_tracker_filename}")
                iteration, release = read_metadata(legacy_tracker_filename)
                tracker_filename = legacy_tracker_filename  # Update for error messages
    if non_persistent_iteration != -1:  # there is a non-persistent checkpoint
        if non_persistent_iteration >= iteration:
            return _load_non_persistent_base_checkpoint(
                non_persistent_global_dir,
                ckpt_cfg,
                rank0,
                sharded_state_dict,
                non_persistent_iteration,
                checkpointing_context,
            )
        else:
            print_rank_0("WARNING: non-persistent checkpoints are older than persistent checkpoint")

    # Otherwise we are dealing with global checkpoints
    # If no tracker file, return nothing
    if iteration == -1:
        if not rank0:
            print_rank_0("WARNING: could not find the metadata file {}".format(tracker_filename))
            print_rank_0("    will not load any checkpoints and will start from random")
        # Conditionally exit if checkpoint not found.
        if ckpt_cfg.exit_on_missing_checkpoint:
            print_rank_0(">> '--exit-on-missing-checkpoint' set ... exiting. <<")
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            sys.exit()

        return None, "", False, None

    # Determine the checkpoint format
    checkpoint_path = get_checkpoint_name(load_dir, iteration, release)
    ckpt_format = _get_checkpoint_format(checkpoint_path)

    if not rank0:
        dist_infix = "distributed " if ckpt_format == "torch_dist" else ""
        if release:
            print_rank_0(f" loading release {dist_infix}checkpoint from {load_dir}")
        else:
            print_rank_0(f" loading {dist_infix}checkpoint from {load_dir} at iteration {iteration}")

    # Handle different checkpoint formats
    if ckpt_format == "torch_dist":
        return _load_global_dist_base_checkpoint(
            load_dir,
            ckpt_cfg,
            rank0,
            sharded_state_dict,
            iteration,
            release,
            checkpointing_context=checkpointing_context,
        )
    elif ckpt_format == "fsdp_dtensor":
        return _load_fsdp_dtensor_base_checkpoint(
            load_dir,
            ckpt_cfg,
            rank0,
            sharded_state_dict,
            iteration,
            release,
            checkpointing_context=checkpointing_context,
        )
    else:
        raise NotImplementedError(f"Checkpoint format {ckpt_format} not supported")


def _load_fsdp_dtensor_base_checkpoint(
    load_dir: str,
    ckpt_cfg: CheckpointConfig,
    rank0: bool,
    sharded_state_dict: dict[str, Any] | None,
    iteration: int,
    release: bool,
    checkpointing_context: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str, bool, CheckpointType]:
    """Load the base state_dict from an FSDP DTensor checkpoint."""
    if rank0:
        # For rank 0, return empty state dict (no common metadata for fsdp_dtensor)
        return {}, get_checkpoint_name(load_dir, iteration, release), release, CheckpointType.FSDP_DTENSOR

    if not HAVE_MEGATRON_FSDP:
        raise RuntimeError("Megatron FSDP is required but not available for loading FSDP DTensor checkpoints.")

    if sharded_state_dict is None:
        raise RuntimeError("sharded_state_dict is required for FSDP DTensor checkpoint loading.")

    checkpoint_name = get_checkpoint_name(load_dir, iteration, release)
    fs_storage_reader = torch.distributed.checkpoint.FileSystemReader(checkpoint_name)

    # Configure partial loading based on strict_fsdp_dtensor_load setting
    allow_partial_load = not getattr(ckpt_cfg, "strict_fsdp_dtensor_load", False)
    if allow_partial_load:
        state_dict_metadata = fs_storage_reader.read_metadata().state_dict_metadata
        rank = torch.distributed.get_rank()
        import time as _time

        _time.sleep(rank * 0.001)  # Prevent log overlap across ranks
        print_diff_in_state_dicts(state_dict_metadata, sharded_state_dict)

    planner = torch.distributed.checkpoint.default_planner.DefaultLoadPlanner(allow_partial_load=allow_partial_load)
    torch.distributed.checkpoint.load_state_dict(
        state_dict=sharded_state_dict,
        storage_reader=fs_storage_reader,
        planner=planner,
    )

    return sharded_state_dict, checkpoint_name, release, CheckpointType.FSDP_DTENSOR


def _build_sharded_state_dict_metadata(
    use_distributed_optimizer: bool, ckpt_fully_parallel_save: bool, ckpt_format: str = "torch_dist"
) -> dict:
    """Builds metadata used for sharded_state_dict versioning.

    The whole content metadata is passed to ``shared_state_dict`` model and optimizer methods
    and therefore affects only the logic behind sharded_state_dict creation.
    The content metadata should be minimalistic, ideally flat (or with a single nesting level)
    and with semantically meaningful flag names (e.g. `distrib_optim_sharding_type`).
    In particular, a simple integer (or SemVer) versioning flag (e.g. `metadata['version'] = 3.4`)
    is discouraged, because the metadata serves for all models and optimizers and it's practically
    impossible to enforce a linearly increasing versioning for this whole space.

    Args:
        use_distributed_optimizer: Whether to use distributed optimizer.
        ckpt_fully_parallel_save: Whether to use fully parallel save.
    """
    metadata = {}
    if use_distributed_optimizer:
        if ckpt_format == "fsdp_dtensor":
            metadata["distrib_optim_sharding_type"] = "fsdp_dtensor"
        elif ckpt_fully_parallel_save:
            metadata["distrib_optim_sharding_type"] = "fully_sharded_model_space"
        else:
            metadata["distrib_optim_sharding_type"] = "dp_zero_gather_scatter"
    metadata["chained_optim_avoid_prefix"] = True
    metadata["singleton_local_shards"] = False
    return metadata


def _get_train_state_from_state_dict(state_dict: dict[str, Any]) -> TrainState:
    """Create a TrainState from the state dict from a Megatron-LM checkpoint."""
    legacy_train_state = TrainState()
    legacy_train_state.step = state_dict.get("iteration", 0)

    # Extract training progress from checkpoint args (like Megatron-LM does)
    checkpoint_args = state_dict.get("args", None)
    if checkpoint_args is not None:
        legacy_train_state.consumed_train_samples = getattr(checkpoint_args, "consumed_train_samples", 0)
        legacy_train_state.skipped_train_samples = getattr(checkpoint_args, "skipped_train_samples", 0)
        legacy_train_state.consumed_valid_samples = getattr(checkpoint_args, "consumed_valid_samples", 0)
    else:
        # Fallback if args not found
        legacy_train_state.consumed_train_samples = 0
        legacy_train_state.skipped_train_samples = 0
        legacy_train_state.consumed_valid_samples = 0

    # Extract floating point operations count from state_dict (like Megatron-LM does)
    legacy_train_state.floating_point_operations_so_far = state_dict.get("num_floating_point_operations_so_far", 0)
    legacy_train_state.do_train = False
    legacy_train_state.do_valid = False
    legacy_train_state.do_test = False
    return legacy_train_state
