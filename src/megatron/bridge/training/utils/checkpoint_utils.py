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

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from megatron.core.msc_utils import MultiStorageClientFeature

from megatron.bridge.training.state import TrainState
from megatron.bridge.training.utils.config_utils import apply_run_config_backward_compat
from megatron.bridge.utils.common_utils import get_rank_safe, get_world_size_safe, print_rank_0


TRAIN_STATE_FILE = "train_state.pt"
TRACKER_PREFIX = "latest"
CONFIG_FILE = "run_config.yaml"

_ITER_PREFIX: str = "iter_"
_ITER_PREFIX_LEN: int = len(_ITER_PREFIX)

logger = logging.getLogger(__name__)
_RUNTIME_ONLY_TARGETS = frozenset({"megatron.core.timers.Timers"})


def file_exists(path: str) -> bool:
    """Check if a file exists.

    Args:
        path: The path to the file. Can be a local path or an MSC URL.

    Returns:
        True if the file exists, False otherwise.
    """
    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        return msc.os.path.exists(path)
    else:
        return os.path.exists(path)


def ensure_directory_exists(filename: str, check_parent: bool = True) -> None:
    """Ensure that the directory for a given filename exists.

    Args:
        filename: The path whose directory should be checked/created.
        check_parent: If True (default), checks the parent directory of the filename.
                      If False, treats the filename itself as the directory path.
    """
    dirname = os.path.dirname(filename) if check_parent else filename
    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        msc.os.makedirs(dirname, exist_ok=True)
    else:
        os.makedirs(dirname, exist_ok=True)


def get_checkpoint_name(checkpoints_path: str, iteration: int, release: bool = False) -> str:
    """Determine the directory name for a specific checkpoint.

    Constructs the path based on iteration number or release flag.

    Args:
        checkpoints_path: Base directory where checkpoints are stored.
        iteration: The training iteration number.
        release: If True, uses 'release' as the directory name instead of iteration.

    Returns:
        The full path to the checkpoint directory.
    """
    if release:
        directory = "release"
    else:
        directory = "iter_{:07d}".format(iteration)

    common_path = os.path.join(checkpoints_path, directory)
    return common_path


def get_checkpoint_train_state_filename(checkpoints_path: str, prefix: Optional[str] = None) -> str:
    """Get the filename for the train state tracker file.

    This file typically stores metadata about the latest checkpoint, like the iteration number.

    Args:
        checkpoints_path: Base directory where checkpoints are stored.
        prefix: Optional prefix (e.g., 'latest') to prepend to the filename.

    Returns:
        The full path to the train state tracker file.
    """
    if prefix is None:
        return os.path.join(checkpoints_path, TRAIN_STATE_FILE)
    else:
        return os.path.join(checkpoints_path, f"{prefix}_{TRAIN_STATE_FILE}")


def get_checkpoint_run_config_filename(checkpoints_path: str) -> str:
    """Get the filename for the run configuration file within a checkpoint directory.

    Args:
        checkpoints_path: Base directory where checkpoints are stored.

    Returns:
        The full path to the run configuration file (e.g., run_config.yaml).
    """
    return os.path.join(checkpoints_path, CONFIG_FILE)


def get_checkpoint_tracker_filename(checkpoints_path: str) -> str:
    """Tracker file rescords the latest chckpoint during training to restart from.

    Supports checkpoints produced by Megatron-LM.

    Args:
        checkpoints_path: Base directory where checkpoints are stored.

    Returns:
        The full path to the checkpoint tracker file (e.g., latest_checkpointed_iteration.txt).
    """
    return os.path.join(checkpoints_path, "latest_checkpointed_iteration.txt")


def is_iteration_directory(path: str) -> bool:
    """Check if a path is a specific iteration checkpoint directory.

    Iteration directories follow the naming convention ``iter_<digits>`` where
    digits is one or more numeric characters (e.g., ``iter_0000005``, ``iter_123``,
    ``iter_1000000000``). The standard format from ``get_checkpoint_name`` is
    ``iter_{:07d}`` (7 zero-padded digits), but this function accepts any digit count
    for flexibility with custom or legacy checkpoints.

    Args:
        path: Path to check.

    Returns:
        True if the path basename matches the iteration directory pattern.
    """
    basename = os.path.basename(path.rstrip(os.sep))
    if not basename.startswith(_ITER_PREFIX):
        return False
    suffix = basename[_ITER_PREFIX_LEN:]
    # Accept any non-empty sequence of digits (flexible for various iteration formats)
    return len(suffix) > 0 and suffix.isdigit()


def checkpoint_exists(checkpoints_path: Optional[str]) -> bool:
    """Check if a checkpoint directory contains valid checkpoint files.

    Args:
        checkpoints_path: Path to the potential checkpoint directory. Can be either
            a top-level checkpoint directory or a specific iteration directory.

    Returns:
        True if the path contains valid checkpoint files, False otherwise.
    """
    if checkpoints_path is None:
        return False

    train_state_filename = os.path.join(checkpoints_path, f"{TRACKER_PREFIX}_{TRAIN_STATE_FILE}")

    if file_exists(train_state_filename):
        return True

    # Fallback to the Megatron-LM tracker file
    path = get_checkpoint_tracker_filename(checkpoints_path)
    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        return msc.os.path.isfile(path)
    else:
        return os.path.isfile(path)


def resolve_checkpoint_path(path: str | os.PathLike[str]) -> str:
    """Resolve a checkpoint path to a specific iteration directory.

    This utility handles both:
    - Top-level checkpoint directories: Resolves to the latest iteration
    - Specific iteration directories: Returns the path as-is after validation

    Args:
        path: Path to a checkpoint. Can be either:
            - A top-level checkpoint directory containing ``iter_*`` subdirectories
            - A specific iteration directory (e.g., ``/path/to/checkpoint/iter_0000005``)

    Returns:
        The full path to the resolved iteration directory.

    Raises:
        FileNotFoundError: If path doesn't exist or no iteration checkpoints found
            in a top-level directory.
        NotADirectoryError: If the path exists but is not a directory.
    """
    path_str = str(path)

    # Validate path exists and is a directory
    exists, is_dir = _path_exists_and_is_dir(path_str)
    if not exists:
        raise FileNotFoundError(f"Checkpoint path does not exist: {path_str}")
    if not is_dir:
        raise NotADirectoryError(f"Checkpoint path must be a directory: {path_str}")

    # If already an iteration directory, return as-is
    if is_iteration_directory(path_str):
        return path_str

    # Find latest iteration in top-level directory
    iter_dirs = _list_iteration_directories(path_str)
    if not iter_dirs:
        raise FileNotFoundError(f"No iteration checkpoints found in: {path_str}")

    return _get_latest_iteration_path(iter_dirs)


def get_hf_model_id_from_checkpoint(path: str | os.PathLike[str]) -> str | None:
    """
    Infer the HuggingFace model identifier recorded in a Megatron Bridge checkpoint.

    Args:
        path: Path to a Megatron checkpoint directory. This can be either the root
            checkpoint directory containing ``iter_*`` subdirectories or a specific
            iteration directory.

    Returns:
        The HuggingFace model identifier/path if present, otherwise ``None``.

    Raises:
        FileNotFoundError: If the provided path does not exist, or if ``path`` is a
            top-level checkpoint directory that contains no ``iter_*`` subdirectories.
        NotADirectoryError: If the provided path is not a directory.
    """
    path_str = str(path)

    # First check: does the path exist?
    exists, is_dir = _path_exists_and_is_dir(path_str)
    if not exists:
        raise FileNotFoundError(f"Checkpoint path '{path}' does not exist.")
    if not is_dir:
        raise NotADirectoryError(f"Checkpoint path '{path}' must be a directory.")

    # Try to find run_config.yaml - check given path first, then resolve to iteration
    run_config_path = os.path.join(path_str, CONFIG_FILE)
    if not file_exists(run_config_path):
        resolved_path = resolve_checkpoint_path(path_str)
        run_config_path = os.path.join(resolved_path, CONFIG_FILE)

    if not file_exists(run_config_path):
        return None

    run_config = read_run_config(run_config_path)
    if not isinstance(run_config, dict):
        return None

    model_section = run_config.get("model")
    if not isinstance(model_section, dict):
        return None

    hf_model_id = model_section.get("hf_model_id")
    if not hf_model_id:
        return None

    return str(hf_model_id)


@lru_cache()
def read_run_config(run_config_filename: str) -> dict[str, Any]:
    """Read the run configuration from a YAML file (rank 0 only).

    Reads the file on rank 0 and broadcasts the result to other ranks.

    Args:
        run_config_filename: Path to the run config YAML file.

    Returns:
        A dictionary containing the run configuration.

    Raises:
        RuntimeError: If reading the config file fails on rank 0.
    """
    if torch.distributed.is_initialized():
        config_obj = [None]

        if get_rank_safe() == 0:
            try:
                if MultiStorageClientFeature.is_enabled():
                    msc = MultiStorageClientFeature.import_package()
                    with msc.open(run_config_filename, "r") as f:
                        config_dict = yaml.safe_load(f)
                else:
                    with open(run_config_filename, "r") as f:
                        config_dict = yaml.safe_load(f)
                config_dict = _sanitize_run_config_object(config_dict)
                config_dict = apply_run_config_backward_compat(config_dict)
                config_obj[0] = config_dict
            except Exception as e:
                error_msg = f"ERROR: Unable to load config file {run_config_filename}: {e}"
                sys.stderr.write(error_msg + "\n")
                config_obj[0] = {"error": True, "msg": error_msg}

        print_rank_0(f"Broadcasting config from rank 0 to all {get_world_size_safe()} ranks")
        torch.distributed.broadcast_object_list(config_obj, src=0)

        if isinstance(config_obj[0], dict) and config_obj[0].get("error", False):
            raise RuntimeError(config_obj[0]["msg"])

        return config_obj[0]
    else:
        try:
            if MultiStorageClientFeature.is_enabled():
                msc = MultiStorageClientFeature.import_package()
                with msc.open(run_config_filename, "r") as f:
                    config_dict = yaml.safe_load(f)
            else:
                with open(run_config_filename, "r") as f:
                    config_dict = yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Unable to load config file {run_config_filename}: {e}") from e

        config_dict = _sanitize_run_config_object(config_dict)
        config_dict = apply_run_config_backward_compat(config_dict)
        return config_dict


@lru_cache()
def read_train_state(train_state_filename: str) -> TrainState:
    """Read the train state metadata from a YAML file (rank 0 only).

    Reads the file on rank 0 and broadcasts the result to other ranks if
    torch.distributed is initialized. Otherwise, loads the file locally.

    Args:
        train_state_filename: Path to the train state YAML file.

    Returns:
        An initialized TrainState object.
    """
    if torch.distributed.is_initialized():
        state_obj = [None]
        if get_rank_safe() == 0:
            try:
                if MultiStorageClientFeature.is_enabled():
                    msc = MultiStorageClientFeature.import_package()
                    state_dict = msc.torch.load(train_state_filename, map_location="cpu")
                else:
                    state_dict = torch.load(train_state_filename, map_location="cpu")
                ts = TrainState()
                ts.load_state_dict(state_dict)
                state_obj[0] = ts
            except Exception as e:
                error_msg = f"ERROR: Unable to load train state file {train_state_filename}: {e}"
                sys.stderr.write(error_msg + "\n")
                state_obj[0] = {"error": True, "msg": error_msg}

        print_rank_0(f"Broadcasting TrainState from rank 0 to all {get_world_size_safe()} ranks")
        torch.distributed.broadcast_object_list(state_obj, src=0)

        if isinstance(state_obj[0], dict) and state_obj[0].get("error", False):
            raise RuntimeError(state_obj[0]["msg"])

        return state_obj[0]

    try:
        if MultiStorageClientFeature.is_enabled():
            msc = MultiStorageClientFeature.import_package()
            state_dict = msc.torch.load(train_state_filename, map_location="cpu")
        else:
            state_dict = torch.load(train_state_filename, map_location="cpu")
        ts = TrainState()
        ts.load_state_dict(state_dict)
        return ts
    except Exception as e:
        raise RuntimeError(f"Unable to load train state file {train_state_filename}: {e}") from e


def _sanitize_run_config_object(obj: Any) -> Any:
    """Remove runtime-only objects from run config dictionaries.

    Timers and other runtime constructs are serialized with `_target_` entries
    that cannot be recreated without additional context (e.g., constructor
    arguments provided at runtime). These objects are not required when loading
    a checkpoint configuration, so we replace them with ``None`` to avoid
    instantiation errors when the config is processed later.
    """

    if isinstance(obj, dict):
        target = obj.get("_target_")
        if isinstance(target, str) and target in _RUNTIME_ONLY_TARGETS:
            return None
        return {key: _sanitize_run_config_object(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_run_config_object(item) for item in obj]
    return obj


def _list_iteration_directories(path: str) -> list[tuple[str, str]]:
    """List valid iteration directories in a checkpoint directory.

    Args:
        path: Path to a checkpoint directory.

    Returns:
        List of (name, full_path) tuples for valid iteration directories.
    """
    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        path_obj = msc.Path(str(path))
        return [
            (child.name, str(child))
            for child in path_obj.iterdir()
            if child.is_dir() and is_iteration_directory(child.name)
        ]
    else:
        checkpoint_path = Path(path)
        return [
            (child.name, str(child))
            for child in checkpoint_path.iterdir()
            if child.is_dir() and is_iteration_directory(child.name)
        ]


def _get_latest_iteration_path(iter_dirs: list[tuple[str, str]]) -> str:
    """Get the path to the latest iteration from a list of iteration directories.

    Args:
        iter_dirs: List of (name, full_path) tuples from ``_list_iteration_directories``.

    Returns:
        The full path to the directory with the highest iteration number.

    Raises:
        ValueError: If iter_dirs is empty.
    """
    if not iter_dirs:
        raise ValueError("Cannot get latest iteration from empty list")

    def _iteration_number(item: tuple[str, str]) -> int:
        # Extract iteration number from "iter_XXXXXXX" format
        return int(item[0][_ITER_PREFIX_LEN:])

    _, latest_path = max(iter_dirs, key=_iteration_number)
    return latest_path


def _path_exists_and_is_dir(path: str) -> tuple[bool, bool]:
    """Check if path exists and is a directory.

    Args:
        path: Path to check.

    Returns:
        Tuple of (exists, is_directory).
    """
    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        path_obj = msc.Path(str(path))
        if not path_obj.exists():
            return False, False
        return True, path_obj.is_dir()
    else:
        p = Path(path)
        if not p.exists():
            return False, False
        return True, p.is_dir()
