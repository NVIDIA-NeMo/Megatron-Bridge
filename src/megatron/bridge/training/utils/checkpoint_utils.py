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
from typing import Any

import torch
import yaml
from megatron.core.msc_utils import MultiStorageClientFeature

from megatron.bridge.training.state import TrainState
from megatron.bridge.utils.common_utils import get_rank_safe, get_world_size_safe, print_rank_0


TRAIN_STATE_FILE = "train_state.pt"
TRACKER_PREFIX = "latest"
CONFIG_FILE = "run_config.yaml"

logger = logging.getLogger(__name__)


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


def get_checkpoint_train_state_filename(checkpoints_path: str, prefix: str | None = None) -> str:
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


def checkpoint_exists(checkpoints_path: str | None) -> bool:
    """Check if a checkpoint directory exists.

    Args:
        checkpoints_path: Path to the potential checkpoint directory.

    Returns:
        True if the path exists, False otherwise.
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
            config_obj[0] = config_dict
        except Exception as e:
            error_msg = f"ERROR: Unable to load config file {run_config_filename}: {e}"
            sys.stderr.write(error_msg + "\n")
            config_obj[0] = {"error": True, "msg": error_msg}

    if torch.distributed.is_initialized():
        print_rank_0(f"Broadcasting config from rank 0 to all {get_world_size_safe()} ranks")
        torch.distributed.broadcast_object_list(config_obj, src=0)

    if isinstance(config_obj[0], dict) and config_obj[0].get("error", False):
        raise RuntimeError(config_obj[0]["msg"])

    return config_obj[0]


@lru_cache()
def read_train_state(train_state_filename: str) -> TrainState:
    """Read the train state metadata from a YAML file (rank 0 only).

    Reads the file on rank 0 and broadcasts the result to other ranks.

    Args:
        train_state_filename: Path to the train state YAML file.

    Returns:
        An initialized TrainState object.
    """
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

    if torch.distributed.is_initialized():
        print_rank_0(f"Broadcasting TrainState from rank 0 to all {get_world_size_safe()} ranks")
        torch.distributed.broadcast_object_list(state_obj, src=0)

    if isinstance(state_obj[0], dict) and state_obj[0].get("error", False):
        raise RuntimeError(state_obj[0]["msg"])

    return state_obj[0]
