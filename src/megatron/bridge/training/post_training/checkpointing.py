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

"""Input/output checkpointing for ModelOpt."""

try:
    from modelopt.torch.opt.plugins import restore_sharded_modelopt_state
except ImportError as e:
    raise ImportError('Required `"nvidia-modelopt[torch]"` is not installed!') from e

import os
from typing import List

from megatron.core import dist_checkpointing
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import unwrap_model


def _get_modelopt_checkpoint_path(checkpoint_path: str) -> str:
    """Get the path to use for ModelOpt operations (handles iteration directories)."""
    if not checkpoint_path or not os.path.isdir(checkpoint_path):
        return checkpoint_path

    # Check for iter_* folders
    try:
        iter_folders = [
            f
            for f in os.listdir(checkpoint_path)
            if os.path.isdir(os.path.join(checkpoint_path, f)) and f.startswith("iter_")
        ]
    except (OSError, FileNotFoundError):
        # Directory doesn't exist or can't be accessed
        return checkpoint_path

    if iter_folders:
        # Find the folder with the largest iteration number from state dict
        latest_iter_num = -1
        latest_iter_folder = None

        for folder in iter_folders:
            folder_path = os.path.join(checkpoint_path, folder)
            try:
                state_dict = dist_checkpointing.load_common_state_dict(folder_path)
                if state_dict is not None:
                    iter_num = state_dict.get("iteration", 0)
                    if iter_num > latest_iter_num:
                        latest_iter_num = iter_num
                        latest_iter_folder = folder
            except Exception:
                # Skip checkpoints that fail to load
                continue

        if latest_iter_folder is not None:
            return os.path.join(checkpoint_path, latest_iter_folder)

    return checkpoint_path  # No iteration dirs, use root


def has_modelopt_state(checkpoint_path: str) -> bool:
    """Check if modelopt_state folder exists inside the checkpoint path.

    Checks for modelopt_state in iteration directories (iter_*) or root directory.

    Args:
        checkpoint_path: Path to the checkpoint directory

    Returns:
        True if modelopt_state folder exists, False otherwise
    """
    modelopt_checkpoint_path = _get_modelopt_checkpoint_path(checkpoint_path)
    modelopt_state_path = os.path.join(modelopt_checkpoint_path, "modelopt_state")
    return os.path.isdir(modelopt_state_path)


def load_modelopt_state(model: List[MegatronModule], checkpoint_path: str) -> None:
    """Load modelopt_state from a checkpoint.
    Args:
        model: The model to load the modelopt_state into
        checkpoint_path: Path to the checkpoint directory
    """
    modelopt_checkpointpath = _get_modelopt_checkpoint_path(checkpoint_path)
    unwrapped_model = unwrap_model(model)
    restore_sharded_modelopt_state(unwrapped_model, modelopt_checkpointpath)
