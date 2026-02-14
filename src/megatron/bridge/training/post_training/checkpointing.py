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

import torch
from megatron.core.dist_checkpointing.strategies.common import COMMON_STATE_FNAME
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import unwrap_model

from megatron.bridge.training.utils.checkpoint_utils import (
    _get_latest_iteration_path,
    _list_iteration_directories,
    is_iteration_directory,
)


def _get_modelopt_checkpoint_path(checkpoint_path: str) -> str:
    """Get the path to use for ModelOpt operations.

    ModelOpt state can be stored either at the root level or inside an iteration
    directory. This function returns the best path to look for modelopt_state:
    - If the path is already an iteration directory, use it
    - If iteration directories exist, use the latest one
    - Otherwise, use the root path (modelopt_state might be at root level)

    Args:
        checkpoint_path: Path to a checkpoint directory.

    Returns:
        The path where modelopt_state is most likely to be found.
    """
    if not checkpoint_path or not os.path.isdir(checkpoint_path):
        return checkpoint_path

    if is_iteration_directory(checkpoint_path):
        return checkpoint_path

    # Check for iteration directories - use latest if any exist
    iter_dirs = _list_iteration_directories(checkpoint_path)
    if iter_dirs:
        return _get_latest_iteration_path(iter_dirs)

    # No iterations found - modelopt_state might be at root level
    return checkpoint_path


def has_modelopt_state(checkpoint_path: str) -> bool:
    """Check if ModelOpt state exists inside the checkpoint path.

    Checks for modelopt_state in iteration directories (iter_*) or root directory.
    NOTE: Ignores distillation state which is deprecated and unused.

    Args:
        checkpoint_path: Path to the checkpoint directory

    Returns:
        True if modelopt_state folder exists and contains nontrivial state, else False.
    """
    modelopt_checkpoint_path = _get_modelopt_checkpoint_path(checkpoint_path)
    modelopt_state_path = os.path.join(modelopt_checkpoint_path, "modelopt_state")
    if not os.path.isdir(modelopt_state_path):
        return False

    modelopt_state = torch.load(modelopt_state_path + "/" + COMMON_STATE_FNAME, weights_only=False)
    modes = modelopt_state["modelopt_state_dict"]
    if len(modes) == 1 and modes[0][0] == "kd_loss":
        # Ignore KD state
        modes.pop()

    return len(modes) > 0


def load_modelopt_state(model: list[MegatronModule], checkpoint_path: str) -> None:
    """Load modelopt_state from a checkpoint.
    Args:
        model: The model to load the modelopt_state into
        checkpoint_path: Path to the checkpoint directory
    """
    modelopt_checkpoint_path = _get_modelopt_checkpoint_path(checkpoint_path)
    unwrapped_model = unwrap_model(model)
    restore_sharded_modelopt_state(unwrapped_model, modelopt_checkpoint_path)
