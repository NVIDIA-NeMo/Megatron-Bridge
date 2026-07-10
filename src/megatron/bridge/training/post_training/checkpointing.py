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

import logging
import os


try:
    import modelopt
    import modelopt.torch.opt as mto
    from modelopt.torch.opt.plugins.mcore_dist_checkpointing import _load_extra_state_from_sharded_checkpoint
except ImportError as e:
    raise ImportError('Required `"nvidia-modelopt[torch]"` is not installed!') from e

from megatron.core import dist_checkpointing
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import unwrap_model


logger = logging.getLogger(__name__)


def restore_sharded_modelopt_state(model: list[MegatronModule], checkpoint_path: str) -> None:
    """Restore ModelOpt state saved in legacy or current MCore checkpoint layouts.

    Args:
        model: Unwrapped model chunks to restore.
        checkpoint_path: Iteration checkpoint directory containing ``modelopt_state``.

    Raises:
        ValueError: If virtual pipeline parallelism produced multiple model chunks.
    """
    if len(model) > 1:
        raise ValueError("sharded_modelopt_state does not support virtual pipeline parallel")

    modelopt_state_path = os.path.join(checkpoint_path, "modelopt_state")
    if not os.path.isdir(modelopt_state_path) or mto.ModeloptStateManager.is_converted(model[0]):
        return

    modelopt_state = dist_checkpointing.load_common_state_dict(modelopt_state_path)
    logger.info(
        "Restoring NVIDIA ModelOpt checkpoint version %s with installed version %s",
        modelopt_state["modelopt_version"],
        modelopt.__version__,
    )
    model[0] = mto.restore_from_modelopt_state(model[0], modelopt_state)
    _load_extra_state_from_sharded_checkpoint(model[0], checkpoint_path, prefix="")


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

    modelopt_state = dist_checkpointing.load_common_state_dict(modelopt_state_path)
    modes = modelopt_state["modelopt_state_dict"]
    return any(mode[0] != "kd_loss" for mode in modes)


def load_modelopt_state(model: list[MegatronModule], checkpoint_path: str) -> None:
    """Load modelopt_state from a checkpoint.
    Args:
        model: The model to load the modelopt_state into
        checkpoint_path: Path to the checkpoint directory
    """
    modelopt_checkpoint_path = _get_modelopt_checkpoint_path(checkpoint_path)
    unwrapped_model = unwrap_model(model)
    restore_sharded_modelopt_state(unwrapped_model, modelopt_checkpoint_path)
