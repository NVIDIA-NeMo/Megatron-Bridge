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

import os.path

from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import unwrap_model


def has_modelopt_state(checkpoint_path: str) -> bool:
    """Check if modelopt_state folder exists inside the checkpoint path.

    Args:
        checkpoint_path: Path to the checkpoint directory

    Returns:
        True if modelopt_state folder exists, False otherwise
    """
    modelopt_state_path = os.path.join(checkpoint_path, "modelopt_state")
    return os.path.isdir(modelopt_state_path)


def load_modelopt_state(model: list[MegatronModule], checkpoint_path: str) -> None:
    """Load modelopt_state from a checkpoint.
    Args:
        model: The model to load the modelopt_state into
        checkpoint_path: Path to the checkpoint directory
    """
    unwrapped_model = unwrap_model(model)
    restore_sharded_modelopt_state(unwrapped_model, checkpoint_path)
