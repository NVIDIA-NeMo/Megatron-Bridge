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
from typing import Any, Mapping

from megatron.training.config.container import ConfigContainerBase as _ConfigContainerBase  # noqa: F401
from megatron.training.config.utils import (
    _get_init_false_fields,  # noqa: F401
    _resolve_target_class,  # noqa: F401
)
from megatron.training.config.utils import (
    sanitize_dataclass_config as _sanitize_dataclass_config,
)


logger = logging.getLogger(__name__)


def create_ddp_config(
    wrap_with_ddp: bool = True,
    use_distributed_optimizer: bool = True,
    use_megatron_fsdp: bool = False,
    overrides: Mapping[str, object] | None = None,
    finalize: bool = True,
) -> object | None:
    """Create a finalized Bridge DDP config for external model construction."""
    if not wrap_with_ddp:
        return None

    from megatron.bridge.training.config import DistributedDataParallelConfig

    ddp_config = {
        "use_distributed_optimizer": use_distributed_optimizer,
    }
    if use_megatron_fsdp:
        ddp_config.update(
            {
                "use_distributed_optimizer": True,
                "check_for_nan_in_grad": True,
                "use_megatron_fsdp": True,
                "data_parallel_sharding_strategy": "optim_grads_params",
                "overlap_grad_reduce": True,
            }
        )
    ddp_config.update(overrides or {})

    config = DistributedDataParallelConfig(**ddp_config)
    if finalize:
        config.finalize()
    return config


def apply_run_config_backward_compat(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Apply backward compatibility transformations to run config.

    This function handles dataclass config fields that should not be passed to
    the constructor when loading older checkpoints. It automatically detects
    init=False fields by inspecting the target class.

    The entire config is sanitized recursively to handle init=False fields in any part of the configuration hierarchy.

    Args:
        config_dict: The full run configuration dictionary.

    Returns:
        The config dictionary with backward compatibility fixes applied.
    """
    return _sanitize_dataclass_config(config_dict)
