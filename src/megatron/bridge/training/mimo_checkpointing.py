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

"""MIMO checkpointing utilities.

Provides wrapper for per-module optimizers to support standard Bridge
checkpointing with heterogeneous MIMO training.
"""

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from megatron.core.optimizer import MegatronOptimizer


class MimoOptimizerWrapper:
    """Wrapper for heterogeneous per-module optimizers to support standard checkpointing.

    Aggregates state from multiple independent optimizers (one per module)
    into a single state dictionary structure that `save_checkpoint` can handle.

    Args:
        optimizers: Dictionary mapping module names to MegatronOptimizer instances.
    """

    def __init__(self, optimizers: Dict[str, "MegatronOptimizer"]):
        self.optimizers = optimizers

    def sharded_state_dict(
        self, model_sharded_state_dict: Dict[str, Any] = None, **kwargs
    ) -> Dict[str, Any]:
        """Collect sharded state dicts from all underlying optimizers.

        Args:
            model_sharded_state_dict: Ignored, kept for API compatibility.
            **kwargs: Passed to underlying optimizers' sharded_state_dict methods.

        Returns:
            Dict mapping module names to their optimizer sharded states.
        """
        aggregated_state = {}

        for name, optimizer in self.optimizers.items():
            if optimizer is None:
                continue
            if getattr(optimizer, 'is_stub_optimizer', False):
                continue

            opt_state = optimizer.sharded_state_dict(model_sharded_state_dict, **kwargs)
            aggregated_state[name] = opt_state

        return aggregated_state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dicts into respective underlying optimizers.

        Args:
            state_dict: Dict mapping module names to optimizer states.
        """
        if state_dict is None:
            return

        for name, optimizer in self.optimizers.items():
            if optimizer is None:
                continue
            if name in state_dict:
                optimizer.load_state_dict(state_dict[name])
