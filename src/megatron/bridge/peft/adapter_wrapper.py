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

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn


if TYPE_CHECKING:
    from megatron.core.dist_checkpointing.mapping import ShardedStateDict


class AdapterWrapper(nn.Module):
    """Abstract base class for wrapping modules with adapters in Parameter-Efficient Fine-Tuning (PEFT).

    This class wraps a module and its associated adapter, providing methods for
    managing the state dictionaries of both the main module and the adapter. It does not
    implement the forward method, which must be implemented by concrete subclasses.

    Attributes:
        to_wrap (nn.Module): The main module to be wrapped.
        adapter (nn.Module): The adapter module to be applied.

    Note:
        This class is abstract and cannot be instantiated directly. Subclasses must
        implement the forward method.

    Example:
        class LoRALinear(AdapterWrapper):
            def __init__(self, to_wrap, adapter):
                super().__init__(to_wrap, adapter)

            def forward(self, x):
                return self.to_wrap(x) + self.adapter(x)

        main_module = nn.Linear(100, 100)
        adapter = nn.Linear(100, 100)
        parallel_adapter = LoRALinear(main_module, adapter)
    """

    def __init__(self, to_wrap: nn.Module, adapter: nn.Module) -> None:
        """Initialize the AdapterWrapper with a main module and adapter.

        Args:
            to_wrap: The main module to be wrapped.
            adapter: The adapter module to be applied.
        """
        super(AdapterWrapper, self).__init__()
        self.to_wrap = to_wrap
        self.adapter = adapter

    def base_linear_forward(
        self, x: torch.Tensor, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Run the forward method of the linear module `to_wrap`.

        This method handles the complex return patterns of Megatron's linear layers,
        which can return different combinations of outputs, biases, and layernorm outputs.

        The flow is: x -> [layernorm/identity] -> layernorm_output -> [linear] -> linear_output, bias

        Args:
            x: Input tensor.
            *args: Additional positional arguments for the wrapped module.
            **kwargs: Additional keyword arguments for the wrapped module.

        Returns:
            A tuple containing:
                - linear_output: The output from the linear layer
                - bias: The bias term (if present, otherwise None)
                - layernorm_output: The output from layernorm (differs from x only for
                  LayerNormColumnParallelLinear, otherwise equals x)

        Note:
            The wrapped module can return values in four different patterns:
            1. nothing: (out, None)
            2. return_bias: (out, bias)
            3. return_layernorm_output: ((out, ln_out), None)
            4. both: (out, bias, ln_out)
        """
        linear_output = self.to_wrap(x, *args, **kwargs)
        assert isinstance(linear_output, tuple), (
            f"{self.to_wrap} should return a tuple but instead returns {linear_output}"
        )

        bias = None
        layernorm_output = x

        if len(linear_output) == 2:
            linear_output, bias = linear_output
            if isinstance(linear_output, tuple) and len(linear_output) == 2:
                linear_output, layernorm_output = linear_output
        elif len(linear_output) == 3:
            linear_output, bias, layernorm_output = linear_output

        return linear_output, bias, layernorm_output

    def state_dict(
        self, destination: dict[str, Any] | None = None, prefix: str = "", keep_vars: bool = False
    ) -> dict[str, Any]:
        """Retrieve the state dictionary of the wrapped module and adapter.

        This method overrides the default state_dict behavior to include both
        the main module's state and the adapter's state under a special 'adapter' prefix.

        Args:
            destination: A dictionary to store the state. If None, a new
                        dictionary is created. Defaults to None.
            prefix: A prefix added to parameter and buffer names. Defaults to ''.
            keep_vars: If True, returns variables instead of tensor values.
                      Defaults to False.

        Returns:
            The state dictionary containing both the main module and adapter states.
        """
        if destination is None:
            destination = {}

        # Get state dict of the main module
        self.to_wrap.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        # Store adapter state dict under the "adapter" prefix in the destination dict
        self.adapter.state_dict(destination=destination, prefix=f"{prefix}adapter.", keep_vars=keep_vars)
        return destination

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple[tuple[int, int, int]] = (),
        metadata: dict[str, Any] | None = None,
    ) -> "ShardedStateDict":
        """Retrieve the sharded state dictionary of the wrapped module and adapter.

        This method is used for distributed checkpointing, combining the sharded states
        of both the main module and the adapter.

        Args:
            prefix: A prefix added to parameter and buffer names. Defaults to ''.
            sharded_offsets: Offsets for sharded parameters. Defaults to an empty tuple.
            metadata: Additional metadata for the sharded state. Defaults to None.

        Returns:
            The combined sharded state dictionary.
        """
        sharded_state_dict = {}
        sharded_state_dict.update(self.to_wrap.sharded_state_dict(prefix, sharded_offsets, metadata))
        sharded_state_dict.update(self.adapter.sharded_state_dict(f"{prefix}adapter.", sharded_offsets, metadata))
        return sharded_state_dict
