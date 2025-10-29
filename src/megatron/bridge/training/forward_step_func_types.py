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

"""Type definitions for forward step function definitions.

This module provides comprehensive type definitions for forward step functions used in
Megatron Bridge training. Forward step functions are the core of the training loop,
responsible for performing a single forward pass and returning both the output tensor
and a loss function.

Key Types:
    - ForwardStepCallable: Union of all supported forward step signatures (functions + functors)
    - LossFunction: The partial function returned by forward step functions
    - LossFunctionReturn: The possible return types when calling a loss function

Example Usage:
    >>> from functools import partial
    >>> from megatron.bridge.training.state import GlobalState
    >>>
    >>> def my_forward_step(state: GlobalState, data_iterator, model, return_schedule_plan=False):
    ...     # Access configuration, timers, and training state
    ...     timers = state.timers
    ...     config = state.cfg
    ...
    ...     # Get batch data
    ...     batch = next(data_iterator)
    ...
    ...     # Forward pass with timing
    ...     timers("forward-step").start()
    ...     output_tensor = model(batch['input_ids'])
    ...     timers("forward-step").stop()
    ...
    ...     # Create loss function
    ...     def loss_func(output_tensor):
    ...         loss = compute_loss(output_tensor, batch['labels'])
    ...         num_tokens = batch['labels'].numel()
    ...         loss_reduced = {"lm_loss": loss.detach()}
    ...         return loss, num_tokens, loss_reduced  # ThreeTupleLossReturn
    ...
    ...     return output_tensor, partial(loss_func)
    ...
    >>> # State injection is automatic - no manual binding needed!
    >>> pretrain(config, my_forward_step)
    >>>
    >>> # Functor example (for stateful forward steps)
    >>> class StatefulForwardStep:
    ...     def __init__(self, loss_scale: float = 1.0):
    ...         self.loss_scale = loss_scale
    ...         self.step_count = 0
    ...
    ...     def __call__(self, state: GlobalState, data_iterator, model, return_schedule_plan=False):
    ...         self.step_count += 1
    ...         # ... forward step logic with state tracking ...
    ...         return output_tensor, partial(loss_func)
    ...
    >>> functor = StatefulForwardStep(loss_scale=2.0)
    >>> pretrain(config, functor)
"""

from functools import partial
from typing import Any, Iterable, Protocol, overload

import torch
from megatron.core.models.gpt import GPTModel

from megatron.bridge.training.state import GlobalState


# Loss function return types
LossReduced = dict[str, torch.Tensor]  # Dictionary of loss metrics for logging
TwoTupleLossReturn = tuple[torch.Tensor, LossReduced]  # (loss, loss_reduced) - legacy format
ThreeTupleLossReturn = tuple[
    torch.Tensor, torch.Tensor, LossReduced
]  # (loss, num_tokens, loss_reduced) - per-token loss
InferenceLossReturn = Any  # Any data for inference/non-loss collection (when collect_non_loss_data=True)

# Union of all possible loss function return types
LossFunctionReturn = TwoTupleLossReturn | ThreeTupleLossReturn | InferenceLossReturn

# Type for the loss function that gets called with output_tensor
# This is a partial function that when called returns one of the LossFunctionReturn types
LossFunction = partial[LossFunctionReturn]


class TwoArgForwardStep(Protocol):
    """Protocol for forward step functions with 2 arguments.

    This represents forward step functions that don't need access to GlobalState
    and don't support schedule plan return mode.

    Args:
        data_iterator: Iterator providing training data batches
        model: The GPT model to train

    Returns:
        Tuple of (output_tensor, loss_function)
    """

    def __call__(
        self,
        data_iterator: Iterable,
        model: GPTModel,
    ) -> tuple[torch.Tensor, LossFunction]: ...


class ThreeArgStateForwardStep(Protocol):
    """Protocol for forward step functions with 3 arguments including state.

    This represents forward step functions that need access to GlobalState
    but don't support schedule plan return mode.

    Args:
        state: Global training state containing configuration and runtime objects
        data_iterator: Iterator providing training data batches
        model: The GPT model to train

    Returns:
        Tuple of (output_tensor, loss_function)
    """

    def __call__(
        self,
        state: GlobalState,
        data_iterator: Iterable,
        model: GPTModel,
    ) -> tuple[torch.Tensor, LossFunction]: ...


class ThreeArgForwardStep(Protocol):
    """Protocol for forward step functions with 3 arguments.

    This represents forward step functions that don't need access to GlobalState
    but support schedule plan return mode. These are typically 4-arg functions
    that have had GlobalState pre-bound via functools.partial.

    Args:
        data_iterator: Iterator providing training data batches
        model: The GPT model to train
        return_schedule_plan: Whether to return schedule plan instead of output tensor

    Returns:
        Tuple of (output_tensor, loss_function) or (schedule_plan, loss_function)
    """

    def __call__(
        self,
        data_iterator: Iterable,
        model: GPTModel,
        return_schedule_plan: bool = False,
    ) -> tuple[torch.Tensor, LossFunction]: ...


class FourArgForwardStep(Protocol):
    """Protocol for forward step functions with 4 arguments.

    This represents forward step functions that need access to GlobalState
    and support schedule plan return mode. These are the most complete
    forward step function signatures.

    Args:
        state: Global training state containing configuration and runtime objects
        data_iterator: Iterator providing training data batches
        model: The GPT model to train
        return_schedule_plan: Whether to return schedule plan instead of output tensor

    Returns:
        Tuple of (output_tensor, loss_function) or (schedule_plan, loss_function)
    """

    def __call__(
        self,
        state: GlobalState,
        data_iterator: Iterable,
        model: GPTModel,
        return_schedule_plan: bool = False,
    ) -> tuple[torch.Tensor, LossFunction]: ...


class ForwardStepFunctor(Protocol):
    """Protocol for forward step functors (callable classes).

    This protocol represents classes that implement __call__ with one of the
    supported forward step function signatures. Functors are useful when you
    need to maintain state between forward step calls or implement complex
    forward step logic that benefits from object-oriented design.

    The __call__ method must match one of the supported signatures:
    - (data_iterator, model)
    - (data_iterator, model, return_schedule_plan=False)
             OR (state: GlobalState, data_iterator, model)
    - (state: GlobalState, data_iterator, model, return_schedule_plan=False)

    RECOMMENDED: Use GlobalState type hint for automatic state injection and full access
    to configuration, timers, and training state.

    Examples:
        >>> class MyForwardFunctor:
        ...     def __init__(self, loss_scale: float = 1.0):
        ...         self.loss_scale = loss_scale
        ...         self.call_count = 0
        ...
        ...     def __call__(self, state: GlobalState, data_iterator, model, return_schedule_plan=False):
        ...         self.call_count += 1
        ...         # Access training infrastructure
        ...         timers = state.timers
        ...         config = state.cfg
        ...         # ... forward step logic ...
        ...         return output_tensor, loss_function
        ...
        >>> functor = MyForwardFunctor(loss_scale=2.0)
        >>> pretrain(config, functor)  # State injection is automatic!
    """

    @overload
    def __call__(
        self,
        data_iterator: Iterable,
        model: GPTModel,
    ) -> tuple[torch.Tensor, LossFunction]:
        """2-argument signature: (data_iterator, model)."""
        ...

    @overload
    def __call__(
        self,
        data_iterator: Iterable,
        model: GPTModel,
        return_schedule_plan: bool = False,
    ) -> tuple[torch.Tensor, LossFunction]:
        """3-argument signature: (data_iterator, model, return_schedule_plan)."""
        ...

    @overload
    def __call__(
        self,
        state: GlobalState,
        data_iterator: Iterable,
        model: GPTModel,
    ) -> tuple[torch.Tensor, LossFunction]:
        """3-argument signature with state: (state, data_iterator, model)."""
        ...

    @overload
    def __call__(
        self,
        state: GlobalState,
        data_iterator: Iterable,
        model: GPTModel,
        return_schedule_plan: bool = False,
    ) -> tuple[torch.Tensor, LossFunction]:
        """4-argument signature: (state, data_iterator, model, return_schedule_plan)."""
        ...

    def __call__(self, *args, **kwargs) -> tuple[torch.Tensor, LossFunction]:
        """Execute the forward step.

        The actual implementation must match one of the overloaded signatures above.
        This fallback signature is required by the Protocol but should not be used
        directly - type checkers will use the @overload signatures for validation.
        """
        ...


# Union type for all supported forward step function signatures
ForwardStepFunc = TwoArgForwardStep | ThreeArgStateForwardStep | ThreeArgForwardStep | FourArgForwardStep

# Type alias that includes both functions and functors
ForwardStepCallable = ForwardStepFunc | ForwardStepFunctor
