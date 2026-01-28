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

"""Training callbacks for Megatron-Bridge.

This module provides a lightweight callback system for injecting custom logic
into the training loop without modifying framework code.

Two registration patterns are supported:

1. Class-based: Subclass `Callback` and override event methods
   ```python
   class MyCallback(Callback):
       def on_train_start(self, context):
           print("Training started!")

   pretrain(config, forward_step_func, callbacks=[MyCallback()])
   ```

2. Functional: Register functions directly with `CallbackManager`
   ```python
   manager = CallbackManager()
   manager.register("on_train_step_end", my_logging_fn)
   pretrain(config, forward_step_func, callbacks=manager)
   ```

Both patterns can be mixed. Callbacks fire in registration order.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import torch
    from megatron.core.optimizer import MegatronOptimizer
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
    from megatron.core.transformer import MegatronModule

    from megatron.bridge.training.state import GlobalState


logger: logging.Logger = logging.getLogger(__name__)


VALID_EVENTS: frozenset[str] = frozenset(
    {
        "on_train_start",
        "on_train_step_start",
        "on_train_step_end",
        "on_train_end",
        "on_eval_start",
        "on_eval_step_start",
        "on_eval_step_end",
        "on_eval_end",
        "on_test_start",
        "on_test_step_start",
        "on_test_step_end",
        "on_test_end",
    }
)


@dataclass
class CallbackContext:
    """Context passed to callbacks.

    Contains framework state and a persistent user_state dict.
    Modifying framework objects is at the user's own risk.

    Attributes:
        state: Global training state (config, train_state, timers, loggers).
        model: List of model chunks.
        user_state: Mutable dict for storing user data across callback invocations.
        optimizer: Optimizer instance. Available during training events only.
        scheduler: Learning rate scheduler. Available during training events only.
        loss_dict: Reduced losses from training step. Available in on_train_step_end.
        grad_norm: Gradient norm. Available in on_train_step_end if computed.
        skipped_iter: Whether the iteration was skipped. Available in on_train_step_end.
        total_loss_dict: Aggregated eval losses. Available in on_eval_end.

    Field Availability by Event:
        All events: state, model, user_state
        Training events: optimizer, scheduler
        on_train_step_end: loss_dict, grad_norm, skipped_iter
        on_eval_end, on_test_end: total_loss_dict
    """

    # Always available
    state: GlobalState
    model: list[MegatronModule]
    user_state: dict = field(default_factory=dict)

    # Training events only
    optimizer: MegatronOptimizer | None = None
    scheduler: OptimizerParamScheduler | None = None

    # on_train_step_end
    loss_dict: dict[str, torch.Tensor] | None = None
    grad_norm: float | None = None
    skipped_iter: bool | None = None

    # on_eval_end
    total_loss_dict: dict[str, torch.Tensor] | None = None


class Callback:
    """Base class for organizing callbacks.

    Subclass and override methods for events you want to handle.
    All methods are no-ops by default.

    Example:
        ```python
        class MyCallback(Callback):
            def on_train_start(self, context):
                context.user_state['start_time'] = time.time()

            def on_train_end(self, context):
                elapsed = time.time() - context.user_state['start_time']
                print(f"Training took {elapsed:.2f}s")

        pretrain(config, forward_step_func, callbacks=[MyCallback()])
        ```
    """

    def on_train_start(self, context: CallbackContext) -> None:
        """
        Invoked once immediately after switching the model to training mode and before the training loop begins.
        
        Parameters:
            context (CallbackContext): Event context. Always includes `state`, `model`, and `user_state`. During training start, `optimizer` and `scheduler` may also be present (or `None`).
        """
        pass

    def on_train_step_start(self, context: CallbackContext) -> None:
        """
        Hook invoked at the start of a training step.
        
        Parameters:
            context (CallbackContext): Execution context for the event. Provides
                access to `state`, `model`, and the persistent `user_state`. During
                this event the current `optimizer` and `scheduler` are available;
                step-end-specific fields (e.g., `loss_dict`, `grad_norm`,
                `skipped_iter`) are not set.
        """
        pass

    def on_train_step_end(self, context: CallbackContext) -> None:
        """
        Handle actions after a training step completes.
        
        Parameters:
            context (CallbackContext): Execution context for this event. Available fields:
                - state: global framework state
                - model: list of model replicas
                - user_state: persistent dict shared across callbacks
                - optimizer: optimizer instance or `None`
                - scheduler: learning-rate scheduler or `None`
                - loss_dict: dict[str, torch.Tensor] of losses for the step or `None`
                - grad_norm: gradient norm for the step or `None`
                - skipped_iter: `True` if the iteration was skipped, otherwise `False` or `None`
        """
        pass

    def on_train_end(self, context: CallbackContext) -> None:
        """
        Invoked after the training loop completes to perform any finalization or teardown actions.
        
        Parameters:
            context (CallbackContext): Execution context exposing global framework state, `model` (list of MegatronModule),
                and the persistent `user_state` dict. For training-related end events, `optimizer` and `scheduler`
                (if available) are present on the context.
        """
        pass

    def on_eval_start(self, context: CallbackContext) -> None:
        """
        Invoked after models are set to evaluation mode and before the evaluation loop begins.
        
        Parameters:
            context (CallbackContext): Context carrying global state, model list, and persistent `user_state`. During evaluation, training-only fields (`optimizer`, `scheduler`, `loss_dict`, `grad_norm`, `skipped_iter`) are `None`.
        """
        pass

    def on_eval_step_start(self, context: CallbackContext) -> None:
        """
        Hook invoked at the start of an evaluation step.
        
        Parameters:
            context (CallbackContext): Context for the current evaluation step. Contains the global `state`, the `model` list, and the persistent `user_state` dict; training-only fields (e.g., `optimizer`, `scheduler`, step/end loss/grad fields) are not applicable for this event.
        """
        pass

    def on_eval_step_end(self, context: CallbackContext) -> None:
        """
        Called after each evaluation step completes.
        
        Parameters:
            context (CallbackContext): The callback context carrying framework state and shared user data. For this event, `state`, `model`, and `user_state` are available; optimizer, scheduler, and loss-related fields may be None or not populated.
        """
        pass

    def on_eval_end(self, context: CallbackContext) -> None:
        """
        Called after evaluation completes and before returning to training mode.
        
        Parameters:
            context (CallbackContext): The callback context carrying global state, models, and a persistent `user_state` dict. For this event `context.total_loss_dict` may contain the aggregated evaluation losses (a dict mapping loss names to tensors) or `None`.
        """
        pass

    def on_test_start(self, context: CallbackContext) -> None:
        """
        Invoked at the start of the test phase after the model has been set to evaluation mode.
        
        Parameters:
            context (CallbackContext): Provides the current framework state, the list of model
                replicas/shards, and the persistent `user_state` dictionary. For the
                `on_test_start` event only `state`, `model`, and `user_state` are guaranteed
                to be available; training-only attributes (e.g., optimizer, scheduler) are
                not provided.
        """
        pass

    def on_test_step_start(self, context: CallbackContext) -> None:
        """
        Called at the start of each test step.
        
        Parameters:
            context (CallbackContext): The callback context. For this event `state`, `model`, and `user_state` are available; training-only fields (e.g., `optimizer`, `scheduler`) and step-end metrics are not provided.
        """
        pass

    def on_test_step_end(self, context: CallbackContext) -> None:
        """
        Hook invoked after a single test step finishes.
        
        Parameters:
            context (CallbackContext): Execution context carrying global framework state, the model list, and the persistent `user_state` dict. (Only `state`, `model`, and `user_state` are guaranteed to be available for this event.)
        """
        pass

    def on_test_end(self, context: CallbackContext) -> None:
        """
        Invoked after a testing run finishes, before the model is switched back to training.
        
        Parameters:
            context (CallbackContext): Execution context containing framework state, the model list, and the persistent `user_state`. For this event `context.total_loss_dict` holds aggregated test losses (or `None`). Training-only fields such as `optimizer` and `scheduler` are not populated.
        """
        pass


class CallbackManager:
    """Manages registration and execution of training callbacks.

    Supports two registration patterns:

    1. Class-based: Use add() with Callback subclass instances
       ```python
       manager.add(MyCallback())
       manager.add([CallbackA(), CallbackB()])
       ```

    2. Functional: Use register() with event name and callable
       ```python
       manager.register("on_train_start", my_function)
       ```

    Both patterns can be mixed. Callbacks fire in registration order.

    The manager also owns a `user_state` dictionary that persists across all
    callback invocations, allowing callbacks to share state.

    Example:
        ```python
        manager = CallbackManager()
        manager.add(MyCallback())
        manager.register("on_eval_end", lambda ctx: print("Eval done!"))
        pretrain(config, forward_step_func, callbacks=manager)
        ```
    """

    def __init__(self) -> None:
        """
        Create a CallbackManager with empty per-event callback lists and a persistent user state.
        
        Initializes:
        - _callbacks: mapping of each valid event name to an empty list of callback functions.
        - _active_events: set tracking which events currently have registered callbacks.
        - _user_state: persistent mutable dict shared across callback invocations.
        """
        self._callbacks: dict[str, list[Callable[[CallbackContext], None]]] = {event: [] for event in VALID_EVENTS}
        self._active_events: set[str] = set()
        self._user_state: dict = {}

    @property
    def user_state(self) -> dict:
        """
        Persistent mutable dictionary for storing arbitrary user data shared across callback invocations.
        
        Returns:
            user_state (dict): The mutable dictionary persisted on the manager and accessible to all callbacks.
        """
        return self._user_state

    def add(self, callback: Callback | list[Callback]) -> None:
        """
        Register one or more Callback instances by wiring their overridden event handlers.
        
        Registers any event methods on the provided Callback instance(s) that override the base Callback implementations, appending them to the manager's callback lists in registration order and marking those events active.
        
        Parameters:
            callback (Callback | list[Callback]): A single Callback or a list of Callbacks to register.
        """
        callbacks = callback if isinstance(callback, list) else [callback]

        for cb in callbacks:
            for event_name in VALID_EVENTS:
                method = getattr(cb, event_name, None)
                base_method = getattr(Callback, event_name, None)
                if method is not None and method.__func__ is not base_method:
                    self._callbacks[event_name].append(method)
                    self._active_events.add(event_name)

    def register(self, event_name: str, fn: Callable[[CallbackContext], None]) -> None:
        """
        Register a function to be invoked for a lifecycle event.
        
        Parameters:
            event_name (str): Name of the event to register; must be one of VALID_EVENTS.
            fn (Callable[[CallbackContext], None]): Callback to invoke with a CallbackContext.
        
        Raises:
            ValueError: If `event_name` is not in VALID_EVENTS.
        """
        if event_name not in VALID_EVENTS:
            raise ValueError(f"Unknown event '{event_name}'. Valid events: {VALID_EVENTS}")
        self._callbacks[event_name].append(fn)
        self._active_events.add(event_name)

    @property
    def events(self) -> frozenset[str]:
        """
        Provides the set of valid callback event names.
        
        Returns:
            frozenset[str]: The set of event name strings accepted for callback registration.
        """
        return VALID_EVENTS

    def list_callbacks(self, event_name: str) -> list[Callable[[CallbackContext], None]]:
        """
        Retrieve the registered callbacks for a given event.
        
        Parameters:
            event_name (str): Name of the lifecycle event to query.
        
        Returns:
            list[Callable[[CallbackContext], None]]: Copy of the registered callables in execution order.
        
        Raises:
            ValueError: If `event_name` is not one of the valid events.
        """
        if event_name not in VALID_EVENTS:
            raise ValueError(f"Unknown event '{event_name}'. Valid events: {VALID_EVENTS}")
        return list(self._callbacks[event_name])

    def has_callbacks(self, event_name: str) -> bool:
        """Check if any callbacks are registered for an event.

        Args:
            event_name: Name of the event.

        Returns:
            True if at least one callback is registered for the event.
        """
        return event_name in self._active_events

    def fire(self, event_name: str, context: CallbackContext) -> None:
        """
        Invoke all registered callbacks for the given lifecycle event.
        
        Callbacks are executed in registration order. Any exception raised by a callback is propagated to the caller.
        
        Parameters:
            event_name (str): The lifecycle event to fire (one of the module's valid event names).
            context (CallbackContext): Context object passed to each callback.
        """
        for fn in self._callbacks[event_name]:
            fn(context)


def normalize_callbacks(
    callbacks: list[Callback] | CallbackManager | None,
) -> CallbackManager | None:
    """
    Normalize a callbacks input into a CallbackManager or preserve `None`.
    
    Converts a None, an existing CallbackManager, or a list of Callback instances into
    the canonical CallbackManager form used by the training loop.
    
    Parameters:
        callbacks (list[Callback] | CallbackManager | None): Either a list of Callback
            instances to register, an existing CallbackManager, or None.
    
    Returns:
        CallbackManager | None: A CallbackManager containing the provided callbacks,
        or `None` if `callbacks` was `None`.
    """
    if callbacks is None:
        return None
    if isinstance(callbacks, CallbackManager):
        return callbacks
    # It's a list of Callback instances
    manager = CallbackManager()
    manager.add(callbacks)
    return manager


def should_fire(callback_manager: CallbackManager | None, event_name: str) -> bool:
    """
    Determine whether callbacks should fire for a given event when a callback manager is present.
    
    Parameters:
        callback_manager: A CallbackManager instance or `None`.
        event_name: Name of the event to check (should be one of `VALID_EVENTS`).
    
    Returns:
        `true` if a callback manager is provided and at least one callback is registered for `event_name`, `false` otherwise.
    """
    return callback_manager is not None and callback_manager.has_callbacks(event_name)