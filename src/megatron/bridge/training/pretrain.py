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

import torch.distributed as dist
from nvidia_resiliency_ext.inprocess import CallWrapper

from megatron.bridge.data.utils import get_dataset_provider
from megatron.bridge.training.callbacks import Callback, CallbackManager, normalize_callbacks
from megatron.bridge.training.checkpointing import save_checkpoint
from megatron.bridge.training.config import ConfigContainer, runtime_config_update
from megatron.bridge.training.eval import evaluate_and_print_results
from megatron.bridge.training.forward_step_func_types import ForwardStepCallable
from megatron.bridge.training.setup import setup
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.train import _finish_train, train
from megatron.bridge.training.utils.log_utils import barrier_and_log
from megatron.bridge.utils.common_utils import print_rank_0
from megatron.bridge.utils.decorators import experimental_fn


@experimental_fn
def pretrain(
    config: ConfigContainer,
    forward_step_func: ForwardStepCallable,
    callbacks: list[Callback] | CallbackManager | None = None,
) -> None:
    """Main function to run the training pipeline.

    Sets up the environment, model, optimizer, scheduler, and data iterators.
    Performs training, validation, and optionally testing based on the provided
    configuration.

    Args:
        config: The main configuration container holding all necessary parameters.
        forward_step_func: A callable (function or functor) that performs a single
                          forward and backward step, returning the loss and any computed
                          metrics. Supports the following signatures:
                          - 2 args: (data_iterator, model)
                          - 3 args: (data_iterator, model, return_schedule_plan=False)
                                   OR (state: GlobalState, data_iterator, model)
                          - 4 args: (state: GlobalState, data_iterator, model, return_schedule_plan=False)
        callbacks: Optional callbacks for custom logic injection. Can be:
                   - list[Callback]: List of Callback subclass instances
                   - CallbackManager: Pre-configured manager with registered callbacks
                   - None: No callbacks (default)

    Note:
        Use the signature with GlobalState type hint for full access to configuration, timers, and training state.
        State injection is automatic based on type hints or parameter names.
        Functors (classes with __call__) are fully supported.

    Warnings:
        This is an experimental API and is subject to change in backwards
        incompatible ways without notice.
    """
    # Apply runtime config updates prior to creating/attaching GlobalState
    runtime_config_update(config)

    # Create a single GlobalState instance regardless of restart path
    state = GlobalState()
    state.cfg = config

    # Normalize callbacks to CallbackManager
    callback_manager = normalize_callbacks(callbacks)

    if config.inprocess_restart and config.inprocess_restart.enabled:
        if dist.is_initialized():
            raise RuntimeError(
                "In-process restart is incompatible with user-initialized process groups. "
                "The in-process restart mechanism expects to manage the process group lifecycle "
                "and will destroy it during fault recovery. Either:\n"
                "1. Disable in-process restart and manage the process group yourself, or\n"
                "2. Let the framework initialize the process group by not calling "
                "torch.distributed.init_process_group() before training."
            )

        # Apply in-process restart wrapper directly to _pretrain
        from megatron.bridge.training.inprocess_restart import maybe_wrap_for_inprocess_restart

        # Wrap _pretrain directly and get the store; state is captured for abort
        wrapped_pretrain, store = maybe_wrap_for_inprocess_restart(_pretrain, config.inprocess_restart, state)

        # Execute the wrapped function - nvidia-resiliency-ext will inject inprocess_call_wrapper
        # Call with positional args matching the adapter signature: (state, forward_step_func, store=None, inprocess_call_wrapper=None)
        wrapped_pretrain(state, forward_step_func, callback_manager, store=store)
    else:
        # Normal execution without in-process restart
        _pretrain(state=state, forward_step_func=forward_step_func, callback_manager=callback_manager)


def _pretrain(
    state: GlobalState,
    forward_step_func: ForwardStepCallable,
    callback_manager: CallbackManager | None = None,
    store: dist.Store | None = None,
    inprocess_call_wrapper: CallWrapper | None = None,
) -> None:
    """Internal function containing the actual pretrain logic.

    Args:
        state: Global training state containing the validated configuration and runtime objects
        forward_step_func: Function or functor that performs a single forward/backward step
        callback_manager: Optional CallbackManager for custom callback execution
        store: Optional distributed Store used by in-process restart for coordination
        inprocess_call_wrapper: Optional wrapper injected by nvrx to expose restart iteration
    """
    # Determine whether the training loop will initialize the process group
    # If the trainer creates the process group, the trainer should destroy it before returning control back to the user
    should_destroy_process_group = not dist.is_initialized()

    # Handle in-process restart store prefix
    if inprocess_call_wrapper is not None:
        restart_attempt = inprocess_call_wrapper.iteration
        store = dist.PrefixStore(str(restart_attempt), store)

    config = state.cfg
    dataset_provider = get_dataset_provider(config.dataset)
    setup_output = setup(state, dataset_provider, restart_store=store, callback_manager=callback_manager)
    state = setup_output.state
    model = setup_output.model
    optimizer = setup_output.optimizer
    scheduler = setup_output.scheduler
    train_data_iterator = setup_output.train_data_iterator
    valid_data_iterator = setup_output.valid_data_iterator
    test_data_iterator = setup_output.test_data_iterator
    ckpt_context = setup_output.checkpointing_context
    pg_collection = setup_output.pg_collection

    # TRAINING
    if not config.validation.skip_train:
        if state.train_state.do_train and config.train.train_iters > 0:
            train(
                forward_step_func,
                model,
                optimizer,
                scheduler,
                train_data_iterator,
                valid_data_iterator,
                state,
                ckpt_context,
                pg_collection,
                callback_manager=callback_manager,
            )

        barrier_and_log("after training is done")
        ckpt_config = config.checkpoint
        if ckpt_config.save and state.train_state.step != 0 and ckpt_config.save_interval != 0:
            save_checkpoint(
                state,
                model,
                optimizer,
                scheduler,
                state.train_state.floating_point_operations_so_far,
                ckpt_context,
                train_data_iterator=train_data_iterator,
            )

    else:
        print_rank_0("skipping training ...")

    iteration = state.train_state.step

    # VALIDATION
    if state.train_state.do_valid:
        prefix = f"iteration {iteration} on validation set"
        evaluate_and_print_results(
            state,
            prefix,
            forward_step_func,
            valid_data_iterator,
            model,
            config.model,
            verbose=True,
            write_to_tensorboard=not config.validation.skip_train,
            callback_manager=callback_manager,
        )
    if state.train_state.do_test:
        prefix = f"iteration {iteration} on test set"
        evaluate_and_print_results(
            state,
            prefix,
            forward_step_func,
            test_data_iterator,
            model,
            config.model,
            verbose=True,
            write_to_tensorboard=not config.validation.skip_train,
            callback_manager=callback_manager,
            is_test=True,
        )

    _finish_train(state)
    _maybe_destroy_process_group(should_destroy_process_group)


def _maybe_destroy_process_group(should_destroy: bool) -> None:
    """Destroy the process group if it was created by this training session.

    Args:
        should_destroy: Whether the process group should be destroyed
    """
    if should_destroy and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
