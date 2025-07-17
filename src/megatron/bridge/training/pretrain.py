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

from typing import Callable

from megatron.bridge.data.utils import get_dataset_provider
from megatron.bridge.training.checkpointing import save_checkpoint
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.eval import evaluate_and_print_results
from megatron.bridge.training.setup import setup
from megatron.bridge.training.train import _finish_train, train
from megatron.bridge.training.utils.log_utils import barrier_and_log
from megatron.bridge.utils.common_utils import print_rank_0
from megatron.bridge.utils.decorators import experimental_fn


@experimental_fn
def pretrain(
    config: ConfigContainer,
    forward_step_func: Callable,
) -> None:
    """Main function to run the pretraining pipeline.

    Sets up the environment, model, optimizer, scheduler, and data iterators.
    Performs training, validation, and optionally testing based on the provided
    configuration.

    Args:
        config: The main configuration container holding all necessary parameters.
        forward_step_func: A callable that performs a single forward and backward
                           step, returning the loss and any computed metrics.

    Warnings:
        This is an experimental API and is subject to change in backwards
        incompatible ways without notice.
    """
    dataset_provider = get_dataset_provider(config.dataset)
    setup_output = setup(config, dataset_provider)
    state = setup_output.state
    model = setup_output.model
    optimizer = setup_output.optimizer
    scheduler = setup_output.scheduler
    train_data_iterator = setup_output.train_data_iterator
    valid_data_iterator = setup_output.valid_data_iterator
    test_data_iterator = setup_output.test_data_iterator
    ckpt_context = setup_output.checkpointing_context

    # TRAINING
    if not config.train.skip_train:
        print_rank_0("Training ...")
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
            write_to_tensorboard=not config.train.skip_train,
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
            write_to_tensorboard=not config.train.skip_train,
        )

    _finish_train(state)
