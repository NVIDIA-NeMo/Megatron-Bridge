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

"""Utilities for recipe functional tests."""

from pathlib import Path
from typing import Callable, Optional

from megatron.bridge.training.config import ConfigContainer, runtime_config_update
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from tests.functional_tests.utils import (
    broadcast_path,
    clear_directories,
    initialize_distributed,
    verify_checkpoint_files,
)


def run_pretrain_recipe_test(
    config_func: Callable,
    recipe_name: str,
    tmp_path: Path,
    tensor_parallelism: Optional[int] = None,
    pipeline_parallelism: Optional[int] = None,
):
    """
    Common test implementation for pretrain recipe configurations.

    This function runs a minimal training session to verify that:
    1. The recipe config can be loaded without errors
    2. Training can start and run for a few iterations
    3. Checkpoints are saved correctly
    4. No crashes occur during the process

    Args:
        config_func: The recipe's pretrain_config function
        recipe_name: Name of the recipe for logging/debugging
        tmp_path: Temporary directory for test outputs
        tensor_parallelism: Override tensor parallelism (None = use recipe default)
        pipeline_parallelism: Override pipeline parallelism (None = use recipe default)
    """
    initialize_distributed()
    shared_base_dir = broadcast_path(tmp_path)

    try:
        config: ConfigContainer = config_func(
            dir=str(shared_base_dir), name=f"{recipe_name}_functional_test", mock=True
        )
        config.train.train_iters = 10
        config.train.eval_interval = 5
        config.train.eval_iters = 2
        config.scheduler.lr_warmup_iters = 2
        test_seq_length = 512
        config.model.seq_length = test_seq_length
        config.dataset.sequence_length = test_seq_length

        train_samples_needed = config.train.train_iters * config.train.global_batch_size
        eval_samples_needed = config.train.eval_iters * config.train.global_batch_size
        test_samples_needed = 100  # Minimal test samples

        total_samples = train_samples_needed + eval_samples_needed + test_samples_needed

        # Set dataset split ratios for minimal dataset
        train_split = train_samples_needed / total_samples
        valid_split = eval_samples_needed / total_samples
        test_split = test_samples_needed / total_samples

        config.dataset.split = [train_split, valid_split, test_split]

        if tensor_parallelism is not None:
            config.model.tensor_parallelism = tensor_parallelism
        if pipeline_parallelism is not None:
            config.model.pipeline_parallelism = pipeline_parallelism

        pretrain(config, forward_step)

        # Basic verification that training completed successfully
        verify_checkpoint_files(config.checkpoint.save, 10)

    finally:
        clear_directories(tmp_path)


def run_pretrain_config_override_test(config_func: Callable):
    """
    Common test implementation for testing pretrain_config with CLI-style overrides *after* instantiation.
    """
    config: ConfigContainer = config_func()

    # apply CLI-style overrides
    config.train.train_iters = 50000
    # FIXME:This should not be needed, but in some pretrain_config functions,
    # the default seq_length does *not* match the model seq_length.
    config.model.seq_length = 512
    config.dataset.sequence_length = 512

    assert config.scheduler.lr_decay_iters is None

    runtime_config_update(config)

    assert config.train.train_iters == 50000
    assert config.scheduler.lr_decay_iters == config.train.train_iters
