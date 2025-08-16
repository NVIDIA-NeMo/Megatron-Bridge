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
from typing import Callable

from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from tests.functional_tests.utils import (
    broadcast_path,
    clear_directories,
    initialize_distributed,
    verify_checkpoint_files,
)


def run_pretrain_recipe_test(config_func: Callable, recipe_name: str, tmp_path: Path):
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
    """
    initialize_distributed()
    shared_base_dir = broadcast_path(tmp_path)

    try:
        # Get recipe config with minimal overrides for fast smoke testing
        config = config_func(dir=str(shared_base_dir), name=f"{recipe_name}_functional_test", mock=True)
        config.train.train_iters = 10
        config.train.eval_interval = 5
        config.train.eval_iters = 2
        test_seq_length = 512
        config.model.seq_length = test_seq_length
        config.dataset.sequence_length = test_seq_length

        # Run the smoke test
        pretrain(config, forward_step)

        # Basic verification that training completed successfully
        verify_checkpoint_files(config.checkpoint.save, 10)

    finally:
        # Clean up test artifacts
        clear_directories(tmp_path)
