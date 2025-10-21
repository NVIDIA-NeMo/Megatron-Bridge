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
    tensor_parallelism: int | None = None,
    pipeline_parallelism: int | None = None,
    expert_parallelism: int | None = None,
    model_overrides: dict | None = None,
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
        expert_parallelism: Override expert parallelism (None = use recipe default)
        model_overrides: Optional mapping of model attribute overrides to apply
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

        # Keep dataloader light-weight for CI
        if hasattr(config.dataset, "pin_memory"):
            config.dataset.pin_memory = False
        if hasattr(config.dataset, "num_workers"):
            config.dataset.num_workers = 0
        if hasattr(config.dataset, "persistent_workers"):
            config.dataset.persistent_workers = False

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
            if hasattr(config.model, "tensor_model_parallel_size"):
                config.model.tensor_model_parallel_size = tensor_parallelism
            else:
                setattr(config.model, "tensor_parallelism", tensor_parallelism)
        if pipeline_parallelism is not None:
            if hasattr(config.model, "pipeline_model_parallel_size"):
                config.model.pipeline_model_parallel_size = pipeline_parallelism
            else:
                setattr(config.model, "pipeline_parallelism", pipeline_parallelism)
        if expert_parallelism is not None:
            if hasattr(config.model, "expert_model_parallel_size"):
                config.model.expert_model_parallel_size = expert_parallelism
            else:
                setattr(config.model, "expert_parallelism", expert_parallelism)

        # Apply any model-specific overrides provided by the caller
        if model_overrides:
            for attribute_name, attribute_value in model_overrides.items():
                setattr(config.model, attribute_name, attribute_value)

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


def run_pretrain_vl_recipe_test(
    config_func: Callable,
    recipe_name: str,
    tmp_path: Path,
    tensor_parallelism: int | None = None,
    pipeline_parallelism: int | None = None,
):
    """
    VLM variant of run_pretrain_recipe_test that uses the VLM forward step.

    Mirrors the llama/qwen functional test utility but routes through
    megatron.bridge.training.vlm_step.forward_step.
    """
    # Import locally to avoid loading VLM stack for non-VL tests
    from megatron.bridge.training.vlm_step import forward_step as vlm_forward_step

    initialize_distributed()
    shared_base_dir = broadcast_path(tmp_path)

    try:
        # Note: qwen_vl recipe config functions do not support 'mock' kwarg
        config: ConfigContainer = config_func(
            dir=str(shared_base_dir), name=f"{recipe_name}_functional_test", dataset_type="mock"
        )
        config.train.train_iters = 2
        config.train.eval_interval = 1
        config.train.eval_iters = 1
        config.scheduler.lr_warmup_iters = 1
        test_seq_length = 1024
        config.model.seq_length = test_seq_length
        config.dataset.sequence_length = test_seq_length

        # Disable pin-memory and worker persistence in tests to avoid
        # pin-memory device mismatches under torchrun+pytest environments.
        config.dataset.pin_memory = False
        config.dataset.num_workers = 0
        config.dataset.persistent_workers = False

        train_samples_needed = config.train.train_iters * config.train.global_batch_size
        eval_samples_needed = config.train.eval_iters * config.train.global_batch_size
        test_samples_needed = 8

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

        if model_overrides:
            for attribute_name, attribute_value in model_overrides.items():
                setattr(config.model, attribute_name, attribute_value)
        pretrain(config, vlm_forward_step)

        # Basic verification that training completed successfully
        verify_checkpoint_files(config.checkpoint.save, config.train.train_iters)

    finally:
        clear_directories(tmp_path)
