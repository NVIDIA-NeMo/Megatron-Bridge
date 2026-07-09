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


def configure_ci_pretraining_dataset(config: ConfigContainer, test_data_root: Path) -> None:
    """Use the small indexed CI corpus instead of generated mock samples.

    The functional-test asset is already downloaded by the session-scoped
    ``ensure_test_data`` fixture. Its token ids fit every model covered by the
    MoE performance proxies, so the production ``GPTDatasetConfig`` path can be
    exercised without adding a model-specific tokenizer or online dependency.
    """
    # torchrun starts one independent pytest process per rank, so each process
    # receives a different tmp_path_factory directory. MCore builds dataset
    # indices on rank 0 only; all ranks therefore need to use rank 0's copy of
    # the downloaded corpus and its shared index cache.
    shared_test_data_root = Path(broadcast_path(test_data_root))
    data_prefix = shared_test_data_root / "datasets" / "fim" / "fim_text_document"
    for suffix in (".bin", ".idx"):
        data_file = data_prefix.with_suffix(suffix)
        if not data_file.is_file():
            raise FileNotFoundError(f"CI pretraining dataset file is missing: {data_file}")

    config.dataset.blend = None
    config.dataset.blend_per_split = None
    config.dataset.data_path = str(data_prefix)
    config.dataset.split = "100,0,0"
    config.dataset.num_workers = 0


def run_pretrain_recipe_test(
    config_func: Callable,
    recipe_name: str,
    tmp_path: Path,
    tensor_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_size: Optional[int] = None,
    expert_model_parallel_size: Optional[int] = None,
    model_overrides: Optional[dict] = None,
    checkpoint_overrides: Optional[dict] = None,
    ddp_overrides: Optional[dict] = None,
):
    """
    Common test implementation for pretrain recipe configurations.

    This function runs a minimal training session to verify that:
    1. The recipe config can be loaded without errors
    2. Training can start and run for a few iterations
    3. Checkpoints are saved correctly
    4. No crashes occur during the process

    Args:
        config_func: The recipe's pretrain_config function (parameterless API)
        recipe_name: Name of the recipe for logging/debugging
        tmp_path: Temporary directory for test outputs
        tensor_model_parallel_size: Override tensor parallelism (None = use recipe default)
        pipeline_model_parallel_size: Override pipeline parallelism (None = use recipe default)
        expert_model_parallel_size: Override expert parallelism (None = use recipe default)
        model_overrides: Optional mapping of model attribute overrides to apply
    """
    initialize_distributed()
    shared_base_dir = Path(broadcast_path(tmp_path))

    try:
        # Pretrain configs use parameterless API - call without arguments
        config: ConfigContainer = config_func()

        # Set up output directories after instantiation
        run_output_dir = shared_base_dir / f"{recipe_name}_functional_test"
        checkpoint_dir = run_output_dir / "checkpoints"
        tensorboard_dir = run_output_dir / "tb_logs"
        config.checkpoint.save = str(checkpoint_dir)
        config.checkpoint.load = str(checkpoint_dir)
        config.logger.tensorboard_dir = str(tensorboard_dir)
        # Keep runs short and consistent across tests
        config.train.train_iters = 10
        config.validation.eval_interval = 5
        config.validation.eval_iters = 2
        # Standardize batch sizes for functional tests
        config.train.micro_batch_size = 1
        config.train.global_batch_size = 8
        config.scheduler.lr_warmup_iters = 2
        test_seq_length = 512
        config.model.seq_length = test_seq_length
        config.dataset.seq_length = test_seq_length
        # Keep dataloader light-weight for CI
        if hasattr(config.dataset, "pin_memory"):
            config.dataset.pin_memory = False
        if hasattr(config.dataset, "num_workers"):
            config.dataset.num_workers = 0
        if hasattr(config.dataset, "persistent_workers"):
            config.dataset.persistent_workers = False

        train_samples_needed = config.train.train_iters * config.train.global_batch_size
        eval_samples_needed = config.validation.eval_iters * config.train.global_batch_size
        test_samples_needed = 100  # Minimal test samples

        total_samples = train_samples_needed + eval_samples_needed + test_samples_needed

        # Set dataset split ratios for minimal dataset
        train_split = train_samples_needed / total_samples
        valid_split = eval_samples_needed / total_samples
        test_split = test_samples_needed / total_samples

        config.dataset.split = [train_split, valid_split, test_split]

        if tensor_model_parallel_size is not None:
            if hasattr(config.model, "tensor_model_parallel_size"):
                config.model.tensor_model_parallel_size = tensor_model_parallel_size
        if pipeline_model_parallel_size is not None:
            if hasattr(config.model, "pipeline_model_parallel_size"):
                config.model.pipeline_model_parallel_size = pipeline_model_parallel_size
        if expert_model_parallel_size is not None:
            if hasattr(config.model, "expert_model_parallel_size"):
                config.model.expert_model_parallel_size = expert_model_parallel_size

        # Apply any model-specific overrides provided by the caller
        if model_overrides:
            for attribute_name, attribute_value in model_overrides.items():
                if not hasattr(config.model, attribute_name):
                    raise ValueError(f"Attempted to test a foreign attribute ({attribute_name}) in {config.model}.")
                setattr(config.model, attribute_name, attribute_value)

        if checkpoint_overrides:
            for attribute_name, attribute_value in checkpoint_overrides.items():
                if not hasattr(config.checkpoint, attribute_name):
                    raise ValueError(
                        f"Attempted to test a foreign attribute ({attribute_name}) in {config.checkpoint}."
                    )
                setattr(config.checkpoint, attribute_name, attribute_value)

        if ddp_overrides:
            for attribute_name, attribute_value in ddp_overrides.items():
                if not hasattr(config.ddp, attribute_name):
                    raise ValueError(f"Attempted to test a foreign attribute ({attribute_name}) in {config.ddp}.")
                setattr(config.ddp, attribute_name, attribute_value)

        pretrain(config, forward_step)

        # Basic verification that training completed successfully
        verify_checkpoint_files(
            config.checkpoint.save,
            10,
            ckpt_format=config.checkpoint.ckpt_format,
            storage_writers_per_rank=config.checkpoint.storage_writers_per_rank,
        )

    finally:
        clear_directories(tmp_path)


def _run_pretrain_without_checkpoint(
    config_func: Callable,
    recipe_name: str,
    config_overrides: Optional[dict] = None,
):
    """
    Common implementation for short pretraining tests without checkpoint I/O.

    This function runs a minimal training session to verify that:
    1. The recipe config can be loaded without errors
    2. Training can start and run for a few iterations
    3. No crashes occur during the process

    Args:
        config_func: The recipe's pretrain_config function (parameterless API)
        recipe_name: Name of the recipe for logging/debugging
        config_overrides: Optional mapping of config attribute overrides to apply
    """
    initialize_distributed()

    # Pretrain configs use parameterless API - call without arguments
    config: ConfigContainer = config_func()
    config.checkpoint.save = None
    config.checkpoint.load = None
    config.checkpoint.pretrained_checkpoint = None
    # Keep runs short and consistent across tests
    config.train.train_iters = 10
    config.validation.eval_interval = 5
    config.validation.eval_iters = 0  # Skip evaluation. TODO: Fix this.
    config.logger.log_interval = 1

    # Standardize batch sizes for functional tests
    config.train.micro_batch_size = 1
    config.train.global_batch_size = 8
    config.scheduler.lr_warmup_iters = 2
    test_seq_length = 512
    config.model.seq_length = test_seq_length
    config.dataset.seq_length = test_seq_length
    config.train.global_batch_size = 8

    # Apply any model-specific overrides provided by the caller
    if config_overrides:
        for obj_name, overrides_dict in config_overrides.items():
            config_obj = getattr(config, obj_name)
            for key, value in overrides_dict.items():
                if not hasattr(config_obj, key):
                    raise ValueError(f"Attempted to test a foreign attribute ({key}) in {config_obj}.")
                setattr(config_obj, key, value)

    pretrain(config, forward_step)


def run_perf_recipe_proxy_test(
    config_func: Callable,
    recipe_name: str,
    config_overrides: Optional[dict] = None,
):
    """Run a compact functional proxy derived from a production performance recipe.

    Callers should keep performance behavior in ``config_func`` and limit
    overrides to reductions required by the CI topology, dataset, and run
    length. This prevents functional tests from growing a second, manually
    maintained copy of the production performance configuration.
    """
    _run_pretrain_without_checkpoint(config_func, recipe_name, config_overrides)


def run_pretrain_feature_test(
    config_func: Callable,
    test_name: str,
    config_overrides: Optional[dict] = None,
):
    """Run a short no-checkpoint pretraining test for a focused training feature."""
    _run_pretrain_without_checkpoint(config_func, test_name, config_overrides)


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
    config.dataset.seq_length = 512

    assert config.scheduler.lr_decay_iters is None

    runtime_config_update(config)

    assert config.train.train_iters == 50000
    assert config.scheduler.lr_decay_iters == config.train.train_iters


def run_pretrain_vl_recipe_test(
    config_func: Callable,
    recipe_name: str,
    tmp_path: Path,
    tensor_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_size: Optional[int] = None,
    model_overrides: Optional[dict] = None,
    dataset_overrides: Optional[dict] = None,
    forward_step_func: Optional[Callable] = None,
):
    """
    VLM variant of run_pretrain_recipe_test that uses the VLM forward step.

    Mirrors the llama/qwen functional test utility but routes through
    megatron.bridge.training.vlm_step.forward_step.

    Args:
        config_func: The recipe's config function (parameterless API for SFT,
                     or takes peft_scheme parameter for PEFT)
        recipe_name: Name of the recipe for logging/debugging
        tmp_path: Temporary directory for test outputs
        tensor_model_parallel_size: Override tensor parallelism (None = use recipe default)
        pipeline_model_parallel_size: Override pipeline parallelism (None = use recipe default)
        model_overrides: Optional mapping of model attribute overrides to apply
        dataset_overrides: Optional mapping of dataset attribute overrides to apply
    """
    from megatron.bridge.data.vlm_datasets.mock_provider import MockVLMConversationProvider

    if forward_step_func is None:
        # Import locally to avoid loading VLM stack for non-VL tests
        from megatron.bridge.training.vlm_step import forward_step as vlm_forward_step
    else:
        vlm_forward_step = forward_step_func

    initialize_distributed()
    shared_base_dir = Path(broadcast_path(tmp_path))

    try:
        # VLM recipe configs use parameterless API - call without arguments
        config: ConfigContainer = config_func()

        # Set up output directories after instantiation
        run_output_dir = shared_base_dir / f"{recipe_name}_functional_test"
        tensorboard_dir = run_output_dir / "tb_logs"
        config.checkpoint.save = None
        config.checkpoint.load = None
        config.logger.tensorboard_dir = str(tensorboard_dir)

        # Keep runs short and consistent across tests
        config.train.train_iters = 10
        config.train.eval_interval = None
        config.train.eval_iters = None
        config.validation.eval_interval = 5
        config.validation.eval_iters = 2
        # Standardize batch sizes for functional tests
        config.train.micro_batch_size = 1
        config.train.global_batch_size = 8
        config.scheduler.lr_warmup_iters = 1
        test_seq_length = 1024
        config.model.seq_length = test_seq_length

        # Get the HF processor path from the original dataset config before replacing
        hf_processor_path = getattr(config.dataset, "hf_processor_path", None)
        enable_in_batch_packing = getattr(config.dataset, "enable_in_batch_packing", False)
        defer_in_batch_packing_to_step = getattr(config.dataset, "defer_in_batch_packing_to_step", False)
        pad_to_max_length = getattr(config.dataset, "pad_to_max_length", False)
        pad_to_multiple_of = getattr(config.dataset, "pad_to_multiple_of", 128)
        in_batch_packing_pad_to_multiple_of = getattr(config.dataset, "in_batch_packing_pad_to_multiple_of", 1)

        # Replace the real dataset with a mock dataset provider for tests
        # MockVLMConversationProvider generates synthetic data and doesn't need a split attribute
        # since the DatasetBuildContext calculates sample counts from training configuration
        config.dataset = MockVLMConversationProvider(
            seq_length=test_seq_length,
            hf_processor_path=hf_processor_path,
            enable_in_batch_packing=enable_in_batch_packing,
            defer_in_batch_packing_to_step=defer_in_batch_packing_to_step,
            pad_to_max_length=pad_to_max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            in_batch_packing_pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
        )

        if tensor_model_parallel_size is not None:
            if hasattr(config.model, "tensor_model_parallel_size"):
                config.model.tensor_model_parallel_size = tensor_model_parallel_size
        if pipeline_model_parallel_size is not None:
            if hasattr(config.model, "pipeline_model_parallel_size"):
                config.model.pipeline_model_parallel_size = pipeline_model_parallel_size

        # Apply any model-specific overrides provided by the caller
        if model_overrides:
            for attribute_name, attribute_value in model_overrides.items():
                if not hasattr(config.model, attribute_name):
                    raise ValueError(f"Attempted to test a foreign attribute ({attribute_name}) in {config.model}.")
                setattr(config.model, attribute_name, attribute_value)

        # Apply any dataset-specific overrides provided by the caller
        if dataset_overrides:
            for attribute_name, attribute_value in dataset_overrides.items():
                if not hasattr(config.dataset, attribute_name):
                    raise ValueError(f"Attempted to test a foreign attribute ({attribute_name}) in {config.dataset}.")
                setattr(config.dataset, attribute_name, attribute_value)

        if hasattr(config.dataset, "enable_in_batch_packing") and config.dataset.enable_in_batch_packing:
            config.train.micro_batch_size = 2

        pretrain(config, vlm_forward_step)

        # Basic verification that training completed successfully
        if config.checkpoint.save is not None:
            verify_checkpoint_files(
                config.checkpoint.save,
                config.train.train_iters,
                ckpt_format=config.checkpoint.ckpt_format,
                storage_writers_per_rank=config.checkpoint.storage_writers_per_rank,
            )

    finally:
        clear_directories(tmp_path)
