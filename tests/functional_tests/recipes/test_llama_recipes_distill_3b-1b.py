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

from pathlib import Path
from typing import Callable, Optional

import pytest

from megatron.bridge.models.gpt_provider import GPTDistillationProvider
from megatron.bridge.recipes.llama import (
    llama32_1b_pretrain_config,
    llama32_3b_pretrain_config,
)
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.distill import distill
from megatron.bridge.training.post_training.distillation import ModelOptDistillConfig
from tests.functional_tests.utils import (
    broadcast_path,
    clear_directories,
    initialize_distributed,
    verify_checkpoint_files,
)


LLAMA_DISTILL_RECIPES = [
    # (student_config_func, teacher_config_func, name, parallelism_overrides)
    (llama32_1b_pretrain_config, llama32_3b_pretrain_config, "llama32_3b-1b", {"tensor_model_parallel_size": 2}),
]


class TestLlamaDistillRecipes:
    """Test class for LLaMA distillation recipe functional tests."""

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "student_config_func,teacher_config_func,recipe_name,parallelism_overrides", LLAMA_DISTILL_RECIPES
    )
    def test_llama_distill_recipes(
        self, student_config_func, teacher_config_func, recipe_name, parallelism_overrides, tmp_path
    ):
        """Functional test for LLaMA distillation recipes with appropriate parallelism configurations."""
        run_distill_recipe_test(
            student_config_func, teacher_config_func, recipe_name, tmp_path, **parallelism_overrides
        )


def run_distill_recipe_test(
    student_config_func: Callable,
    teacher_config_func: Callable,
    recipe_name: str,
    tmp_path: Path,
    tensor_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_size: Optional[int] = None,
    expert_model_parallel_size: Optional[int] = None,
    model_overrides: Optional[dict] = None,
):
    """
    Common test implementation for distillation recipe configurations.

    This function runs a minimal distillation session to verify that:
    1. The recipe config can be loaded without errors
    2. Distillation can start and run for a few iterations
    3. Checkpoints are saved correctly
    4. No crashes occur during the process

    Args:
        student_config_func: The student model's pretrain_config function
        teacher_config_func: The teacher model's pretrain_config function
        recipe_name: Name of the recipe for logging/debugging
        tmp_path: Temporary directory for test outputs
        tensor_model_parallel_size: Override tensor parallelism (None = use recipe default)
        pipeline_model_parallel_size: Override pipeline parallelism (None = use recipe default)
        expert_model_parallel_size: Override expert parallelism (None = use recipe default)
        model_overrides: Optional mapping of model attribute overrides to apply
    """
    initialize_distributed()
    shared_base_dir = broadcast_path(tmp_path)

    try:
        # Load student config and wrap it with GPTDistillationProvider
        config: ConfigContainer = student_config_func(
            dir=str(shared_base_dir), name=f"{recipe_name}_functional_test", mock=True
        )
        config.model.__class__ = GPTDistillationProvider

        # Load teacher config and add to student config
        teacher_config = teacher_config_func(
            dir=str(shared_base_dir), name=f"{recipe_name}_teacher_functional_test", mock=True
        )
        config.model.teacher = teacher_config.model

        # Set default distillation
        config.model.kd_config = ModelOptDistillConfig()

        config.train.train_iters = 10
        config.train.eval_interval = 5
        config.train.eval_iters = 2
        config.scheduler.lr_warmup_iters = 2
        test_seq_length = 512
        config.model.seq_length = test_seq_length
        config.model.teacher.seq_length = test_seq_length
        config.dataset.sequence_length = test_seq_length
        config.train.global_batch_size = 8
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

        # Apply parallelism overrides to both student and teacher models
        if tensor_model_parallel_size is not None:
            if hasattr(config.model, "tensor_model_parallel_size"):
                config.model.tensor_model_parallel_size = tensor_model_parallel_size
            if hasattr(config.model.teacher, "tensor_model_parallel_size"):
                config.model.teacher.tensor_model_parallel_size = tensor_model_parallel_size
        if pipeline_model_parallel_size is not None:
            if hasattr(config.model, "pipeline_model_parallel_size"):
                config.model.pipeline_model_parallel_size = pipeline_model_parallel_size
            if hasattr(config.model.teacher, "pipeline_model_parallel_size"):
                config.model.teacher.pipeline_model_parallel_size = pipeline_model_parallel_size
        if expert_model_parallel_size is not None:
            if hasattr(config.model, "expert_model_parallel_size"):
                config.model.expert_model_parallel_size = expert_model_parallel_size
            if hasattr(config.model.teacher, "expert_model_parallel_size"):
                config.model.teacher.expert_model_parallel_size = expert_model_parallel_size

        # Apply any model-specific overrides provided by the caller
        if model_overrides:
            for attribute_name, attribute_value in model_overrides.items():
                setattr(config.model, attribute_name, attribute_value)

        distill(config=config)

        # Basic verification that training completed successfully
        verify_checkpoint_files(config.checkpoint.save, 10)

    finally:
        clear_directories(tmp_path)
