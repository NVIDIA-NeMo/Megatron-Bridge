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

import importlib
import logging
from dataclasses import dataclass
from typing import Dict, Optional


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParallelismAndBatchConfig:
    """Container for model parallelism and batch size overrides."""

    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    context_parallel_size: int
    virtual_pipeline_model_parallel_size: int | None
    expert_model_parallel_size: int
    expert_tensor_parallel_size: int | None
    global_batch_size: int
    micro_batch_size: int
    sequence_parallel: Optional[bool] = None

    def sequence_parallel_enabled(self) -> bool:
        if self.sequence_parallel is not None:
            return self.sequence_parallel
        return self.tensor_model_parallel_size > 1


def get_model_recipe(
    model_name: str,
    model_size: str,
    gpu: str,
    num_gpus: int,
    compute_dtype: str,
    fp8_recipe: Optional[str] = None,
):
    """Get the model recipe factory by its name."""
    recipe_name = f"{model_name}_{model_size}_{gpu}_{num_gpus}gpus_{compute_dtype}_config"
    module_name = f"configs.{model_name}.{model_name}_llm_pretrain"
    try:
        module = importlib.import_module(module_name)
        logger.debug("Imported configuration module '%s' to load recipe '%s'.", module_name, recipe_name)
    except ModuleNotFoundError as exc:
        raise ValueError(f"Failed to import configuration module '{module_name}'") from exc

    try:
        recipe_builder = getattr(module, recipe_name)
    except AttributeError as err:
        raise ValueError(f"Failed to get recipe builder '{recipe_name}' from module '{module_name}'") from err

    if compute_dtype == "fp8" and fp8_recipe is not None:
        return recipe_builder(fp8_recipe=fp8_recipe)
    elif compute_dtype == "bf16":
        return recipe_builder()
    else:
        raise ValueError(f"Invalid compute dtype: {compute_dtype} and FP8 recipe: {fp8_recipe}")


def get_parallelism_defaults(
    model_name: str,
    model_size: str,
    gpu: str,
    num_gpus: int,
    compute_dtype: str,
    fp8_recipe: Optional[str] = None,
) -> Dict[str, int]:
    """Get the parallelism defaults for a given model, size, GPU, number of GPUs, compute dtype, and FP8 recipe."""
    parallelism_name = f"{model_name}_{model_size}_{gpu}_{num_gpus}gpus_{compute_dtype}"
    if compute_dtype == "fp8":
        fp8_recipe = "cs" if fp8_recipe == "ds" else fp8_recipe  # we don't maintain FP8-DS parallelism configs
        parallelism_name += f"_{fp8_recipe}"
    parallelism_name = parallelism_name.upper() + "_PARALLEL_CONFIG"

    module_name = f"configs.{model_name}.parallelism_configs"
    try:
        module = importlib.import_module(module_name)
        logger.info(
            "Imported configuration module '%s' to load parallelism config '%s'.", module_name, parallelism_name
        )
    except ModuleNotFoundError as exc:
        raise ValueError(f"Failed to import configuration module '{module_name}'") from exc

    try:
        parallelism_config = getattr(module, parallelism_name)
        logger.info(f"Loaded parallelism config: {parallelism_config}")
    except AttributeError:
        logger.error(f"Failed to get parallelism config '{parallelism_name}' from module '{module_name}'")
        parallelism_config = ParallelismAndBatchConfig(1, 1, 1, None, 1, None, 1, 1)

    return parallelism_config
