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

import ast
import importlib
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from parse_default_config_ast import ParallelismExtractor

from megatron.bridge.training.config import ConfigContainer


logger = logging.getLogger(__name__)


def get_model_recipe(
    model_name: str,
    model_size: str,
    gpu: str,
    num_gpus: int,
    compute_dtype: str,
    fp8_recipe: Optional[str] = None,
) -> ConfigContainer:
    """Get the model recipe factory by its name."""
    recipe_name = f"{model_name}_{model_size}_{gpu}_{num_gpus}gpus_{compute_dtype}_config"
    module_name = f"configs.{model_name}.{model_name}_{model_size}_llm_pretrain"
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


def _config_module_path(model_name: str, model_size: str) -> Path:
    configs_root = Path(__file__).resolve().parent.parent / "configs"
    return configs_root / model_name / f"{model_name}_{model_size}_llm_pretrain.py"


@lru_cache(maxsize=None)
def get_parallelism_defaults(
    model_name: str,
    model_size: str,
    gpu: str,
    num_gpus: int,
    compute_dtype: str,
    fp8_recipe: Optional[str] = None,
) -> Dict[str, int]:
    """Get the parallelism defaults for a given model, size, GPU, number of GPUs, compute dtype, and FP8 recipe."""
    config_path = _config_module_path(model_name, model_size)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration module path not found: {config_path}")

    function_name = f"{model_name}_{model_size}_{gpu}_{num_gpus}gpus_{compute_dtype}_config"

    try:
        tree = ast.parse(config_path.read_text())
    except SyntaxError as exc:
        raise RuntimeError(f"Failed to parse configuration module: {config_path}") from exc

    context = {
        "fp8_recipe": fp8_recipe,
        "compute_dtype": compute_dtype,
        "gpu": gpu,
        "num_gpus": num_gpus,
        "model_name": model_name,
        "model_size": model_size,
    }

    extractor = ParallelismExtractor(function_name, context)
    extractor.visit(tree)

    if not extractor.defaults:
        raise ValueError(
            f"Unable to extract parallelism defaults for '{function_name}' in '{config_path}'. "
            "Ensure the configuration uses explicit assignments for tensor/pipeline/context parallel sizes."
        )

    tp_size = extractor.defaults.get("tensor_model_parallel_size", 1)
    pp_size = extractor.defaults.get("pipeline_model_parallel_size", 1)
    cp_size = extractor.defaults.get("context_parallel_size", 1)

    return {
        "tp_size": tp_size,
        "pp_size": pp_size,
        "cp_size": cp_size,
    }
