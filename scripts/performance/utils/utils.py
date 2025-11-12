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
from typing import Dict, List, Optional


logger = logging.getLogger(__name__)


@dataclass
class WorkloadBaseConfig:
    """Container for workload base configs."""

    # NOTE: `num_gpus` is for representation purposes only. It is only meant to
    # communicate number of GPUs to be used for a specific workload in the file-
    # "scripts/performance/configs/<model_name>/workload_base_configs.py".

    # NOTE: You can specify number of GPUs to use for a SLURM job from command
    # line like `-ng/--num_gpus <num_gpus>` ("scripts/performance/README.md")
    # or update your sbatch script.
    num_gpus: int = 1

    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: int | None = None
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int | None = None

    global_batch_size: int = 1
    micro_batch_size: int = 1

    use_megatron_fsdp: Optional[bool] = None
    cuda_graph_impl: Optional[str] = None
    cuda_graph_scope: Optional[str] = None
    cpu_offloading_num_layers: Optional[int] = None
    recompute_num_layers: Optional[int] = None
    recompute_modules: Optional[List[str]] = None

    @property
    def sequence_parallel(self) -> bool:
        """Get the sequence parallel flag."""
        return bool(self.tensor_model_parallel_size > 1)

    @property
    def gbs_scaling_factor(self) -> float:
        """Get the global batch size scaling factor."""
        return self.global_batch_size / self.num_gpus


def get_model_recipe(
    model_name: str,
    model_size: str,
    gpu: str,
    compute_dtype: str,
):
    """Get the model recipe factory by its name."""
    recipe_name = f"{model_name}_{model_size}_{gpu}_config"
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

    return recipe_builder(precision=compute_dtype)


def get_workload_base_config(
    model_name: str,
    model_size: str,
    gpu: str,
    compute_dtype: str,
) -> Dict[str, int]:
    """Get the workload base config for a given model, size, GPU, compute dtype, and FP8 recipe."""
    workload_base_config_name = f"{model_name}_{model_size}_{gpu}_{compute_dtype}"
    workload_base_config_name = workload_base_config_name.upper() + "_BASE_CONFIG"

    module_name = f"configs.{model_name}.workload_base_configs"
    try:
        module = importlib.import_module(module_name)
        logger.info(f"Imported module '{module_name}'.")
    except ModuleNotFoundError as exc:
        raise ValueError(f"Failed to import module '{module_name}'") from exc

    try:
        workload_base_config = getattr(module, workload_base_config_name)
        logger.info(f"Loaded {workload_base_config=}")
    except AttributeError:
        logger.error(f"Failed to get {workload_base_config_name=} from {module_name=}")
        workload_base_config = WorkloadBaseConfig()

    return workload_base_config
