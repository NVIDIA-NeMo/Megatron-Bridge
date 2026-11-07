# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""TrainingEngine recipes: model and execution configuration per supported model and hardware."""

from megatron.bridge.engine.recipes.base import (
    TrainingRecipe,
    check_supported,
    data_parallel_size,
    finalize_recipe,
    min_gpus,
)
from megatron.bridge.engine.recipes.qwen35 import qwen35_text_35b_a3b_rl_16gpu_gb200_bf16
from megatron.bridge.engine.recipes.registry import (
    KNOWN_HARDWARE,
    REGISTRY,
    RecipeEntry,
    apply_recipe_overrides,
    detect_hardware,
    resolve_recipe,
)


__all__ = [
    "KNOWN_HARDWARE",
    "REGISTRY",
    "RecipeEntry",
    "TrainingRecipe",
    "apply_recipe_overrides",
    "check_supported",
    "data_parallel_size",
    "detect_hardware",
    "finalize_recipe",
    "min_gpus",
    "qwen35_text_35b_a3b_rl_16gpu_gb200_bf16",
    "resolve_recipe",
]
