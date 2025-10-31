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

from megatron.bridge.training.config import ConfigContainer


MODEL_RECIPES: dict[str, Callable[[], "ConfigContainer"]] = {}


def register_model_recipe(func: Callable[[], "ConfigContainer"]):
    """Decorator that registers a model recipe factory by its function name."""
    name = func.__name__
    MODEL_RECIPES[name] = func
    return func


def get_model_recipe(
    model_name: str,
    model_size: str,
    gpu: str,
    num_gpus: int,
    compute_dtype: str,
) -> ConfigContainer:
    """Get the model recipe factory by its name."""
    recipe_name = f"{model_name}_{model_size}_{gpu}_{num_gpus}gpus_{compute_dtype}_config"
    try:
        return MODEL_RECIPES[recipe_name]()
    except KeyError as err:
        valid = ", ".join(sorted(MODEL_RECIPES.keys()))
        raise ValueError(f"Unknown model recipe '{recipe_name}'. Available recipes: {valid}.") from err
