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

"""Functional smoke tests for Nemotron H and Nemotron Nano v2 recipe configurations."""

import pytest

from megatron.bridge.recipes.nemotronh import (
    nemotron_nano_9b_v2_pretrain_config,
    nemotronh_4b_pretrain_config,
)
from tests.functional_tests.recipes.utils import run_pretrain_recipe_test


NEMOTRONH_PRETRAIN_RECIPES = [
    # (config_func, name, parallelism_overrides, model_overrides)
    (
        nemotronh_4b_pretrain_config,
        "nemotronh_4b",
        {"tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1},
        {"num_layers": 3, "hybrid_override_pattern": "M*-"},
    ),
]


NEMOTRON_NANO_V2_PRETRAIN_RECIPES = [
    # (config_func, name, parallelism_overrides, model_overrides)
    (
        nemotron_nano_9b_v2_pretrain_config,
        "nemotron_nano_9b_v2",
        {"tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1},
        {"num_layers": 3, "hybrid_override_pattern": "M*-", "sequence_parallel": False},
    ),
]


class TestNemotronHRecipes:
    """Test class for Nemotron H recipe functional tests."""

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "config_func,recipe_name,parallelism_overrides,model_overrides", NEMOTRONH_PRETRAIN_RECIPES
    )
    def test_nemotronh_pretrain_recipes(
        self, config_func, recipe_name, parallelism_overrides, model_overrides, tmp_path
    ):
        """Functional test for Nemotron H recipes with appropriate parallelism configurations."""
        run_pretrain_recipe_test(
            config_func,
            recipe_name,
            tmp_path,
            model_overrides=model_overrides,
            **parallelism_overrides,
        )


class TestNemotronNanoV2Recipes:
    """Test class for Nemotron Nano v2 recipe functional tests."""

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "config_func,recipe_name,parallelism_overrides,model_overrides", NEMOTRON_NANO_V2_PRETRAIN_RECIPES
    )
    def test_nemotron_nano_v2_pretrain_recipes(
        self, config_func, recipe_name, parallelism_overrides, model_overrides, tmp_path
    ):
        """Functional test for Nemotron Nano v2 recipes with appropriate parallelism configurations."""
        run_pretrain_recipe_test(
            config_func,
            recipe_name,
            tmp_path,
            model_overrides=model_overrides,
            **parallelism_overrides,
        )
