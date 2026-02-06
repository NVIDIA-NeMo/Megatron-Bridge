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

"""Functional smoke tests for Gemma3 recipe configurations."""

import pytest

from megatron.bridge.recipes.gemma import (
    gemma3_1b_pretrain_config as gemma3_1b_config,
)
from tests.functional_tests.recipes.utils import run_pretrain_config_override_test, run_pretrain_recipe_test


GEMMA3_PRETRAIN_RECIPES = [
    # (config_func, name, parallelism_overrides)
    (gemma3_1b_config, "gemma3_1b", {}),  # Small model, use recipe defaults
]


class TestGemma3Recipes:
    """Test class for Gemma3 recipe functional tests."""

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("config_func,recipe_name,parallelism_overrides", GEMMA3_PRETRAIN_RECIPES)
    def test_gemma3_pretrain_recipes(self, config_func, recipe_name, parallelism_overrides, tmp_path):
        """Functional test for Gemma3 recipes with appropriate parallelism configurations."""
        run_pretrain_recipe_test(config_func, recipe_name, tmp_path, **parallelism_overrides)

    @pytest.mark.parametrize("config_func,recipe_name,parallelism_overrides", GEMMA3_PRETRAIN_RECIPES)
    def test_pretrain_config_override_after_instantiation(self, config_func, recipe_name, parallelism_overrides):
        """Functional test for overriding Gemma3 recipes from CLI"""
        run_pretrain_config_override_test(config_func)
