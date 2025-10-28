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

"""Functional smoke tests for Gemma3-VL recipe configurations."""

import pytest

from megatron.bridge.recipes.gemma3_vl.gemma3_vl import (
    gemma3_vl_4b_finetune_config,
)
from tests.functional_tests.recipes.utils import run_pretrain_vl_recipe_test


GEMMA3_VL_FINETUNE_RECIPES = [
    # Small model, only use 2 layers
    (
        gemma3_vl_4b_finetune_config,
        "gemma3_vl_4b",
        {"tensor_parallelism": 1, "pipeline_parallelism": 1, "num_layers": 2},
    ),
]


class TestGemma3VLRecipes:
    """Test class for Gemma3-VL recipe functional tests."""

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("config_func,recipe_name,parallelism_overrides", GEMMA3_VL_FINETUNE_RECIPES)
    def test_gemma3_vl_finetune_recipes(self, config_func, recipe_name, model_overrides, tmp_path):
        """Functional test for Gemma3-VL recipes with appropriate parallelism configurations."""
        run_pretrain_vl_recipe_test(config_func, recipe_name, tmp_path, model_overrides=model_overrides)
