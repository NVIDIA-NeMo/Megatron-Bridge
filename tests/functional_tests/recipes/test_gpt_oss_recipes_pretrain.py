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

"""Functional smoke tests for GPT-OSS recipe configurations."""

import pytest

from megatron.bridge.recipes.gpt_oss.gpt_oss import gpt_oss_20b_pretrain_config
from tests.functional_tests.recipes.utils import run_pretrain_recipe_test


GPT_OSS_PRETRAIN_RECIPES = [
    # (config_func, name, parallelism_overrides, model_overrides)
    (
        gpt_oss_20b_pretrain_config,
        "gpt_oss_20b",
        {"tensor_parallelism": 1, "pipeline_parallelism": 1, "expert_parallelism": 1},
        {"num_layers": 2},
    ),
]


class TestGPTOSSRecipes:
    """Test class for GPT-OSS recipe functional tests."""

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("config_func,recipe_name,parallelism_overrides,model_overrides", GPT_OSS_PRETRAIN_RECIPES)
    def test_gpt_oss_pretrain_recipes(
        self, config_func, recipe_name, parallelism_overrides, model_overrides, tmp_path
    ):
        """Functional test for GPT-OSS recipes with appropriate parallelism configurations."""
        run_pretrain_recipe_test(
            config_func,
            recipe_name,
            tmp_path,
            model_overrides=model_overrides,
            **parallelism_overrides,
        )
