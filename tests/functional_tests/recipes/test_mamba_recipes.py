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

"""Functional smoke tests for Mamba recipe configurations."""

import pytest

from megatron.bridge.recipes.mamba.mamba2_130m import pretrain_config as mamba2_130m_config
from megatron.bridge.recipes.mamba.mamba2_370m import pretrain_config as mamba2_370m_config
from megatron.bridge.recipes.mamba.mamba2_780m import pretrain_config as mamba2_780m_config
from tests.functional_tests.recipes.utils import run_pretrain_recipe_test


MAMBA_PRETRAIN_RECIPES = [
    (mamba2_130m_config, "mamba2_130m"),
    (mamba2_370m_config, "mamba2_370m"),
    (mamba2_780m_config, "mamba2_780m"),
]


class TestMambaRecipes:
    """Test class for Mamba recipe smoke tests."""

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("config_func,recipe_name", MAMBA_PRETRAIN_RECIPES)
    def test_mamba_pretrain_recipes(self, config_func, recipe_name, tmp_path):
        """Functional test for Mamba recipes with default configurations."""
        run_pretrain_recipe_test(config_func, recipe_name, tmp_path)
