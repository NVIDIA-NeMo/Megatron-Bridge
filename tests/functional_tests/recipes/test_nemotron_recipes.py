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

"""Functional smoke tests for Nemotron recipe configurations."""

import pytest

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import (
    nemotron_3_nano_pretrain_config as nemotron_3_nano_config,
)
from tests.functional_tests.recipes.utils import run_pretrain_recipe_test


def patched_nemotron_3_nano_config(*args, **kwargs):
    """Wrapper function that patches the hidden size to 672 for testing."""
    # Call the original config function
    cfg = nemotron_3_nano_config(*args, **kwargs)
    
    # to fit the test environment with 2 GPUs:
    cfg.model.hidden_size = 672
    cfg.train.global_batch_size = 4
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.expert_model_parallel_size = 2
    
    return cfg


NEMOTRON_PRETRAIN_RECIPES = [
    (patched_nemotron_3_nano_config, "nemotron_3_nano"),
]


class TestNemotronRecipes:
    """Test class for Nemotron recipe smoke tests."""

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("config_func,recipe_name", NEMOTRON_PRETRAIN_RECIPES)
    def test_nemotron_pretrain_recipes(self, config_func, recipe_name, tmp_path):
        """Functional test for Nemotron recipes with default configurations."""
        run_pretrain_recipe_test(config_func, recipe_name, tmp_path)
