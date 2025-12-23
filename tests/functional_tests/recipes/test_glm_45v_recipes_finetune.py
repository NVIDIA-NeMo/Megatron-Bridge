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

"""Functional smoke tests for Ministral 3 recipe configurations."""

from functools import partial

import pytest

from megatron.bridge.recipes.glm_vl.glm_45v import (
    glm_45v_finetune_config,
)
from tests.functional_tests.recipes.utils import run_pretrain_vl_recipe_test


GLM_45V_FINETUNE_RECIPES = [
    # Small model, only use 2 layers for quick functional test
    (
        partial(glm_45v_finetune_config, peft=None),
        "glm_45v",
        {
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "num_layers": 2,
            "num_moe_experts": 8,
            "hidden_size": 4096,
            "ffn_hidden_size": 512,
            "moe_layer_freq": [0, 1],
            "pipeline_model_parallel_layout": None,
        },
    ),
]


class TestGLM45VRecipes:
    """Test class for GLM 4.5V recipe functional tests."""

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("config_func,recipe_name,model_overrides", GLM_45V_FINETUNE_RECIPES)
    def test_glm_45v_finetune_recipes(self, config_func, recipe_name, model_overrides, tmp_path):
        """Functional test for GLM 4.5V recipes with appropriate parallelism configurations."""
        run_pretrain_vl_recipe_test(config_func, recipe_name, tmp_path, model_overrides=model_overrides)
