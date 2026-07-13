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

"""Functional smoke tests for Nemotron Nano V2 VL recipe configurations."""

import pytest

from megatron.bridge.recipes.nemotron_vl.nemotron_nano_v2_vl import (
    nemotron_nano_v2_vl_12b_sft_config,
)
from megatron.bridge.training import llava_step
from tests.functional_tests.test_groups.recipes.utils import run_pretrain_vl_recipe_test


NEMOTRON_VL_FINETUNE_RECIPES = [
    # Small model, only use 2 layers
    (
        nemotron_nano_v2_vl_12b_sft_config,
        "nemotron_vl_nano_v2_sft",
        {
            "hybrid_layer_pattern": "M*-",
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
        },
    ),
]


class TestNemotronVLRecipes:
    """Test class for Nemotron VL recipe functional tests."""

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("config_func,recipe_name,model_overrides", NEMOTRON_VL_FINETUNE_RECIPES)
    def test_nemotron_vl_finetune_recipes(self, config_func, recipe_name, model_overrides, tmp_path):
        """Functional test for Nemotron VL recipes with minimal parallelism."""
        run_pretrain_vl_recipe_test(
            config_func,
            recipe_name,
            tmp_path,
            model_overrides=model_overrides,
            forward_step_func=llava_step.forward_step,
        )
