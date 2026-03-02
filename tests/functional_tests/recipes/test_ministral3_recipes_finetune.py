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

from megatron.bridge.recipes.ministral3.ministral3 import (
    ministral3_3b_finetune_config,
)
from tests.functional_tests.recipes.utils import run_pretrain_vl_recipe_test


MINISTRAL3_FINETUNE_RECIPES = [
    # Small model, only use 2 layers for quick functional test
    (
        partial(ministral3_3b_finetune_config, peft=None),
        "ministral3_3b",
        {"tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1, "num_layers": 2},
    ),
]

MINISTRAL3_FINETUNE_PACKED_RECIPES = [
    # Small model with packed sequences, only use 2 layers
    (
        partial(ministral3_3b_finetune_config, peft=None),
        "ministral3_3b_packed",
        {"tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1, "num_layers": 2},
        {"pack_sequences_in_batch": True},
    ),
]


class TestMinistral3Recipes:
    """Test class for Ministral 3 recipe functional tests."""

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("config_func,recipe_name,model_overrides", MINISTRAL3_FINETUNE_RECIPES)
    def test_ministral3_finetune_recipes(self, config_func, recipe_name, model_overrides, tmp_path):
        """Functional test for Ministral 3 recipes with appropriate parallelism configurations."""
        try:
            from transformers import Ministral3ForCausalLM, Mistral3ForConditionalGeneration  # noqa: F401
            from transformers.models.mistral3.configuration_mistral3 import Mistral3Config  # noqa: F401
        except ImportError:
            pytest.skip("Ministral 3 not available in transformers")
        run_pretrain_vl_recipe_test(config_func, recipe_name, tmp_path, model_overrides=model_overrides)

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "config_func,recipe_name,model_overrides,dataset_overrides", MINISTRAL3_FINETUNE_PACKED_RECIPES
    )
    def test_ministral3_finetune_packed_recipes(
        self, config_func, recipe_name, model_overrides, dataset_overrides, tmp_path
    ):
        """Functional test for Ministral 3 recipes with packed sequences enabled."""
        try:
            from transformers import Ministral3ForCausalLM, Mistral3ForConditionalGeneration  # noqa: F401
            from transformers.models.mistral3.configuration_mistral3 import Mistral3Config  # noqa: F401
        except ImportError:
            pytest.skip("Ministral 3 not available in transformers")
        run_pretrain_vl_recipe_test(
            config_func,
            recipe_name,
            tmp_path,
            model_overrides=model_overrides,
            dataset_overrides=dataset_overrides,
        )
