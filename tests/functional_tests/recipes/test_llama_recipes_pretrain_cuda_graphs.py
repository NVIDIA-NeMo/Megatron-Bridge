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

"""Functional smoke tests for LLaMA recipe configurations."""

import pytest

from megatron.bridge.recipes.llama import (
    llama32_1b_pretrain_config as llama32_1b_config,
)
from tests.functional_tests.recipes.utils import run_pretrain_recipe_perf_test


LLAMA_PRETRAIN_RECIPES = [
    # (config_func, name, config_overrides)
    (
        llama32_1b_config,
        "llama32_1b",
        {
            "model": {
                "num_layers": 2,
                "cuda_graph_impl": "local",
                "cuda_graph_scope": ["full_iteration"],
                "check_for_nan_in_grad": False,
                "use_te_rng_tracker": True,
            },
            "rerun_state_machine": {"check_for_nan_in_loss": False},
            "ddp": {"check_for_nan_in_grad": False},
        },
    ),
    (
        llama32_1b_config,
        "llama32_1b",
        {
            "model": {
                "num_layers": 2,
                "cuda_graph_impl": "transformer_engine",
                "cuda_graph_scope": ["attn"],
                "check_for_nan_in_grad": False,
                "use_te_rng_tracker": True,
            },
            "rerun_state_machine": {"check_for_nan_in_loss": False},
            "ddp": {"check_for_nan_in_grad": False},
        },
    ),
]


class TestLlamaCudaGraphRecipes:
    """Test class for LLaMA recipe functional tests."""

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("config_func,recipe_name,config_overrides", LLAMA_PRETRAIN_RECIPES)
    def test_llama_pretrain_recipes(self, config_func, recipe_name, config_overrides):
        """Functional test for LLaMA recipes with appropriate parallelism configurations."""
        run_pretrain_recipe_perf_test(
            config_func,
            recipe_name,
            config_overrides=config_overrides,
        )
