# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Functional smoke tests for Qwen3.5-VL pretrain mock recipes.

Verifies that pretrain configs using MockVLMConversationProvider can:
1. Build a valid config and instantiate the model
2. Run a short training loop without errors
3. Save checkpoints correctly

Run with:
    uv run torchrun --nproc_per_node=2 -m pytest \
        tests/functional_tests/test_groups/recipes/test_qwen35_vl_recipes_pretrain.py -v
"""

import pytest

from megatron.bridge.recipes.qwen_vl.qwen35_vl import qwen35_vl_9b_pretrain_mock_config
from tests.functional_tests.test_groups.recipes.utils import run_pretrain_vl_recipe_test


pytestmark = pytest.mark.integration

_TP2_PP1 = {"tensor_model_parallel_size": 2, "pipeline_model_parallel_size": 1}
_TINY_MODEL = {"num_layers": 4}

QWEN35_VL_PRETRAIN_RECIPES = [
    (
        qwen35_vl_9b_pretrain_mock_config,
        "qwen35_vl_9b_pretrain_mock",
        _TP2_PP1,
        _TINY_MODEL,
    ),
    (
        qwen35_vl_9b_pretrain_mock_config,
        "qwen35_vl_9b_pretrain_mock_unfrozen_proj",
        _TP2_PP1,
        {
            **_TINY_MODEL,
            "freeze_language_model": True,
            "freeze_vision_model": True,
            "freeze_vision_projection": False,
        },
    ),
]


class TestQwen35VLPretrainRecipes:
    """Functional tests for Qwen3.5-VL pretrain mock recipes."""

    @pytest.fixture(autouse=True)
    def _reset_microbatch_calculator(self):
        """Ensure the global microbatch calculator is cleared between tests."""
        from megatron.core.num_microbatches_calculator import (
            _GLOBAL_NUM_MICROBATCHES_CALCULATOR,
            destroy_num_microbatches_calculator,
        )

        if _GLOBAL_NUM_MICROBATCHES_CALCULATOR is not None:
            destroy_num_microbatches_calculator()

        yield

        if _GLOBAL_NUM_MICROBATCHES_CALCULATOR is not None:
            destroy_num_microbatches_calculator()

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "config_func,recipe_name,parallelism_overrides,model_overrides",
        QWEN35_VL_PRETRAIN_RECIPES,
    )
    def test_qwen35_vl_pretrain_mock(self, config_func, recipe_name, parallelism_overrides, model_overrides, tmp_path):
        """Pretrain mock recipe: verify training loop completes and checkpoints are saved."""
        run_pretrain_vl_recipe_test(
            config_func,
            recipe_name,
            tmp_path,
            model_overrides=model_overrides,
            **parallelism_overrides,
        )
