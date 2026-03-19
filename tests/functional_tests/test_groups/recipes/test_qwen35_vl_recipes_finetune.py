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

"""Functional smoke tests for Qwen3.5-VL finetuning recipes.

Covers three training scenarios:
1. SFT with nothing frozen (all modules trainable)
2. SFT with language model frozen (train vision + projection)
3. SFT with vision + language frozen (train projection only)
4. SFT with activation recomputation

Run with:
    uv run torchrun --nproc_per_node=2 -m pytest tests/functional_tests/recipes/test_qwen35_vl_recipes_finetune.py -v
"""

import pytest

from megatron.bridge.recipes.qwen_vl.qwen35_vl import qwen35_vl_27b_sft_config
from tests.functional_tests.recipes.utils import run_pretrain_vl_recipe_test


pytestmark = pytest.mark.integration


_TP2_PP1 = {"tensor_model_parallel_size": 2, "pipeline_model_parallel_size": 1}
_TINY_MODEL = {"num_layers": 4}


# ---------------------------------------------------------------------------
# Scenario 1: SFT — nothing frozen
# ---------------------------------------------------------------------------

QWEN35_VL_SFT_NONE_FROZEN = [
    (
        qwen35_vl_27b_sft_config,
        "qwen35_vl_27b_sft_none_frozen",
        _TP2_PP1,
        {
            **_TINY_MODEL,
            "freeze_language_model": False,
            "freeze_vision_model": False,
            "freeze_vision_projection": False,
        },
    ),
]

# ---------------------------------------------------------------------------
# Scenario 2: SFT — language model frozen
# ---------------------------------------------------------------------------

QWEN35_VL_SFT_LM_FROZEN = [
    (
        qwen35_vl_27b_sft_config,
        "qwen35_vl_27b_sft_lm_frozen",
        _TP2_PP1,
        {
            **_TINY_MODEL,
            "freeze_language_model": True,
            "freeze_vision_model": False,
            "freeze_vision_projection": False,
        },
    ),
]

# ---------------------------------------------------------------------------
# Scenario 3: SFT — vision + language frozen (train projection only)
# ---------------------------------------------------------------------------

QWEN35_VL_SFT_PROJ_ONLY = [
    (
        qwen35_vl_27b_sft_config,
        "qwen35_vl_27b_sft_projection_only",
        _TP2_PP1,
        {
            **_TINY_MODEL,
            "freeze_language_model": True,
            "freeze_vision_model": True,
            "freeze_vision_projection": False,
        },
    ),
]

# ---------------------------------------------------------------------------
# Scenario 4: SFT — activation recomputation
# ---------------------------------------------------------------------------

QWEN35_VL_SFT_RECOMPUTE = [
    (
        qwen35_vl_27b_sft_config,
        "qwen35_vl_27b_sft_recompute",
        _TP2_PP1,
        {
            **_TINY_MODEL,
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
        },
    ),
]


class TestQwen35VLFinetuneRecipes:
    """Functional tests covering SFT freeze combos and recompute."""

    @pytest.fixture(autouse=True)
    def _reset_microbatch_calculator(self):
        """Ensure the global microbatch calculator is cleared between tests.

        If a previous test fails mid-pretrain, destroy_global_state() never
        runs and the calculator leaks into the next test.
        """
        from megatron.core.num_microbatches_calculator import (
            _GLOBAL_NUM_MICROBATCHES_CALCULATOR,
            destroy_num_microbatches_calculator,
        )

        if _GLOBAL_NUM_MICROBATCHES_CALCULATOR is not None:
            destroy_num_microbatches_calculator()

        yield

        if _GLOBAL_NUM_MICROBATCHES_CALCULATOR is not None:
            destroy_num_microbatches_calculator()

    # -----------------------------------------------------------------------
    # SFT scenarios
    # -----------------------------------------------------------------------

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "config_func,recipe_name,parallelism_overrides,model_overrides",
        QWEN35_VL_SFT_NONE_FROZEN,
    )
    def test_sft_nothing_frozen(self, config_func, recipe_name, parallelism_overrides, model_overrides, tmp_path):
        """Scenario 1: all modules trainable."""
        run_pretrain_vl_recipe_test(
            config_func,
            recipe_name,
            tmp_path,
            model_overrides=model_overrides,
            **parallelism_overrides,
        )

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "config_func,recipe_name,parallelism_overrides,model_overrides",
        QWEN35_VL_SFT_LM_FROZEN,
    )
    def test_sft_language_model_frozen(
        self, config_func, recipe_name, parallelism_overrides, model_overrides, tmp_path
    ):
        """Scenario 2: language model frozen, train vision + projection."""
        run_pretrain_vl_recipe_test(
            config_func,
            recipe_name,
            tmp_path,
            model_overrides=model_overrides,
            **parallelism_overrides,
        )

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "config_func,recipe_name,parallelism_overrides,model_overrides",
        QWEN35_VL_SFT_PROJ_ONLY,
    )
    def test_sft_vision_and_language_frozen(
        self, config_func, recipe_name, parallelism_overrides, model_overrides, tmp_path
    ):
        """Scenario 3: vision + language frozen, train projection only."""
        run_pretrain_vl_recipe_test(
            config_func,
            recipe_name,
            tmp_path,
            model_overrides=model_overrides,
            **parallelism_overrides,
        )

    # -----------------------------------------------------------------------
    # Recompute
    # -----------------------------------------------------------------------

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "config_func,recipe_name,parallelism_overrides,model_overrides",
        QWEN35_VL_SFT_RECOMPUTE,
    )
    def test_recompute(self, config_func, recipe_name, parallelism_overrides, model_overrides, tmp_path):
        """SFT with activation recomputation."""
        run_pretrain_vl_recipe_test(
            config_func,
            recipe_name,
            tmp_path,
            model_overrides=model_overrides,
            **parallelism_overrides,
        )
