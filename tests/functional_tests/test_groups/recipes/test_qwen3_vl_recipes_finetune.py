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

"""Functional smoke tests for Qwen3-VL finetuning recipes.

This test ensures that:
1. Qwen3-VL model forward pass works with all required parameters (including loss_mask)
2. Training loop completes without errors
3. Checkpoints are saved correctly

This catches regressions like missing parameters in the forward pass signature.

Run with:
    uv run torchrun --nproc_per_node=2 -m pytest tests/functional_tests/recipes/test_qwen3_vl_recipes_finetune.py -v
"""

import pytest

from megatron.bridge.models.qwen_vl.qwen3_vl_step import forward_step as qwen3_vl_forward_step
from megatron.bridge.recipes.qwen_vl.qwen3_vl import qwen3_vl_8b_sft_config
from tests.functional_tests.test_groups.recipes.utils import run_pretrain_vl_recipe_test


# Variants that route through the Qwen3-VL-specific forward step (``qwen3_vl_step``).
# Covers the step function used by the examples / ``run_recipe.py --step_func qwen3_vl_step``,
# which is otherwise only exercised by the DistTrain smoke test that requires 8 GPUs.
# Qwen3-VL is incompatible with the generic vlm_step because preprocess_packed_seqs
# inside Qwen3VLModel.forward always pads to tp_size alignment, which the generic
# vlm_step packing does not match. Use qwen3_vl_step for all Qwen3-VL training.
QWEN3_VL_FINETUNE_RECIPES = [
    # (config_func, recipe_name, parallelism_overrides, model_overrides)
    # Qwen3-VL 8B finetune - uses TP=2 for 2-GPU CI
    # Note: deepstack_visual_indexes must have len <= num_layers
    (
        qwen3_vl_8b_sft_config,
        "qwen3_vl_8b_sft_qwen3_vl_step",
        {"tensor_model_parallel_size": 2, "pipeline_model_parallel_size": 1},
        {"num_layers": 4, "deepstack_visual_indexes": [0, 1, 2]},
    ),
    (
        qwen3_vl_8b_sft_config,
        "qwen3_vl_8b_sft_qwen3_vl_step",
        {
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 1,
        },
        {
            "freeze_language_model": False,
            "freeze_vision_model": False,
            "freeze_vision_projection": False,
            "num_layers": 4,
            "deepstack_visual_indexes": [0, 1, 2],
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
        },
    ),
    (
        qwen3_vl_8b_sft_config,
        "qwen3_vl_8b_sft_qwen3_vl_step",
        {
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 1,
        },
        {
            "num_layers": 4,
            "deepstack_visual_indexes": [0, 1, 2],
        },
    ),
]


class TestQwen3VLFinetuneRecipes:
    """Test class for Qwen3-VL finetune recipe functional tests."""

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "config_func,recipe_name,parallelism_overrides,model_overrides",
        QWEN3_VL_FINETUNE_RECIPES,
    )
    def test_qwen3_vl_finetune_recipes(
        self,
        config_func,
        recipe_name,
        parallelism_overrides,
        model_overrides,
        tmp_path,
    ):
        """Functional test for Qwen3-VL finetune recipes using ``qwen3_vl_step.forward_step``.

        Exercises the Qwen3-VL-specific forward/step path (not the generic VLM step),
        which handles Qwen3-VL visual inputs, deepstack indexing, and batch padding.
        Qwen3-VL must use qwen3_vl_step — the generic vlm_step is incompatible because
        preprocess_packed_seqs inside Qwen3VLModel always pads to tp_size alignment.
        """
        run_pretrain_vl_recipe_test(
            config_func,
            recipe_name,
            tmp_path,
            model_overrides=model_overrides,
            forward_step_func=qwen3_vl_forward_step,
            **parallelism_overrides,
        )
