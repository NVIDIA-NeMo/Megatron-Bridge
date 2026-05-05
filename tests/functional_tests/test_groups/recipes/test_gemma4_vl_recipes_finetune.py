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

"""Functional smoke tests for Gemma 4 VL recipe configurations.

Uses a minimal 2-layer / 4-expert model to exercise the full provider stack
(Gemma4TransformerLayer, Gemma4SelfAttention, Gemma4RotaryEmbedding, MoE, etc.)
without requiring the full 26B-A4B weights.

``interleaved_attn_pattern=(1, 1)`` ensures both a sliding-window layer (layer 1)
and a global-attention layer (layer 2) are included in the 2-layer run, so the
K=V tying path in Gemma4SelfAttention is exercised.
"""

import pytest

from megatron.bridge.recipes.gemma4_vl.gemma4_vl import (
    gemma4_vl_26b_peft_config,
    gemma4_vl_26b_sft_config,
)
from tests.functional_tests.test_groups.recipes.utils import run_pretrain_vl_recipe_test


# Shared model overrides: trim the 26B-A4B architecture down to fit on 2 GPUs.
# - num_layers=2:              keep only 2 transformer layers
# - num_moe_experts=4:         reduce from 128 → 4 (critical for memory)
# - interleaved_attn_pattern:  (1,1) → layer 1 sliding, layer 2 global;
#                              ensures Gemma4SelfAttention global path is covered
# - expert/tensor/pipeline:    all-1 for single-node simplicity
_SMALL_MODEL_OVERRIDES = {
    "tensor_model_parallel_size": 1,
    "pipeline_model_parallel_size": 1,
    "expert_model_parallel_size": 1,
    "num_layers": 2,
    "num_moe_experts": 4,
    "moe_router_topk": 2,
    "interleaved_attn_pattern": (1, 1),
}

GEMMA4_VL_FINETUNE_RECIPES = [
    (
        gemma4_vl_26b_sft_config,
        "gemma4_vl_26b_sft",
        _SMALL_MODEL_OVERRIDES,
    ),
]

GEMMA4_VL_PEFT_RECIPES = [
    (
        gemma4_vl_26b_peft_config,
        "gemma4_vl_26b_peft_lora",
        _SMALL_MODEL_OVERRIDES,
    ),
]


class TestGemma4VLRecipes:
    """Functional smoke tests for Gemma 4 VL recipe configurations."""

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("config_func,recipe_name,model_overrides", GEMMA4_VL_FINETUNE_RECIPES)
    def test_gemma4_vl_sft_recipes(self, config_func, recipe_name, model_overrides, tmp_path):
        """Smoke test for Gemma4 VL SFT recipe: 2 layers, 4 experts, sliding+global attention."""
        run_pretrain_vl_recipe_test(config_func, recipe_name, tmp_path, model_overrides=model_overrides)
