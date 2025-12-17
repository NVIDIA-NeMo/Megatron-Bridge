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

import functools

import pytest

from megatron.bridge.recipes.nemotron_vl import nemotron_nano_v2_vl as nemotron_recipe
from megatron.bridge.training import llava_step
from tests.functional_tests.recipes.utils import run_pretrain_vl_recipe_test


def _finetune_wrapper(**kwargs):
    """Wrapper to adapt Nemotron VL finetune_config to the test runner signature.

    The runner will pass (dir, name, dataset_type=mock) among others; we forward
    everything to finetune_config and inject a dummy pretrained_checkpoint.
    """
    kwargs.setdefault("pretrained_checkpoint", "/tmp/fake_nemotron_vl_ckpt")
    return nemotron_recipe.nemotron_nano_v2_vl_12b_finetune_config(**kwargs)


NEMOTRON_VL_FINETUNE_RECIPES = [
    # Small model, only use 2 layers
    (
        functools.partial(_finetune_wrapper, hf_model_path="nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"),
        "nemotron_vl_nano_v2",
        {
            "num_layers": 3,
            "hybrid_override_pattern": "M*-",
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
