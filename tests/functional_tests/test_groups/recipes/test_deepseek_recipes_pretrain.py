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

"""Functional smoke tests for DeepSeek recipe configurations."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch

from megatron.bridge.recipes.deepseek import (
    deepseek_v2_lite_pretrain_config as deepseek_v2_lite_config,
)
from megatron.bridge.recipes.deepseek import (
    deepseek_v4_flash_pretrain_config,
    deepseek_v4_flash_pretrain_muon_config,
    deepseek_v4_flash_pretrain_mxfp8_config,
)
from megatron.bridge.recipes.deepseek.h100 import deepseek_v4 as deepseek_v4_h100_module
from tests.functional_tests.test_groups.recipes.utils import run_pretrain_recipe_test


DEEPSEEK_V4_TEST_MODEL_ENV = "DEEPSEEK_V4_TOY_HF_PATH"
DEEPSEEK_V4_TEST_MODEL_PATH = Path("/home/TestData/megatron_bridge/models/deepseek_v4_toy")


def _has_dsv4_in_mcore() -> bool:
    try:
        if not all(
            importlib.util.find_spec(mod) is not None
            for mod in (
                "megatron.core.transformer.hyper_connection",
                "megatron.core.transformer.experimental_attention_variant.csa",
                "megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention",
            )
        ):
            return False
        # The DSv4 HybridModel stack spec ships in the "Enable DeepSeek-v4 hybrid_model"
        # megatron-core series; older cores have the attention modules above but not this,
        # so the DSv4 hybrid model cannot be built. Skip rather than fail on such cores.
        from megatron.core.models.hybrid import hybrid_layer_specs

        return hasattr(hybrid_layer_specs, "hybrid_dsv4_stack_spec")
    except (ImportError, ValueError):
        return False


def _deepseek_v4_toy_model_path() -> str:
    model_path = Path(os.environ.get(DEEPSEEK_V4_TEST_MODEL_ENV, DEEPSEEK_V4_TEST_MODEL_PATH))
    if not model_path.exists():
        pytest.skip(
            f"DeepSeek-V4 toy HF model not found at {model_path}. "
            f"Set {DEEPSEEK_V4_TEST_MODEL_ENV} or upload the synthetic model to CI test data."
        )
    return str(model_path)


DEEPSEEK_PRETRAIN_RECIPES = [
    # (config_func, name, parallelism_overrides, model_overrides)
    (
        deepseek_v2_lite_config,
        "deepseek_v2_lite",
        {"tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1, "expert_model_parallel_size": 1},
        {"num_layers": 2, "num_moe_experts": 8, "moe_router_topk": 1, "moe_layer_freq": [0, 1]},
    ),
    # (
    #     deepseek_v3_config,
    #     "deepseek_v3",
    #     {"tensor_model_parallel_size": 2, "pipeline_model_parallel_size": 1, "expert_model_parallel_size": 1},
    #     {
    #         "num_layers": 2,
    #         "num_moe_experts": 8,
    #         "moe_router_topk": 1,
    #         "moe_layer_freq": [0, 1],
    #         "pipeline_model_parallel_layout": [["embedding"] + ["decoder"] * 2 + ["mtp", "loss"]],
    #     },
    # ),
]

# On HybridModel, DeepSeek-V4's layer count, per-layer compression ratios, and MoE
# placement are all derived from the toy HF config via the hybrid layer pattern, so we
# must NOT override num_layers / csa_compress_ratios / moe_layer_freq here (they would
# conflict with the pattern-derived values and trip finalize()'s consistency check).
# We only disable the Blackwell-only fused kernels and any explicit pipeline layout so
# the smoke test runs unfused on a single rank.
DEEPSEEK_V4_MODEL_OVERRIDES = {
    "mtp_num_layers": None,
    "pipeline_model_parallel_layout": None,
    "use_fused_mhc": False,
    "apply_rope_fusion": False,
    "dsa_indexer_loss_coeff": 0.0,
    "dsa_indexer_use_sparse_loss": False,
    "recompute_granularity": None,
    "recompute_modules": None,
}

DEEPSEEK_V4_PRETRAIN_RECIPES = [
    # (config_func, name, requires_blackwell, checkpoint_overrides)
    (
        deepseek_v4_flash_pretrain_config,
        "deepseek_v4_flash_adam_bf16",
        False,
        None,
    ),
    (
        deepseek_v4_flash_pretrain_muon_config,
        "deepseek_v4_flash_muon_bf16",
        False,
        None,
    ),
    (
        deepseek_v4_flash_pretrain_mxfp8_config,
        "deepseek_v4_flash_adam_mxfp8",
        True,
        {"save_optim": False, "load_optim": False},
    ),
]


class TestDeepSeekRecipes:
    """Test class for DeepSeek recipe functional tests."""

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "config_func,recipe_name,parallelism_overrides,model_overrides", DEEPSEEK_PRETRAIN_RECIPES
    )
    def test_deepseek_pretrain_recipes(
        self, config_func, recipe_name, parallelism_overrides, model_overrides, tmp_path
    ):
        """Functional test for DeepSeek recipes with appropriate parallelism configurations."""
        run_pretrain_recipe_test(
            config_func,
            recipe_name,
            tmp_path,
            model_overrides=model_overrides,
            **parallelism_overrides,
        )

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.skipif(not _has_dsv4_in_mcore(), reason="megatron-core does not yet ship DSv4 prerequisites.")
    @pytest.mark.parametrize(
        "config_func,recipe_name,requires_blackwell,checkpoint_overrides", DEEPSEEK_V4_PRETRAIN_RECIPES
    )
    def test_deepseek_v4_pretrain_recipes(
        self,
        config_func,
        recipe_name,
        requires_blackwell,
        checkpoint_overrides,
        tmp_path,
    ):
        """Functional test for DeepSeek-V4 Flash pretraining recipes."""
        if requires_blackwell and torch.cuda.get_device_capability()[0] < 10:
            pytest.skip("DeepSeek-V4 MXFP8 recipe requires Blackwell GPUs.")

        hf_path = _deepseek_v4_toy_model_path()

        def recipe_with_test_model():
            with pytest.MonkeyPatch.context() as monkeypatch:
                monkeypatch.setattr(deepseek_v4_h100_module, "DEEPSEEK_V4_FLASH_HF_PATH", hf_path)
                return config_func()

        run_pretrain_recipe_test(
            recipe_with_test_model,
            recipe_name,
            tmp_path,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            model_overrides=DEEPSEEK_V4_MODEL_OVERRIDES,
            checkpoint_overrides=checkpoint_overrides,
        )
