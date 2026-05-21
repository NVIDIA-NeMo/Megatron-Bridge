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
import json

import pytest
import torch

from megatron.bridge.recipes.deepseek import (
    deepseek_v2_lite_pretrain_config as deepseek_v2_lite_config,
)
from megatron.bridge.recipes.deepseek import (
    deepseek_v4_flash_mtp_proxy_pretrain_config as deepseek_v4_mtp_proxy_config,
)
from megatron.bridge.recipes.deepseek import (
    deepseek_v4_flash_proxy_pretrain_config as deepseek_v4_proxy_config,
)
from megatron.bridge.recipes.deepseek import (
    deepseek_v4_tiny_pretrain_config as deepseek_v4_tiny_config,
)
from tests.functional_tests.test_groups.recipes.utils import run_pretrain_recipe_test


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


def _has_fast_hadamard_transform() -> bool:
    return importlib.util.find_spec("fast_hadamard_transform") is not None


@pytest.fixture()
def deepseek_v4_toy_hf_path(tmp_path):
    """Local DSv4 HF config for self-contained recipe functional smoke tests."""
    model_dir = tmp_path / "deepseek_v4_toy_hf"
    model_dir.mkdir()
    config = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "compress_ratios": [0, 4, 128, 4],
        "first_k_dense_replace": 0,
        "head_dim": 16,
        "hidden_act": "silu",
        "hidden_dropout": 0.0,
        "hidden_size": 512,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "index_head_dim": 128,
        "index_n_heads": 64,
        "index_topk": 512,
        "initializer_range": 0.006,
        "intermediate_size": 2048,
        "max_position_embeddings": 1024,
        "mlp_bias": False,
        "moe_intermediate_size": 512,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "norm_topk_prob": True,
        "num_attention_heads": 8,
        "num_experts_per_tok": 1,
        "num_hash_layers": 1,
        "num_hidden_layers": 4,
        "num_key_value_heads": 1,
        "num_nextn_predict_layers": 1,
        "o_groups": 8,
        "o_lora_rank": 192,
        "q_lora_rank": 192,
        "qk_rope_head_dim": 8,
        "rms_norm_eps": 1e-6,
        "rope_scaling": {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 1,
            "original_max_position_embeddings": 1024,
            "type": "yarn",
        },
        "rope_theta": 10000.0,
        "routed_scaling_factor": 1.0,
        "scoring_func": "sqrtsoftplus",
        "sliding_window": 128,
        "swiglu_limit": 10.0,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "use_qk_norm": True,
        "vocab_size": 32000,
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return model_dir


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
    def test_deepseek_v4_tiny_pretrain_recipe(self, deepseek_v4_toy_hf_path, tmp_path):
        """Functional smoke for DSv4 hybrid attention without mHC/MoE/MTP."""
        run_pretrain_recipe_test(
            lambda: deepseek_v4_tiny_config(hf_path=str(deepseek_v4_toy_hf_path)),
            "deepseek_v4_tiny",
            tmp_path,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            checkpoint_overrides={"save": None, "load": None},
        )

    @pytest.mark.run_only_on("GPU")
    def test_deepseek_v4_proxy_pretrain_recipe(self, deepseek_v4_toy_hf_path, tmp_path):
        """Functional smoke for the DSv4 proxy recipe that exercises mHC and hash MoE."""
        if torch.cuda.device_count() < 2:
            pytest.skip("DeepSeek-V4 proxy recipe uses PP=2 and requires two visible GPUs.")
        if not _has_fast_hadamard_transform():
            pytest.skip("DeepSeek-V4 proxy recipe requires fast_hadamard_transform for the DSA path.")

        run_pretrain_recipe_test(
            lambda: deepseek_v4_proxy_config(hf_path=str(deepseek_v4_toy_hf_path)),
            "deepseek_v4_proxy",
            tmp_path,
            checkpoint_overrides={"save": None, "load": None},
        )

    @pytest.mark.run_only_on("GPU")
    def test_deepseek_v4_mtp_proxy_pretrain_recipe(self, deepseek_v4_toy_hf_path, tmp_path):
        """Functional smoke for the DSv4 proxy recipe that exercises mHC and MTP."""
        if torch.cuda.device_count() < 2:
            pytest.skip("DeepSeek-V4 MTP proxy recipe uses PP=2 and requires two visible GPUs.")
        if not _has_fast_hadamard_transform():
            pytest.skip("DeepSeek-V4 MTP proxy recipe requires fast_hadamard_transform for the DSA path.")

        run_pretrain_recipe_test(
            lambda: deepseek_v4_mtp_proxy_config(hf_path=str(deepseek_v4_toy_hf_path)),
            "deepseek_v4_mtp_proxy",
            tmp_path,
            checkpoint_overrides={"save": None, "load": None},
        )
