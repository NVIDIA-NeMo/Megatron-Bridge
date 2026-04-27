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

"""Functional perf smoke tests for DeepSeek V3 recipe configurations.

Test FSDP, EP=4, and MoE performance optimizations.
"""

import os

import pytest

from megatron.bridge.recipes.deepseek import (
    deepseek_v3_pretrain_config,
)
from tests.functional_tests.test_groups.recipes.utils import run_pretrain_recipe_perf_test


DEEPSEEK_PRETRAIN_PERF_RECIPES = [
    # (config_func, name, config_overrides)
    # GB200 proxy: FSDP + EP=4 + MoE perf optimizations (small model)
    # Mirrors config_GB300_proxy_1x4x4xfsdp4ep4.sh + config_common.sh + config_common_fsdp.sh
    (
        deepseek_v3_pretrain_config,
        "deepseek_v3_GB200_proxy",
        {
            "model": {
                "num_layers": 2,
                "moe_layer_freq": [0, 1],
                "num_moe_experts": 256,
                "pipeline_model_parallel_layout": None,
                "moe_router_topk": 8,
                "tensor_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "virtual_pipeline_model_parallel_size": None,
                "context_parallel_size": 1,
                "expert_model_parallel_size": 4,
                "sequence_parallel": False,
                "moe_token_dispatcher_type": "flex",
                "moe_flex_dispatcher_backend": "hybridep",
                "moe_permute_fusion": True,
                "moe_router_fusion": True,
                "moe_router_force_load_balancing": True,
                "moe_router_dtype": "bf16",
                "apply_rope_fusion": True,
                "recompute_granularity": "selective",
                "recompute_modules": ["layernorm", "mla_up_proj", "moe_act"],
                "fine_grained_activation_offloading": True,
                "offload_modules": ["core_attn", "attn_proj"],
                "gradient_accumulation_fusion": False,
                "init_model_with_meta_device": True,
                "fused_residual_rmsnorm": False,
                "moe_hybridep_num_sms": 32,
                "seq_length": 4096,
                "use_te_rng_tracker": True,
            },
            "ddp": {
                "use_megatron_fsdp": True,
                "use_distributed_optimizer": True,
                "data_parallel_sharding_strategy": "optim_grads_params",
                "num_distributed_optimizer_instances": 1,
                "outer_dp_sharding_strategy": "no_shard",
                "check_for_nan_in_grad": False,
                "grad_reduce_in_fp32": False,
                "average_in_collective": True,
            },
            "mixed_precision": {
                "grad_reduce_in_fp32": False,
                "fp8": "e4m3",
                "fp8_recipe": "mxfp8",
                "fp8_param_gather": True,
                "reuse_grad_buf_for_mxfp8_param_ag": False,
            },
            "comm_overlap": {
                "defer_embedding_wgrad_compute": False,
                "overlap_param_gather_with_optimizer_step": False,
                "overlap_grad_reduce": True,
                "overlap_param_gather": True,
            },
            "rerun_state_machine": {"check_for_nan_in_loss": False},
            "dist": {"enable_megatron_core_experimental": True},
            "checkpoint": {
                "save": None,
                "ckpt_format": "fsdp_dtensor",
            },
            "logger": {
                "save_config_filepath": "/workspace/llm/mbridge_test.yaml",
                "log_interval": 1,
            },
            "train": {
                "train_iters": 50,
                "global_batch_size": 4,
                "manual_gc_interval": 100,
            },
            "validation": {
                "eval_global_batch_size": 4,
                "eval_interval": 10,
                "eval_iters": 2,
            },
            "dataset": {
                "seq_length": 4096,
            },
        },
    ),
]


class TestDeepSeekRecipesPerf:
    """Test class for DeepSeek V3 perf recipe functional tests."""

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("config_func,recipe_name,config_overrides", DEEPSEEK_PRETRAIN_PERF_RECIPES)
    def test_deepseek_pretrain_recipes(self, config_func, recipe_name, config_overrides):
        """Functional test for DeepSeek V3 with GB200 proxy perf configurations."""
        os.environ["NVTE_CPU_OFFLOAD_V1"] = "1"
        os.environ["NVTE_FWD_LAYERNORM_SM_MARGIN"] = "0"
        os.environ["NVTE_BWD_LAYERNORM_SM_MARGIN"] = "0"
        os.environ["NVTE_NORM_FWD_USE_CUDNN"] = "1"
        os.environ["NVTE_NORM_BWD_USE_CUDNN"] = "1"
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "32"
        run_pretrain_recipe_perf_test(
            config_func,
            recipe_name,
            config_overrides=config_overrides,
        )
