# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compact compatibility proxy for the production DeepSeek V3 FSDP recipe.

The production recipe targets GB300. Functional CI currently exercises the
same NVL72 recipe path on a four-GPU GB200 runner, so only topology and model
size are reduced here. Performance features stay owned by ``perf_recipes``.
"""

import os

import pytest
import torch

from megatron.bridge.perf_recipes.deepseek import deepseek_v3_pretrain_64gpu_gb300_fp8mx_fsdp_config
from tests.functional_tests.test_groups.recipes.utils import (
    configure_ci_pretraining_dataset,
    run_perf_recipe_proxy_test,
)


def _deepseek_v3_fsdp_4gpu_compat_config():
    config = deepseek_v3_pretrain_64gpu_gb300_fp8mx_fsdp_config()

    # Compact-only overrides: retain the production FSDP, MXFP8, HybridEP,
    # activation-offload, and overlap configuration.
    config.model.num_layers = 2
    config.model.moe_layer_freq = [0, 1]
    config.model.pipeline_model_parallel_layout = None
    config.model.expert_model_parallel_size = 4

    assert config.ddp.use_megatron_fsdp is True
    assert config.ddp.data_parallel_sharding_strategy == "optim_grads_params"
    assert config.model.moe_token_dispatcher_type == "flex"
    assert config.model.moe_flex_dispatcher_backend == "hybridep"
    assert config.model.moe_router_force_load_balancing is False
    assert config.model.fine_grained_activation_offloading is True
    assert config.model.offload_modules == ["core_attn", "attn_proj"]
    assert config.mixed_precision.fp8_recipe == "mxfp8"
    assert config.model.fp8_param_gather is True
    assert config.comm_overlap.overlap_grad_reduce is True

    return config


class TestDeepSeekFSDPPerfProxy:
    """Exercise the canonical DeepSeek FSDP performance recipe on compact CI topology."""

    @pytest.mark.run_only_on("GPU")
    def test_gb300_recipe_on_gb200_compat_proxy(self, ensure_test_data):
        if torch.cuda.get_device_capability()[0] < 10:
            pytest.skip("The DeepSeek FSDP MXFP8 compatibility proxy requires Blackwell GPUs.")

        assert os.environ.get("NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN") == "4"
        assert os.environ.get("NVLINK_DOMAIN_SIZE") == "4"

        os.environ["NVTE_CPU_OFFLOAD_V1"] = "1"
        os.environ["NVTE_FWD_LAYERNORM_SM_MARGIN"] = "0"
        os.environ["NVTE_BWD_LAYERNORM_SM_MARGIN"] = "0"
        os.environ["NVTE_NORM_FWD_USE_CUDNN"] = "1"
        os.environ["NVTE_NORM_BWD_USE_CUDNN"] = "1"
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "32"
        os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
        os.environ["NCCL_ALGO"] = "Ring"

        def proxy_config():
            config = _deepseek_v3_fsdp_4gpu_compat_config()
            configure_ci_pretraining_dataset(config, ensure_test_data)
            return config

        run_perf_recipe_proxy_test(
            proxy_config,
            "deepseek_v3_gb300_fsdp_on_gb200_compat_proxy",
            config_overrides={
                "model": {"seq_length": 4096},
                "train": {"train_iters": 50, "global_batch_size": 4},
                "validation": {"eval_global_batch_size": 4, "eval_interval": 10, "eval_iters": 2},
                "dataset": {"seq_length": 4096},
            },
        )
