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

"""Functional proxies for the Qwen3 30B-A3B performance recipes."""

import os
from collections.abc import Callable

import pytest
import torch

from megatron.bridge.perf_recipes.qwen import (
    qwen3_30b_a3b_pretrain_8gpu_gb200_fp8mx_config,
    qwen3_30b_a3b_pretrain_16gpu_h100_fp8cs_config,
)
from megatron.bridge.training.config import ConfigContainer
from tests.functional_tests.test_groups.recipes.utils import run_pretrain_recipe_perf_test


def _qwen3_moe_proxy(
    config_func: Callable[[], ConfigContainer],
    *,
    expert_model_parallel_size: int,
) -> ConfigContainer:
    """Reduce model depth while retaining the production MoE execution path."""
    config = config_func()
    config.model.num_layers = 2
    config.model.expert_model_parallel_size = expert_model_parallel_size
    config.tokenizer.tokenizer_type = "NullTokenizer"
    config.tokenizer.tokenizer_model = None
    config.tokenizer.vocab_size = config.model.vocab_size

    assert config.model.num_moe_experts == 128
    assert config.model.moe_router_topk == 8
    assert config.model.moe_grouped_gemm is True
    assert config.model.moe_token_dispatcher_type == "flex"
    assert config.model.moe_flex_dispatcher_backend == "hybridep"
    assert config.model.moe_router_force_load_balancing is False
    return config


class TestQwen3MoePerfProxy:
    """Train reduced production configs on matching eight-GPU runners."""

    @pytest.mark.run_only_on("GPU")
    def test_h100_fp8cs(self):
        assert torch.cuda.get_device_capability()[0] == 9, "The H100 proxy requires Hopper GPUs."

        def proxy_config() -> ConfigContainer:
            config = _qwen3_moe_proxy(
                qwen3_30b_a3b_pretrain_16gpu_h100_fp8cs_config,
                expert_model_parallel_size=8,
            )
            assert config.mixed_precision.fp8 is not None
            assert config.mixed_precision.fp8_recipe == "tensorwise"
            return config

        run_pretrain_recipe_perf_test(proxy_config, "qwen3_30b_a3b_h100_fp8cs_proxy")

    @pytest.mark.run_only_on("GPU")
    def test_gb200_fp8mx(self):
        assert torch.cuda.get_device_capability()[0] >= 10, "The GB200 MXFP8 proxy requires Blackwell GPUs."

        def proxy_config() -> ConfigContainer:
            assert os.environ.get("NVTE_CUTEDSL_FUSED_GROUPED_MLP") == "1"
            config = _qwen3_moe_proxy(
                qwen3_30b_a3b_pretrain_8gpu_gb200_fp8mx_config,
                expert_model_parallel_size=8,
            )
            assert config.mixed_precision.fp8_recipe == "mxfp8"
            assert config.mixed_precision.fp8_dot_product_attention is True
            assert config.model.use_transformer_engine_op_fuser is True
            assert config.model.moe_single_grouped_weight is False
            assert config.model.moe_mlp_glu_interleave_size == 32
            assert config.model.high_priority_a2a_comm_stream is True
            assert config.model.cuda_graph_impl == "transformer_engine"
            assert config.model.cuda_graph_scope == ["attn", "moe_router", "moe_preprocess"]
            assert config.model.moe_paged_stash is False
            assert config.model.moe_pad_experts_for_cuda_graph_inference is False
            assert config.model.moe_expert_rank_capacity_factor is None
            assert config.comm_overlap.overlap_moe_expert_parallel_comm is True
            assert config.comm_overlap.delay_wgrad_compute is True
            return config

        run_pretrain_recipe_perf_test(proxy_config, "qwen3_30b_a3b_gb200_fp8mx_proxy")
