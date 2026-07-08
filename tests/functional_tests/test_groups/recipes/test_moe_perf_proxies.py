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

"""Functional proxies for production MoE performance recipes.

These proxies cover the model families that failed together in nemo-ci issue
#4094.  They start from the exact production performance constructors and only
reduce topology or model depth enough to fit one eight-GPU runner.  Assertions
pin the MXFP8, natural-routing, HybridEP, TE op-fuser, and CUDA-graph behavior
that exposed the historical grouped-MLP failure.
"""

import os
from collections.abc import Callable

import pytest
import torch

from megatron.bridge.perf_recipes.deepseek import deepseek_v3_pretrain_256gpu_gb200_fp8mx_config
from megatron.bridge.perf_recipes.gpt_oss import gpt_oss_120b_pretrain_64gpu_gb200_fp8mx_config
from megatron.bridge.perf_recipes.qwen import (
    qwen3_30b_a3b_pretrain_8gpu_gb200_fp8mx_config,
    qwen3_30b_a3b_pretrain_16gpu_h100_fp8cs_config,
)
from megatron.bridge.training.config import ConfigContainer
from tests.functional_tests.test_groups.recipes.utils import run_pretrain_recipe_perf_test


def _use_null_tokenizer(config: ConfigContainer) -> None:
    """Keep mock-data proxies independent of external tokenizer downloads."""
    config.tokenizer.tokenizer_type = "NullTokenizer"
    config.tokenizer.tokenizer_model = None
    config.tokenizer.vocab_size = config.model.vocab_size


def _assert_hybridep_natural_routing(config: ConfigContainer) -> None:
    """Guard the shared dispatch path implicated in the historical failures."""
    assert config.model.moe_grouped_gemm is True
    assert config.model.moe_token_dispatcher_type == "flex"
    assert config.model.moe_flex_dispatcher_backend == "hybridep"
    assert config.model.moe_router_force_load_balancing is False


def _assert_blackwell_mxfp8_path(config: ConfigContainer) -> None:
    """Guard the exact Blackwell TE grouped-MLP path fixed by the paired PR."""
    assert os.environ.get("NVTE_CUTEDSL_FUSED_GROUPED_MLP") == "1"
    assert config.mixed_precision.fp8_recipe == "mxfp8"
    assert config.model.use_transformer_engine_op_fuser is True
    assert config.model.moe_single_grouped_weight is False
    assert config.model.moe_mlp_glu_interleave_size == 32
    assert config.model.high_priority_a2a_comm_stream is True
    assert config.comm_overlap.overlap_moe_expert_parallel_comm is True
    assert config.comm_overlap.delay_wgrad_compute is True


def _qwen3_moe_proxy(
    config_func: Callable[[], ConfigContainer],
    *,
    expert_model_parallel_size: int,
) -> ConfigContainer:
    config = config_func()
    config.model.num_layers = 2
    config.model.expert_model_parallel_size = expert_model_parallel_size
    _use_null_tokenizer(config)

    assert config.model.num_moe_experts == 128
    assert config.model.moe_router_topk == 8
    _assert_hybridep_natural_routing(config)
    return config


def _deepseek_v3_gb200_proxy() -> ConfigContainer:
    """Shrink DeepSeek V3 while preserving PP, MTP, and the production MoE path."""
    config = deepseek_v3_pretrain_256gpu_gb200_fp8mx_config()

    # Four decoder layers allow every PP rank to own useful work while keeping
    # MTP and loss colocated on the final stage, as in the production layout.
    config.model.num_layers = 4
    config.model.moe_layer_freq = [0, 1, 1, 1]
    config.model.num_moe_experts = 16
    config.model.tensor_model_parallel_size = 1
    config.model.pipeline_model_parallel_size = 4
    config.model.virtual_pipeline_model_parallel_size = None
    config.model.context_parallel_size = 1
    config.model.expert_model_parallel_size = 2
    config.model.pipeline_model_parallel_layout = [
        ["embedding", "decoder"],
        ["decoder"],
        ["decoder"],
        ["decoder", "mtp", "loss"],
    ]
    _use_null_tokenizer(config)

    assert config.model.mtp_num_layers == 1
    assert config.model.moe_router_topk == 8
    assert config.model.cuda_graph_impl == "full_iteration"
    assert config.model.cuda_graph_scope == []
    assert config.model.moe_paged_stash is True
    assert config.model.moe_pad_experts_for_cuda_graph_inference is True
    assert config.model.moe_expert_rank_capacity_factor == 1.5
    assert config.model.fp8_output_proj is True
    _assert_hybridep_natural_routing(config)
    _assert_blackwell_mxfp8_path(config)
    return config


def _gpt_oss_120b_gb200_proxy() -> ConfigContainer:
    """Shrink GPT-OSS depth while preserving its 120B provider and EP path."""
    config = gpt_oss_120b_pretrain_64gpu_gb200_fp8mx_config()
    config.model.num_layers = 2
    config.model.tensor_model_parallel_size = 1
    config.model.pipeline_model_parallel_size = 1
    config.model.virtual_pipeline_model_parallel_size = None
    config.model.context_parallel_size = 1
    config.model.expert_model_parallel_size = 8
    _use_null_tokenizer(config)

    assert config.model.num_moe_experts == 128
    assert config.model.moe_router_topk == 4
    assert config.model.window_attn_skip_freq == 2
    assert config.model.cuda_graph_impl == "full_iteration"
    assert config.model.cuda_graph_scope == []
    assert config.model.moe_paged_stash is True
    assert config.model.moe_pad_experts_for_cuda_graph_inference is True
    assert config.model.moe_expert_rank_capacity_factor == 1.5
    _assert_hybridep_natural_routing(config)
    _assert_blackwell_mxfp8_path(config)
    return config


class TestQwen3MoePerfProxy:
    """Train reduced Qwen3 production configs on matching GPU runners."""

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
            config = _qwen3_moe_proxy(
                qwen3_30b_a3b_pretrain_8gpu_gb200_fp8mx_config,
                expert_model_parallel_size=8,
            )
            assert config.mixed_precision.fp8_dot_product_attention is True
            assert config.model.cuda_graph_impl == "transformer_engine"
            assert config.model.cuda_graph_scope == ["attn", "moe_router", "moe_preprocess"]
            assert config.model.moe_paged_stash is False
            assert config.model.moe_pad_experts_for_cuda_graph_inference is False
            assert config.model.moe_expert_rank_capacity_factor is None
            _assert_blackwell_mxfp8_path(config)
            return config

        run_pretrain_recipe_perf_test(proxy_config, "qwen3_30b_a3b_gb200_fp8mx_proxy")


class TestAdditionalMoePerfProxies:
    """Cover the DeepSeek V3 and GPT-OSS 120B failures missed by Qwen alone."""

    @pytest.mark.run_only_on("GPU")
    def test_deepseek_v3_gb200_fp8mx(self):
        assert torch.cuda.get_device_capability()[0] >= 10, "The GB200 MXFP8 proxy requires Blackwell GPUs."
        run_pretrain_recipe_perf_test(_deepseek_v3_gb200_proxy, "deepseek_v3_gb200_fp8mx_proxy")

    @pytest.mark.run_only_on("GPU")
    def test_gpt_oss_120b_gb200_fp8mx(self):
        assert torch.cuda.get_device_capability()[0] >= 10, "The GB200 MXFP8 proxy requires Blackwell GPUs."
        run_pretrain_recipe_perf_test(_gpt_oss_120b_gb200_proxy, "gpt_oss_120b_gb200_fp8mx_proxy")
