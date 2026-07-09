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

"""Functional proxy for the DeepSeek V3 GB200 MXFP8 performance recipe."""

import os

import pytest
import torch

from megatron.bridge.perf_recipes.deepseek import deepseek_v3_pretrain_256gpu_gb200_fp8mx_config
from megatron.bridge.training.config import ConfigContainer
from tests.functional_tests.test_groups.recipes.utils import (
    configure_ci_pretraining_dataset,
    run_perf_recipe_proxy_test,
)


def _deepseek_v3_gb200_proxy() -> ConfigContainer:
    """Reduce model scale while preserving PP, MTP, and the production MoE path."""
    config = deepseek_v3_pretrain_256gpu_gb200_fp8mx_config()

    config.model.num_layers = 4
    config.model.moe_layer_freq = [0, 1, 1, 1]
    config.model.num_moe_experts = 16
    config.model.tensor_model_parallel_size = 1
    config.model.pipeline_model_parallel_size = 2
    config.model.virtual_pipeline_model_parallel_size = 2
    config.model.context_parallel_size = 1
    config.model.expert_model_parallel_size = 2
    config.model.pipeline_model_parallel_layout = [
        ["embedding", "decoder"],
        ["decoder"],
        ["decoder"],
        ["decoder", "mtp", "loss"],
    ]
    config.tokenizer.tokenizer_type = "NullTokenizer"
    config.tokenizer.tokenizer_model = None
    config.tokenizer.vocab_size = config.model.vocab_size

    assert os.environ.get("NVTE_CUTEDSL_FUSED_GROUPED_MLP") == "1"
    assert config.model.mtp_num_layers == 1
    assert config.model.moe_router_topk == 8
    assert config.model.moe_grouped_gemm is True
    assert config.model.moe_token_dispatcher_type == "flex"
    assert config.model.moe_flex_dispatcher_backend == "hybridep"
    assert config.model.moe_router_force_load_balancing is False
    assert config.mixed_precision.fp8_recipe == "mxfp8"
    assert config.model.use_transformer_engine_op_fuser is True
    assert config.model.moe_single_grouped_weight is False
    assert config.model.moe_mlp_glu_interleave_size == 32
    assert config.model.high_priority_a2a_comm_stream is True
    assert config.model.cuda_graph_impl == "full_iteration"
    assert config.model.cuda_graph_scope == []
    assert config.model.moe_paged_stash is True
    assert config.model.moe_pad_experts_for_cuda_graph_inference is True
    assert config.model.moe_expert_rank_capacity_factor == 1.5
    assert config.model.fp8_output_proj is True
    assert config.comm_overlap.overlap_moe_expert_parallel_comm is True
    assert config.comm_overlap.delay_wgrad_compute is True
    return config


class TestDeepSeekV3PerfProxy:
    """Train the reduced DeepSeek V3 production config on four GB200 GPUs."""

    @pytest.mark.run_only_on("GPU")
    def test_gb200_fp8mx(self, ensure_test_data):
        assert torch.cuda.get_device_capability()[0] >= 10, "The GB200 MXFP8 proxy requires Blackwell GPUs."

        def proxy_config() -> ConfigContainer:
            config = _deepseek_v3_gb200_proxy()
            configure_ci_pretraining_dataset(config, ensure_test_data)
            return config

        run_perf_recipe_proxy_test(proxy_config, "deepseek_v3_gb200_fp8mx_proxy")
