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

"""Functional proxy for the GPT-OSS 120B GB200 MXFP8 performance recipe."""

import os

import pytest
import torch

from megatron.bridge.perf_recipes.gpt_oss import gpt_oss_120b_pretrain_64gpu_gb200_fp8mx_config
from megatron.bridge.training.config import ConfigContainer
from tests.functional_tests.test_groups.recipes.utils import run_pretrain_recipe_perf_test


def _gpt_oss_120b_gb200_proxy() -> ConfigContainer:
    """Reduce model depth while retaining the production 120B provider and EP path."""
    config = gpt_oss_120b_pretrain_64gpu_gb200_fp8mx_config()
    config.model.num_layers = 2
    config.model.tensor_model_parallel_size = 1
    config.model.pipeline_model_parallel_size = 1
    config.model.virtual_pipeline_model_parallel_size = None
    config.model.context_parallel_size = 1
    config.model.expert_model_parallel_size = 4
    config.tokenizer.tokenizer_type = "NullTokenizer"
    config.tokenizer.tokenizer_model = None
    config.tokenizer.vocab_size = config.model.vocab_size

    assert os.environ.get("NVTE_CUTEDSL_FUSED_GROUPED_MLP") == "1"
    assert config.model.num_moe_experts == 128
    assert config.model.moe_router_topk == 4
    assert config.model.window_attn_skip_freq == 2
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
    assert config.comm_overlap.overlap_moe_expert_parallel_comm is True
    assert config.comm_overlap.delay_wgrad_compute is True
    return config


class TestGPTOSS120BPerfProxy:
    """Train the reduced GPT-OSS 120B production config on four GB200 GPUs."""

    @pytest.mark.run_only_on("GPU")
    def test_gb200_fp8mx(self):
        assert torch.cuda.get_device_capability()[0] >= 10, "The GB200 MXFP8 proxy requires Blackwell GPUs."
        run_pretrain_recipe_perf_test(_gpt_oss_120b_gb200_proxy, "gpt_oss_120b_gb200_fp8mx_proxy")
