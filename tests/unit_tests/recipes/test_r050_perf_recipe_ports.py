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

"""Regression tests for performance recipes manually ported from r0.5.0."""

import pytest
import torch

from megatron.bridge.perf_recipes.deepseek import (
    deepseek_v3_pretrain_256gpu_b300_fp8mx_config,
    deepseek_v3_pretrain_256gpu_gb200_fp8mx_large_scale_config,
    deepseek_v3_pretrain_256gpu_gb300_fp8mx_large_scale_config,
    deepseek_v4_pro_pretrain_256gpu_gb300_fp8mx_config,
)
from megatron.bridge.perf_recipes.qwen import (
    qwen3_30b_a3b_pretrain_8gpu_b200_fp8mx_config,
    qwen3_235b_a22b_pretrain_256gpu_b200_fp8mx_config,
    qwen3_235b_a22b_pretrain_256gpu_b200_fp8mx_large_scale_config,
    qwen3_235b_a22b_pretrain_256gpu_b300_fp8mx_config,
    qwen3_235b_a22b_pretrain_256gpu_b300_fp8mx_large_scale_config,
)
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _keep_recipe_construction_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


def _assert_full_iteration_hybridep_mxfp8(cfg, *, fp8_dot_product_attention: bool = True) -> None:
    """Check the normalized r0.5.0 full-iteration HybridEP settings."""
    assert cfg.model.cuda_graph_impl == "full_iteration"
    assert cfg.model.cuda_graph_scope == []
    assert cfg.model.use_te_rng_tracker is True
    assert cfg.rng.te_rng_tracker is True
    assert cfg.model.offload_modules == []
    assert cfg.model.moe_pad_experts_for_cuda_graph_inference is True
    assert cfg.model.moe_paged_stash is True
    assert cfg.model.moe_expert_rank_capacity_factor == 1.5
    assert cfg.model.moe_paged_stash_buffer_size_factor_cuda == 1.2
    assert cfg.model.moe_paged_stash_buffer_size_factor_cpu == 1.0
    assert cfg.model.moe_shared_expert_overlap is False
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_hybridep_num_sms == 32
    assert cfg.model.high_priority_a2a_comm_stream is True
    assert cfg.model.use_transformer_engine_op_fuser is True
    assert cfg.model.moe_mlp_glu_interleave_size == 32
    assert cfg.model.moe_hybridep_num_sms_preprocessing == 32
    assert cfg.mixed_precision.fp8_dot_product_attention is fp8_dot_product_attention
    assert cfg.comm_overlap.overlap_moe_expert_parallel_comm is True
    assert cfg.comm_overlap.delay_wgrad_compute is True


@pytest.mark.parametrize(
    ("recipe", "expected_vpp"),
    [
        (qwen3_30b_a3b_pretrain_8gpu_b200_fp8mx_config, None),
        (qwen3_235b_a22b_pretrain_256gpu_b200_fp8mx_config, 3),
        (qwen3_235b_a22b_pretrain_256gpu_b300_fp8mx_config, 3),
    ],
)
def test_qwen_mxfp8_recipes_match_r050_full_iteration_settings(recipe, expected_vpp) -> None:
    cfg = recipe()

    _assert_full_iteration_hybridep_mxfp8(cfg)
    assert cfg.model.virtual_pipeline_model_parallel_size == expected_vpp
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.env_vars["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == 1
    assert cfg.env_vars["TORCH_NCCL_AVOID_RECORD_STREAMS"] == 0
    assert "graph_capture_record_stream_reuse:True" in cfg.env_vars["PYTORCH_CUDA_ALLOC_CONF"]


@pytest.mark.parametrize(
    "recipe",
    [
        qwen3_235b_a22b_pretrain_256gpu_b200_fp8mx_large_scale_config,
        qwen3_235b_a22b_pretrain_256gpu_b300_fp8mx_large_scale_config,
    ],
)
def test_qwen_large_scale_recipes_only_override_global_batch_size(recipe) -> None:
    cfg = recipe()

    _assert_full_iteration_hybridep_mxfp8(cfg)
    assert cfg.train.global_batch_size == 512
    assert cfg.model.virtual_pipeline_model_parallel_size == 3
    assert cfg.env_vars["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == 1


def test_deepseek_v3_gb200_large_scale_matches_r050_fp8mx_base() -> None:
    cfg = deepseek_v3_pretrain_256gpu_gb200_fp8mx_large_scale_config()

    _assert_full_iteration_hybridep_mxfp8(cfg, fp8_dot_product_attention=False)
    assert cfg.train.global_batch_size == 256
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.virtual_pipeline_model_parallel_size == 4
    assert cfg.model.expert_model_parallel_size == 64
    assert cfg.model.recompute_modules == ["mla_up_proj"]
    assert cfg.model.cuda_graph_impl == "full_iteration"
    assert cfg.model.fp8_output_proj is True
    assert cfg.mixed_precision.fp8_dot_product_attention is False
    assert cfg.env_vars["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == 1


def test_deepseek_v3_gb300_large_scale_matches_final_r050_config() -> None:
    cfg = deepseek_v3_pretrain_256gpu_gb300_fp8mx_large_scale_config()

    _assert_full_iteration_hybridep_mxfp8(cfg)
    assert cfg.train.global_batch_size == 256
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.virtual_pipeline_model_parallel_size == 4
    assert cfg.model.expert_model_parallel_size == 64
    assert cfg.model.pipeline_model_parallel_layout == "Et*4|(t*4|)*14tmL"
    assert cfg.model.recompute_modules == []
    assert cfg.model.cuda_graph_impl == "full_iteration"
    assert cfg.model.fp8_output_proj is True
    assert cfg.mixed_precision.fp8_dot_product_attention is True
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 64


def test_deepseek_v3_b300_mxfp8_preserves_r050_hybridep_settings() -> None:
    cfg = deepseek_v3_pretrain_256gpu_b300_fp8mx_config()

    _assert_full_iteration_hybridep_mxfp8(cfg)


def test_deepseek_v4_pro_gb300_matches_r050_performance_config() -> None:
    cfg = deepseek_v4_pro_pretrain_256gpu_gb300_fp8mx_config()

    assert cfg.train.global_batch_size == 4096
    assert cfg.train.micro_batch_size == 1
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.virtual_pipeline_model_parallel_size == 4
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 64
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_layout == "Et*4|(tttt|)*14tmL"
    assert cfg.model.recompute_modules == ["mla_up_proj", "mhc"]
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_shared_expert_overlap is False

    assert cfg.model.cuda_graph_impl == "full_iteration"
    assert cfg.model.cuda_graph_scope == []
    assert cfg.model.moe_pad_experts_for_cuda_graph_inference is True
    assert cfg.model.moe_paged_stash is True
    assert cfg.model.moe_expert_rank_capacity_factor == 1.5
    assert cfg.model.moe_paged_stash_buffer_size_factor_cuda == 1.2
    assert cfg.model.moe_paged_stash_buffer_size_factor_cpu == 0.0

    assert cfg.model.apply_dsa_kernel_fusion is True
    assert cfg.model.use_transformer_engine_op_fuser is True
    assert cfg.model.cross_entropy_fusion_impl == "native"
    assert cfg.model.moe_mlp_glu_interleave_size == 32
    assert cfg.model.quant_recipe is None
    assert cfg.mixed_precision.fp8_param_gather is True
    assert cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag is True
    assert cfg.model.dsa_indexer_loss_coeff == 0.01
    assert cfg.model.dsa_indexer_use_sparse_loss is True
    assert cfg.optimizer.main_grads_dtype == torch.bfloat16
    assert cfg.mixed_precision.grad_reduce_in_fp32 is False
    assert cfg.ddp.grad_reduce_in_fp32 is False

    assert cfg.model.fine_grained_activation_offloading is True
    assert cfg.model.offload_modules == ["core_attn", "attn_proj"]
    assert cfg.model.fine_grained_offloading_max_inflight_offloads == 2
    assert cfg.comm_overlap.overlap_grad_reduce is True
    assert cfg.comm_overlap.overlap_moe_expert_parallel_comm is False
    assert cfg.comm_overlap.delay_wgrad_compute is False
