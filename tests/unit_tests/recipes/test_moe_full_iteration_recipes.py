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

"""Configuration checks for MoE full-iteration performance recipes."""

from collections.abc import Callable

import pytest

from megatron.bridge.perf_recipes.deepseek import deepseek_v3_pretrain_256gpu_vr200_nvfp4_config
from megatron.bridge.perf_recipes.nemotronh import (
    nemotron_3_super_pretrain_64gpu_gb200_fp8mx_config,
    nemotron_3_super_pretrain_64gpu_gb200_nvfp4_config,
    nemotron_3_super_pretrain_64gpu_gb300_fp8mx_config,
    nemotron_3_super_pretrain_64gpu_gb300_nvfp4_config,
    nemotron_3_super_pretrain_64gpu_vr200_fp8mx_config,
    nemotron_3_super_pretrain_64gpu_vr200_nvfp4_config,
)
from megatron.bridge.perf_recipes.qwen import qwen3_235b_a22b_pretrain_256gpu_vr200_nvfp4_config
from megatron.bridge.training.config import ConfigContainer
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _keep_recipe_construction_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


def _assert_full_iteration_hybridep(cfg: ConfigContainer) -> None:
    assert cfg.model.cuda_graph_impl == "full_iteration"
    assert cfg.model.cuda_graph_scope == []
    assert cfg.model.use_te_rng_tracker is True
    assert cfg.rng.te_rng_tracker is True
    assert cfg.ddp.check_for_nan_in_grad is False
    assert cfg.rerun_state_machine.check_for_nan_in_loss is False

    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_shared_expert_overlap is False
    assert cfg.model.moe_pad_experts_for_cuda_graph_inference is True
    assert cfg.model.moe_paged_stash is True
    assert cfg.model.moe_expert_rank_capacity_factor == 1.5
    assert cfg.model.moe_paged_stash_buffer_size_factor_cuda == 1.2
    assert cfg.model.moe_paged_stash_buffer_size_factor_cpu == 1.0
    assert cfg.model.use_transformer_engine_op_fuser is True
    assert cfg.model.moe_mlp_glu_interleave_size == 32
    assert cfg.model.moe_hybridep_num_sms == 32
    assert cfg.model.moe_hybridep_num_sms_preprocessing == 32

    assert cfg.env_vars["NCCL_GRAPH_REGISTER"] == 0
    assert cfg.env_vars["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == 1
    assert "graph_capture_record_stream_reuse:True" in cfg.env_vars["PYTORCH_CUDA_ALLOC_CONF"]


def test_qwen3_235b_vr200_nvfp4_uses_full_iteration_stack() -> None:
    cfg = qwen3_235b_a22b_pretrain_256gpu_vr200_nvfp4_config()

    _assert_full_iteration_hybridep(cfg)
    assert cfg.mixed_precision.fp4 == "e2m1"
    assert cfg.mixed_precision.fp4_param_gather is True
    assert cfg.mixed_precision.fp8_dot_product_attention is False
    assert cfg.model.high_priority_a2a_comm_stream is False
    assert cfg.comm_overlap is not None
    assert cfg.comm_overlap.tp_comm_overlap is False
    assert cfg.comm_overlap.overlap_moe_expert_parallel_comm is True
    assert cfg.comm_overlap.delay_wgrad_compute is True
    assert cfg.env_vars["TORCH_NCCL_AVOID_RECORD_STREAMS"] == 0
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 32
    assert (
        cfg.model.tensor_model_parallel_size,
        cfg.model.pipeline_model_parallel_size,
        cfg.model.virtual_pipeline_model_parallel_size,
        cfg.model.expert_model_parallel_size,
    ) == (1, 4, 12, 32)


def test_deepseek_v3_256gpu_vr200_nvfp4_uses_full_iteration_stack() -> None:
    cfg = deepseek_v3_pretrain_256gpu_vr200_nvfp4_config()

    _assert_full_iteration_hybridep(cfg)
    assert cfg.mixed_precision.fp4 == "e2m1"
    assert cfg.model.fp8_output_proj is False
    assert cfg.mixed_precision.fp8_dot_product_attention is True
    assert cfg.model.mla_down_proj_fusion is True
    assert cfg.model.high_priority_a2a_comm_stream is True
    assert cfg.comm_overlap is not None
    assert cfg.comm_overlap.overlap_moe_expert_parallel_comm is True
    assert cfg.comm_overlap.delay_wgrad_compute is True
    assert cfg.model.recompute_modules == []
    assert cfg.env_vars["TORCH_NCCL_AVOID_RECORD_STREAMS"] == 1
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 32
    assert cfg.env_vars["NVTE_DPA_FP8_RECIPE"] == "MXFP8BlockScaling"
    assert cfg.env_vars["NVTE_DPA_FP8_FORMAT"] == "E4M3"
    assert (
        cfg.model.tensor_model_parallel_size,
        cfg.model.pipeline_model_parallel_size,
        cfg.model.virtual_pipeline_model_parallel_size,
        cfg.model.expert_model_parallel_size,
    ) == (1, 2, 8, 32)


@pytest.mark.parametrize(
    ("recipe", "expected_tp", "expected_ep", "expected_micro_batch_size"),
    [
        (nemotron_3_super_pretrain_64gpu_gb200_fp8mx_config, 2, 64, 2),
        (nemotron_3_super_pretrain_64gpu_gb300_fp8mx_config, 1, 32, 1),
        (nemotron_3_super_pretrain_64gpu_vr200_fp8mx_config, 1, 32, 1),
    ],
)
def test_nemotron_3_super_mxfp8_uses_full_iteration_stack(
    recipe: Callable[[], ConfigContainer], expected_tp: int, expected_ep: int, expected_micro_batch_size: int
) -> None:
    cfg = recipe()

    _assert_full_iteration_hybridep(cfg)
    assert cfg.mixed_precision.fp8 == "e4m3"
    assert cfg.mixed_precision.fp8_recipe == "mxfp8"
    assert cfg.mixed_precision.fp8_param_gather is True
    assert cfg.mixed_precision.fp8_dot_product_attention is True
    assert cfg.model.moe_router_padding_for_quantization is True
    assert cfg.model.high_priority_a2a_comm_stream is False
    assert cfg.model.recompute_granularity is None
    assert cfg.model.recompute_modules is None
    assert cfg.model.offload_modules == []
    assert cfg.model.mtp_num_layers == 2
    assert cfg.model.overlap_moe_expert_parallel_comm is False
    assert cfg.model.delay_wgrad_compute is False
    assert cfg.comm_overlap is None
    assert cfg.env_vars["TORCH_NCCL_AVOID_RECORD_STREAMS"] == 0
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == expected_ep
    assert cfg.train.micro_batch_size == expected_micro_batch_size
    assert (
        cfg.model.tensor_model_parallel_size,
        cfg.model.pipeline_model_parallel_size,
        cfg.model.virtual_pipeline_model_parallel_size,
        cfg.model.expert_model_parallel_size,
    ) == (expected_tp, 1, None, expected_ep)


@pytest.mark.parametrize(
    ("recipe", "expected_tp", "expected_ep", "expected_micro_batch_size"),
    [
        (nemotron_3_super_pretrain_64gpu_gb200_nvfp4_config, 2, 64, 2),
        (nemotron_3_super_pretrain_64gpu_gb300_nvfp4_config, 1, 32, 1),
        (nemotron_3_super_pretrain_64gpu_vr200_nvfp4_config, 1, 32, 1),
    ],
)
def test_nemotron_3_super_nvfp4_uses_full_iteration_stack(
    recipe: Callable[[], ConfigContainer], expected_tp: int, expected_ep: int, expected_micro_batch_size: int
) -> None:
    cfg = recipe()

    _assert_full_iteration_hybridep(cfg)
    assert cfg.mixed_precision.fp4 == "e2m1"
    assert cfg.mixed_precision.fp4_param_gather is False
    assert cfg.mixed_precision.fp8_dot_product_attention is False
    assert cfg.model.quant_recipe is not None
    assert cfg.model.moe_router_padding_for_quantization is True
    assert cfg.model.high_priority_a2a_comm_stream is False
    assert cfg.model.recompute_granularity is None
    assert cfg.model.recompute_modules is None
    assert cfg.model.offload_modules == []
    assert cfg.model.mtp_num_layers == 2
    assert cfg.model.overlap_moe_expert_parallel_comm is False
    assert cfg.model.delay_wgrad_compute is False
    assert cfg.comm_overlap is None
    assert cfg.env_vars["TORCH_NCCL_AVOID_RECORD_STREAMS"] == 0
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == expected_ep
    assert cfg.train.micro_batch_size == expected_micro_batch_size
    assert cfg.env_vars["NVTE_USE_FAST_MATH"] == 1
    assert (
        cfg.model.tensor_model_parallel_size,
        cfg.model.pipeline_model_parallel_size,
        cfg.model.virtual_pipeline_model_parallel_size,
        cfg.model.expert_model_parallel_size,
    ) == (expected_tp, 1, None, expected_ep)
