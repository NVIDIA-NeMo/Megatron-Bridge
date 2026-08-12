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

"""Tests for Qwen3.5-VL performance workload presets."""

import sys
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from megatron.bridge.perf_recipes.qwen_vl.gb200.qwen35_vl import (
    qwen35_vl_35b_a3b_pretrain_8gpu_gb200_bf16_config,
    qwen35_vl_35b_a3b_pretrain_8gpu_gb200_fp8cs_config,
    qwen35_vl_35b_a3b_pretrain_8gpu_gb200_fp8mx_config,
)
from megatron.bridge.perf_recipes.qwen_vl.h100.qwen35_vl import (
    qwen35_vl_35b_a3b_pretrain_16gpu_h100_bf16_config,
    qwen35_vl_122b_a10b_pretrain_128gpu_h100_bf16_config,
    qwen35_vl_122b_a10b_pretrain_128gpu_h100_fp8cs_config,
)
from megatron.bridge.utils.cuda_graph import cuda_graph_module_names
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit

_PERF_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts" / "performance"
if str(_PERF_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_SCRIPTS_DIR))


@pytest.mark.parametrize(
    ("recipe_fn", "expected_pp_size", "expected_vp_size"),
    [
        (
            qwen35_vl_122b_a10b_pretrain_128gpu_h100_bf16_config,
            8,
            2,
        ),
        (
            qwen35_vl_122b_a10b_pretrain_128gpu_h100_fp8cs_config,
            8,
            2,
        ),
    ],
)
def test_qwen35_vl_122b_h100_pipeline_layout(
    recipe_fn: Callable,
    expected_pp_size: int,
    expected_vp_size: int,
) -> None:
    num_layers = 48
    config = recipe_fn()
    pp_size = config.model.pipeline_model_parallel_size
    vp_size = config.model.virtual_pipeline_model_parallel_size

    assert num_layers % pp_size == 0
    assert vp_size is not None
    assert (num_layers // pp_size) % vp_size == 0
    assert pp_size == expected_pp_size
    assert vp_size == expected_vp_size


def test_qwen35_vl_35b_h100_measured_performance_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """The H100 factory should retain the measured 185-TFLOP execution policy."""
    patch_recipe_construction_dependencies(monkeypatch)

    config = qwen35_vl_35b_a3b_pretrain_16gpu_h100_bf16_config()

    assert config.model.tensor_model_parallel_size == 1
    assert config.model.pipeline_model_parallel_size == 2
    assert config.model.context_parallel_size == 1
    assert config.model.virtual_pipeline_model_parallel_size is None
    assert config.model.num_layers_in_first_pipeline_stage == 17
    assert config.model.num_layers_in_last_pipeline_stage == 23
    assert config.model.expert_model_parallel_size == 8
    assert config.model.expert_tensor_parallel_size == 1
    assert config.model.sequence_parallel is False
    assert config.train.micro_batch_size == 1
    assert config.train.global_batch_size == 512

    assert config.model.moe_token_dispatcher_type == "flex"
    assert config.model.moe_flex_dispatcher_backend == "hybridep"
    assert config.model.moe_flex_dispatcher_num_sms == 16
    assert config.model.moe_hybridep_num_sms is None
    assert config.model.moe_hybridep_num_sms_preprocessing == 16
    assert config.model.moe_permute_fusion is True
    assert config.model.moe_permute_fusion_into_hybridep is True
    assert config.model.moe_router_force_load_balancing is True
    assert config.model.moe_shared_expert_overlap is False
    assert config.model.overlap_dispatch_backward_with_experts_wgrad is True

    assert config.model.recompute_granularity is None
    assert config.model.recompute_method is None
    assert config.model.recompute_num_layers is None
    assert config.model.recompute_modules == []
    assert config.model.cuda_graph_impl == "transformer_engine"
    assert cuda_graph_module_names(config.model) == ["attn", "moe_router", "moe_preprocess"]
    assert config.model.vision_cuda_graph_impl == "transformer_engine"
    assert config.model.vision_cuda_graph_scope == ["attn", "mlp"]
    assert config.model.max_vision_cuda_graph_seq_length == 784
    assert config.model.use_te_rng_tracker is True
    assert config.rng.te_rng_tracker is True

    assert config.optimizer.use_precision_aware_optimizer is True
    assert config.optimizer.main_params_dtype == torch.float32
    assert config.optimizer.main_grads_dtype == torch.float32
    assert config.optimizer.exp_avg_dtype == torch.bfloat16
    assert config.optimizer.exp_avg_sq_dtype == torch.bfloat16
    assert config.optimizer.overlap_param_gather is False
    assert config.optimizer.overlap_param_gather_with_optimizer_step is False
    assert config.ddp.overlap_grad_reduce is False
    assert config.ddp.overlap_param_gather is False
    assert config.comm_overlap.tp_comm_overlap is False
    assert config.comm_overlap.overlap_grad_reduce is False
    assert config.comm_overlap.overlap_param_gather is False
    assert config.comm_overlap.overlap_param_gather_with_optimizer_step is False
    assert config.comm_overlap.overlap_moe_expert_parallel_comm is False
    assert config.comm_overlap.delay_wgrad_compute is False
    assert config.model.batch_p2p_sync is False

    assert config.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] == 32
    assert config.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 8
    assert config.env_vars["NUM_OF_TOKENS_PER_CHUNK_COMBINE_API"] == 64
    assert config.env_vars["NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API"] == 64
    assert config.env_vars["NUM_OF_TOKENS_PER_CHUNK_PREPROCESSING_API"] == 64
    assert config.env_vars["NVLINK_DOMAIN_SIZE"] == 8
    assert config.env_vars["USE_MNNVL"] == 0
    assert config.env_vars["NVTE_BWD_LAYERNORM_SM_MARGIN"] == 0
    assert config.env_vars["NVTE_FWD_LAYERNORM_SM_MARGIN"] == 0
    assert config.env_vars["NVTE_NORM_BWD_USE_CUDNN"] == 1
    assert config.env_vars["NVTE_NORM_FWD_USE_CUDNN"] == 1


@pytest.mark.parametrize(
    ("recipe_fn", "expected_micro_batch_size", "expected_graph_modules"),
    [
        (
            qwen35_vl_35b_a3b_pretrain_8gpu_gb200_bf16_config,
            2,
            [],
        ),
        (
            qwen35_vl_35b_a3b_pretrain_8gpu_gb200_fp8cs_config,
            3,
            [],
        ),
        (
            qwen35_vl_35b_a3b_pretrain_8gpu_gb200_fp8mx_config,
            3,
            ["moe_router", "moe_preprocess"],
        ),
    ],
)
def test_qwen35_vl_35b_gb200_measured_performance_defaults(
    recipe_fn: Callable,
    expected_micro_batch_size: int,
    expected_graph_modules: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real GB200 factories should retain their precision-specific defaults."""
    patch_recipe_construction_dependencies(monkeypatch)

    config = recipe_fn()

    assert config.train.micro_batch_size == expected_micro_batch_size
    assert config.train.global_batch_size == 480
    assert config.model.moe_router_force_load_balancing is True
    assert config.model.moe_flex_dispatcher_backend == "hybridep"
    expected_graph_impl = "transformer_engine" if expected_graph_modules else "none"
    assert config.model.cuda_graph_impl == expected_graph_impl
    assert config.model.cuda_graph_scope is None
    assert bool(config.model.cuda_graph_modules) is bool(expected_graph_modules)
    assert cuda_graph_module_names(config.model) == expected_graph_modules
    assert config.env_vars["NVTE_NORM_BWD_USE_CUDNN"] == 1
    assert config.env_vars["NVTE_NORM_FWD_USE_CUDNN"] == 1
