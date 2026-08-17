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

"""Regression tests for the DeepSeek V4 Flash performance recipes."""

import pytest
import torch

from megatron.bridge.perf_recipes.deepseek import (
    deepseek_v4_flash_pretrain_128gpu_gb200_fp8mx_config,
    deepseek_v4_flash_pretrain_128gpu_gb300_fp8mx_config,
)
from megatron.bridge.utils.cuda_graph import cuda_graph_module_names, is_full_iteration_cuda_graph
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _keep_recipe_construction_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


def test_deepseek_v4_flash_128gpu_gb200_fp8mx_config() -> None:
    cfg = deepseek_v4_flash_pretrain_128gpu_gb200_fp8mx_config()

    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.virtual_pipeline_model_parallel_size is None
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 64
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.sequence_parallel is False
    assert cfg.model.pipeline_model_parallel_layout is None
    assert cfg.train.global_batch_size == 2048
    assert cfg.train.micro_batch_size == 1

    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.moe_router_force_load_balancing is True
    assert cfg.model.moe_flex_dispatcher_num_sms == 32
    assert cfg.model.moe_hybridep_num_sms is None
    assert cfg.model.moe_hybridep_num_sms_preprocessing == 108
    assert cfg.model.recompute_granularity == "selective"
    assert cfg.model.recompute_modules == ["mla_up_proj"]

    assert is_full_iteration_cuda_graph(cfg.model)
    assert cuda_graph_module_names(cfg.model) == []
    assert cfg.model.cuda_graph_warmup_steps == 3
    assert cfg.model.use_te_rng_tracker is True
    assert cfg.rng.te_rng_tracker is True
    assert cfg.rerun_state_machine.check_for_nan_in_loss is False
    assert cfg.ddp.check_for_nan_in_grad is False

    assert cfg.model.moe_expert_rank_capacity_factor == 1.5
    assert cfg.model.moe_pad_experts_for_cuda_graph_inference is True
    assert cfg.model.moe_paged_stash is True
    assert cfg.model.moe_paged_stash_page_size == 64
    assert cfg.model.moe_paged_stash_buffer_size_factor_cpu == 0.0
    assert cfg.model.moe_paged_stash_buffer_size_factor_cuda == 1.2
    assert cfg.model.moe_mlp_glu_interleave_size == 32
    assert cfg.model.use_transformer_engine_op_fuser is True

    assert cfg.model.csa_compress_rotary_base == 40_000
    assert cfg.model.rotary_scaling_factor == 4
    assert cfg.model.apply_dsa_kernel_fusion is True
    assert cfg.model.dsa_indexer_loss_coeff == 0.01
    assert cfg.model.dsa_indexer_use_sparse_loss is True
    assert cfg.model.cross_entropy_fusion_impl == "native"
    assert cfg.model.quant_recipe is None
    assert cfg.model.moe_router_padding_for_fp8 is False
    assert cfg.model.moe_router_padding_for_quantization is True

    assert cfg.mixed_precision.fp8_param_gather is True
    assert cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag is True
    assert cfg.optimizer.main_grads_dtype == torch.float32
    assert cfg.optimizer.main_params_dtype == torch.float32
    assert cfg.optimizer.exp_avg_dtype == torch.bfloat16
    assert cfg.optimizer.exp_avg_sq_dtype == torch.bfloat16
    assert cfg.optimizer.optimizer_offload_fraction == 1.0
    assert cfg.optimizer.offload_optimizer_states is False
    assert cfg.ddp.grad_reduce_in_fp32 is False
    assert cfg.ddp.average_in_collective is False
    assert cfg.comm_overlap.overlap_moe_expert_parallel_comm is False
    assert cfg.comm_overlap.delay_wgrad_compute is False

    assert cfg.checkpoint.load_optim is False
    assert cfg.checkpoint.load_rng is False
    assert cfg.checkpoint.save_optim is False
    assert cfg.env_vars["PYTORCH_CUDA_ALLOC_CONF"] == (
        "expandable_segments:True,graph_capture_record_stream_reuse:True"
    )
    assert cfg.env_vars["NCCL_GRAPH_REGISTER"] == 0
    assert cfg.env_vars["TORCH_NCCL_AVOID_RECORD_STREAMS"] == 0
    assert cfg.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] == 32
    assert cfg.env_vars["NVTE_FWD_LAYERNORM_SM_MARGIN"] == 20
    assert cfg.env_vars["NVTE_BWD_LAYERNORM_SM_MARGIN"] == 20
    assert cfg.env_vars["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == 1
    assert cfg.env_vars["NVLINK_DOMAIN_SIZE"] == 72
    assert cfg.env_vars["USE_MNNVL"] == 1


def test_deepseek_v4_flash_128gpu_gb300_fp8mx_config() -> None:
    cfg = deepseek_v4_flash_pretrain_128gpu_gb300_fp8mx_config()

    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.virtual_pipeline_model_parallel_size is None
    assert cfg.model.expert_model_parallel_size == 64
    assert cfg.train.global_batch_size == 2048
    assert cfg.train.micro_batch_size == 1
    assert cfg.model.recompute_modules == ["mla_up_proj"]
    assert cfg.model.moe_flex_dispatcher_num_sms == 32
    assert cfg.model.moe_hybridep_num_sms is None
    assert is_full_iteration_cuda_graph(cfg.model)
