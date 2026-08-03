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

"""Regression tests for the DeepSeek V4 Flash GB300 performance recipes."""

import pytest
import torch

from megatron.bridge.perf_recipes.deepseek import (
    deepseek_v4_flash_pretrain_128gpu_gb300_fp8mx_config,
    deepseek_v4_flash_pretrain_128gpu_gb300_fp8mx_mbs2_offload_optimizer_expert_fc1_config,
    deepseek_v4_flash_pretrain_128gpu_gb300_fp8mx_mbs2_recompute_moe_act_config,
)
from megatron.bridge.utils.cuda_graph import cuda_graph_module_names
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _keep_recipe_construction_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


@pytest.mark.parametrize(
    ("recipe", "micro_batch_size", "recompute_granularity", "recompute_modules", "offload_optimizer_states"),
    [
        (
            deepseek_v4_flash_pretrain_128gpu_gb300_fp8mx_config,
            1,
            "selective",
            ["moe_act", "layernorm", "mla_up_proj", "shared_experts", "mhc"],
            False,
        ),
        (
            deepseek_v4_flash_pretrain_128gpu_gb300_fp8mx_mbs2_recompute_moe_act_config,
            2,
            "selective",
            ["moe_act"],
            False,
        ),
        (
            deepseek_v4_flash_pretrain_128gpu_gb300_fp8mx_mbs2_offload_optimizer_expert_fc1_config,
            2,
            None,
            None,
            True,
        ),
    ],
)
def test_deepseek_v4_flash_gb300_configs(
    recipe,
    micro_batch_size: int,
    recompute_granularity: str | None,
    recompute_modules: list[str] | None,
    offload_optimizer_states: bool,
) -> None:
    cfg = recipe()

    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.virtual_pipeline_model_parallel_size is None
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 64
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.sequence_parallel is False
    assert cfg.model.pipeline_model_parallel_layout is None
    assert cfg.train.global_batch_size == 2048
    assert cfg.train.micro_batch_size == micro_batch_size

    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.moe_router_force_load_balancing is True
    assert cfg.model.moe_hybridep_num_sms == 32
    assert cfg.model.recompute_granularity == recompute_granularity
    assert cfg.model.recompute_modules == recompute_modules
    assert cfg.model.cuda_graph_impl == "transformer_engine"
    assert cuda_graph_module_names(cfg.model) == ["attn", "moe_router", "moe_preprocess"]
    assert cfg.model.cuda_graph_warmup_steps == 1
    assert cfg.model.use_te_rng_tracker is True
    assert cfg.rng.te_rng_tracker is True
    assert cfg.env_vars["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert cfg.env_vars["NCCL_GRAPH_REGISTER"] == 0
    assert cfg.env_vars["TORCH_NCCL_AVOID_RECORD_STREAMS"] == 1

    assert cfg.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] == 32
    assert cfg.env_vars["NVTE_FWD_LAYERNORM_SM_MARGIN"] == 20
    assert cfg.env_vars["NVTE_BWD_LAYERNORM_SM_MARGIN"] == 20

    assert cfg.env_vars["NVLINK_DOMAIN_SIZE"] == 72
    assert cfg.env_vars["USE_MNNVL"] == 1
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 64
    assert cfg.env_vars["NUM_OF_TOKENS_PER_CHUNK_COMBINE_API"] == 128
    assert "NVTE_CUTEDSL_FUSED_GROUPED_MLP" not in cfg.env_vars

    assert cfg.model.csa_compress_rotary_base == 40_000
    assert cfg.model.rotary_scaling_factor == 4
    assert cfg.model.apply_dsa_kernel_fusion is True
    assert cfg.model.dsa_indexer_loss_coeff == 0.01
    assert cfg.model.dsa_indexer_use_sparse_loss is True
    assert cfg.model.quant_recipe is None
    assert cfg.model.moe_router_padding_for_fp8 is False
    assert cfg.model.moe_router_padding_for_quantization is True
    assert cfg.mixed_precision.fp8_param_gather is True
    assert cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag is True
    assert cfg.optimizer.main_grads_dtype == torch.float32
    assert cfg.ddp.grad_reduce_in_fp32 is False
    assert cfg.ddp.average_in_collective is False

    assert getattr(cfg.optimizer, "offload_optimizer_states", False) is offload_optimizer_states
    assert cfg.model.fine_grained_activation_offloading is offload_optimizer_states
    assert cfg.model.offload_modules == (["expert_fc1"] if offload_optimizer_states else [])
    assert (cfg.env_vars.get("NVTE_CPU_OFFLOAD_V1") == 1) is offload_optimizer_states
