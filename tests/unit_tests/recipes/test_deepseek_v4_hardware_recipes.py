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

"""Execution and convergence guardrails for DeepSeek V4 hardware recipes."""

from collections.abc import Callable

import pytest
import torch

import megatron.bridge.recipes as recipes
from megatron.bridge.recipes.deepseek.gb200.deepseek_v4 import (
    deepseek_v4_flash_peft_openmath_thinking_packed_gb200_config as flash_packed_peft_config,
)
from megatron.bridge.recipes.deepseek.gb200.deepseek_v4 import (
    deepseek_v4_flash_pretrain_64gpu_gb200_bf16_config as flash_bf16_base_config,
)
from megatron.bridge.recipes.deepseek.gb200.deepseek_v4 import (
    deepseek_v4_flash_pretrain_64gpu_gb200_fp8mx_config as flash_fp8_config,
)
from megatron.bridge.recipes.deepseek.gb200.deepseek_v4 import (
    deepseek_v4_flash_pretrain_128gpu_gb200_fp8mx_library_config as flash_library_config,
)
from megatron.bridge.recipes.deepseek.gb200.deepseek_v4 import (
    deepseek_v4_flash_sft_openmath_thinking_packed_gb200_config as flash_packed_sft_config,
)
from megatron.bridge.recipes.deepseek.gb300.deepseek_v4 import (
    deepseek_v4_pro_pretrain_32gpu_gb300_fp8mx_config as pro_fp8_config,
)
from megatron.bridge.training.config import ConfigContainer
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _keep_recipe_construction_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


def test_flash_base_recipe_enables_precision_independent_fusions() -> None:
    cfg = flash_bf16_base_config()

    assert cfg.model.apply_dsa_kernel_fusion is True
    assert cfg.model.moe_pad_experts_for_cuda_graph_inference is True
    assert getattr(cfg.model, "moe_mlp_glu_interleave_size", None) is None
    assert cfg.model.use_transformer_engine_op_fuser is False
    assert cfg.model.cross_entropy_fusion_impl == "native"
    assert cfg.comm_overlap.overlap_grad_reduce is True
    assert "NVTE_CUTEDSL_FUSED_GROUPED_MLP" not in cfg.env_vars

    assert getattr(cfg.model, "moe_expert_rank_capacity_factor", None) is None
    assert getattr(cfg.model, "moe_paged_stash", False) is False
    assert cfg.model.cuda_graph_impl == "none"
    assert cfg.rerun_state_machine.check_for_nan_in_loss is True
    assert cfg.ddp.check_for_nan_in_grad is True
    assert not getattr(cfg.model, "fp8", False)
    assert getattr(cfg.model, "quant_recipe", None) is None
    assert cfg.optimizer.optimizer_offload_fraction == 0.0


def test_flash_base_recipe_ports_flash_fusions() -> None:
    cfg = flash_bf16_base_config()

    assert cfg.model.attention_backend == "auto"
    assert cfg.model.moe_router_fusion is True
    assert cfg.train.manual_gc_interval == 5
    assert cfg.model.fine_grained_activation_offloading is True
    assert cfg.model.offload_modules == ["core_attn", "attn_proj"]
    assert cfg.model.fine_grained_offloading_max_inflight_offloads == 2
    assert cfg.env_vars["NVTE_CPU_OFFLOAD_V1"] == 1


@pytest.mark.parametrize(
    "fp8_factory",
    [flash_fp8_config, pro_fp8_config],
    ids=["flash-mxfp8", "pro-mxfp8"],
)
def test_mxfp8_recipes_keep_training_precision_contract(
    fp8_factory: Callable[[], ConfigContainer],
) -> None:
    cfg = fp8_factory()

    assert cfg.model.quant_recipe is not None
    assert cfg.model.moe_router_padding_for_fp8 is True
    assert cfg.mixed_precision.fp8_param_gather is False
    assert cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag is False
    assert cfg.mixed_precision.grad_reduce_in_fp32 is True
    assert cfg.ddp.grad_reduce_in_fp32 is True
    assert cfg.optimizer.main_grads_dtype == torch.float32


def test_flash_mxfp8_recipe_uses_activation_offload_to_fit() -> None:
    cfg = flash_fp8_config()

    assert cfg.model.recompute_modules == ["moe_act", "mhc", "mla_up_proj"]
    assert cfg.model.fine_grained_activation_offloading is True
    assert cfg.model.offload_modules == ["core_attn", "attn_proj"]
    assert cfg.model.fine_grained_offloading_max_inflight_offloads == 2
    assert cfg.env_vars["NVTE_CPU_OFFLOAD_V1"] == 1


def test_flash_packed_sft_recipe_uses_gb200_training_contract() -> None:
    cfg = flash_packed_sft_config()

    assert cfg.model.cp_partition_mode == "contiguous"
    assert cfg.dataset.offline_packing_specs.pad_seq_to_mult == 4
    assert cfg.dataset.offline_packing_specs.pad_cu_seqlens is True
    assert cfg.dataset.dataset_kwargs == {"pad_to_max_length": True}
    assert cfg.model.apply_dsa_kernel_fusion is True
    assert cfg.model.dsa_indexer_loss_coeff == 0.0
    assert cfg.model.dsa_indexer_use_sparse_loss is False
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.moe_flex_dispatcher_num_sms == 16
    assert cfg.model.moe_hybridep_pad_uneven_dispatch_inputs is True
    assert cfg.model.moe_shared_expert_overlap is False
    assert cfg.model.moe_permute_fusion is True
    assert cfg.model.moe_router_fusion is True
    assert cfg.model.moe_grouped_gemm is True
    assert cfg.model.cross_entropy_fusion_impl == "native"
    assert cfg.model.recompute_granularity == "selective"
    assert cfg.model.recompute_modules == ["moe", "mhc", "mla_up_proj", "layernorm"]
    assert cfg.model.recompute_method is None
    assert cfg.model.recompute_num_layers is None
    assert cfg.model.calculate_per_token_loss is True
    assert cfg.model.fine_grained_activation_offloading is True
    assert cfg.model.offload_modules == ["core_attn", "attn_proj"]
    assert cfg.model.fine_grained_offloading_max_inflight_offloads == 2
    assert cfg.env_vars["NVTE_CPU_OFFLOAD_V1"] == 1


def test_flash_packed_peft_recipe_adapts_mla_and_unshared_experts() -> None:
    cfg = flash_packed_peft_config()

    assert cfg.peft.target_modules == [
        "linear_q_down_proj",
        "linear_q_up_proj",
        "linear_kv_proj",
        "linear_proj",
        "linear_fc1",
        "linear_fc2",
    ]
    assert cfg.peft.dim == 32
    assert cfg.peft.alpha == 32
    assert cfg.peft.dropout == 0.0
    assert cfg.peft.share_expert_adapters is False
    assert cfg.optimizer.lr == 1.0e-4
    assert cfg.optimizer.min_lr == 0.0

    sft_cfg = flash_packed_sft_config()
    assert cfg.dataset == sft_cfg.dataset
    assert cfg.train == sft_cfg.train
    assert cfg.rng == sft_cfg.rng
    assert cfg.model.seq_length == sft_cfg.model.seq_length
    assert cfg.model.pipeline_model_parallel_size == sft_cfg.model.pipeline_model_parallel_size
    assert cfg.model.expert_model_parallel_size == sft_cfg.model.expert_model_parallel_size
    assert cfg.model.moe_token_dispatcher_type == sft_cfg.model.moe_token_dispatcher_type
    assert cfg.model.recompute_modules == sft_cfg.model.recompute_modules


def test_flash_high_scale_recipe_preserves_real_training_contract() -> None:
    cfg = flash_library_config()

    assert cfg.train.train_iters == 1_000_000
    assert cfg.train.global_batch_size == 256
    assert cfg.validation.eval_interval == 2000
    assert cfg.validation.eval_iters == 32
    assert cfg.model.moe_router_force_load_balancing is False
    assert cfg.model.dsa_indexer_loss_coeff == 0.0
    assert cfg.model.dsa_indexer_use_sparse_loss is False
    assert cfg.model.apply_dsa_kernel_fusion is True
    assert cfg.model.quant_recipe is not None
    assert cfg.model.moe_router_padding_for_fp8 is True
    assert cfg.mixed_precision.fp8_param_gather is True
    assert cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag is True
    assert cfg.mixed_precision.grad_reduce_in_fp32 is True
    assert cfg.ddp.grad_reduce_in_fp32 is True
    assert cfg.optimizer.main_grads_dtype == torch.float32
    assert cfg.ddp.check_for_nan_in_grad is True
    assert cfg.model.cuda_graph_impl == "none"
    assert cfg.model.use_transformer_engine_op_fuser is True
    assert cfg.model.moe_mlp_glu_interleave_size == 32
    assert cfg.model.moe_flex_dispatcher_num_sms == 32
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_layout is None
    assert cfg.model.recompute_modules == ["moe", "mhc", "mla_up_proj", "layernorm"]
    assert cfg.model.fine_grained_activation_offloading is True
    assert cfg.model.offload_modules == ["core_attn", "attn_proj"]
    assert cfg.model.fine_grained_offloading_max_inflight_offloads == 2
    assert getattr(cfg.model, "moe_expert_rank_capacity_factor", None) is None
    assert getattr(cfg.model, "moe_paged_stash", False) is False
    assert cfg.env_vars["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert cfg.env_vars["TORCH_NCCL_AVOID_RECORD_STREAMS"] == 1
    assert cfg.env_vars["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == 1
    assert cfg.env_vars["NVTE_CPU_OFFLOAD_V1"] == 1


def test_high_scale_deepseek_v4_recipes_are_exported() -> None:
    assert recipes.deepseek_v4_flash_peft_openmath_thinking_packed_gb200_config is flash_packed_peft_config
    assert recipes.deepseek_v4_flash_pretrain_128gpu_gb200_fp8mx_library_config is flash_library_config
    assert recipes.deepseek_v4_flash_sft_openmath_thinking_packed_gb200_config is flash_packed_sft_config
