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

"""Tests for the GPT-OSS 120B NCCL EP performance recipe variants.

Each NCCL EP variant derives from the HybridEP/DeepEP recipe of the same name and is expected to
change only the MoE dispatch stack, so the parity test below pins the parallelism, batch sizes,
precision and recompute settings to the parent's.
"""

from collections.abc import Callable

import pytest

from megatron.bridge.perf_recipes.gpt_oss import (
    gpt_oss_120b_pretrain_64gpu_gb200_bf16_config,
    gpt_oss_120b_pretrain_64gpu_gb200_bf16_ncclep_config,
    gpt_oss_120b_pretrain_64gpu_gb200_fp8mx_config,
    gpt_oss_120b_pretrain_64gpu_gb200_fp8mx_ncclep_config,
    gpt_oss_120b_pretrain_64gpu_gb300_bf16_config,
    gpt_oss_120b_pretrain_64gpu_gb300_bf16_ncclep_config,
    gpt_oss_120b_pretrain_64gpu_gb300_fp8mx_config,
    gpt_oss_120b_pretrain_64gpu_gb300_fp8mx_ncclep_config,
)
from megatron.bridge.training.config import ConfigContainer


pytestmark = pytest.mark.unit

_HYBRID_EP_ENV_NAMES = {
    "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN",
    "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API",
    "NVLINK_DOMAIN_SIZE",
    "USE_MNNVL",
}

# NCCL EP variant -> the recipe it derives from
_NCCLEP_TO_PARENT = {
    gpt_oss_120b_pretrain_64gpu_gb300_bf16_ncclep_config: gpt_oss_120b_pretrain_64gpu_gb300_bf16_config,
    gpt_oss_120b_pretrain_64gpu_gb300_fp8mx_ncclep_config: gpt_oss_120b_pretrain_64gpu_gb300_fp8mx_config,
    gpt_oss_120b_pretrain_64gpu_gb200_bf16_ncclep_config: gpt_oss_120b_pretrain_64gpu_gb200_bf16_config,
    gpt_oss_120b_pretrain_64gpu_gb200_fp8mx_ncclep_config: gpt_oss_120b_pretrain_64gpu_gb200_fp8mx_config,
}
_NCCLEP_RECIPES = tuple(_NCCLEP_TO_PARENT)


@pytest.mark.parametrize("recipe_factory", _NCCLEP_RECIPES, ids=lambda recipe: recipe.__name__)
def test_gpt_oss_ncclep_dispatch_stack(recipe_factory: Callable[[], ConfigContainer]) -> None:
    """Every NCCL EP variant selects the ncclep flex backend, whatever its precision."""
    cfg = recipe_factory()

    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "ncclep"
    assert cfg.model.moe_shared_expert_overlap is False
    assert cfg.model.high_priority_a2a_comm_stream is True
    assert cfg.model.moe_hybridep_num_sms is None
    assert cfg.model.moe_flex_dispatcher_num_sms is None
    assert cfg.model.moe_ncclep_zero_copy is False

    assert cfg.model.offload_modules == []
    assert cfg.comm_overlap is not None
    assert cfg.comm_overlap.delay_wgrad_compute is False
    assert cfg.env_vars.keys().isdisjoint(_HYBRID_EP_ENV_NAMES)


@pytest.mark.parametrize("recipe_factory", _NCCLEP_RECIPES, ids=lambda recipe: recipe.__name__)
def test_gpt_oss_ncclep_fused_grouped_mlp_is_mxfp8_only(
    recipe_factory: Callable[[], ConfigContainer],
) -> None:
    """The CuTe DSL fused grouped MLP exists only for MXFP8; BF16 runs eager NCCL EP.

    On BF16 the op fuser makes TE reject GPT-OSS's clamped quick-GeLU outright
    (``ScaledClampedQGeGLU(...) requires the fused grouped MLP path``) when moe_act recompute is
    on, and run ~9x slower when it is not.
    """
    cfg = recipe_factory()

    if "fp8mx" in recipe_factory.__name__:
        parent = _NCCLEP_TO_PARENT[recipe_factory]().model
        assert cfg.model.moe_grouped_gemm is True
        assert cfg.model.use_transformer_engine_op_fuser is True
        assert cfg.model.moe_mlp_glu_interleave_size == 32
        assert cfg.env_vars["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == 1
        assert cfg.model.moe_router_padding_for_quantization is True
        assert cfg.model.moe_paged_stash is True
        # Static shapes: the fused grouped GEMM consumes device-side per-expert counts. Both the
        # receive budget and the capacity-factor-independent stash pool track the HybridEP parent,
        # so this arm differs from its baseline in the dispatcher alone. Pinning the literals as
        # well keeps an accidental change to both sides at once from passing unnoticed.
        assert cfg.model.moe_expert_rank_capacity_factor == 1.5
        assert cfg.model.moe_paged_stash_buffer_size_factor_cuda == 1.2
        assert cfg.model.moe_paged_stash_buffer_size_factor_cpu == 1.0
        for field in (
            "moe_expert_rank_capacity_factor",
            "moe_paged_stash_buffer_size_factor_cuda",
            "moe_paged_stash_buffer_size_factor_cpu",
        ):
            assert getattr(cfg.model, field) == getattr(parent, field), field
    else:
        assert cfg.model.use_transformer_engine_op_fuser is False
        assert "NVTE_CUTEDSL_FUSED_GROUPED_MLP" not in cfg.env_vars
        # Eager NCCL EP sizes the receive buffer per step, so no static capacity factor and
        # therefore no paged stash either.
        assert cfg.model.moe_expert_rank_capacity_factor is None
        assert cfg.model.moe_paged_stash is False


@pytest.mark.parametrize("recipe_factory", _NCCLEP_RECIPES, ids=lambda recipe: recipe.__name__)
def test_gpt_oss_ncclep_matches_parent_outside_the_dispatch_stack(
    recipe_factory: Callable[[], ConfigContainer],
) -> None:
    """Everything the NCCL EP switch should not touch stays identical to the parent recipe."""
    cfg = recipe_factory()
    parent = _NCCLEP_TO_PARENT[recipe_factory]()

    for field in (
        "expert_model_parallel_size",
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "sequence_parallel",
        "seq_length",
        "num_moe_experts",
        "moe_router_topk",
        "moe_router_force_load_balancing",
        "recompute_granularity",
        "recompute_modules",
        "cuda_graph_impl",
    ):
        assert getattr(cfg.model, field) == getattr(parent.model, field), field

    assert cfg.train.micro_batch_size == parent.train.micro_batch_size
    assert cfg.train.global_batch_size == parent.train.global_batch_size
    assert type(cfg.mixed_precision) is type(parent.mixed_precision)


def test_gpt_oss_ncclep_recipes_keep_router_force_load_balancing() -> None:
    """The static receive-capacity factor assumes the router is balanced by construction."""
    for recipe_factory in _NCCLEP_RECIPES:
        cfg = recipe_factory()
        assert cfg.model.moe_router_force_load_balancing is True, recipe_factory.__name__
