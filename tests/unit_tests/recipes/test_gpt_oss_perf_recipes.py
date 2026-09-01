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

"""Tests for the GPT-OSS 120B GB200 BF16 NCCL EP performance recipe.

The NCCL EP recipe derives from the corresponding parent recipe and changes only the MoE dispatch
stack, so the parity test below pins parallelism, batch sizes, precision and recompute settings.
"""

import pytest

from megatron.bridge.perf_recipes.gpt_oss import (
    gpt_oss_120b_pretrain_64gpu_gb200_bf16_config,
    gpt_oss_120b_pretrain_64gpu_gb200_bf16_ncclep_config,
)


pytestmark = pytest.mark.unit

_HYBRID_EP_ENV_NAMES = {
    "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN",
    "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API",
    "NVLINK_DOMAIN_SIZE",
    "USE_MNNVL",
}


def test_gpt_oss_gb200_bf16_ncclep_dispatch_stack() -> None:
    """The BF16 example selects eager NCCL EP without HybridEP environment settings."""
    cfg = gpt_oss_120b_pretrain_64gpu_gb200_bf16_ncclep_config()

    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "ncclep"
    assert cfg.model.moe_shared_expert_overlap is False
    assert cfg.model.high_priority_a2a_comm_stream is True
    assert cfg.model.moe_hybridep_num_sms is None
    assert cfg.model.moe_flex_dispatcher_num_sms is None
    assert cfg.model.moe_ncclep_zero_copy is False

    assert cfg.model.offload_modules == []
    assert cfg.comm_overlap is not None
    assert cfg.comm_overlap.overlap_moe_expert_parallel_comm is True
    assert cfg.comm_overlap.delay_wgrad_compute is False
    assert cfg.env_vars.keys().isdisjoint(_HYBRID_EP_ENV_NAMES)

    assert cfg.model.use_transformer_engine_op_fuser is False
    assert "NVTE_CUTEDSL_FUSED_GROUPED_MLP" not in cfg.env_vars
    assert cfg.model.moe_expert_rank_capacity_factor is None
    assert cfg.model.moe_paged_stash is False


def test_gpt_oss_gb200_bf16_ncclep_matches_parent_outside_dispatch_stack() -> None:
    """Everything the NCCL EP switch should not touch stays identical to the parent recipe."""
    cfg = gpt_oss_120b_pretrain_64gpu_gb200_bf16_ncclep_config()
    parent = gpt_oss_120b_pretrain_64gpu_gb200_bf16_config()

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
