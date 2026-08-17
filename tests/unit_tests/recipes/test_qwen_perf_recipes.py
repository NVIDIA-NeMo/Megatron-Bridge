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

"""Tests for Qwen3 NVFP4 overlap and NCCL EP performance recipe settings.

NVFP4's fp4_param_gather path is incompatible with TP comm overlap, so every
NVFP4 pretrain config must set ``comm_overlap.tp_comm_overlap = False`` while
non-NVFP4 (FP8 current-scaling) siblings keep it enabled.
"""

import pytest

from megatron.bridge.perf_recipes.qwen import (
    qwen3_30b_a3b_pretrain_8gpu_b200_fp8cs_config,
    qwen3_30b_a3b_pretrain_8gpu_b200_nvfp4_config,
    qwen3_30b_a3b_pretrain_8gpu_b300_fp8cs_config,
    qwen3_30b_a3b_pretrain_8gpu_b300_nvfp4_config,
    qwen3_30b_a3b_pretrain_8gpu_gb200_bf16_ncclep_config,
    qwen3_30b_a3b_pretrain_8gpu_gb200_fp8cs_config,
    qwen3_30b_a3b_pretrain_8gpu_gb200_fp8mx_ncclep_config,
    qwen3_30b_a3b_pretrain_8gpu_gb200_nvfp4_config,
    qwen3_30b_a3b_pretrain_8gpu_gb300_bf16_ncclep_config,
    qwen3_30b_a3b_pretrain_8gpu_gb300_fp8cs_config,
    qwen3_30b_a3b_pretrain_8gpu_gb300_fp8mx_ncclep_config,
    qwen3_30b_a3b_pretrain_8gpu_gb300_nvfp4_config,
    qwen3_30b_a3b_pretrain_8gpu_vr200_nvfp4_config,
    qwen3_235b_a22b_pretrain_64gpu_b200_fp8cs_config,
    qwen3_235b_a22b_pretrain_64gpu_b200_nvfp4_config,
    qwen3_235b_a22b_pretrain_64gpu_b300_fp8cs_config,
    qwen3_235b_a22b_pretrain_64gpu_b300_nvfp4_config,
    qwen3_235b_a22b_pretrain_64gpu_gb200_fp8cs_config,
    qwen3_235b_a22b_pretrain_64gpu_gb200_nvfp4_config,
    qwen3_235b_a22b_pretrain_64gpu_gb300_fp8cs_config,
    qwen3_235b_a22b_pretrain_64gpu_gb300_nvfp4_config,
    qwen3_235b_a22b_pretrain_256gpu_b200_fp8cs_config,
    qwen3_235b_a22b_pretrain_256gpu_b200_nvfp4_config,
    qwen3_235b_a22b_pretrain_256gpu_b300_fp8cs_config,
    qwen3_235b_a22b_pretrain_256gpu_b300_nvfp4_config,
    qwen3_235b_a22b_pretrain_256gpu_gb200_fp8cs_config,
    qwen3_235b_a22b_pretrain_256gpu_gb200_nvfp4_config,
    qwen3_235b_a22b_pretrain_256gpu_gb300_fp8cs_config,
    qwen3_235b_a22b_pretrain_256gpu_gb300_nvfp4_config,
    qwen3_235b_a22b_pretrain_256gpu_vr200_nvfp4_config,
)


_HYBRID_EP_ENV_NAMES = {
    "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN",
    "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API",
    "NVLINK_DOMAIN_SIZE",
    "USE_MNNVL",
}


def test_qwen3_30b_gb200_bf16_ncclep_config():
    cfg = qwen3_30b_a3b_pretrain_8gpu_gb200_bf16_ncclep_config()

    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "ncclep"
    assert cfg.model.moe_expert_rank_capacity_factor == 1.05
    assert cfg.model.moe_grouped_gemm is True
    assert cfg.model.use_transformer_engine_op_fuser is True
    assert cfg.model.moe_ncclep_zero_copy is False
    assert cfg.model.moe_hybridep_num_sms is None
    assert cfg.model.moe_flex_dispatcher_num_sms is None
    assert cfg.comm_overlap.tp_comm_overlap is False
    assert cfg.env_vars.keys().isdisjoint(_HYBRID_EP_ENV_NAMES)


def test_qwen3_30b_gb200_mxfp8_ncclep_config():
    cfg = qwen3_30b_a3b_pretrain_8gpu_gb200_fp8mx_ncclep_config()

    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "ncclep"
    assert cfg.model.moe_expert_rank_capacity_factor == 1.05
    assert cfg.model.moe_ncclep_zero_copy is False
    assert cfg.model.moe_paged_stash is True
    assert cfg.model.moe_grouped_gemm is True
    assert cfg.model.use_transformer_engine_op_fuser is True
    assert cfg.model.moe_router_padding_for_quantization is True
    assert cfg.model.moe_hybridep_num_sms is None
    assert cfg.model.moe_flex_dispatcher_num_sms is None
    assert cfg.comm_overlap.overlap_moe_expert_parallel_comm is True
    assert cfg.env_vars["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == 1
    assert cfg.env_vars.keys().isdisjoint(_HYBRID_EP_ENV_NAMES)


def test_qwen3_30b_gb300_bf16_ncclep_config():
    cfg = qwen3_30b_a3b_pretrain_8gpu_gb300_bf16_ncclep_config()

    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "ncclep"
    assert cfg.model.moe_expert_rank_capacity_factor == 1.05
    # Paged stashing only captures TE's quantized grouped tensors, so it is a no-op in BF16.
    assert cfg.model.moe_paged_stash is False
    assert cfg.model.moe_grouped_gemm is True
    assert cfg.model.use_transformer_engine_op_fuser is True
    assert cfg.model.moe_ncclep_zero_copy is False
    assert cfg.model.moe_hybridep_num_sms is None
    assert cfg.model.moe_flex_dispatcher_num_sms is None
    assert cfg.model.cross_entropy_fusion_impl == "native"
    assert cfg.model.cuda_graph_scope == ["moe_router", "moe_preprocess"]
    assert cfg.train.micro_batch_size == 8
    assert cfg.comm_overlap.tp_comm_overlap is False
    assert cfg.env_vars.keys().isdisjoint(_HYBRID_EP_ENV_NAMES)


def test_qwen3_30b_gb300_mxfp8_ncclep_config():
    cfg = qwen3_30b_a3b_pretrain_8gpu_gb300_fp8mx_ncclep_config()

    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "ncclep"
    assert cfg.model.moe_expert_rank_capacity_factor == 1.05
    assert cfg.model.moe_ncclep_zero_copy is False
    assert cfg.model.moe_paged_stash is True
    assert cfg.model.moe_grouped_gemm is True
    assert cfg.model.use_transformer_engine_op_fuser is True
    assert cfg.model.moe_router_padding_for_quantization is True
    assert cfg.model.moe_hybridep_num_sms is None
    assert cfg.model.moe_flex_dispatcher_num_sms is None
    assert cfg.model.cross_entropy_fusion_impl == "native"
    assert cfg.train.micro_batch_size == 8
    assert cfg.comm_overlap.overlap_moe_expert_parallel_comm is True
    assert cfg.env_vars["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == 1
    assert cfg.env_vars.keys().isdisjoint(_HYBRID_EP_ENV_NAMES)


@pytest.mark.parametrize(
    "nvfp4_config_fn",
    [
        qwen3_30b_a3b_pretrain_8gpu_gb200_nvfp4_config,
        qwen3_30b_a3b_pretrain_8gpu_gb300_nvfp4_config,
        qwen3_30b_a3b_pretrain_8gpu_b200_nvfp4_config,
        qwen3_30b_a3b_pretrain_8gpu_b300_nvfp4_config,
        qwen3_30b_a3b_pretrain_8gpu_vr200_nvfp4_config,
        qwen3_235b_a22b_pretrain_64gpu_gb200_nvfp4_config,
        qwen3_235b_a22b_pretrain_256gpu_gb200_nvfp4_config,
        qwen3_235b_a22b_pretrain_64gpu_gb300_nvfp4_config,
        qwen3_235b_a22b_pretrain_256gpu_gb300_nvfp4_config,
        qwen3_235b_a22b_pretrain_64gpu_b200_nvfp4_config,
        qwen3_235b_a22b_pretrain_256gpu_b200_nvfp4_config,
        qwen3_235b_a22b_pretrain_64gpu_b300_nvfp4_config,
        qwen3_235b_a22b_pretrain_256gpu_b300_nvfp4_config,
        qwen3_235b_a22b_pretrain_256gpu_vr200_nvfp4_config,
    ],
)
def test_nvfp4_disables_tp_comm_overlap(nvfp4_config_fn):
    cfg = nvfp4_config_fn()
    assert cfg.comm_overlap.tp_comm_overlap is False, (
        f"{nvfp4_config_fn.__name__}: expected tp_comm_overlap=False for NVFP4, got {cfg.comm_overlap.tp_comm_overlap}"
    )


@pytest.mark.parametrize(
    "fp8cs_config_fn",
    [
        qwen3_30b_a3b_pretrain_8gpu_gb200_fp8cs_config,
        qwen3_30b_a3b_pretrain_8gpu_gb300_fp8cs_config,
        qwen3_30b_a3b_pretrain_8gpu_b200_fp8cs_config,
        qwen3_30b_a3b_pretrain_8gpu_b300_fp8cs_config,
        qwen3_235b_a22b_pretrain_64gpu_gb200_fp8cs_config,
        qwen3_235b_a22b_pretrain_256gpu_gb200_fp8cs_config,
        qwen3_235b_a22b_pretrain_64gpu_gb300_fp8cs_config,
        qwen3_235b_a22b_pretrain_256gpu_gb300_fp8cs_config,
        qwen3_235b_a22b_pretrain_64gpu_b200_fp8cs_config,
        qwen3_235b_a22b_pretrain_256gpu_b200_fp8cs_config,
        qwen3_235b_a22b_pretrain_64gpu_b300_fp8cs_config,
        qwen3_235b_a22b_pretrain_256gpu_b300_fp8cs_config,
    ],
)
def test_non_nvfp4_preserves_tp_comm_overlap(fp8cs_config_fn):
    """Regression: the NVFP4 fix must not affect FP8 current-scaling siblings."""
    cfg = fp8cs_config_fn()
    assert cfg.comm_overlap.tp_comm_overlap is True, (
        f"{fp8cs_config_fn.__name__}: expected tp_comm_overlap=True for FP8-CS, got {cfg.comm_overlap.tp_comm_overlap}"
    )
