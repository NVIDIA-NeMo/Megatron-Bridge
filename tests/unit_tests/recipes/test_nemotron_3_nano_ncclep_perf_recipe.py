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

"""Tests for the Nemotron 3 Nano GB300 MXFP8 NCCL EP performance recipe."""

import pytest

from megatron.bridge.perf_recipes.nemotronh import nemotron_3_nano_pretrain_8gpu_gb300_fp8mx_ncclep_config
from megatron.bridge.utils.cuda_graph import cuda_graph_module_names


pytestmark = pytest.mark.unit

_HYBRID_EP_ENV_NAMES = {
    "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN",
    "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API",
    "NVLINK_DOMAIN_SIZE",
    "USE_MNNVL",
}


def test_nemotron_3_nano_gb300_mxfp8_ncclep_defaults() -> None:
    """The MXFP8 example uses static fused NCCL EP and no HybridEP environment."""
    cfg = nemotron_3_nano_pretrain_8gpu_gb300_fp8mx_ncclep_config()

    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "ncclep"
    assert cfg.model.moe_shared_expert_overlap is False
    assert cfg.model.high_priority_a2a_comm_stream is True
    assert cfg.model.moe_hybridep_num_sms is None
    assert cfg.model.moe_flex_dispatcher_num_sms is None
    assert cfg.model.moe_ncclep_zero_copy is False

    assert cfg.model.moe_grouped_gemm is True
    assert cfg.model.use_transformer_engine_op_fuser is True
    assert cfg.model.moe_mlp_glu_interleave_size == 32

    assert cfg.model.offload_modules == []
    assert cfg.model.moe_expert_rank_capacity_factor == 1.05
    assert cfg.model.moe_paged_stash is True
    assert cfg.model.moe_paged_stash_buffer_size_factor_cuda == 1.2
    assert cfg.model.moe_paged_stash_buffer_size_factor_cpu == 1.0

    assert cfg.model.cuda_graph_impl == "transformer_engine"
    assert cuda_graph_module_names(cfg.model) == ["attn", "mamba", "moe_router", "moe_preprocess"]
    assert cfg.model.use_te_rng_tracker is True
    assert cfg.rng.te_rng_tracker is True
    assert cfg.env_vars.keys().isdisjoint(_HYBRID_EP_ENV_NAMES)
    assert cfg.env_vars["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == 1

    assert cfg.comm_overlap.overlap_moe_expert_parallel_comm is False
    assert cfg.comm_overlap.delay_wgrad_compute is False
    assert cfg.model.moe_router_padding_for_quantization is True

    cfg.model.finalize()
