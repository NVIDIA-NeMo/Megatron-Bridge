# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from megatron.bridge.perf_recipes.nemotronh.b200.nemotronh import (
    nemotron_3_ultra_pretrain_256gpu_b200_fp8mx_config,
)
from megatron.bridge.perf_recipes.nemotronh.b300.nemotronh import (
    nemotron_3_ultra_pretrain_256gpu_b300_fp8mx_config,
)
from megatron.bridge.perf_recipes.nemotronh.gb300.nemotronh import (
    nemotron_3_ultra_pretrain_256gpu_gb300_fp8mx_config,
)
from megatron.bridge.perf_recipes.nemotronh.h100.nemotronh import (
    nemotron_3_ultra_pretrain_24gpu_h100_bf16_config,
)
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _keep_recipe_construction_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


def test_nemotron_3_ultra_b300_uses_single_node_nvlink_domains() -> None:
    """The B300 port should retain GB300 model settings but use 8-GPU HSDP domains."""
    cfg = nemotron_3_ultra_pretrain_256gpu_b300_fp8mx_config()

    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.seq_length == 8192
    assert cfg.train.global_batch_size == 256
    assert cfg.model.cuda_graph_impl == "none"
    assert cfg.model.cuda_graph_scope == []

    assert cfg.ddp.use_megatron_fsdp is True
    assert cfg.ddp.num_distributed_optimizer_instances == 32
    assert cfg.ddp.outer_dp_sharding_strategy == "optim"
    assert cfg.ddp.megatron_fsdp_cuda_graph_mode is False
    assert cfg.ddp.fsdp_all_gather_in_start_param_sync is True
    expert_data_parallel_size = 256 // (
        cfg.model.expert_tensor_parallel_size
        * cfg.model.expert_model_parallel_size
        * cfg.model.pipeline_model_parallel_size
    )
    assert expert_data_parallel_size == cfg.ddp.num_distributed_optimizer_instances

    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 8
    assert cfg.env_vars["NVLINK_DOMAIN_SIZE"] == 8
    assert cfg.env_vars["USE_MNNVL"] == 0
    assert cfg.env_vars["NVTE_CPU_OFFLOAD_V1"] == 1
    assert cfg.env_vars["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == 1
    assert cfg.env_vars["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert cfg.env_vars["TORCH_NCCL_AVOID_RECORD_STREAMS"] == 1


def test_nemotron_3_ultra_b200_uses_wider_fsdp_shards() -> None:
    """B200 should preserve EP8/HybridEP8 while reducing training-state memory."""
    cfg = nemotron_3_ultra_pretrain_256gpu_b200_fp8mx_config()

    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.cuda_graph_impl == "none"
    assert cfg.ddp.use_megatron_fsdp is True
    assert cfg.ddp.num_distributed_optimizer_instances == 4
    assert cfg.ddp.outer_dp_sharding_strategy == "optim"

    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 8
    assert cfg.env_vars["NVLINK_DOMAIN_SIZE"] == 8
    assert cfg.env_vars["USE_MNNVL"] == 0
    assert cfg.env_vars["NVTE_CPU_OFFLOAD_V1"] == 1
    assert cfg.env_vars["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == 1
    assert cfg.env_vars["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert cfg.env_vars["TORCH_NCCL_AVOID_RECORD_STREAMS"] == 1


def test_nemotron_3_ultra_gb300_keeps_mnnvl_hsdp_domains() -> None:
    """Making HSDP topology-aware should preserve the existing GB300 configuration."""
    cfg = nemotron_3_ultra_pretrain_256gpu_gb300_fp8mx_config()

    assert cfg.ddp.num_distributed_optimizer_instances == 4
    assert cfg.model.cuda_graph_impl == "none"
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 64
    assert cfg.env_vars["NVLINK_DOMAIN_SIZE"] == 72
    assert cfg.env_vars["USE_MNNVL"] == 1
    assert cfg.env_vars["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert cfg.env_vars["TORCH_NCCL_AVOID_RECORD_STREAMS"] == 1


def test_nemotron_3_ultra_h100_keeps_the_validated_bf16_topology() -> None:
    """H100 should use the existing BF16/PP3 baseline without Blackwell-only features."""
    cfg = nemotron_3_ultra_pretrain_24gpu_h100_bf16_config()

    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 3
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.model.seq_length == 8192
    assert cfg.train.global_batch_size == 3072
    assert cfg.train.micro_batch_size == 1
    assert cfg.model.cuda_graph_impl == "none"
    assert cfg.model.fine_grained_activation_offloading is False
    assert cfg.model.use_transformer_engine_op_fuser is False
    assert cfg.ddp.use_megatron_fsdp is False

    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 8
    assert cfg.env_vars["USE_MNNVL"] == 0
    assert "NVTE_CPU_OFFLOAD_V1" not in cfg.env_vars
    assert "NVTE_CUTEDSL_FUSED_GROUPED_MLP" not in cfg.env_vars
