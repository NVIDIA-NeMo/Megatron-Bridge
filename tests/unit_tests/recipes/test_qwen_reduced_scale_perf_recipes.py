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

import torch
from scripts.common.benchmark_parallelism import data_parallel_size, topology_from_config

from megatron.bridge.perf_recipes.qwen import (
    qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_common_contract_config,
    qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_config,
    qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_config,
    qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_no_recompute_config,
    qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_selective_config,
    qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_selective_graph_config,
    qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_tp2_config,
    qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_tp2_full_graph_config,
    qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_tp2_selective_config,
    qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_tp2_selective_graph_config,
    qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero3_full_graph_config,
    qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_config,
    qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_chunk64_config,
    qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_chunk256_config,
    qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_config,
    qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_mbs2_config,
    qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_sms16_config,
    qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_sms24_config,
    qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_sms40_config,
    qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_sms48_config,
    qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_tp2_pp1_config,
    qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_tp2_pp1_mbs2_config,
)


def test_qwen3_235b_32gpu_gb300_bf16_topology() -> None:
    cfg = qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_config()

    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 2
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 16
    assert cfg.model.expert_tensor_parallel_size == 1
    assert data_parallel_size(num_gpus=32, topology=topology_from_config(cfg.model)) == 16
    assert cfg.train.global_batch_size == 2048
    assert cfg.train.micro_batch_size == 1
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 16


def test_qwen3_235b_16gpu_gb300_bf16_uses_fsdp_zero3() -> None:
    cfg = qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_config()

    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 16
    assert cfg.model.expert_tensor_parallel_size == 1
    assert data_parallel_size(num_gpus=16, topology=topology_from_config(cfg.model)) == 16
    assert cfg.train.global_batch_size == 2048
    assert cfg.train.micro_batch_size == 1

    assert cfg.dist.use_megatron_fsdp is True
    assert cfg.ddp.use_megatron_fsdp is True
    assert cfg.ddp.data_parallel_sharding_strategy == "optim_grads_params"
    assert cfg.ddp.expert_data_parallel_sharding_strategy == "optim_grads_params"
    assert cfg.ddp.average_in_collective is False
    assert cfg.checkpoint.ckpt_format == "fsdp_dtensor"
    assert cfg.model.init_model_with_meta_device is True

    assert cfg.model.recompute_granularity == "full"
    assert cfg.model.recompute_method == "uniform"
    assert cfg.model.recompute_num_layers == 1
    assert cfg.model.cuda_graph_impl == "none"
    assert cfg.model.cuda_graph_scope == []
    assert cfg.model.fine_grained_activation_offloading is False
    assert cfg.model.offload_modules is None

    assert cfg.optimizer.main_params_dtype == torch.float32
    assert cfg.optimizer.exp_avg_dtype == torch.float32
    assert cfg.optimizer.exp_avg_sq_dtype == torch.float32
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 16


def test_qwen3_235b_16gpu_gb300_bf16_common_contract() -> None:
    cfg = qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_common_contract_config()

    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 16
    assert cfg.model.cuda_graph_impl == "transformer_engine"
    assert cfg.model.cuda_graph_scope == ["moe_router", "moe_preprocess"]
    assert cfg.model.recompute_granularity is None
    assert cfg.model.recompute_modules == []
    assert cfg.optimizer.main_params_dtype == torch.float32
    assert cfg.optimizer.main_grads_dtype == torch.float32
    assert cfg.optimizer.exp_avg_dtype == torch.bfloat16
    assert cfg.optimizer.exp_avg_sq_dtype == torch.bfloat16
    assert cfg.ddp.use_distributed_optimizer is True
    assert cfg.ddp.use_megatron_fsdp is False
    assert cfg.dist.use_megatron_fsdp is False


def test_qwen3_235b_16gpu_gb300_bf16_zero2_variant() -> None:
    cfg = qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_config()

    assert cfg.ddp.use_megatron_fsdp is True
    assert cfg.ddp.data_parallel_sharding_strategy == "optim_grads"
    assert cfg.ddp.expert_data_parallel_sharding_strategy == "optim_grads"
    assert cfg.train.micro_batch_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 16


def test_qwen3_235b_16gpu_gb300_bf16_recompute_variants() -> None:
    selective = qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_selective_config()
    assert selective.model.recompute_granularity == "selective"
    assert selective.model.recompute_modules == ["moe_act"]
    assert selective.model.cuda_graph_impl == "none"

    no_recompute = qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_no_recompute_config()
    assert no_recompute.model.recompute_granularity is None
    assert no_recompute.model.recompute_modules == []

    graph = qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_selective_graph_config()
    assert graph.model.recompute_granularity == "selective"
    assert graph.model.recompute_modules == ["moe_act"]
    assert graph.model.cuda_graph_impl == "transformer_engine"
    assert graph.model.cuda_graph_scope == ["attn", "moe_router", "moe_preprocess"]
    assert graph.model.use_te_rng_tracker is True
    assert graph.rng.te_rng_tracker is True


def test_qwen3_235b_16gpu_gb300_bf16_tp2_candidates() -> None:
    full = qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_tp2_config()
    assert full.model.tensor_model_parallel_size == 2
    assert full.model.pipeline_model_parallel_size == 1
    assert full.model.expert_model_parallel_size == 16
    assert full.model.sequence_parallel is True
    assert data_parallel_size(num_gpus=16, topology=topology_from_config(full.model)) == 8
    assert full.model.recompute_granularity == "full"

    selective = qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_tp2_selective_config()
    assert selective.model.recompute_granularity == "selective"
    assert selective.model.recompute_modules == ["moe_act"]


def test_qwen3_235b_32gpu_gb300_bf16_pp1_candidates() -> None:
    tp1 = qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_config()
    assert tp1.model.tensor_model_parallel_size == 1
    assert tp1.model.pipeline_model_parallel_size == 1
    assert tp1.model.expert_model_parallel_size == 32
    assert data_parallel_size(num_gpus=32, topology=topology_from_config(tp1.model)) == 32

    tp1_mbs2 = qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_mbs2_config()
    assert tp1_mbs2.train.micro_batch_size == 2

    tp2 = qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_tp2_pp1_config()
    assert tp2.model.tensor_model_parallel_size == 2
    assert tp2.model.sequence_parallel is True
    assert data_parallel_size(num_gpus=32, topology=topology_from_config(tp2.model)) == 16

    tp2_mbs2 = qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_tp2_pp1_mbs2_config()
    assert tp2_mbs2.train.micro_batch_size == 2

    sms16 = qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_sms16_config()
    sms24 = qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_sms24_config()
    sms40 = qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_sms40_config()
    sms48 = qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_sms48_config()
    assert sms16.model.moe_hybridep_num_sms == 16
    assert sms24.model.moe_hybridep_num_sms == 24
    assert sms40.model.moe_hybridep_num_sms == 40
    assert sms48.model.moe_hybridep_num_sms == 48

    chunk64 = qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_chunk64_config()
    chunk256 = qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_pp1_chunk256_config()
    assert chunk64.env_vars["NUM_OF_TOKENS_PER_CHUNK_COMBINE_API"] == 64
    assert chunk256.env_vars["NUM_OF_TOKENS_PER_CHUNK_COMBINE_API"] == 256


def test_qwen3_235b_16gpu_gb300_bf16_graph_candidates() -> None:
    selective_graph = qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_tp2_selective_graph_config()
    assert selective_graph.model.cuda_graph_impl == "transformer_engine"
    assert selective_graph.model.recompute_granularity == "selective"

    for config_fn in (
        qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_tp2_full_graph_config,
        qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero3_full_graph_config,
    ):
        cfg = config_fn()
        assert cfg.model.cuda_graph_impl == "full_iteration"
        assert cfg.model.cuda_graph_scope == []
        assert cfg.ddp.megatron_fsdp_cuda_graph_mode is True
        assert cfg.model.moe_paged_stash is True
        assert cfg.model.moe_use_grouped_tensor is True
