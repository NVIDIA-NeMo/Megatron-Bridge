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
    qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_config,
    qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_config,
    qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_config,
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


def test_qwen3_235b_16gpu_gb300_bf16_zero2_variant() -> None:
    cfg = qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_config()

    assert cfg.ddp.use_megatron_fsdp is True
    assert cfg.ddp.data_parallel_sharding_strategy == "optim_grads"
    assert cfg.ddp.expert_data_parallel_sharding_strategy == "optim_grads"
    assert cfg.train.micro_batch_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 16
