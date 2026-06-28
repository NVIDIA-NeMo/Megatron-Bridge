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
"""H100 performance recipes for Qwen3 MoE."""

from megatron.bridge.perf_recipes.qwen.common import (
    CommOverlapConfig,
    ConfigContainer,
    _benchmark_common,
    _enable_overlap_param_gather_with_optimizer_step,
    _perf_precision,
    _with_global_batch_size,
    qwen3_30b_a3b_pretrain_config,
    qwen3_235b_a22b_pretrain_config,
    qwen3_next_80b_a3b_pretrain_config,
)


def qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config() -> ConfigContainer:
    """Qwen3 30B-A3B pretrain: 16× H100, BF16, EP=16."""
    cfg = qwen3_30b_a3b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.model.bias_activation_fusion = True
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.moe_router_fusion = True
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 16
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_a2a_overlap = False

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    return cfg


def qwen3_30b_a3b_pretrain_16gpu_h100_fp8cs_config() -> ConfigContainer:
    """Qwen3 30B-A3B pretrain: 16× H100, FP8 current-scaling, EP=16."""
    cfg = qwen3_30b_a3b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.model.bias_activation_fusion = True
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.moe_router_fusion = True
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 16
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_a2a_overlap = False

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    return cfg


def qwen3_235b_a22b_pretrain_256gpu_h100_bf16_config() -> ConfigContainer:
    """Qwen3 235B-A22B pretrain: 256× H100, BF16, TP=2 PP=8 VP=4 EP=32."""
    cfg = qwen3_235b_a22b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.model.bias_activation_fusion = True
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.moe_router_fusion = True
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.expert_model_parallel_size = 32
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 8192
    cfg.train.micro_batch_size = 1

    cfg.model.moe_flex_dispatcher_backend = None
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_a2a_overlap = True
    cfg.model.moe_shared_expert_overlap = False

    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=False,
        overlap_moe_expert_parallel_comm=True,
        delay_wgrad_compute=True,
    )

    _benchmark_common(cfg)
    _enable_overlap_param_gather_with_optimizer_step(cfg)
    return cfg


def qwen3_235b_a22b_pretrain_256gpu_h100_fp8cs_config() -> ConfigContainer:
    """Qwen3 235B-A22B pretrain: 256× H100, FP8 current-scaling, TP=2 PP=8 VP=4 EP=32."""
    cfg = qwen3_235b_a22b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.model.bias_activation_fusion = True
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.moe_router_fusion = True
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.expert_model_parallel_size = 32
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 8192
    cfg.train.micro_batch_size = 1

    cfg.model.moe_flex_dispatcher_backend = None
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_a2a_overlap = True
    cfg.model.moe_shared_expert_overlap = False

    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=False,
        overlap_moe_expert_parallel_comm=True,
        delay_wgrad_compute=True,
    )

    _benchmark_common(cfg)
    _enable_overlap_param_gather_with_optimizer_step(cfg)
    return cfg


def qwen3_235b_a22b_pretrain_256gpu_h100_fp8cs_large_scale_config() -> ConfigContainer:
    """Qwen3 235B A22B pretrain: 256× H100, FP8-CS, large-scale proxy (GBS=512)."""
    cfg = qwen3_235b_a22b_pretrain_256gpu_h100_fp8cs_config()
    cfg.train.global_batch_size = 512
    return cfg


def qwen3_30b_a3b_pretrain_64gpu_h100_bf16_config() -> ConfigContainer:
    """Qwen3 30B-A3B pretrain: 64× H100, BF16, legacy-scaled GBS."""
    return _with_global_batch_size(qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config(), 4096)


def qwen3_30b_a3b_pretrain_64gpu_h100_fp8cs_config() -> ConfigContainer:
    """Qwen3 30B-A3B pretrain: 64× H100, FP8 current-scaling, legacy-scaled GBS."""
    return _with_global_batch_size(qwen3_30b_a3b_pretrain_16gpu_h100_fp8cs_config(), 4096)


def qwen3_next_80b_a3b_pretrain_128gpu_h100_bf16_config() -> ConfigContainer:
    """Qwen3 Next 80B-A3B pretrain: 128× H100, BF16, EP=128, MBS=1."""
    cfg = qwen3_next_80b_a3b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.model.bias_activation_fusion = True
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.moe_router_fusion = True
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 128
    cfg.model.expert_tensor_parallel_size = 1
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1

    cfg.model.moe_token_dispatcher_type = "alltoall"

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    return cfg


def qwen3_next_80b_a3b_pretrain_128gpu_h100_fp8cs_config() -> ConfigContainer:
    """Qwen3 Next 80B-A3B pretrain: 128× H100, FP8-CS (same layout as BF16)."""
    cfg = qwen3_next_80b_a3b_pretrain_128gpu_h100_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    return cfg
