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
"""B200 performance recipes for Qwen3 MoE."""

from megatron.bridge.perf_recipes.qwen.b300.qwen3_moe import (
    qwen3_next_80b_a3b_pretrain_64gpu_b300_bf16_config,
)
from megatron.bridge.perf_recipes.qwen.common import (
    CommOverlapConfig,
    ConfigContainer,
    _benchmark_common,
    _enable_overlap_param_gather_with_optimizer_step,
    _perf_precision,
    _with_global_batch_size,
    qwen3_30b_a3b_pretrain_config,
    qwen3_235b_a22b_pretrain_config,
)


def qwen3_235b_a22b_pretrain_64gpu_b200_bf16_config() -> ConfigContainer:
    """Qwen3 235B-A22B pretrain: 64× B200, BF16, PP=8 VP=4 EP=8."""
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

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.expert_model_parallel_size = 8
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 1

    cfg.model.moe_flex_dispatcher_backend = None
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_a2a_overlap = False

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    _enable_overlap_param_gather_with_optimizer_step(cfg)
    return cfg


def qwen3_235b_a22b_pretrain_64gpu_b200_fp8cs_config() -> ConfigContainer:
    """Qwen3 235B-A22B pretrain: 64× B200, FP8 current-scaling, PP=8 EP=8."""
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

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 8
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 1

    cfg.model.moe_flex_dispatcher_backend = None
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_a2a_overlap = False

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    return cfg


def qwen3_235b_a22b_pretrain_64gpu_b200_fp8mx_config() -> ConfigContainer:
    """Qwen3 235B-A22B pretrain: 64× B200, MXFP8, PP=8 EP=8."""
    cfg = qwen3_235b_a22b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    cfg.model.bias_activation_fusion = True
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.moe_router_fusion = True
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 8
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 1

    cfg.model.moe_flex_dispatcher_backend = None
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_a2a_overlap = False

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    return cfg


def qwen3_30b_a3b_pretrain_8gpu_b200_bf16_config() -> ConfigContainer:
    """Qwen3 30B-A3B pretrain: 8× B200, BF16, EP=8."""
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
    cfg.model.expert_model_parallel_size = 8
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 4

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    return cfg


def qwen3_30b_a3b_pretrain_8gpu_b200_fp8cs_config() -> ConfigContainer:
    """Qwen3 30B-A3B pretrain: 8× B200, FP8 current-scaling, EP=8."""
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
    cfg.model.expert_model_parallel_size = 8
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 4

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    return cfg


def qwen3_30b_a3b_pretrain_8gpu_b200_fp8mx_config() -> ConfigContainer:
    """Qwen3 30B-A3B pretrain: 8× B200, MXFP8, EP=8."""
    cfg = qwen3_30b_a3b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
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
    cfg.model.expert_model_parallel_size = 8
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 4

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    return cfg


def qwen3_235b_a22b_pretrain_256gpu_b200_bf16_config() -> ConfigContainer:
    """Qwen3 235B-A22B pretrain: 256× B200, BF16, PP=8 VP=4 EP=8."""
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

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.expert_model_parallel_size = 8
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 8192
    cfg.train.micro_batch_size = 1

    cfg.model.moe_flex_dispatcher_backend = None
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_a2a_overlap = False

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    _enable_overlap_param_gather_with_optimizer_step(cfg)
    return cfg


def qwen3_235b_a22b_pretrain_256gpu_b200_fp8cs_config() -> ConfigContainer:
    """Qwen3 235B-A22B pretrain: 256× B200, FP8 current-scaling, PP=8 EP=8."""
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

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 8
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 8192
    cfg.train.micro_batch_size = 1

    cfg.model.moe_flex_dispatcher_backend = None
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_a2a_overlap = False

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    return cfg


def qwen3_235b_a22b_pretrain_256gpu_b200_fp8mx_config() -> ConfigContainer:
    """Qwen3 235B-A22B pretrain: 256× B200, MXFP8, PP=8 EP=8."""
    cfg = qwen3_235b_a22b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    cfg.model.bias_activation_fusion = True
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.moe_router_fusion = True
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 8
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 8192
    cfg.train.micro_batch_size = 1

    cfg.model.moe_flex_dispatcher_backend = None
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_a2a_overlap = False

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    return cfg


def qwen3_235b_a22b_pretrain_256gpu_b200_fp8mx_large_scale_config() -> ConfigContainer:
    """Qwen3 235B A22B pretrain: 256× B200, FP8-MX, large-scale proxy (GBS=512)."""
    cfg = qwen3_235b_a22b_pretrain_256gpu_b200_fp8mx_config()
    cfg.train.global_batch_size = 512
    return cfg


def qwen3_235b_a22b_pretrain_64gpu_b200_nvfp4_config() -> ConfigContainer:
    """Qwen3 235B A22B pretrain: 64× B200, NVFP4 (same layout as FP8-CS)."""
    cfg = qwen3_235b_a22b_pretrain_64gpu_b200_fp8cs_config()
    cfg.mixed_precision = _perf_precision("nvfp4")
    return cfg


def qwen3_235b_a22b_pretrain_256gpu_b200_nvfp4_config() -> ConfigContainer:
    """Qwen3 235B A22B pretrain: 256× B200, NVFP4 (same layout as FP8-CS)."""
    cfg = qwen3_235b_a22b_pretrain_256gpu_b200_fp8cs_config()
    cfg.mixed_precision = _perf_precision("nvfp4")
    return cfg


def qwen3_30b_a3b_pretrain_64gpu_b200_bf16_config() -> ConfigContainer:
    """Qwen3 30B-A3B pretrain: 64× B200, BF16, legacy-scaled GBS."""
    return _with_global_batch_size(qwen3_30b_a3b_pretrain_8gpu_b200_bf16_config(), 4096)


def qwen3_30b_a3b_pretrain_64gpu_b200_fp8cs_config() -> ConfigContainer:
    """Qwen3 30B-A3B pretrain: 64× B200, FP8 current-scaling, legacy-scaled GBS."""
    return _with_global_batch_size(qwen3_30b_a3b_pretrain_8gpu_b200_fp8cs_config(), 4096)


def qwen3_next_80b_a3b_pretrain_64gpu_b200_bf16_config() -> ConfigContainer:
    """Qwen3 Next 80B-A3B pretrain: 64× B200, BF16 (same layout as B300 BF16)."""
    return qwen3_next_80b_a3b_pretrain_64gpu_b300_bf16_config()


def qwen3_next_80b_a3b_pretrain_64gpu_b200_fp8mx_config() -> ConfigContainer:
    """Qwen3 Next 80B-A3B pretrain: 64× B200, MXFP8, EP=64, deepep, MBS=1."""
    cfg = qwen3_next_80b_a3b_pretrain_64gpu_b300_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    return cfg
