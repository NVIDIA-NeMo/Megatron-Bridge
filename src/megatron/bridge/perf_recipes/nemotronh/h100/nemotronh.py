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
"""H100 performance recipes for NemotronH and Nemotron 3."""

from megatron.bridge.perf_recipes.nemotronh.common import (
    ConfigContainer,
    _benchmark_common,
    _perf_precision,
    nemotron_3_nano_pretrain_config,
    nemotronh_56b_pretrain_config,
)


def nemotronh_56b_pretrain_64gpu_h100_fp8cs_config() -> ConfigContainer:
    """NemotronH 56B pretrain: 64× H100, FP8 current-scaling."""
    cfg = nemotronh_56b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")

    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 192
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["mamba"]

    _benchmark_common(cfg)
    return cfg


def nemotron_3_nano_pretrain_16gpu_h100_bf16_config() -> ConfigContainer:
    """Nemotron 3 Nano pretrain: 16× H100, BF16, recompute MoE+layernorm."""
    cfg = nemotron_3_nano_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.model.recompute_granularity = "selective"

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1

    cfg.model.moe_router_force_load_balancing = True

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "mamba"]

    cfg.model.recompute_modules = ["moe", "layernorm"]

    cfg.comm_overlap.tp_comm_overlap = True

    _benchmark_common(cfg)
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    return cfg


def nemotron_3_nano_pretrain_16gpu_h100_fp8cs_config() -> ConfigContainer:
    """Nemotron 3 Nano pretrain: 16× H100, FP8 current-scaling, recompute."""
    cfg = nemotron_3_nano_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.model.recompute_granularity = "selective"

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1

    cfg.model.moe_router_force_load_balancing = True

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["mamba"]

    cfg.model.recompute_modules = ["moe", "layernorm", "core_attn", "moe_act"]

    cfg.comm_overlap.tp_comm_overlap = True

    _benchmark_common(cfg)
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    return cfg
