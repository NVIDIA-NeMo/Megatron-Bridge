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
"""GB200 performance recipes for GPT-OSS."""

from megatron.bridge.perf_recipes.gpt_oss.common import (
    CommOverlapConfig,
    ConfigContainer,
    _apply_gpt_oss_20b_common_configs,
    _apply_gpt_oss_20b_local_graph_configs,
    _apply_gpt_oss_20b_transformer_engine_graph_configs,
    _apply_gpt_oss_120b_full_iter_fp8mx_configs,
    _benchmark_common,
    _gpt_oss_20b_fp8mx_precision,
    _gpt_oss_20b_nvfp4_precision,
    _perf_precision,
    gpt_oss_20b_pretrain_config,
    gpt_oss_120b_pretrain_config,
)


def gpt_oss_20b_pretrain_8gpu_gb200_nvfp4_config() -> ConfigContainer:
    """GPT-OSS 20B pretrain: 8× GB200, NVFP4."""
    cfg = gpt_oss_20b_pretrain_config()
    cfg.mixed_precision = _gpt_oss_20b_nvfp4_precision()

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.context_parallel_size = 2
    cfg.model.expert_model_parallel_size = 4
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 4
    cfg.train.micro_batch_size = 1

    _benchmark_common(cfg)
    _apply_gpt_oss_20b_common_configs(cfg)
    _apply_gpt_oss_20b_transformer_engine_graph_configs(cfg)

    cfg.optimizer.lr = 0.0006
    cfg.optimizer.min_lr = 0.0004
    cfg.validation.eval_interval = 341
    cfg.validation.eval_iters = 29
    cfg.scheduler.lr_warmup_iters = 64
    return cfg


def gpt_oss_20b_pretrain_72gpu_gb200_nvfp4_config() -> ConfigContainer:
    """GPT-OSS 20B pretrain: 72× GB200, NVFP4."""
    cfg = gpt_oss_20b_pretrain_config()
    cfg.mixed_precision = _gpt_oss_20b_nvfp4_precision()

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.context_parallel_size = 2
    cfg.model.expert_model_parallel_size = 4
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 36
    cfg.train.micro_batch_size = 1

    _benchmark_common(cfg)
    _apply_gpt_oss_20b_common_configs(cfg)
    _apply_gpt_oss_20b_transformer_engine_graph_configs(cfg)

    cfg.optimizer.lr = 0.0006
    cfg.optimizer.min_lr = 0.0004
    cfg.validation.eval_interval = 341
    cfg.validation.eval_iters = 29
    cfg.scheduler.lr_warmup_iters = 64
    return cfg


def gpt_oss_20b_pretrain_512gpu_gb200_fp8mx_config() -> ConfigContainer:
    """GPT-OSS 20B pretrain: 512× GB200, MXFP8."""
    cfg = gpt_oss_20b_pretrain_config()
    cfg.mixed_precision = _gpt_oss_20b_fp8mx_precision()

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.context_parallel_size = 4
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 64
    cfg.train.micro_batch_size = 1

    _benchmark_common(cfg)
    _apply_gpt_oss_20b_common_configs(cfg)
    _apply_gpt_oss_20b_local_graph_configs(cfg)

    cfg.model.moe_expert_rank_capacity_factor = 7
    cfg.model.sequence_parallel = True
    cfg.optimizer.lr = 0.00052
    cfg.optimizer.min_lr = 0.00052
    cfg.validation.eval_interval = 192
    cfg.validation.eval_iters = 16
    cfg.scheduler.lr_warmup_iters = 32
    return cfg


def gpt_oss_120b_pretrain_64gpu_gb200_bf16_config() -> ConfigContainer:
    """GPT-OSS 120B pretrain: 64× GB200, BF16, GBS=1280."""
    cfg = gpt_oss_120b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.model.moe_router_fusion = True
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 1280
    cfg.train.micro_batch_size = 4

    cfg.model.recompute_modules = ["layernorm", "moe_act"]
    cfg.model.recompute_granularity = "selective"

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.comm_overlap.delay_wgrad_compute = True
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = True

    _benchmark_common(cfg)
    return cfg


def gpt_oss_120b_pretrain_64gpu_gb200_fp8mx_config() -> ConfigContainer:
    """GPT-OSS 120B pretrain: 64× GB200, FP8-MX."""
    cfg = gpt_oss_120b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    cfg.model.moe_router_fusion = True
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 1280
    cfg.train.micro_batch_size = 4

    cfg.model.recompute_modules = ["layernorm", "moe_act"]
    cfg.model.recompute_granularity = "selective"

    _benchmark_common(cfg)
    _apply_gpt_oss_120b_full_iter_fp8mx_configs(cfg)
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = []
    return cfg
