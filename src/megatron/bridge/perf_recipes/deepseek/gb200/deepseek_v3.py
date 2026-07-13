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
"""GB200 performance recipes for DeepSeek V3."""

from megatron.bridge.perf_recipes.deepseek.common import (
    ConfigContainer,
    _benchmark_common,
    _deepseek_v3_common,
    _enable_deepseek_full_iteration_mxfp8,
    _enable_deepseek_precision_aware_optimizer,
    _enable_overlap_param_gather_with_optimizer_step,
    _perf_precision,
    deepseek_v3_pretrain_config,
    set_deepseek_v3_pipeline_model_parallel_layout,
)


def deepseek_v3_pretrain_256gpu_gb200_bf16_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× GB200, BF16."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    _deepseek_v3_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = ["mla_up_proj"]

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]

    cfg.ddp.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_grad_reduce = True

    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    _benchmark_common(cfg)
    _enable_overlap_param_gather_with_optimizer_step(cfg)
    return cfg


def deepseek_v3_pretrain_256gpu_gb200_fp8cs_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× GB200, FP8 current-scaling."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    _deepseek_v3_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = ["mlp"]

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]

    cfg.ddp.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_grad_reduce = True

    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    _benchmark_common(cfg)
    _enable_overlap_param_gather_with_optimizer_step(cfg)
    return cfg


def deepseek_v3_pretrain_256gpu_gb200_fp8mx_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× GB200, MXFP8."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    _deepseek_v3_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = ["mla_up_proj"]

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]

    cfg.ddp.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_grad_reduce = True

    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    _benchmark_common(cfg)
    _enable_deepseek_full_iteration_mxfp8(cfg, fp8_output_proj=True)
    return cfg


def deepseek_v3_pretrain_256gpu_gb200_nvfp4_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× GB200, NVFP4 (same layout as BF16, mlp recompute)."""
    cfg = deepseek_v3_pretrain_256gpu_gb200_bf16_config()
    cfg.mixed_precision = _perf_precision("nvfp4")
    cfg.model.recompute_modules = ["mlp"]
    cfg.optimizer.overlap_param_gather_with_optimizer_step = False
    cfg.comm_overlap.overlap_param_gather_with_optimizer_step = None
    return cfg


def deepseek_v3_pretrain_256gpu_gb200_fp8mx_large_scale_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× GB200, MXFP8, large-scale proxy (GBS=256)."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    _deepseek_v3_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 256
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = ["mlp"]

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]

    cfg.ddp.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_grad_reduce = True

    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    _benchmark_common(cfg)
    cfg.model.fp8_output_proj = True
    return cfg


def deepseek_v3_pretrain_256gpu_gb200_fp8mx_partial_cg_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× GB200, MXFP8, scoped CUDA graphs and offloading."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    _benchmark_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.virtual_pipeline_model_parallel_size = 2
    cfg.model.expert_model_parallel_size = 64
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.kv_channels = 128
    cfg.model.make_vocab_size_divisible_by = 1280
    cfg.model.moe_router_force_load_balancing = True
    cfg.model.moe_router_fusion = True
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_router_padding_for_quantization = True
    cfg.model.moe_hybridep_num_sms = 32
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["mla_up_proj"]
    cfg.model.fine_grained_activation_offloading = True
    cfg.model.offload_modules = ["expert_fc1"]
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
    cfg.model.cuda_graph_warmup_steps = 1
    cfg.model.use_te_rng_tracker = True

    cfg.comm_overlap.delay_wgrad_compute = True
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = True
    cfg.ddp.reuse_grad_buf_for_mxfp8_param_ag = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True

    cfg.train.micro_batch_size = 1
    cfg.train.global_batch_size = 8192
    cfg.train.exit_duration_in_mins = 220
    cfg.train.manual_gc_interval = 10

    cfg.rng.te_rng_tracker = True
    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model, "Et*4|(tttt|)*14tmL")
    _enable_deepseek_precision_aware_optimizer(cfg)
    return cfg
