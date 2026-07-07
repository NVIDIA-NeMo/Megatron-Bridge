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
"""GB300 performance recipes for Qwen3 MoE."""

from megatron.bridge.perf_recipes.qwen.common import (
    CommOverlapConfig,
    ConfigContainer,
    _apply_qwen3_moe_tuned_defaults,
    _benchmark_common,
    _enable_hybridep_full_iteration_mxfp8,
    _enable_overlap_param_gather_with_optimizer_step,
    _enable_qwen_precision_aware_optimizer,
    _perf_precision,
    _with_global_batch_size,
    qwen3_30b_a3b_pretrain_config,
    qwen3_235b_a22b_pretrain_config,
    qwen3_next_80b_a3b_pretrain_config,
)


def qwen3_235b_a22b_pretrain_64gpu_gb300_bf16_config() -> ConfigContainer:
    """Qwen3 235B-A22B pretrain: 64× GB300, BF16, EP=64."""
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
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 32
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 2

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    return cfg


def qwen3_235b_a22b_pretrain_64gpu_gb300_fp8cs_config() -> ConfigContainer:
    """Qwen3 235B-A22B pretrain: 64× GB300, FP8 current-scaling, EP=64."""
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
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 32
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 2

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    return cfg


def qwen3_235b_a22b_pretrain_64gpu_gb300_fp8mx_config() -> ConfigContainer:
    """Qwen3 235B-A22B pretrain: 64× GB300, MXFP8, EP=64."""
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
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 12
    cfg.model.expert_model_parallel_size = 32
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 2

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"

    _benchmark_common(cfg)
    _enable_hybridep_full_iteration_mxfp8(cfg)
    return cfg


def qwen3_30b_a3b_pretrain_8gpu_gb300_bf16_config() -> ConfigContainer:
    """Qwen3 30B-A3B pretrain: 8× GB300, BF16, EP=8."""
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
    cfg.train.micro_batch_size = 8

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    return cfg


def qwen3_30b_a3b_pretrain_8gpu_gb300_fp8cs_config() -> ConfigContainer:
    """Qwen3 30B-A3B pretrain: 8× GB300, FP8 current-scaling, EP=8."""
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
    cfg.train.micro_batch_size = 8

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    return cfg


def qwen3_30b_a3b_pretrain_8gpu_gb300_fp8mx_config() -> ConfigContainer:
    """Qwen3 30B-A3B pretrain: 8× GB300, MXFP8, EP=8."""
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
    cfg.train.micro_batch_size = 8

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"

    _benchmark_common(cfg)
    _enable_hybridep_full_iteration_mxfp8(cfg)
    return cfg


def qwen3_235b_a22b_pretrain_256gpu_gb300_bf16_config() -> ConfigContainer:
    """Qwen3 235B-A22B pretrain: 256× GB300, BF16, PP=4 EP=32."""
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
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 32
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 8192
    cfg.train.micro_batch_size = 2

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    return cfg


def qwen3_235b_a22b_pretrain_256gpu_gb300_fp8cs_config() -> ConfigContainer:
    """Qwen3 235B-A22B pretrain: 256× GB300, FP8 current-scaling, PP=4 EP=32."""
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
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 32
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 8192
    cfg.train.micro_batch_size = 2

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    return cfg


def qwen3_235b_a22b_pretrain_256gpu_gb300_fp8mx_config() -> ConfigContainer:
    """Qwen3 235B-A22B pretrain: 256× GB300, MXFP8, PP=4 EP=32."""
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
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 12
    cfg.model.expert_model_parallel_size = 32
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 8192
    cfg.train.micro_batch_size = 2

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"

    _benchmark_common(cfg)
    _enable_hybridep_full_iteration_mxfp8(cfg)
    return cfg


def qwen3_235b_a22b_pretrain_256gpu_gb300_fp8mx_large_scale_config() -> ConfigContainer:
    """Qwen3 235B A22B pretrain: 256× GB300, FP8-MX, large-scale proxy (GBS=512)."""
    cfg = qwen3_235b_a22b_pretrain_256gpu_gb300_fp8mx_config()
    cfg.train.global_batch_size = 512
    return cfg


def qwen3_235b_a22b_pretrain_64gpu_gb300_nvfp4_config() -> ConfigContainer:
    """Qwen3 235B A22B pretrain: 64× GB300, NVFP4 (same layout as FP8-CS)."""
    cfg = qwen3_235b_a22b_pretrain_64gpu_gb300_fp8cs_config()
    cfg.mixed_precision = _perf_precision("nvfp4")
    return cfg


def qwen3_235b_a22b_pretrain_256gpu_gb300_nvfp4_config() -> ConfigContainer:
    """Qwen3 235B A22B pretrain: 256× GB300, NVFP4 (same layout as FP8-CS)."""
    cfg = qwen3_235b_a22b_pretrain_256gpu_gb300_fp8cs_config()
    cfg.mixed_precision = _perf_precision("nvfp4")
    return cfg


def qwen3_30b_a3b_pretrain_32gpu_gb300_bf16_config() -> ConfigContainer:
    """Qwen3 30B-A3B pretrain: 32× GB300, BF16, legacy-scaled GBS."""
    return _with_global_batch_size(qwen3_30b_a3b_pretrain_8gpu_gb300_bf16_config(), 2048)


def qwen3_30b_a3b_pretrain_32gpu_gb300_fp8cs_config() -> ConfigContainer:
    """Qwen3 30B-A3B pretrain: 32× GB300, FP8 current-scaling, legacy-scaled GBS."""
    return _with_global_batch_size(qwen3_30b_a3b_pretrain_8gpu_gb300_fp8cs_config(), 2048)


def qwen3_next_80b_a3b_pretrain_64gpu_gb300_bf16_config() -> ConfigContainer:
    """Qwen3 Next 80B-A3B pretrain: 64× GB300, BF16, PP=2 VP=4 EP=32, hybridep."""
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

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 4

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
    cfg.model.pipeline_model_parallel_layout = "Et*4|(t*7|)*5t*8|tmL"

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    cfg.model.moe_hybridep_num_sms = 16
    _enable_overlap_param_gather_with_optimizer_step(cfg)
    return cfg


def qwen3_next_80b_a3b_pretrain_64gpu_gb300_fp8mx_config() -> ConfigContainer:
    """Qwen3 Next 80B-A3B pretrain: 64× GB300, MXFP8 (same layout as BF16)."""
    cfg = qwen3_next_80b_a3b_pretrain_64gpu_gb300_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    cfg.optimizer.overlap_param_gather_with_optimizer_step = False
    cfg.comm_overlap.overlap_param_gather_with_optimizer_step = None
    return cfg


def qwen3_235b_a22b_128gpu_gb300_fp8mx_paged_stash_pretrain_config() -> ConfigContainer:
    """Qwen3 235B-A22B pretrain: 128× GB300, MXFP8 and paged-stash CUDA graphs."""
    cfg = qwen3_235b_a22b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    _apply_qwen3_moe_tuned_defaults(cfg, original_max_position_embeddings=4096)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_router_padding_for_quantization = True
    cfg.model.moe_hybridep_num_sms = 32
    cfg.model.moe_paged_stash = True
    cfg.model.moe_expert_rank_capacity_factor = 1.2
    cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.0
    cfg.model.moe_pad_experts_for_cuda_graph_inference = True
    cfg.model.moe_mlp_glu_interleave_size = 32
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.model.cuda_graph_impl = "local"
    cfg.model.cuda_graph_scope = "full_iteration"
    cfg.model.cuda_graph_warmup_steps = 2
    cfg.model.use_te_rng_tracker = True
    cfg.model.offload_modules = []

    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=False,
        overlap_moe_expert_parallel_comm=True,
        delay_wgrad_compute=True,
    )
    cfg.train.micro_batch_size = 1
    cfg.train.global_batch_size = 8192
    cfg.ddp.reuse_grad_buf_for_mxfp8_param_ag = True
    cfg.ddp.check_for_nan_in_grad = False
    cfg.checkpoint.dist_ckpt_strictness = "log_all"
    cfg.rng.te_rng_tracker = True
    cfg.dist.distributed_timeout_minutes = 220
    _enable_qwen_precision_aware_optimizer(cfg)
    return cfg
