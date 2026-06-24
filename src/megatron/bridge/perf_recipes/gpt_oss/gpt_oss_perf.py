# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Flat performance benchmark recipes for GPT-OSS.

Each function is self-contained: call library recipe, override fields, call
``_benchmark_common()``, return.
"""

from megatron.bridge.perf_recipes._common import _benchmark_common, _perf_precision
from megatron.bridge.recipes.gpt_oss.gpt_oss import gpt_oss_20b_pretrain_config, gpt_oss_120b_pretrain_config
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer


def _apply_full_iter_fp8mx_overrides(
    cfg: ConfigContainer, *, expert_model_parallel_size: int, clear_recompute: bool = False
) -> None:
    """Apply legacy GPT-OSS 120B FP8-MX full-iteration CUDA graph settings."""
    cfg.model.expert_model_parallel_size = expert_model_parallel_size
    cfg.model.cuda_graph_impl = "full_iteration"
    cfg.model.cuda_graph_scope = []
    cfg.model.cuda_graph_warmup_steps = 2
    cfg.model.fp8_output_proj = True
    cfg.model.high_priority_a2a_comm_stream = True
    cfg.model.moe_expert_rank_capacity_factor = 1.5
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_hybridep_num_sms = 32
    cfg.model.moe_hybridep_num_sms_preprocessing = 32
    cfg.model.moe_mlp_glu_interleave_size = 32
    cfg.model.moe_pad_experts_for_cuda_graph_inference = True
    cfg.model.moe_paged_stash = True
    cfg.model.moe_paged_stash_buffer_size_factor_cpu = 1.0
    cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.2
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.use_te_rng_tracker = True
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.model.offload_modules = []
    cfg.mixed_precision.fp8_dot_product_attention = True
    cfg.rng.te_rng_tracker = True

    if cfg.comm_overlap is None:
        cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.comm_overlap.delay_wgrad_compute = True
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = True

    if clear_recompute:
        cfg.model.recompute_granularity = "selective"
        cfg.model.recompute_modules = []


def _gpt_oss_20b_precision(precision: str):
    """Return legacy GPT-OSS 20B perf precision settings."""
    precision_config = _perf_precision(precision)
    precision_config.fp4_param = False
    precision_config.fp4_param_gather = False
    precision_config.fp8_param = False
    precision_config.fp8_param_gather = False
    precision_config.reuse_grad_buf_for_mxfp8_param_ag = False
    if precision == "fp8_mx":
        precision_config.first_last_layers_bf16 = False
        precision_config.num_layers_at_start_in_bf16 = 0
    elif precision == "nvfp4":
        precision_config.first_last_layers_bf16 = True
        precision_config.num_layers_at_start_in_bf16 = 0
        precision_config.num_layers_at_end_in_bf16 = 4
    return precision_config


def _apply_gpt_oss_20b_common_configs(cfg: ConfigContainer) -> None:
    """Apply legacy GPT-OSS 20B perf defaults."""
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.model.apply_rope_fusion = False
    cfg.model.attention_backend = "auto"
    cfg.model.calculate_per_token_loss = False
    cfg.model.cpu_offloading_num_layers = 95
    cfg.model.cuda_graph_warmup_steps = 2
    cfg.model.fused_single_qkv_rope = True
    cfg.model.moe_aux_loss_coeff = 0.0
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_hybridep_num_sms = 128
    cfg.model.moe_permute_fusion = False
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.moe_router_fusion = False
    cfg.model.moe_router_padding_for_quantization = True
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.position_embedding_type = "rope"
    cfg.model.seq_length = 8192
    cfg.model.use_te_rng_tracker = True
    cfg.model.tp_only_amax_red = True
    cfg.model.vocab_size = 128256
    cfg.train.check_optimizer_step_success = False
    cfg.train.skip_sync_grad_norm_across_mp = False
    cfg.checkpoint.dist_ckpt_strictness = "log_all"
    cfg.checkpoint.fully_parallel_load = True
    cfg.checkpoint.load_optim = False
    cfg.tokenizer.hf_tokenizer_kwargs = {"use_fast": True}
    cfg.tokenizer.vocab_size = 128256
    cfg.optimizer.adam_eps = 1e-05
    cfg.dataset.create_attention_mask = False
    cfg.dataset.defer_npy_index_mmap = True
    cfg.dataset.fast_cache_load = True
    cfg.ddp.bucket_size = 768000000
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.fsdp_double_buffer = True
    cfg.ddp.nccl_ub = True
    cfg.rng.te_rng_tracker = True
    cfg.scheduler.start_weight_decay = 0.1
    cfg.scheduler.end_weight_decay = 0.1
    cfg.scheduler.override_opt_param_scheduler = False


def _apply_gpt_oss_20b_variant_configs(cfg: ConfigContainer, *, gpu: str, precision: str, variant: str) -> None:
    """Apply legacy GPT-OSS 20B GPU/precision-specific tuning."""
    if gpu == "b300":
        if precision == "nvfp4" and variant == "v1":
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.optimizer.lr = 0.0004
            cfg.optimizer.min_lr = 0.0004
            cfg.validation.eval_interval = 512
            cfg.validation.eval_iters = 43
            cfg.scheduler.lr_warmup_iters = 192
        elif precision == "fp8_mx" and variant == "v1":
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_modules = "full_iteration"
            cfg.model.cuda_graph_scope = None
            cfg.model.use_transformer_engine_op_fuser = True
            cfg.model.moe_expert_rank_capacity_factor = 1.5
            cfg.model.moe_mlp_glu_interleave_size = 32
            cfg.optimizer.lr = 0.0005
            cfg.optimizer.min_lr = 0.0005
            cfg.validation.eval_interval = 512
            cfg.validation.eval_iters = 43
            cfg.scheduler.lr_warmup_iters = 256
        elif precision == "nvfp4" and variant == "v2":
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.optimizer.lr = 0.0006
            cfg.optimizer.min_lr = 0.0006
            cfg.validation.eval_interval = 384
            cfg.validation.eval_iters = 32
            cfg.scheduler.lr_warmup_iters = 64
        elif precision == "fp8_mx" and variant == "v2":
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_modules = "full_iteration"
            cfg.model.cuda_graph_scope = None
            cfg.model.use_transformer_engine_op_fuser = True
            cfg.model.moe_expert_rank_capacity_factor = 5
            cfg.model.moe_mlp_glu_interleave_size = 32
            cfg.optimizer.lr = 0.0004
            cfg.optimizer.min_lr = 0.0004
            cfg.validation.eval_interval = 384
            cfg.validation.eval_iters = 32
            cfg.scheduler.lr_warmup_iters = 512
    elif gpu == "gb200":
        if precision == "nvfp4" and variant == "v1":
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.optimizer.lr = 0.0006
            cfg.optimizer.min_lr = 0.0006
            cfg.validation.eval_interval = 768
            cfg.validation.eval_iters = 64
            cfg.scheduler.lr_warmup_iters = 128
        elif precision == "fp8_mx" and variant == "v1":
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_modules = "full_iteration"
            cfg.model.cuda_graph_scope = None
            cfg.model.use_transformer_engine_op_fuser = True
            cfg.model.moe_expert_rank_capacity_factor = 1.2
            cfg.model.moe_mlp_glu_interleave_size = 32
            cfg.model.cuda_graph_warmup_steps = 5
            cfg.ddp.average_in_collective = True
            cfg.ddp.overlap_param_gather = True
            cfg.optimizer.overlap_param_gather = True
            cfg.optimizer.lr = 0.0004
            cfg.optimizer.min_lr = 0.0004
            cfg.validation.eval_interval = 768
            cfg.validation.eval_iters = 64
            cfg.scheduler.lr_warmup_iters = 128
            cfg.mixed_precision.fp8_param_gather = True
            cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = True
        elif precision == "nvfp4" and variant == "v2":
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.optimizer.lr = 0.0006
            cfg.optimizer.min_lr = 0.0004
            cfg.validation.eval_interval = 341
            cfg.validation.eval_iters = 29
            cfg.scheduler.lr_warmup_iters = 64
        elif precision == "fp8_mx" and variant == "v2":
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_modules = "full_iteration"
            cfg.model.cuda_graph_scope = None
            cfg.model.use_transformer_engine_op_fuser = True
            cfg.model.moe_expert_rank_capacity_factor = 5
            cfg.model.moe_mlp_glu_interleave_size = 32
            cfg.optimizer.lr = 0.0004
            cfg.optimizer.min_lr = 0.0004
            cfg.validation.eval_interval = 341
            cfg.validation.eval_iters = 29
            cfg.scheduler.lr_warmup_iters = 256
        elif precision == "fp8_mx" and variant == "v3":
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_modules = "full_iteration"
            cfg.model.cuda_graph_scope = None
            cfg.model.use_transformer_engine_op_fuser = True
            cfg.model.moe_expert_rank_capacity_factor = 7
            cfg.model.sequence_parallel = True
            cfg.model.moe_mlp_glu_interleave_size = 32
            cfg.optimizer.lr = 0.00052
            cfg.optimizer.min_lr = 0.00052
            cfg.validation.eval_interval = 192
            cfg.validation.eval_iters = 16
            cfg.scheduler.lr_warmup_iters = 32
    elif gpu == "gb300":
        if precision == "nvfp4" and variant == "v1":
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.optimizer.lr = 0.0004
            cfg.optimizer.min_lr = 0.0004
            cfg.validation.eval_interval = 512
            cfg.validation.eval_iters = 43
            cfg.scheduler.lr_warmup_iters = 192
        elif precision == "fp8_mx" and variant == "v1":
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_modules = "full_iteration"
            cfg.model.cuda_graph_scope = None
            cfg.model.use_transformer_engine_op_fuser = True
            cfg.model.moe_expert_rank_capacity_factor = 2
            cfg.model.moe_mlp_glu_interleave_size = 32
            cfg.ddp.average_in_collective = True
            cfg.optimizer.lr = 0.0005
            cfg.optimizer.min_lr = 0.0005
            cfg.validation.eval_interval = 512
            cfg.validation.eval_iters = 43
            cfg.scheduler.lr_warmup_iters = 256
        elif precision == "nvfp4" and variant == "v2":
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.optimizer.lr = 0.0006
            cfg.optimizer.min_lr = 0.0006
            cfg.validation.eval_interval = 341
            cfg.validation.eval_iters = 29
            cfg.scheduler.lr_warmup_iters = 64
        elif precision == "fp8_mx" and variant == "v2":
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_modules = "full_iteration"
            cfg.model.cuda_graph_scope = None
            cfg.model.use_transformer_engine_op_fuser = True
            cfg.model.moe_expert_rank_capacity_factor = 5
            cfg.model.moe_mlp_glu_interleave_size = 32
            cfg.optimizer.lr = 0.0004
            cfg.optimizer.min_lr = 0.0004
            cfg.validation.eval_interval = 341
            cfg.validation.eval_iters = 29
            cfg.scheduler.lr_warmup_iters = 256
        elif precision == "fp8_mx" and variant == "v3":
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_modules = "full_iteration"
            cfg.model.cuda_graph_scope = None
            cfg.model.use_transformer_engine_op_fuser = True
            cfg.model.moe_expert_rank_capacity_factor = 7
            cfg.model.sequence_parallel = True
            cfg.model.moe_mlp_glu_interleave_size = 32
            cfg.optimizer.lr = 0.00052
            cfg.optimizer.min_lr = 0.00052
            cfg.validation.eval_interval = 192
            cfg.validation.eval_iters = 16
            cfg.scheduler.lr_warmup_iters = 32
    elif gpu == "vr200":
        cfg.model.cuda_graph_impl = "transformer_engine"
        cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
        cfg.model.cuda_graph_warmup_steps = 1
        if precision == "nvfp4" and variant == "v1":
            cfg.optimizer.lr = 0.0004
            cfg.optimizer.min_lr = 0.0004
            cfg.validation.eval_interval = 512
            cfg.validation.eval_iters = 43
            cfg.scheduler.lr_warmup_iters = 192
        elif precision == "fp8_mx" and variant == "v1":
            cfg.optimizer.lr = 0.0005
            cfg.optimizer.min_lr = 0.0005
            cfg.validation.eval_interval = 512
            cfg.validation.eval_iters = 43
            cfg.scheduler.lr_warmup_iters = 192
        elif precision == "nvfp4" and variant == "v2":
            cfg.optimizer.lr = 0.0006
            cfg.optimizer.min_lr = 0.0006
            cfg.validation.eval_interval = 384
            cfg.validation.eval_iters = 43
            cfg.scheduler.lr_warmup_iters = 64


def _gpt_oss_20b_config(
    precision: str,
    gpu: str,
    variant: str,
    *,
    tensor_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    global_batch_size: int = 16,
    micro_batch_size: int = 2,
) -> ConfigContainer:
    """Build a flat GPT-OSS 20B perf recipe matching the legacy workload preset."""
    cfg = gpt_oss_20b_pretrain_config()
    cfg.mixed_precision = _gpt_oss_20b_precision(precision)

    cfg.model.tensor_model_parallel_size = tensor_model_parallel_size
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = context_parallel_size
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = expert_model_parallel_size
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = tensor_model_parallel_size > 1
    cfg.train.global_batch_size = global_batch_size
    cfg.train.micro_batch_size = micro_batch_size

    _benchmark_common(cfg)
    _apply_gpt_oss_20b_common_configs(cfg)
    _apply_gpt_oss_20b_variant_configs(cfg, gpu=gpu, precision=precision, variant=variant)
    return cfg


# =============================================================================
# GPT-OSS 20B pretrain — B300
# =============================================================================


def gpt_oss_20b_pretrain_8gpu_b300_nvfp4_config() -> ConfigContainer:
    """GPT-OSS 20B pretrain: 8× B300, NVFP4."""
    return _gpt_oss_20b_config(
        "nvfp4", "b300", "v1", expert_model_parallel_size=2, global_batch_size=24, micro_batch_size=3
    )


def gpt_oss_20b_pretrain_8gpu_b300_fp8mx_config() -> ConfigContainer:
    """GPT-OSS 20B pretrain: 8× B300, MXFP8."""
    return _gpt_oss_20b_config("fp8_mx", "b300", "v1", global_batch_size=24, micro_batch_size=3)


def gpt_oss_20b_pretrain_64gpu_b300_nvfp4_config() -> ConfigContainer:
    """GPT-OSS 20B pretrain: 64× B300, NVFP4."""
    return _gpt_oss_20b_config(
        "nvfp4",
        "b300",
        "v2",
        context_parallel_size=2,
        expert_model_parallel_size=4,
        global_batch_size=32,
        micro_batch_size=1,
    )


def gpt_oss_20b_pretrain_64gpu_b300_fp8mx_config() -> ConfigContainer:
    """GPT-OSS 20B pretrain: 64× B300, MXFP8."""
    return _gpt_oss_20b_config(
        "fp8_mx",
        "b300",
        "v2",
        context_parallel_size=2,
        expert_model_parallel_size=4,
        global_batch_size=32,
        micro_batch_size=1,
    )


# =============================================================================
# GPT-OSS 20B pretrain — GB200
# =============================================================================


def gpt_oss_20b_pretrain_8gpu_gb200_nvfp4_config() -> ConfigContainer:
    """GPT-OSS 20B pretrain: 8× GB200, NVFP4."""
    return _gpt_oss_20b_config("nvfp4", "gb200", "v1", expert_model_parallel_size=2)


def gpt_oss_20b_pretrain_8gpu_gb200_fp8mx_config() -> ConfigContainer:
    """GPT-OSS 20B pretrain: 8× GB200, MXFP8."""
    return _gpt_oss_20b_config("fp8_mx", "gb200", "v1")


def gpt_oss_20b_pretrain_72gpu_gb200_nvfp4_config() -> ConfigContainer:
    """GPT-OSS 20B pretrain: 72× GB200, NVFP4."""
    return _gpt_oss_20b_config(
        "nvfp4",
        "gb200",
        "v2",
        context_parallel_size=2,
        expert_model_parallel_size=4,
        global_batch_size=36,
        micro_batch_size=1,
    )


def gpt_oss_20b_pretrain_72gpu_gb200_fp8mx_config() -> ConfigContainer:
    """GPT-OSS 20B pretrain: 72× GB200, MXFP8."""
    return _gpt_oss_20b_config(
        "fp8_mx",
        "gb200",
        "v2",
        context_parallel_size=2,
        expert_model_parallel_size=4,
        global_batch_size=36,
        micro_batch_size=1,
    )


def gpt_oss_20b_pretrain_512gpu_gb200_fp8mx_v3_config() -> ConfigContainer:
    """GPT-OSS 20B pretrain: 512× GB200, MXFP8, v3."""
    return _gpt_oss_20b_config(
        "fp8_mx",
        "gb200",
        "v3",
        tensor_model_parallel_size=2,
        context_parallel_size=4,
        expert_model_parallel_size=8,
        global_batch_size=64,
        micro_batch_size=1,
    )


# =============================================================================
# GPT-OSS 20B pretrain — GB300
# =============================================================================


def gpt_oss_20b_pretrain_8gpu_gb300_nvfp4_config() -> ConfigContainer:
    """GPT-OSS 20B pretrain: 8× GB300, NVFP4."""
    return _gpt_oss_20b_config(
        "nvfp4", "gb300", "v1", expert_model_parallel_size=2, global_batch_size=24, micro_batch_size=3
    )


def gpt_oss_20b_pretrain_8gpu_gb300_fp8mx_config() -> ConfigContainer:
    """GPT-OSS 20B pretrain: 8× GB300, MXFP8."""
    return _gpt_oss_20b_config("fp8_mx", "gb300", "v1", global_batch_size=24, micro_batch_size=3)


def gpt_oss_20b_pretrain_72gpu_gb300_nvfp4_config() -> ConfigContainer:
    """GPT-OSS 20B pretrain: 72× GB300, NVFP4."""
    return _gpt_oss_20b_config(
        "nvfp4",
        "gb300",
        "v2",
        context_parallel_size=2,
        expert_model_parallel_size=4,
        global_batch_size=36,
        micro_batch_size=1,
    )


def gpt_oss_20b_pretrain_72gpu_gb300_fp8mx_config() -> ConfigContainer:
    """GPT-OSS 20B pretrain: 72× GB300, MXFP8."""
    return _gpt_oss_20b_config(
        "fp8_mx",
        "gb300",
        "v2",
        context_parallel_size=2,
        expert_model_parallel_size=4,
        global_batch_size=36,
        micro_batch_size=1,
    )


def gpt_oss_20b_pretrain_512gpu_gb300_fp8mx_v3_config() -> ConfigContainer:
    """GPT-OSS 20B pretrain: 512× GB300, MXFP8, v3."""
    return _gpt_oss_20b_config(
        "fp8_mx",
        "gb300",
        "v3",
        tensor_model_parallel_size=2,
        context_parallel_size=4,
        expert_model_parallel_size=8,
        global_batch_size=64,
        micro_batch_size=1,
    )


# =============================================================================
# GPT-OSS 20B pretrain — VR200
# =============================================================================


def gpt_oss_20b_pretrain_8gpu_vr200_nvfp4_config() -> ConfigContainer:
    """GPT-OSS 20B pretrain: 8× VR200, NVFP4."""
    return _gpt_oss_20b_config("nvfp4", "vr200", "v1", global_batch_size=24, micro_batch_size=3)


def gpt_oss_20b_pretrain_8gpu_vr200_fp8mx_config() -> ConfigContainer:
    """GPT-OSS 20B pretrain: 8× VR200, MXFP8."""
    return _gpt_oss_20b_config("fp8_mx", "vr200", "v1", global_batch_size=24, micro_batch_size=3)


def gpt_oss_20b_pretrain_64gpu_vr200_nvfp4_config() -> ConfigContainer:
    """GPT-OSS 20B pretrain: 64× VR200, NVFP4."""
    return _gpt_oss_20b_config(
        "nvfp4",
        "vr200",
        "v2",
        context_parallel_size=2,
        expert_model_parallel_size=4,
        global_batch_size=32,
        micro_batch_size=1,
    )


# =============================================================================
# GPT-OSS 120B pretrain (GBS=1280) — 64 GPU, GB300
# =============================================================================


def gpt_oss_120b_pretrain_64gpu_gb300_bf16_config() -> ConfigContainer:
    """GPT-OSS 120B pretrain: 64× GB300, BF16, GBS=1280."""
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

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# GPT-OSS 120B pretrain (GBS=1280) — 64 GPU, GB200
# =============================================================================


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


# =============================================================================
# GPT-OSS 120B pretrain (GBS=1280) — 64 GPU, B300
# =============================================================================


def gpt_oss_120b_pretrain_64gpu_b300_bf16_config() -> ConfigContainer:
    """GPT-OSS 120B pretrain: 64× B300, BF16, GBS=1280."""
    cfg = gpt_oss_120b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.model.moe_router_fusion = True
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 1280
    cfg.train.micro_batch_size = 4

    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_hybridep_num_sms = 32

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# GPT-OSS 120B pretrain (GBS=1280) — 64 GPU, B200
# =============================================================================


def gpt_oss_120b_pretrain_64gpu_b200_bf16_config() -> ConfigContainer:
    """GPT-OSS 120B pretrain: 64× B200, BF16, GBS=1280."""
    cfg = gpt_oss_120b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.model.moe_router_fusion = True
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 1280
    cfg.train.micro_batch_size = 4

    cfg.model.recompute_modules = ["layernorm", "moe_act"]
    cfg.model.recompute_granularity = "selective"

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# GPT-OSS 120B pretrain (GBS=1280) — 64 GPU, H100
# =============================================================================


def gpt_oss_120b_pretrain_64gpu_h100_bf16_config() -> ConfigContainer:
    """GPT-OSS 120B pretrain: 64× H100, BF16, PP=4 EP=8, GBS=1280."""
    cfg = gpt_oss_120b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.model.moe_router_fusion = True
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.expert_model_parallel_size = 8
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 1280
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = ["layernorm", "moe_act"]
    cfg.model.recompute_granularity = "selective"

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# GPT-OSS 120B — FP8-MX variants: same parallelism as BF16, MXFP8 precision
# =============================================================================


def gpt_oss_120b_pretrain_64gpu_gb300_fp8mx_config() -> ConfigContainer:
    """GPT-OSS 120B pretrain: 64× GB300, FP8-MX."""
    cfg = gpt_oss_120b_pretrain_64gpu_gb300_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    _apply_full_iter_fp8mx_overrides(cfg, expert_model_parallel_size=16)
    return cfg


def gpt_oss_120b_pretrain_64gpu_gb200_fp8mx_config() -> ConfigContainer:
    """GPT-OSS 120B pretrain: 64× GB200, FP8-MX."""
    cfg = gpt_oss_120b_pretrain_64gpu_gb200_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    _apply_full_iter_fp8mx_overrides(cfg, expert_model_parallel_size=64, clear_recompute=True)
    return cfg


def gpt_oss_120b_pretrain_64gpu_b300_fp8mx_config() -> ConfigContainer:
    """GPT-OSS 120B pretrain: 64× B300, FP8-MX."""
    cfg = gpt_oss_120b_pretrain_64gpu_b300_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    return cfg


def gpt_oss_120b_pretrain_64gpu_b200_fp8mx_config() -> ConfigContainer:
    """GPT-OSS 120B pretrain: 64× B200, FP8-MX."""
    cfg = gpt_oss_120b_pretrain_64gpu_b200_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    return cfg


# =============================================================================
# GPT-OSS 120B — FP8-CS variants: same parallelism as BF16, FP8 current-scaling
# =============================================================================


def gpt_oss_120b_pretrain_64gpu_h100_fp8cs_config() -> ConfigContainer:
    """GPT-OSS 120B pretrain: 64× H100, FP8-CS."""
    cfg = gpt_oss_120b_pretrain_64gpu_h100_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    return cfg
