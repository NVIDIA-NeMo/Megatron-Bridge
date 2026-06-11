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

import logging

from utils.overrides import set_workload_base_configs
from utils.precision import get_precision_config
from utils.utils import get_workload_base_config

from megatron.bridge.recipes.gpt_oss import gpt_oss_20b_pretrain_config, gpt_oss_120b_pretrain_config
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.flex_dispatcher_backend import apply_flex_dispatcher_backend
from megatron.bridge.utils.cuda_graph import is_full_iteration_cuda_graph


logger = logging.getLogger(__name__)


def set_gpt_oss_common_configs(cfg: ConfigContainer) -> None:
    """Set common performance configurations for all GPT-OSS configs."""
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.model.moe_router_fusion = True
    cfg.model.moe_router_force_load_balancing = True


def set_full_iter_cg_configs(cfg: ConfigContainer) -> None:
    """Apply defaults required by full-iteration CUDA graph capture with dropless MoE.

    Dropless MoE produces variable-shaped per-expert tensors that CG cannot
    capture; we pad to a fixed capacity (pad_experts + capacity factor) and use
    MCore PR #4247 paged stashing to recover memory. Callers should gate on
    `is_full_iteration_cuda_graph(cfg.model)`.
    """
    cfg.model.moe_pad_experts_for_cuda_graph_inference = True
    cfg.model.moe_paged_stash = True
    if cfg.model.moe_expert_rank_capacity_factor is None:
        cfg.model.moe_expert_rank_capacity_factor = 1.5
    cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.2
    cfg.model.moe_paged_stash_buffer_size_factor_cpu = 1.0
    cfg.model.cuda_graph_warmup_steps = 2
    if cfg.model.offload_modules is None:
        cfg.model.offload_modules = []


def set_gpt_oss_20b_common_configs(cfg: ConfigContainer) -> None:
    """Set common performance configurations for all GPT-OSS 20b configs."""
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


def get_gpt_oss_20b_precision_config(compute_dtype: str):
    """Get the precision configs for the given compute dtype and FP8 recipe."""
    precision_config = get_precision_config(compute_dtype)
    precision_config.fp4_param = False
    precision_config.fp4_param_gather = False
    precision_config.fp8_param = False
    precision_config.fp8_param_gather = False
    precision_config.reuse_grad_buf_for_mxfp8_param_ag = False
    if compute_dtype == "fp8_mx":
        precision_config.first_last_layers_bf16 = False
        precision_config.num_layers_at_start_in_bf16 = 0
    elif compute_dtype == "nvfp4":
        precision_config.first_last_layers_bf16 = True
        precision_config.num_layers_at_start_in_bf16 = 0
        precision_config.num_layers_at_end_in_bf16 = 4
    return precision_config


def gpt_oss_20b_pretrain_config_b300(
    precision: str = "nvfp4", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B300, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="gpt_oss",
        model_recipe_name="gpt_oss_20b",
        gpu="b300",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    cfg = gpt_oss_20b_pretrain_config()
    set_workload_base_configs(cfg, base_cfg)
    precision_config = get_gpt_oss_20b_precision_config(precision)
    cfg.mixed_precision = precision_config
    if base_cfg.moe_flex_dispatcher_backend is not None:
        apply_flex_dispatcher_backend(cfg.model, base_cfg.moe_flex_dispatcher_backend)
    set_gpt_oss_20b_common_configs(cfg)

    # 8 GPUs
    if precision == "nvfp4" and config_variant == "v1":
        cfg.model.cuda_graph_impl = "transformer_engine"
        cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
        cfg.optimizer.lr = 0.0004
        cfg.optimizer.min_lr = 0.0004
        cfg.validation.eval_interval = 512
        cfg.validation.eval_iters = 43
        cfg.scheduler.lr_warmup_iters = 192
    elif precision == "fp8_mx" and config_variant == "v1":
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
    # 64 GPUs
    elif precision == "nvfp4" and config_variant == "v2":
        cfg.model.cuda_graph_impl = "transformer_engine"
        cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
        cfg.optimizer.lr = 0.0006
        cfg.optimizer.min_lr = 0.0006
        cfg.validation.eval_interval = 384
        cfg.validation.eval_iters = 32
        cfg.scheduler.lr_warmup_iters = 64
    elif precision == "fp8_mx" and config_variant == "v2":
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

    return cfg


def gpt_oss_20b_pretrain_config_gb200(
    precision: str = "nvfp4", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB200, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="gpt_oss",
        model_recipe_name="gpt_oss_20b",
        gpu="gb200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    cfg = gpt_oss_20b_pretrain_config()
    set_workload_base_configs(cfg, base_cfg)
    precision_config = get_gpt_oss_20b_precision_config(precision)
    cfg.mixed_precision = precision_config
    if base_cfg.moe_flex_dispatcher_backend is not None:
        apply_flex_dispatcher_backend(cfg.model, base_cfg.moe_flex_dispatcher_backend)
    set_gpt_oss_20b_common_configs(cfg)

    # 8 GPUs
    if precision == "nvfp4" and config_variant == "v1":
        cfg.model.cuda_graph_impl = "transformer_engine"
        cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
        cfg.optimizer.lr = 0.0006
        cfg.optimizer.min_lr = 0.0006
        cfg.validation.eval_interval = 768
        cfg.validation.eval_iters = 64
        cfg.scheduler.lr_warmup_iters = 128
    elif precision == "fp8_mx" and config_variant == "v1":
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
    # 72 GPUs
    elif precision == "nvfp4" and config_variant == "v2":
        cfg.model.cuda_graph_impl = "transformer_engine"
        cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
        cfg.optimizer.lr = 0.0006
        cfg.optimizer.min_lr = 0.0004
        cfg.validation.eval_interval = 341
        cfg.validation.eval_iters = 29
        cfg.scheduler.lr_warmup_iters = 64
    elif precision == "fp8_mx" and config_variant == "v2":
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
    # 512 GPUs
    elif precision == "fp8_mx" and config_variant == "v3":
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

    return cfg


def gpt_oss_20b_pretrain_config_gb300(
    precision: str = "nvfp4", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB300, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="gpt_oss",
        model_recipe_name="gpt_oss_20b",
        gpu="gb300",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    cfg = gpt_oss_20b_pretrain_config()
    set_workload_base_configs(cfg, base_cfg)
    precision_config = get_gpt_oss_20b_precision_config(precision)
    cfg.mixed_precision = precision_config
    if base_cfg.moe_flex_dispatcher_backend is not None:
        apply_flex_dispatcher_backend(cfg.model, base_cfg.moe_flex_dispatcher_backend)
    set_gpt_oss_20b_common_configs(cfg)

    # 8 GPUs
    if precision == "nvfp4" and config_variant == "v1":
        cfg.model.cuda_graph_impl = "transformer_engine"
        cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
        cfg.optimizer.lr = 0.0004
        cfg.optimizer.min_lr = 0.0004
        cfg.validation.eval_interval = 512
        cfg.validation.eval_iters = 43
        cfg.scheduler.lr_warmup_iters = 192
    elif precision == "fp8_mx" and config_variant == "v1":
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
    # 72 GPUs
    elif precision == "nvfp4" and config_variant == "v2":
        cfg.model.cuda_graph_impl = "transformer_engine"
        cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
        cfg.optimizer.lr = 0.0006
        cfg.optimizer.min_lr = 0.0006
        cfg.validation.eval_interval = 341
        cfg.validation.eval_iters = 29
        cfg.scheduler.lr_warmup_iters = 64
    elif precision == "fp8_mx" and config_variant == "v2":
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
    # 512 GPUs
    elif precision == "fp8_mx" and config_variant == "v3":
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

    return cfg


def gpt_oss_20b_pretrain_config_vr200(
    precision: str = "nvfp4", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """VR200, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="gpt_oss",
        model_recipe_name="gpt_oss_20b",
        gpu="vr200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    cfg = gpt_oss_20b_pretrain_config()
    set_workload_base_configs(cfg, base_cfg)
    precision_config = get_gpt_oss_20b_precision_config(precision)
    cfg.mixed_precision = precision_config
    if base_cfg.moe_flex_dispatcher_backend is not None:
        apply_flex_dispatcher_backend(cfg.model, base_cfg.moe_flex_dispatcher_backend)
    set_gpt_oss_20b_common_configs(cfg)

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
    cfg.model.cuda_graph_warmup_steps = 1

    # 8 GPUs
    if precision == "nvfp4" and config_variant == "v1":
        cfg.optimizer.lr = 0.0004
        cfg.optimizer.min_lr = 0.0004
        cfg.validation.eval_interval = 512
        cfg.validation.eval_iters = 43
        cfg.scheduler.lr_warmup_iters = 192
    elif precision == "fp8_mx" and config_variant == "v1":
        cfg.optimizer.lr = 0.0005
        cfg.optimizer.min_lr = 0.0005
        cfg.validation.eval_interval = 512
        cfg.validation.eval_iters = 43
        cfg.scheduler.lr_warmup_iters = 192
    # 64 GPUs
    elif precision == "nvfp4" and config_variant == "v2":
        cfg.optimizer.lr = 0.0006
        cfg.optimizer.min_lr = 0.0006
        cfg.validation.eval_interval = 384
        cfg.validation.eval_iters = 43
        cfg.scheduler.lr_warmup_iters = 64

    return cfg


def gpt_oss_120b_pretrain_config_gb300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB300, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="gpt_oss",
        model_recipe_name="gpt_oss_120b",
        gpu="gb300",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = gpt_oss_120b_pretrain_config()
    cfg.mixed_precision = precision_config
    if base_cfg.moe_flex_dispatcher_backend is not None:
        apply_flex_dispatcher_backend(cfg.model, base_cfg.moe_flex_dispatcher_backend)
    set_gpt_oss_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    if is_full_iteration_cuda_graph(cfg.model):
        set_full_iter_cg_configs(cfg)

    if cfg.mixed_precision.fp8_recipe == "mxfp8":
        cfg.model.fp8_output_proj = True

    return cfg


def gpt_oss_120b_pretrain_config_gb200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB200, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="gpt_oss",
        model_recipe_name="gpt_oss_120b",
        gpu="gb200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = gpt_oss_120b_pretrain_config()
    cfg.mixed_precision = precision_config
    if base_cfg.moe_flex_dispatcher_backend is not None:
        apply_flex_dispatcher_backend(cfg.model, base_cfg.moe_flex_dispatcher_backend)
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=bool(base_cfg.tensor_model_parallel_size > 1))
    cfg.comm_overlap.tp_comm_overlap = False if precision == "nvfp4" else cfg.comm_overlap.tp_comm_overlap
    set_gpt_oss_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    if is_full_iteration_cuda_graph(cfg.model):
        set_full_iter_cg_configs(cfg)

    if cfg.mixed_precision.fp8_recipe == "mxfp8":
        cfg.model.fp8_output_proj = True

    return cfg


def gpt_oss_120b_pretrain_config_vr200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """VR200, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="gpt_oss",
        model_recipe_name="gpt_oss_120b",
        gpu="vr200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = gpt_oss_120b_pretrain_config()
    cfg.mixed_precision = precision_config
    if base_cfg.moe_flex_dispatcher_backend is not None:
        apply_flex_dispatcher_backend(cfg.model, base_cfg.moe_flex_dispatcher_backend)
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=bool(base_cfg.tensor_model_parallel_size > 1))
    cfg.comm_overlap.tp_comm_overlap = False if precision == "nvfp4" else cfg.comm_overlap.tp_comm_overlap
    set_gpt_oss_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def gpt_oss_120b_pretrain_config_b300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B300, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="gpt_oss",
        model_recipe_name="gpt_oss_120b",
        gpu="b300",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = gpt_oss_120b_pretrain_config()
    cfg.mixed_precision = precision_config
    if base_cfg.moe_flex_dispatcher_backend is not None:
        apply_flex_dispatcher_backend(cfg.model, base_cfg.moe_flex_dispatcher_backend)
    set_gpt_oss_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    if is_full_iteration_cuda_graph(cfg.model):
        set_full_iter_cg_configs(cfg)

    return cfg


def gpt_oss_120b_pretrain_config_b200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B200, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="gpt_oss",
        model_recipe_name="gpt_oss_120b",
        gpu="b200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = gpt_oss_120b_pretrain_config()
    cfg.mixed_precision = precision_config
    if base_cfg.moe_flex_dispatcher_backend is not None:
        apply_flex_dispatcher_backend(cfg.model, base_cfg.moe_flex_dispatcher_backend)
    set_gpt_oss_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    if is_full_iteration_cuda_graph(cfg.model):
        set_full_iter_cg_configs(cfg)

    return cfg


def gpt_oss_120b_pretrain_config_h100(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """H100, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="gpt_oss",
        model_recipe_name="gpt_oss_120b",
        gpu="h100",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = gpt_oss_120b_pretrain_config()
    cfg.mixed_precision = precision_config
    if base_cfg.moe_flex_dispatcher_backend is not None:
        apply_flex_dispatcher_backend(cfg.model, base_cfg.moe_flex_dispatcher_backend)
    set_gpt_oss_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg
