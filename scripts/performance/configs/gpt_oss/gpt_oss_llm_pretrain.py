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


logger = logging.getLogger(__name__)


def set_gpt_oss_common_configs(cfg: ConfigContainer) -> None:
    """Set common performance configurations for all GPT-OSS configs."""
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.model.moe_router_fusion = True

    cfg.model.moe_router_force_load_balancing = True

def gpt_oss_20b_pretrain_config_b200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B200, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="gpt_oss",
        model_recipe_name="gpt_oss_20b",
        gpu="b200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = gpt_oss_20b_pretrain_config()
    cfg.mixed_precision = precision_config
    if base_cfg.moe_flex_dispatcher_backend is not None:
        apply_flex_dispatcher_backend(cfg.model, base_cfg.moe_flex_dispatcher_backend)

    # 8 GPUs
    if precision == "bf16" and config_variant == "v1":
        cfg.model.use_te_rng_tracker = True
        cfg.model.cuda_graph_impl = "transformer_engine"
        cfg.model.cuda_graph_scope = ["attn","moe_router","moe_preprocess"]
        cfg.optimizer.lr = 0.0006
        cfg.validation.eval_interval = 768
        cfg.scheduler.lr_warmup_iters = 128
    elif precision == "fp8_mx" and config_variant == "v1":
        cfg.model.use_transformer_engine_op_fuser = True
        cfg.model.moe_expert_rank_capacity_factor = 1.2
        cfg.mixed_precision.fp8_param_gather = True
        cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = True
        cfg.model.cuda_graph_warmup_steps = 5
        cfg.optimizer.lr = 0.0004
        cfg.validation.eval_interval = 768
        cfg.scheduler.lr_warmup_iters = 128
    # 64 GPUs
    elif precision == "bf16" and config_variant == "v2":
        cfg.model.use_te_rng_tracker = True
        cfg.model.cuda_graph_impl = "transformer_engine"
        cfg.model.cuda_graph_scope = ["attn","moe_router","moe_preprocess"]
        cfg.optimizer.lr = 0.0006
        cfg.validation.eval_interval = 384
        cfg.scheduler.lr_warmup_iters = 64
    elif precision == "fp8_mx" and config_variant == "v2":
        cfg.model.use_transformer_engine_op_fuser = True
        cfg.model.moe_expert_rank_capacity_factor = 5
        cfg.optimizer.lr = 0.0004
        cfg.validation.eval_interval = 384
        cfg.scheduler.lr_warmup_iters = 512

    set_gpt_oss_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg

def gpt_oss_20b_pretrain_config_b300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
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
    precision_config = get_precision_config(precision)

    cfg = gpt_oss_20b_pretrain_config()
    cfg.mixed_precision = precision_config
    if base_cfg.moe_flex_dispatcher_backend is not None:
        apply_flex_dispatcher_backend(cfg.model, base_cfg.moe_flex_dispatcher_backend)

    # 8 GPUs
    if precision == "bf16" and config_variant == "v1":
        cfg.model.use_te_rng_tracker = True
        cfg.model.cuda_graph_impl = "transformer_engine"
        cfg.model.cuda_graph_scope = ["attn","moe_router","moe_preprocess"]
        cfg.optimizer.lr = 0.0004
        cfg.validation.eval_interval = 512
        cfg.scheduler.lr_warmup_iters = 192
    elif precision == "fp8_mx" and config_variant == "v1":
        cfg.model.use_transformer_engine_op_fuser = True
        cfg.model.moe_expert_rank_capacity_factor = 1.5
        cfg.optimizer.lr = 0.0005
        cfg.validation.eval_interval = 512
        cfg.scheduler.lr_warmup_iters = 256
    # 64 GPUs
    elif precision == "bf16" and config_variant == "v2":
        cfg.model.use_te_rng_tracker = True
        cfg.model.cuda_graph_impl = "transformer_engine"
        cfg.model.cuda_graph_scope = ["attn","moe_router","moe_preprocess"]
        cfg.optimizer.lr = 0.0006
        cfg.validation.eval_interval = 384
        cfg.scheduler.lr_warmup_iters = 64
    elif precision == "fp8_mx" and config_variant == "v2":
        cfg.model.use_transformer_engine_op_fuser = True
        cfg.model.moe_expert_rank_capacity_factor = 5
        cfg.optimizer.lr = 0.0004
        cfg.validation.eval_interval = 384
        cfg.scheduler.lr_warmup_iters = 512

    set_gpt_oss_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg

def gpt_oss_20b_pretrain_config_gb200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
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
    precision_config = get_precision_config(precision)

    cfg = gpt_oss_20b_pretrain_config()
    cfg.mixed_precision = precision_config
    if base_cfg.moe_flex_dispatcher_backend is not None:
        apply_flex_dispatcher_backend(cfg.model, base_cfg.moe_flex_dispatcher_backend)

    # 8 GPUs
    if precision == "bf16" and config_variant == "v1":
        cfg.model.use_te_rng_tracker = True
        cfg.model.cuda_graph_impl = "transformer_engine"
        cfg.model.cuda_graph_scope = ["attn","moe_router","moe_preprocess"]
        cfg.optimizer.lr = 0.0006
        cfg.validation.eval_interval = 768
        cfg.scheduler.lr_warmup_iters = 128
    elif precision == "fp8_mx" and config_variant == "v1":
        cfg.model.use_transformer_engine_op_fuser = True
        cfg.model.moe_expert_rank_capacity_factor = 1.2
        cfg.mixed_precision.fp8_param_gather = True
        cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = True
        cfg.model.cuda_graph_warmup_steps = 5
        cfg.ddp.average_in_collective = True
        cfg.optimizer.lr = 0.0004
        cfg.validation.eval_interval = 768
        cfg.scheduler.lr_warmup_iters = 128
    # 72 GPUs
    elif precision == "bf16" and config_variant == "v2":
        cfg.model.use_te_rng_tracker = True
        cfg.model.cuda_graph_impl = "transformer_engine"
        cfg.model.cuda_graph_scope = ["attn","moe_router","moe_preprocess"]
        cfg.optimizer.lr = 0.0006
        cfg.validation.eval_interval = 341
        cfg.scheduler.lr_warmup_iters = 64
    elif precision == "fp8_mx" and config_variant == "v2":
        cfg.model.use_transformer_engine_op_fuser = True
        cfg.model.moe_expert_rank_capacity_factor = 5
        cfg.optimizer.lr = 0.0004
        cfg.validation.eval_interval = 341
        cfg.scheduler.lr_warmup_iters = 256
    # 512 GPUs
    elif precision == "fp8_mx" and config_variant == "v3":
        cfg.model.use_transformer_engine_op_fuser = True
        cfg.model.moe_expert_rank_capacity_factor = 7
        cfg.model.sequence_parallel = True
        cfg.optimizer.lr = 0.00052
        cfg.validation.eval_interval = 192
        cfg.scheduler.lr_warmup_iters = 32

    set_gpt_oss_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg

def gpt_oss_20b_pretrain_config_gb300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
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
    precision_config = get_precision_config(precision)

    cfg = gpt_oss_20b_pretrain_config()
    cfg.mixed_precision = precision_config
    if base_cfg.moe_flex_dispatcher_backend is not None:
        apply_flex_dispatcher_backend(cfg.model, base_cfg.moe_flex_dispatcher_backend)

    # 8 GPUs
    if precision == "bf16" and config_variant == "v1":
        cfg.model.use_te_rng_tracker = True
        cfg.model.cuda_graph_impl = "transformer_engine"
        cfg.model.cuda_graph_scope = ["attn","moe_router","moe_preprocess"]
        cfg.optimizer.lr = 0.0004
        cfg.validation.eval_interval = 512
        cfg.scheduler.lr_warmup_iters = 192
    elif precision == "fp8_mx" and config_variant == "v1":
        cfg.model.use_transformer_engine_op_fuser = True
        cfg.ddp.average_in_collective = True
        cfg.optimizer.lr = 0.0005
        cfg.validation.eval_interval = 512
        cfg.scheduler.lr_warmup_iters = 256
    # 72 GPUs
    elif precision == "bf16" and config_variant == "v2":
        cfg.model.use_te_rng_tracker = True
        cfg.model.cuda_graph_impl = "transformer_engine"
        cfg.model.cuda_graph_scope = ["attn","moe_router","moe_preprocess"]
        cfg.optimizer.lr = 0.0006
        cfg.validation.eval_interval = 341
        cfg.scheduler.lr_warmup_iters = 64
    elif precision == "fp8_mx" and config_variant == "v2":
        cfg.model.use_transformer_engine_op_fuser = True
        cfg.model.moe_expert_rank_capacity_factor = 5
        cfg.optimizer.lr = 0.0004
        cfg.validation.eval_interval = 341
        cfg.scheduler.lr_warmup_iters = 256
    # 512 GPUs
    elif precision == "fp8_mx" and config_variant == "v3":
        cfg.model.use_transformer_engine_op_fuser = True
        cfg.model.moe_expert_rank_capacity_factor = 7
        cfg.model.sequence_parallel = True
        cfg.optimizer.lr = 0.00052
        cfg.validation.eval_interval = 192
        cfg.scheduler.lr_warmup_iters = 32

    set_gpt_oss_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg

def gpt_oss_20b_pretrain_config_vr200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
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
    precision_config = get_precision_config(precision)

    cfg = gpt_oss_20b_pretrain_config()
    cfg.mixed_precision = precision_config
    if base_cfg.moe_flex_dispatcher_backend is not None:
        apply_flex_dispatcher_backend(cfg.model, base_cfg.moe_flex_dispatcher_backend)

    # 8 GPUs
    if precision == "bf16" and config_variant == "v1":
        cfg.model.use_te_rng_tracker = True
        cfg.model.cuda_graph_impl = "transformer_engine"
        cfg.model.cuda_graph_scope = ["attn","moe_router","moe_preprocess"]
        cfg.optimizer.lr = 0.0004
        cfg.validation.eval_interval = 512
        cfg.scheduler.lr_warmup_iters = 192
    elif precision == "fp8_mx" and config_variant == "v1":
        cfg.model.use_te_rng_tracker = True
        cfg.model.cuda_graph_impl = "transformer_engine"
        cfg.model.cuda_graph_scope = ["attn","moe_router","moe_preprocess"]
        cfg.optimizer.lr = 0.0005
        cfg.validation.eval_interval = 512
        cfg.scheduler.lr_warmup_iters = 192
    # 64 GPUs
    elif precision == "bf16" and config_variant == "v2":
        cfg.model.use_te_rng_tracker = True
        cfg.model.cuda_graph_impl = "transformer_engine"
        cfg.model.cuda_graph_scope = ["attn","moe_router","moe_preprocess"]
        cfg.optimizer.lr = 0.0006
        cfg.validation.eval_interval = 384
        cfg.scheduler.lr_warmup_iters = 64

    set_gpt_oss_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

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
