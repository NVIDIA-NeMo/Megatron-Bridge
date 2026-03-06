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

import logging

from utils.overrides import set_workload_base_configs
from utils.precision import get_precision_config
from utils.utils import get_workload_base_config

from megatron.bridge.recipes.kimi.kimi_k2 import _get_kimi_k2_pipeline_layout
from megatron.bridge.recipes.kimi.kimi_k2 import kimi_k2_pretrain_config as pretrain_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.flex_dispatcher_backend import apply_flex_dispatcher_backend


logger = logging.getLogger(__name__)


def set_kimi_k2_common_configs(cfg: ConfigContainer) -> None:
    """Set common performance configurations for all Kimi-K2 configs."""
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    # WAR: MXFP8's fp8_param_gather and reuse_grad_buf_for_mxfp8_param_ag require
    # DistributedOptimizer infrastructure, incompatible with Muon's LayerWiseDistributedOptimizer.
    # Only disable for Muon + MXFP8; for Adam leave them on.
    if (
        cfg.optimizer.optimizer == "dist_muon"
        and cfg.mixed_precision is not None
        and cfg.mixed_precision.fp8_recipe == "mxfp8"
    ):
        cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
        cfg.mixed_precision.fp8_param_gather = False

    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = "selective"
    cfg.dist.enable_megatron_core_experimental = True

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.model.moe_router_force_load_balancing = True
    cfg.model.qk_clip = False  # disable qk_clip for now, enable after MCORE fix drop in


def kimi_k2_pretrain_config_gb300(
    precision: str = "bf16",
    mock: bool = True,
    config_variant: str = "v1",
    optimizer_type: str = "muon",
) -> ConfigContainer:
    """GB300, baseline config. optimizer_type: 'adam' or 'muon' (default)."""
    base_cfg = get_workload_base_config(
        model_family_name="kimi",
        model_recipe_name="kimi_k2",
        gpu="gb300",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )

    cfg = pretrain_config(optimizer_type=optimizer_type)
    precision_config = get_precision_config(precision)
    cfg.mixed_precision = precision_config

    if base_cfg.moe_flex_dispatcher_backend is not None:
        cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend
    apply_flex_dispatcher_backend(cfg.model, cfg.model.moe_flex_dispatcher_backend)

    if base_cfg.pp_layout:
        cfg.model.pipeline_model_parallel_layout = base_cfg.pp_layout
    else:
        # Recompute layout based on updated PP/VP sizes
        pp_size = base_cfg.pipeline_model_parallel_size
        vp_size = base_cfg.virtual_pipeline_model_parallel_size
        layout = _get_kimi_k2_pipeline_layout(pp_size, vp_size)
        cfg.model.pipeline_model_parallel_layout = layout

    set_kimi_k2_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap.overlap_grad_reduce = True
    cfg.rng.te_rng_tracker = True

    # Setting num_workers and pin_memory to 0 and False respectively gives better performance.
    # we are debugging this and might change this in the future.
    cfg.dataset.num_workers = 0
    cfg.dataset.pin_memory = False

    return cfg


def kimi_k2_pretrain_config_gb200(
    precision: str = "bf16",
    mock: bool = True,
    config_variant: str = "v1",
    optimizer_type: str = "muon",
) -> ConfigContainer:
    """GB200, baseline config. optimizer_type: 'adam' or 'muon' (default)."""
    base_cfg = get_workload_base_config(
        model_family_name="kimi",
        model_recipe_name="kimi_k2",
        gpu="gb200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )

    cfg = pretrain_config(optimizer_type=optimizer_type)
    precision_config = get_precision_config(precision)
    cfg.mixed_precision = precision_config

    if base_cfg.moe_flex_dispatcher_backend is not None:
        cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend
    apply_flex_dispatcher_backend(cfg.model, cfg.model.moe_flex_dispatcher_backend)

    if base_cfg.pp_layout:
        cfg.model.pipeline_model_parallel_layout = base_cfg.pp_layout
    else:
        # Recompute layout based on updated PP/VP sizes
        pp_size = base_cfg.pipeline_model_parallel_size
        vp_size = base_cfg.virtual_pipeline_model_parallel_size
        layout = _get_kimi_k2_pipeline_layout(pp_size, vp_size)
        cfg.model.pipeline_model_parallel_layout = layout

    set_kimi_k2_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap.overlap_grad_reduce = True

    # Setting num_workers and pin_memory to 0 and False respectively gives better performance.
    # we are debugging this and might change this in the future.
    cfg.dataset.num_workers = 0
    cfg.dataset.pin_memory = False

    return cfg


def kimi_k2_pretrain_config_b200(
    precision: str = "bf16",
    mock: bool = True,
    config_variant: str = "v1",
    optimizer_type: str = "muon",
) -> ConfigContainer:
    """B200, baseline config. optimizer_type: 'adam' or 'muon' (default)."""
    base_cfg = get_workload_base_config(
        model_family_name="kimi",
        model_recipe_name="kimi_k2",
        gpu="b200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )

    cfg = pretrain_config(optimizer_type=optimizer_type)
    precision_config = get_precision_config(precision)
    cfg.mixed_precision = precision_config

    if base_cfg.moe_flex_dispatcher_backend is not None:
        cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend
    apply_flex_dispatcher_backend(cfg.model, cfg.model.moe_flex_dispatcher_backend)

    if base_cfg.pp_layout:
        cfg.model.pipeline_model_parallel_layout = base_cfg.pp_layout
    else:
        # Recompute layout based on updated PP/VP sizes
        pp_size = base_cfg.pipeline_model_parallel_size
        vp_size = base_cfg.virtual_pipeline_model_parallel_size
        layout = _get_kimi_k2_pipeline_layout(pp_size, vp_size)
        cfg.model.pipeline_model_parallel_layout = layout

    set_kimi_k2_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap.overlap_grad_reduce = True

    return cfg


def kimi_k2_pretrain_config_h100(
    precision: str = "bf16",
    mock: bool = True,
    config_variant: str = "v1",
    optimizer_type: str = "muon",
) -> ConfigContainer:
    """H100, baseline config. optimizer_type: 'adam' or 'muon' (default)."""
    base_cfg = get_workload_base_config(
        model_family_name="kimi",
        model_recipe_name="kimi_k2",
        gpu="h100",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )

    cfg = pretrain_config(optimizer_type=optimizer_type)
    precision_config = get_precision_config(precision)
    cfg.mixed_precision = precision_config

    if base_cfg.moe_flex_dispatcher_backend is not None:
        cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend
    apply_flex_dispatcher_backend(cfg.model, cfg.model.moe_flex_dispatcher_backend)

    if base_cfg.pp_layout:
        cfg.model.pipeline_model_parallel_layout = base_cfg.pp_layout
    else:
        # Recompute layout based on updated PP/VP sizes
        pp_size = base_cfg.pipeline_model_parallel_size
        vp_size = base_cfg.virtual_pipeline_model_parallel_size
        layout = _get_kimi_k2_pipeline_layout(pp_size, vp_size)
        cfg.model.pipeline_model_parallel_layout = layout

    set_kimi_k2_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    # Disabling to avoid functional errors. TODO: Test with it enabled and keep it enabled if it works.
    cfg.comm_overlap.overlap_grad_reduce = False

    return cfg
