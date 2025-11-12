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

from utils.helpers import (
    get_precision_config,
    set_moe_a2a_overlap_overrides,
    set_workload_base_configs,
)

from megatron.bridge.recipes.deepseek.deepseek_v3 import deepseek_v3_pretrain_config as pretrain_config
from megatron.bridge.training.config import ConfigContainer

from . import workload_base_configs as base_cfgs


logger = logging.getLogger(__name__)


def set_deepseek_v3_common_configs(cfg: ConfigContainer) -> None:
    """Set common performance configurations for all DeepSeek-V3 configs."""
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = "selective"
    cfg.dist.enable_megatron_core_experimental = True

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.model.moe_router_force_load_balancing = True


def deepseek_v3_gb300_config(precision: str = "bf16") -> ConfigContainer:
    """GB300, baseline config."""
    if precision == "bf16":
        base_cfg = base_cfgs.DEEPSEEK_V3_GB300_BF16_BASE_CONFIG
        precision_config = get_precision_config(precision)
    else:
        base_cfg = base_cfgs.DEEPSEEK_V3_GB300_FP8_CS_BASE_CONFIG
        if precision == "fp8_mx":
            base_cfg = base_cfgs.DEEPSEEK_V3_GB300_FP8_MX_BASE_CONFIG
        precision_config = get_precision_config(precision)

    cfg = pretrain_config(
        mock=True,
        precision_config=precision_config,
        pipeline_model_parallel_size=base_cfg.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=base_cfg.virtual_pipeline_model_parallel_size,
        enable_deepep=False,
        layout=None,
    )
    set_deepseek_v3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap.overlap_grad_reduce = True

    # Setting num_workers and pin_memory to 0 and False respectively gives better performance.
    # we are debugging this and might change this in the future.
    cfg.dataset.num_workers = 0
    cfg.dataset.pin_memory = False

    return cfg


def deepseek_v3_gb200_config(precision: str = "bf16") -> ConfigContainer:
    """GB200, baseline config."""
    if precision == "bf16":
        base_cfg = base_cfgs.DEEPSEEK_V3_GB200_BF16_BASE_CONFIG
        precision_config = get_precision_config(precision)
    else:
        base_cfg = base_cfgs.DEEPSEEK_V3_GB200_FP8_CS_BASE_CONFIG
        if precision == "fp8_mx":
            base_cfg = base_cfgs.DEEPSEEK_V3_GB200_FP8_MX_BASE_CONFIG
        precision_config = get_precision_config(precision)

    cfg = pretrain_config(
        mock=True,
        precision_config=precision_config,
        pipeline_model_parallel_size=base_cfg.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=base_cfg.virtual_pipeline_model_parallel_size,
        enable_deepep=False,
        layout=None,
    )
    set_deepseek_v3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap.overlap_grad_reduce = True

    # Setting num_workers and pin_memory to 0 and False respectively gives better performance.
    # we are debugging this and might change this in the future.
    cfg.dataset.num_workers = 0
    cfg.dataset.pin_memory = False

    return cfg


def deepseek_v3_b200_config(precision: str = "bf16") -> ConfigContainer:
    """B200, baseline config."""
    if precision == "bf16":
        base_cfg = base_cfgs.DEEPSEEK_V3_B200_BF16_BASE_CONFIG
        precision_config = get_precision_config(precision)
    else:
        base_cfg = base_cfgs.DEEPSEEK_V3_B200_FP8_CS_BASE_CONFIG
        if precision == "fp8_mx":
            base_cfg = base_cfgs.DEEPSEEK_V3_B200_FP8_MX_BASE_CONFIG
        precision_config = get_precision_config(precision)

    cfg = pretrain_config(
        mock=True,
        precision_config=precision_config,
        pipeline_model_parallel_size=base_cfg.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=base_cfg.virtual_pipeline_model_parallel_size,
        enable_deepep=False,
        layout=None,
    )
    set_deepseek_v3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap.overlap_grad_reduce = True

    return cfg


def deepseek_v3_h100_config(precision: str = "bf16") -> ConfigContainer:
    """H100, baseline config."""
    if precision == "bf16":
        base_cfg = base_cfgs.DEEPSEEK_V3_H100_BF16_BASE_CONFIG
        precision_config = get_precision_config(precision)
    else:
        base_cfg = base_cfgs.DEEPSEEK_V3_H100_FP8_CS_BASE_CONFIG
        if precision == "fp8_sc":
            base_cfg = base_cfgs.DEEPSEEK_V3_H100_FP8_SC_BASE_CONFIG
        precision_config = get_precision_config(precision)

    cfg = pretrain_config(
        mock=True,
        precision_config=precision_config,
        pipeline_model_parallel_size=base_cfg.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=base_cfg.virtual_pipeline_model_parallel_size,
        enable_deepep=True,
        layout="Et|(tt|)*30mL",
    )
    set_deepseek_v3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    set_moe_a2a_overlap_overrides(cfg)

    # Disabling to avoid functional errors. TODO: Test with it enabled and keep it enabled if it works.
    cfg.comm_overlap.overlap_grad_reduce = False

    return cfg
