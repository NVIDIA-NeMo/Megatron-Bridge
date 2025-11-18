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
    set_workload_base_configs,
)

from megatron.bridge.recipes.gpt_oss import gpt_oss_120b_pretrain_config
from megatron.bridge.training.config import ConfigContainer

from . import workload_base_configs as base_cfgs


logger = logging.getLogger(__name__)


def set_gpt_oss_common_configs(cfg: ConfigContainer) -> None:
    """Set common performance configurations for all GPT-OSS configs."""
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.model.moe_router_force_load_balancing = True


def gpt_oss_120b_gb300_config(precision: str = "bf16") -> ConfigContainer:
    """GB300, baseline config."""
    if precision == "bf16":
        base_cfg = base_cfgs.GPT_OSS_120B_GB300_BF16_BASE_CONFIG
        precision_config = get_precision_config(precision)
    else:
        base_cfg = base_cfgs.GPT_OSS_120B_GB300_FP8_MX_BASE_CONFIG
        precision_config = get_precision_config(precision)

    cfg = gpt_oss_120b_pretrain_config(
        mock=True,
        precision_config=precision_config,
    )
    set_gpt_oss_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def gpt_oss_120b_gb200_config(precision: str = "bf16") -> ConfigContainer:
    """GB200, baseline config."""
    if precision == "bf16":
        base_cfg = base_cfgs.GPT_OSS_120B_GB200_BF16_BASE_CONFIG
        precision_config = get_precision_config(precision)
    else:
        base_cfg = base_cfgs.GPT_OSS_120B_GB200_FP8_MX_BASE_CONFIG
        precision_config = get_precision_config(precision)

    cfg = gpt_oss_120b_pretrain_config(
        mock=True,
        precision_config=precision_config,
    )
    set_gpt_oss_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def gpt_oss_120b_b200_config(precision: str = "bf16") -> ConfigContainer:
    """B200, baseline config."""
    if precision == "bf16":
        base_cfg = base_cfgs.GPT_OSS_120B_B200_BF16_BASE_CONFIG
        precision_config = get_precision_config(precision)
    else:
        base_cfg = base_cfgs.GPT_OSS_120B_B200_FP8_MX_BASE_CONFIG
        precision_config = get_precision_config(precision)

    cfg = gpt_oss_120b_pretrain_config(
        mock=True,
        precision_config=precision_config,
    )
    set_gpt_oss_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def gpt_oss_120b_h100_config(precision: str = "bf16") -> ConfigContainer:
    """H100, baseline config."""
    if precision == "bf16":
        base_cfg = base_cfgs.GPT_OSS_120B_H100_BF16_BASE_CONFIG
        precision_config = get_precision_config(precision)
    else:
        base_cfg = base_cfgs.GPT_OSS_120B_H100_FP8_CS_BASE_CONFIG
        precision_config = get_precision_config(precision)

    cfg = gpt_oss_120b_pretrain_config(
        mock=True,
        precision_config=precision_config,
    )
    set_gpt_oss_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg
