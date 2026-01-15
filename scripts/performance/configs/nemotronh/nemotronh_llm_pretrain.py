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

from megatron.bridge.recipes.nemotronh import nemotronh_56b_pretrain_config
from megatron.bridge.training.config import ConfigContainer

from .nemotronh_workload_base_configs import (
    NEMOTRONH_56B_PRETRAIN_CONFIG_B200_FP8_CS,
    NEMOTRONH_56B_PRETRAIN_CONFIG_B300_FP8_CS,
    NEMOTRONH_56B_PRETRAIN_CONFIG_GB200_FP8_CS,
    NEMOTRONH_56B_PRETRAIN_CONFIG_GB300_FP8_CS,
    NEMOTRONH_56B_PRETRAIN_CONFIG_H100_FP8_CS,
)


logger = logging.getLogger(__name__)


def set_nemotronh_common_configs(cfg: ConfigContainer) -> None:
    """Set common performance configurations for all NemotronH configs."""
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False


def nemotronh_56b_pretrain_config_gb300(precision: str = "bf16", mock: bool = True) -> ConfigContainer:
    """GB300, baseline config."""
    base_cfg = NEMOTRONH_56B_PRETRAIN_CONFIG_GB300_FP8_CS
    precision_config = get_precision_config(precision)

    cfg = nemotronh_56b_pretrain_config(
        mock=mock,
        precision_config=precision_config,
    )
    set_nemotronh_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def nemotronh_56b_pretrain_config_gb200(precision: str = "bf16", mock: bool = True) -> ConfigContainer:
    """GB200, baseline config."""
    base_cfg = NEMOTRONH_56B_PRETRAIN_CONFIG_GB200_FP8_CS
    precision_config = get_precision_config(precision)

    cfg = nemotronh_56b_pretrain_config(
        mock=mock,
        precision_config=precision_config,
    )
    set_nemotronh_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def nemotronh_56b_pretrain_config_b300(precision: str = "bf16", mock: bool = True) -> ConfigContainer:
    """B300, baseline config."""
    base_cfg = NEMOTRONH_56B_PRETRAIN_CONFIG_B300_FP8_CS
    precision_config = get_precision_config(precision)

    cfg = nemotronh_56b_pretrain_config(
        mock=mock,
        precision_config=precision_config,
    )
    set_nemotronh_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def nemotronh_56b_pretrain_config_b200(precision: str = "bf16", mock: bool = True) -> ConfigContainer:
    """B200, baseline config."""
    base_cfg = NEMOTRONH_56B_PRETRAIN_CONFIG_B200_FP8_CS
    precision_config = get_precision_config(precision)

    cfg = nemotronh_56b_pretrain_config(
        mock=mock,
        precision_config=precision_config,
    )
    set_nemotronh_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def nemotronh_56b_pretrain_config_h100(precision: str = "bf16", mock: bool = True) -> ConfigContainer:
    """H100, baseline config."""
    precision_config = get_precision_config(precision)

    base_cfg = NEMOTRONH_56B_PRETRAIN_CONFIG_H100_FP8_CS
    cfg = nemotronh_56b_pretrain_config(
        mock=mock,
        precision_config=precision_config,
    )
    set_nemotronh_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg
