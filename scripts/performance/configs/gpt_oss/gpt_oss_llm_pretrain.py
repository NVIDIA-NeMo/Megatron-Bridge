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

from megatron.bridge.recipes.gpt_oss import gpt_oss_120b_pretrain_config
from megatron.bridge.training.config import ConfigContainer


logger = logging.getLogger(__name__)


def set_gpt_oss_common_configs(cfg: ConfigContainer) -> None:
    """Set common performance configurations for all GPT-OSS configs."""
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.model.moe_router_fusion = True

    cfg.model.moe_router_force_load_balancing = True


def gpt_oss_120b_pretrain_config_gb300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB300, baseline config."""
    # GPT-OSS currently only has BF16 base configs enabled
    base_cfg = get_workload_base_config(
        model_family_name="gpt_oss",
        model_recipe_name="gpt_oss_120b",
        gpu="gb300",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = gpt_oss_120b_pretrain_config(
        mock=mock,
        precision_config=precision_config,
    )
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

    cfg = gpt_oss_120b_pretrain_config(
        mock=mock,
        precision_config=precision_config,
    )
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

    cfg = gpt_oss_120b_pretrain_config(
        mock=mock,
        precision_config=precision_config,
    )
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

    cfg = gpt_oss_120b_pretrain_config(
        mock=mock,
        precision_config=precision_config,
    )
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

    cfg = gpt_oss_120b_pretrain_config(
        mock=mock,
        precision_config=precision_config,
    )
    set_gpt_oss_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg
