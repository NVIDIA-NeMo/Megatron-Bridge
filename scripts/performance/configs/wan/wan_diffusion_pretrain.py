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
from utils.utils import get_workload_base_config

from megatron.bridge.diffusion.recipes.wan.wan import wan_14b_pretrain_config
from megatron.bridge.training.config import ConfigContainer


logger = logging.getLogger(__name__)


# Wan 14B pretrain configs ---------------------------------------------------


def wan_14b_pretrain_config_gb200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB200, Wan 14B pretrain: TP=1, CP=4, GBS=64."""
    base_cfg = get_workload_base_config(
        model_family_name="wan",
        model_recipe_name="wan_14b",
        gpu="gb200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    cfg = wan_14b_pretrain_config()
    set_workload_base_configs(cfg, base_cfg)
    return cfg


def wan_14b_pretrain_config_gb300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB300, Wan 14B pretrain: TP=1, CP=2, GBS=64."""
    base_cfg = get_workload_base_config(
        model_family_name="wan",
        model_recipe_name="wan_14b",
        gpu="gb300",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    cfg = wan_14b_pretrain_config()
    set_workload_base_configs(cfg, base_cfg)
    return cfg


def wan_14b_pretrain_config_h100(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """H100, Wan 14B pretrain: TP=2, CP=4, GBS=128, activation recompute (block/8 layers)."""
    base_cfg = get_workload_base_config(
        model_family_name="wan",
        model_recipe_name="wan_14b",
        gpu="h100",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    cfg = wan_14b_pretrain_config()
    set_workload_base_configs(cfg, base_cfg)
    return cfg
