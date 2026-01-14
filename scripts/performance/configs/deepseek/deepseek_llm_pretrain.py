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

from megatron.bridge.recipes.deepseek.deepseek_v3 import deepseek_v3_pretrain_config as pretrain_config
from megatron.bridge.training.config import ConfigContainer


logger = logging.getLogger(__name__)


def set_deepseek_v3_common_configs(cfg: ConfigContainer, moe_a2a_overlap: bool = False) -> None:
    """Set common performance configurations for all DeepSeek-V3 configs."""
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = "selective"
    cfg.dist.enable_megatron_core_experimental = True

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.model.moe_router_force_load_balancing = True


def deepseek_v3_pretrain_config_gb300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB300, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v3",
        gpu="gb300",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = pretrain_config(
        mock=mock,
        precision_config=precision_config,
        pipeline_model_parallel_size=base_cfg.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=base_cfg.virtual_pipeline_model_parallel_size,
        moe_flex_dispatcher_backend=base_cfg.moe_flex_dispatcher_backend,
        layout=None,
    )
    set_deepseek_v3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap.overlap_grad_reduce = True

    # Setting num_workers and pin_memory to 0 and False respectively gives better performance.
    # we are debugging this and might change this in the future.
    cfg.dataset.num_workers = 0
    cfg.dataset.pin_memory = False

    if precision == "fp8_mx":  # keeping this eanbled causes NaN grad norm
        cfg.comm_overlap.overlap_param_gather = False
        cfg.ddp.overlap_param_gather = False
        cfg.optimizer.overlap_param_gather = False

    return cfg


def deepseek_v3_pretrain_config_gb200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB200, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v3",
        gpu="gb200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = pretrain_config(
        mock=mock,
        precision_config=precision_config,
        pipeline_model_parallel_size=base_cfg.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=base_cfg.virtual_pipeline_model_parallel_size,
        moe_flex_dispatcher_backend=base_cfg.moe_flex_dispatcher_backend,
        layout=None,
    )
    set_deepseek_v3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap.overlap_grad_reduce = True

    # Setting num_workers and pin_memory to 0 and False respectively gives better performance.
    # we are debugging this and might change this in the future.
    cfg.dataset.num_workers = 0
    cfg.dataset.pin_memory = False

    if precision == "fp8_mx":  # keeping this eanbled causes NaN grad norm
        cfg.comm_overlap.overlap_param_gather = False
        cfg.ddp.overlap_param_gather = False
        cfg.optimizer.overlap_param_gather = False

    return cfg


def deepseek_v3_pretrain_config_b300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B300, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v3",
        gpu="b300",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = pretrain_config(
        mock=mock,
        precision_config=precision_config,
        pipeline_model_parallel_size=base_cfg.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=base_cfg.virtual_pipeline_model_parallel_size,
        moe_flex_dispatcher_backend=base_cfg.moe_flex_dispatcher_backend,
        layout=None,
    )
    set_deepseek_v3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap.overlap_grad_reduce = True

    return cfg


def deepseek_v3_pretrain_config_b200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B200, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v3",
        gpu="b200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = pretrain_config(
        mock=mock,
        precision_config=precision_config,
        pipeline_model_parallel_size=base_cfg.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=base_cfg.virtual_pipeline_model_parallel_size,
        moe_flex_dispatcher_backend=base_cfg.moe_flex_dispatcher_backend,
        layout=None,
    )
    set_deepseek_v3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap.overlap_grad_reduce = True

    return cfg


def deepseek_v3_pretrain_config_h100(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """H100, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v3",
        gpu="h100",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = pretrain_config(
        mock=mock,
        precision_config=precision_config,
        pipeline_model_parallel_size=base_cfg.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=base_cfg.virtual_pipeline_model_parallel_size,
        moe_flex_dispatcher_backend=base_cfg.moe_flex_dispatcher_backend,
        layout="Et|(tt|)*30mL",
    )
    set_deepseek_v3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    # Disabling to avoid functional errors. TODO: Test with it enabled and keep it enabled if it works.
    cfg.comm_overlap.overlap_grad_reduce = False

    return cfg
