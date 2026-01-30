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

from megatron.bridge.recipes.qwen.qwen3_moe import qwen3_30b_a3b_pretrain_config, qwen3_235b_a22b_pretrain_config
from megatron.bridge.recipes.qwen.qwen3_next import qwen3_next_80b_a3b_pretrain_config
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer


logger = logging.getLogger(__name__)


def set_qwen3_common_configs(cfg: ConfigContainer) -> None:
    """Set common performance configurations for all Qwen3 configs."""
    cfg.model.bias_activation_fusion = True
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.moe_router_fusion = True

    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.model.moe_router_force_load_balancing = True  # required for token dropless


def set_qwen3_next_common_configs(cfg: ConfigContainer) -> None:
    """Set common performance configurations for all Qwen3 next configs."""
    cfg.model.bias_activation_fusion = True
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.moe_router_fusion = True

    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.model.moe_router_force_load_balancing = True


def qwen3_235b_a22b_pretrain_config_gb300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB300, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="qwen",
        model_recipe_name="qwen3_235b_a22b",
        gpu="gb300",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = qwen3_235b_a22b_pretrain_config(
        mock=mock,
        precision_config=precision_config,
        comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
        moe_flex_dispatcher_backend=base_cfg.moe_flex_dispatcher_backend,
    )
    set_qwen3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def qwen3_235b_a22b_pretrain_config_gb200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB200, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="qwen",
        model_recipe_name="qwen3_235b_a22b",
        gpu="gb200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = qwen3_235b_a22b_pretrain_config(
        mock=mock,
        precision_config=precision_config,
        comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
        moe_flex_dispatcher_backend=base_cfg.moe_flex_dispatcher_backend,
    )
    set_qwen3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def qwen3_235b_a22b_pretrain_config_b300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B300, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="qwen",
        model_recipe_name="qwen3_235b_a22b",
        gpu="b300",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = qwen3_235b_a22b_pretrain_config(
        mock=mock,
        precision_config=precision_config,
        comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
        moe_flex_dispatcher_backend=base_cfg.moe_flex_dispatcher_backend,
    )
    set_qwen3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    if precision == "fp8_mx":  # keeping this eanbled causes NaN grad norm
        cfg.comm_overlap.overlap_param_gather = False
        cfg.ddp.overlap_param_gather = False
        cfg.optimizer.overlap_param_gather = False

    return cfg


def qwen3_235b_a22b_pretrain_config_b200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B200, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="qwen",
        model_recipe_name="qwen3_235b_a22b",
        gpu="b200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = qwen3_235b_a22b_pretrain_config(
        mock=mock,
        precision_config=precision_config,
        comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
        moe_flex_dispatcher_backend=base_cfg.moe_flex_dispatcher_backend,
    )
    set_qwen3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    if precision == "fp8_mx":  # keeping this eanbled causes NaN grad norm
        cfg.comm_overlap.overlap_param_gather = False
        cfg.ddp.overlap_param_gather = False
        cfg.optimizer.overlap_param_gather = False

    return cfg


def qwen3_235b_a22b_pretrain_config_h100(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """H100, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="qwen",
        model_recipe_name="qwen3_235b_a22b",
        gpu="h100",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = qwen3_235b_a22b_pretrain_config(
        mock=mock,
        precision_config=precision_config,
        comm_overlap_config=CommOverlapConfig(tp_comm_overlap=False),
        moe_flex_dispatcher_backend=base_cfg.moe_flex_dispatcher_backend,
    )
    set_qwen3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def qwen3_30b_a3b_pretrain_config_gb300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB300, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="qwen",
        model_recipe_name="qwen3_30b_a3b",
        gpu="gb300",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = qwen3_30b_a3b_pretrain_config(
        mock=mock,
        precision_config=precision_config,
        comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
        moe_flex_dispatcher_backend=base_cfg.moe_flex_dispatcher_backend,
    )
    set_qwen3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def qwen3_30b_a3b_pretrain_config_gb200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB200, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="qwen",
        model_recipe_name="qwen3_30b_a3b",
        gpu="gb200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = qwen3_30b_a3b_pretrain_config(
        mock=mock,
        precision_config=precision_config,
        comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
        moe_flex_dispatcher_backend=base_cfg.moe_flex_dispatcher_backend,
    )
    set_qwen3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def qwen3_30b_a3b_pretrain_config_b300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B300, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="qwen",
        model_recipe_name="qwen3_30b_a3b",
        gpu="b300",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = qwen3_30b_a3b_pretrain_config(
        mock=mock,
        precision_config=precision_config,
        comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
        moe_flex_dispatcher_backend=base_cfg.moe_flex_dispatcher_backend,
    )
    set_qwen3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def qwen3_30b_a3b_pretrain_config_b200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B200, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="qwen",
        model_recipe_name="qwen3_30b_a3b",
        gpu="b200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = qwen3_30b_a3b_pretrain_config(
        mock=mock,
        precision_config=precision_config,
        comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
        moe_flex_dispatcher_backend=base_cfg.moe_flex_dispatcher_backend,
    )
    set_qwen3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def qwen3_30b_a3b_pretrain_config_h100(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """H100, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="qwen",
        model_recipe_name="qwen3_30b_a3b",
        gpu="h100",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = qwen3_30b_a3b_pretrain_config(
        mock=mock,
        precision_config=precision_config,
        comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
        moe_flex_dispatcher_backend=base_cfg.moe_flex_dispatcher_backend,
    )
    set_qwen3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def qwen3_next_80b_a3b_pretrain_config_gb200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB200, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="qwen",
        model_recipe_name="qwen3_next_80b_a3b",
        gpu="gb200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = qwen3_next_80b_a3b_pretrain_config(
        mock=mock,
        precision_config=precision_config,
        comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
    )
    set_qwen3_next_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def qwen3_next_80b_a3b_pretrain_config_b300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B300, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="qwen",
        model_recipe_name="qwen3_next_80b_a3b",
        gpu="b300",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = qwen3_next_80b_a3b_pretrain_config(
        mock=mock,
        precision_config=precision_config,
        comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
    )
    set_qwen3_next_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def qwen3_next_80b_a3b_pretrain_config_b200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B200, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="qwen",
        model_recipe_name="qwen3_next_80b_a3b",
        gpu="b200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = qwen3_next_80b_a3b_pretrain_config(
        mock=mock,
        precision_config=precision_config,
        comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
    )
    set_qwen3_next_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def qwen3_next_80b_a3b_pretrain_config_gb300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB300, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="qwen",
        model_recipe_name="qwen3_next_80b_a3b",
        gpu="gb300",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = qwen3_next_80b_a3b_pretrain_config(
        mock=mock,
        precision_config=precision_config,
        comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
    )
    set_qwen3_next_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def qwen3_next_80b_a3b_pretrain_config_h100(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """H100, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="qwen",
        model_recipe_name="qwen3_next_80b_a3b",
        gpu="h100",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = qwen3_next_80b_a3b_pretrain_config(
        mock=mock,
        precision_config=precision_config,
        comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
    )
    set_qwen3_next_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg
