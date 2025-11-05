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

from megatron.bridge.recipes.qwen.qwen3_moe import qwen3_30b_a3b_pretrain_config, qwen3_235b_a22b_pretrain_config
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.utils.moe_token_drop import apply_moe_token_drop

from . import workload_base_configs as base_cfgs


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

    cfg.model = apply_moe_token_drop(cfg.model)

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False


def qwen3_235b_a22b_gb300_64gpus_config(precision: str = "bf16", fp8_recipe: str = "cs") -> ConfigContainer:
    """GB300, 64xGPU, baseline config."""
    if precision == "bf16":
        base_cfg = base_cfgs.QWEN3_235B_A22B_GB300_64GPUS_BF16_PARALLEL_CONFIG
        precision_config = get_precision_config(precision)
    else:
        base_cfg = base_cfgs.QWEN3_235B_A22B_GB300_64GPUS_FP8_CS_PARALLEL_CONFIG
        if fp8_recipe == "mx":
            base_cfg = base_cfgs.QWEN3_235B_A22B_GB300_64GPUS_FP8_MX_PARALLEL_CONFIG
        precision_config = get_precision_config(precision, fp8_recipe)

    cfg = qwen3_235b_a22b_pretrain_config(
        mock=True,
        precision_config=precision_config,
        comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
    )
    set_qwen3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def qwen3_235b_a22b_gb200_64gpus_config(precision: str = "bf16", fp8_recipe: str = "cs") -> ConfigContainer:
    """GB200, 64xGPU, baseline config."""
    if precision == "bf16":
        base_cfg = base_cfgs.QWEN3_235B_A22B_GB200_64GPUS_BF16_PARALLEL_CONFIG
        precision_config = get_precision_config(precision)
    else:
        base_cfg = base_cfgs.QWEN3_235B_A22B_GB200_64GPUS_FP8_CS_PARALLEL_CONFIG
        if fp8_recipe == "mx":
            base_cfg = base_cfgs.QWEN3_235B_A22B_GB200_64GPUS_FP8_MX_PARALLEL_CONFIG
        precision_config = get_precision_config(precision, fp8_recipe)

    cfg = qwen3_235b_a22b_pretrain_config(
        mock=True,
        precision_config=precision_config,
        comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
    )
    set_qwen3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def qwen3_235b_a22b_b200_64gpus_config(precision: str = "bf16", fp8_recipe: str = "cs") -> ConfigContainer:
    """B200, 64xGPU, baseline config."""
    if precision == "bf16":
        base_cfg = base_cfgs.QWEN3_235B_A22B_B200_64GPUS_BF16_PARALLEL_CONFIG
        precision_config = get_precision_config(precision)
    else:
        base_cfg = base_cfgs.QWEN3_235B_A22B_B200_64GPUS_FP8_CS_PARALLEL_CONFIG
        if fp8_recipe == "mx":
            base_cfg = base_cfgs.QWEN3_235B_A22B_B200_64GPUS_FP8_MX_PARALLEL_CONFIG
        precision_config = get_precision_config(precision, fp8_recipe)

    cfg = qwen3_235b_a22b_pretrain_config(
        mock=True,
        precision_config=precision_config,
        comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
    )
    set_qwen3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def qwen3_235b_a22b_h100_256gpus_config(precision: str = "bf16", fp8_recipe: str = "cs") -> ConfigContainer:
    """H100, 256xGPU, baseline config."""
    if precision == "bf16":
        base_cfg = base_cfgs.QWEN3_235B_A22B_H100_256GPUS_BF16_PARALLEL_CONFIG
        precision_config = get_precision_config(precision)
    else:
        base_cfg = base_cfgs.QWEN3_235B_A22B_H100_256GPUS_FP8_CS_PARALLEL_CONFIG
        precision_config = get_precision_config(precision, fp8_recipe)

    cfg = qwen3_235b_a22b_pretrain_config(
        mock=True,
        precision_config=precision_config,
        comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
    )
    set_qwen3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def qwen3_30b_a3b_gb300_8gpus_config(precision: str = "bf16", fp8_recipe: str = "cs") -> ConfigContainer:
    """GB300, 8xGPU, baseline config."""
    if precision == "bf16":
        base_cfg = base_cfgs.QWEN3_30B_A3B_GB300_8GPUS_BF16_PARALLEL_CONFIG
        precision_config = get_precision_config(precision)
    else:
        base_cfg = base_cfgs.QWEN3_30B_A3B_GB300_8GPUS_FP8_CS_PARALLEL_CONFIG
        if fp8_recipe == "mx":
            base_cfg = base_cfgs.QWEN3_30B_A3B_GB300_8GPUS_FP8_MX_PARALLEL_CONFIG
        precision_config = get_precision_config(precision, fp8_recipe)

    cfg = qwen3_30b_a3b_pretrain_config(
        mock=True,
        precision_config=precision_config,
        comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
    )
    set_qwen3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def qwen3_30b_a3b_gb200_8gpus_config(precision: str = "bf16", fp8_recipe: str = "cs") -> ConfigContainer:
    """GB200, 8xGPU, baseline config."""
    if precision == "bf16":
        base_cfg = base_cfgs.QWEN3_30B_A3B_GB200_8GPUS_BF16_PARALLEL_CONFIG
        precision_config = get_precision_config(precision)
    else:
        base_cfg = base_cfgs.QWEN3_30B_A3B_GB200_8GPUS_FP8_CS_PARALLEL_CONFIG
        if fp8_recipe == "mx":
            base_cfg = base_cfgs.QWEN3_30B_A3B_GB200_8GPUS_FP8_MX_PARALLEL_CONFIG
        precision_config = get_precision_config(precision, fp8_recipe)

    cfg = qwen3_30b_a3b_pretrain_config(
        mock=True,
        precision_config=precision_config,
        comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
    )
    set_qwen3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def qwen3_30b_a3b_b200_8gpus_config(precision: str = "bf16", fp8_recipe: str = "cs") -> ConfigContainer:
    """B200, 8xGPU, baseline config."""
    if precision == "bf16":
        base_cfg = base_cfgs.QWEN3_30B_A3B_B200_8GPUS_BF16_PARALLEL_CONFIG
        precision_config = get_precision_config(precision)
    else:
        base_cfg = base_cfgs.QWEN3_30B_A3B_B200_8GPUS_FP8_CS_PARALLEL_CONFIG
        if fp8_recipe == "mx":
            base_cfg = base_cfgs.QWEN3_30B_A3B_B200_8GPUS_FP8_MX_PARALLEL_CONFIG
        precision_config = get_precision_config(precision, fp8_recipe)

    cfg = qwen3_30b_a3b_pretrain_config(
        mock=True,
        precision_config=precision_config,
        comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
    )
    set_qwen3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def qwen3_30b_a3b_h100_16gpus_config(precision: str = "bf16", fp8_recipe: str = "cs") -> ConfigContainer:
    """H100, 16xGPU, baseline config."""
    if precision == "bf16":
        base_cfg = base_cfgs.QWEN3_30B_A3B_H100_16GPUS_BF16_PARALLEL_CONFIG
        precision_config = get_precision_config(precision)
    else:
        base_cfg = base_cfgs.QWEN3_30B_A3B_H100_16GPUS_FP8_CS_PARALLEL_CONFIG
        precision_config = get_precision_config(precision, fp8_recipe)

    cfg = qwen3_30b_a3b_pretrain_config(
        mock=True,
        precision_config=precision_config,
        comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
    )
    set_qwen3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg
