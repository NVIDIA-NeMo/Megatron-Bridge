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

from megatron.bridge.recipes.llama import llama3_8b_pretrain_config, llama3_70b_pretrain_config
from megatron.bridge.training.comm_overlap import (
    CommOverlapConfig,
    userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192,
    userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
    userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192,
    userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192,
)
from megatron.bridge.training.config import ConfigContainer

from .llama3_workload_base_configs import (
    LLAMA3_8B_PRETRAIN_CONFIG_B200_BF16,
    LLAMA3_8B_PRETRAIN_CONFIG_B200_FP8_CS,
    LLAMA3_8B_PRETRAIN_CONFIG_B200_FP8_MX,
    LLAMA3_8B_PRETRAIN_CONFIG_B200_NVFP4,
    LLAMA3_8B_PRETRAIN_CONFIG_B300_BF16,
    LLAMA3_8B_PRETRAIN_CONFIG_B300_FP8_CS,
    LLAMA3_8B_PRETRAIN_CONFIG_B300_FP8_MX,
    LLAMA3_8B_PRETRAIN_CONFIG_B300_NVFP4,
    LLAMA3_8B_PRETRAIN_CONFIG_GB200_BF16,
    LLAMA3_8B_PRETRAIN_CONFIG_GB200_FP8_CS,
    LLAMA3_8B_PRETRAIN_CONFIG_GB200_FP8_MX,
    LLAMA3_8B_PRETRAIN_CONFIG_GB200_NVFP4,
    LLAMA3_8B_PRETRAIN_CONFIG_GB300_BF16,
    LLAMA3_8B_PRETRAIN_CONFIG_GB300_FP8_CS,
    LLAMA3_8B_PRETRAIN_CONFIG_GB300_FP8_MX,
    LLAMA3_8B_PRETRAIN_CONFIG_GB300_NVFP4,
    LLAMA3_8B_PRETRAIN_CONFIG_H100_BF16,
    LLAMA3_8B_PRETRAIN_CONFIG_H100_FP8_CS,
    LLAMA3_70B_PRETRAIN_CONFIG_B200_BF16,
    LLAMA3_70B_PRETRAIN_CONFIG_B200_FP8_CS,
    LLAMA3_70B_PRETRAIN_CONFIG_B200_FP8_MX,
    LLAMA3_70B_PRETRAIN_CONFIG_B200_NVFP4,
    LLAMA3_70B_PRETRAIN_CONFIG_B300_BF16,
    LLAMA3_70B_PRETRAIN_CONFIG_B300_FP8_CS,
    LLAMA3_70B_PRETRAIN_CONFIG_B300_FP8_MX,
    LLAMA3_70B_PRETRAIN_CONFIG_B300_NVFP4,
    LLAMA3_70B_PRETRAIN_CONFIG_GB200_BF16,
    LLAMA3_70B_PRETRAIN_CONFIG_GB200_FP8_CS,
    LLAMA3_70B_PRETRAIN_CONFIG_GB200_FP8_MX,
    LLAMA3_70B_PRETRAIN_CONFIG_GB200_NVFP4,
    LLAMA3_70B_PRETRAIN_CONFIG_GB300_BF16,
    LLAMA3_70B_PRETRAIN_CONFIG_GB300_FP8_CS,
    LLAMA3_70B_PRETRAIN_CONFIG_GB300_FP8_MX,
    LLAMA3_70B_PRETRAIN_CONFIG_GB300_NVFP4,
    LLAMA3_70B_PRETRAIN_CONFIG_H100_BF16,
    LLAMA3_70B_PRETRAIN_CONFIG_H100_FP8_CS,
)


logger = logging.getLogger(__name__)


def set_llama3_common_configs(cfg: ConfigContainer) -> None:
    """Set common performance configurations for all Llama3 configs."""
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False


# Llama3 70B configs ---------------------------------------------------------


def llama3_70b_pretrain_config_gb300(precision: str = "bf16", mock: bool = True) -> ConfigContainer:
    """GB300, baseline config."""
    if precision == "bf16":
        base_cfg = LLAMA3_70B_PRETRAIN_CONFIG_GB300_BF16
        precision_config = get_precision_config(precision)
        comm_overlap_cfg = userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192
    else:
        base_cfg = LLAMA3_70B_PRETRAIN_CONFIG_GB300_FP8_CS
        if precision == "fp8_mx":
            base_cfg = LLAMA3_70B_PRETRAIN_CONFIG_GB300_FP8_MX
        elif precision == "nvfp4":
            base_cfg = LLAMA3_70B_PRETRAIN_CONFIG_GB300_NVFP4
        precision_config = get_precision_config(precision)
        comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192

    cfg = llama3_70b_pretrain_config(mock=mock, precision_config=precision_config)
    set_llama3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    if cfg.ddp.use_megatron_fsdp:
        cfg.ddp.fsdp_double_buffer = True
        cfg.model.gradient_accumulation_fusion = False  # Disabled to avoid functional errors
        cfg.ddp.suggested_communication_unit_size = 800000000

    cfg.comm_overlap.tp_comm_overlap_cfg = comm_overlap_cfg
    cfg.comm_overlap.tp_comm_overlap = False if precision == "nvfp4" else cfg.comm_overlap.tp_comm_overlap

    if precision == "fp8_mx":  # keeping this eanbled causes NaN grad norm
        cfg.comm_overlap.overlap_param_gather = False
        cfg.ddp.overlap_param_gather = False
        cfg.optimizer.overlap_param_gather = False

    return cfg


def llama3_70b_pretrain_config_gb200(precision: str = "bf16", mock: bool = True) -> ConfigContainer:
    """GB200, baseline config."""
    if precision == "bf16":
        base_cfg = LLAMA3_70B_PRETRAIN_CONFIG_GB200_BF16
        precision_config = get_precision_config(precision)
        comm_overlap_cfg = userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192
    else:
        base_cfg = LLAMA3_70B_PRETRAIN_CONFIG_GB200_FP8_CS
        if precision == "fp8_mx":
            base_cfg = LLAMA3_70B_PRETRAIN_CONFIG_GB200_FP8_MX
        elif precision == "nvfp4":
            base_cfg = LLAMA3_70B_PRETRAIN_CONFIG_GB200_NVFP4
        precision_config = get_precision_config(precision)
        comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192

    cfg = llama3_70b_pretrain_config(mock=mock, precision_config=precision_config)
    set_llama3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    if cfg.ddp.use_megatron_fsdp:
        cfg.ddp.fsdp_double_buffer = True
        cfg.model.gradient_accumulation_fusion = False  # Disabled to avoid functional errors
        cfg.ddp.suggested_communication_unit_size = 800000000

    cfg.comm_overlap.tp_comm_overlap_cfg = comm_overlap_cfg
    cfg.comm_overlap.tp_comm_overlap = False if precision == "nvfp4" else cfg.comm_overlap.tp_comm_overlap

    if precision == "fp8_mx":  # keeping this eanbled causes NaN grad norm
        cfg.comm_overlap.overlap_param_gather = False
        cfg.ddp.overlap_param_gather = False
        cfg.optimizer.overlap_param_gather = False

    return cfg


def llama3_70b_pretrain_config_b300(precision: str = "bf16", mock: bool = True) -> ConfigContainer:
    """B300, baseline config."""
    if precision == "bf16":
        base_cfg = LLAMA3_70B_PRETRAIN_CONFIG_B300_BF16
        precision_config = get_precision_config(precision)
        comm_overlap_cfg = userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192
    else:
        base_cfg = LLAMA3_70B_PRETRAIN_CONFIG_B300_FP8_CS
        if precision == "fp8_mx":
            base_cfg = LLAMA3_70B_PRETRAIN_CONFIG_B300_FP8_MX
        elif precision == "nvfp4":
            base_cfg = LLAMA3_70B_PRETRAIN_CONFIG_B300_NVFP4
        precision_config = get_precision_config(precision)
        comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192

    cfg = llama3_70b_pretrain_config(mock=mock, precision_config=precision_config)
    set_llama3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    if cfg.ddp.use_megatron_fsdp:
        cfg.ddp.fsdp_double_buffer = True
        cfg.model.gradient_accumulation_fusion = False  # Disabled to avoid functional errors
        cfg.ddp.suggested_communication_unit_size = 800000000

    cfg.comm_overlap.tp_comm_overlap_cfg = comm_overlap_cfg
    cfg.comm_overlap.tp_comm_overlap = False if precision == "nvfp4" else cfg.comm_overlap.tp_comm_overlap

    if precision == "fp8_mx":  # keeping this eanbled causes NaN grad norm
        cfg.comm_overlap.overlap_param_gather = False
        cfg.ddp.overlap_param_gather = False
        cfg.optimizer.overlap_param_gather = False

    return cfg


def llama3_70b_pretrain_config_b200(precision: str = "bf16", mock: bool = True) -> ConfigContainer:
    """B200, baseline config."""
    if precision == "bf16":
        base_cfg = LLAMA3_70B_PRETRAIN_CONFIG_B200_BF16
        precision_config = get_precision_config(precision)
        comm_overlap_cfg = userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192
    else:
        base_cfg = LLAMA3_70B_PRETRAIN_CONFIG_B200_FP8_CS
        if precision == "fp8_mx":
            base_cfg = LLAMA3_70B_PRETRAIN_CONFIG_B200_FP8_MX
        elif precision == "nvfp4":
            base_cfg = LLAMA3_70B_PRETRAIN_CONFIG_B200_NVFP4
        precision_config = get_precision_config(precision)
        comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192

    cfg = llama3_70b_pretrain_config(mock=mock, precision_config=precision_config)
    set_llama3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    if cfg.ddp.use_megatron_fsdp:
        cfg.ddp.fsdp_double_buffer = True
        cfg.model.gradient_accumulation_fusion = False  # Disabled to avoid functional errors
        cfg.ddp.suggested_communication_unit_size = 800000000

    cfg.comm_overlap.tp_comm_overlap_cfg = comm_overlap_cfg
    cfg.comm_overlap.tp_comm_overlap = False if precision == "nvfp4" else cfg.comm_overlap.tp_comm_overlap

    if precision == "fp8_mx":  # keeping this eanbled causes NaN grad norm
        cfg.comm_overlap.overlap_param_gather = False
        cfg.ddp.overlap_param_gather = False
        cfg.optimizer.overlap_param_gather = False

    return cfg


def llama3_70b_pretrain_config_h100(precision: str = "bf16", mock: bool = True) -> ConfigContainer:
    """H100, baseline config."""
    if precision == "bf16":
        base_cfg = LLAMA3_70B_PRETRAIN_CONFIG_H100_BF16
        precision_config = get_precision_config(precision)
        comm_overlap_cfg = userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192
    else:
        base_cfg = LLAMA3_70B_PRETRAIN_CONFIG_H100_FP8_CS
        precision_config = get_precision_config(precision)
        comm_overlap_cfg = userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192

    cfg = llama3_70b_pretrain_config(mock=mock, precision_config=precision_config)
    set_llama3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap.tp_comm_overlap_cfg = comm_overlap_cfg

    return cfg


# Llama3 8B configs ---------------------------------------------------------


def llama3_8b_pretrain_config_gb300(precision: str = "bf16", mock: bool = True) -> ConfigContainer:
    """GB300, baseline config."""
    if precision == "bf16":
        base_cfg = LLAMA3_8B_PRETRAIN_CONFIG_GB300_BF16
        precision_config = get_precision_config(precision)
    else:
        base_cfg = LLAMA3_8B_PRETRAIN_CONFIG_GB300_FP8_CS
        if precision == "fp8_mx":
            base_cfg = LLAMA3_8B_PRETRAIN_CONFIG_GB300_FP8_MX
        elif precision == "nvfp4":
            base_cfg = LLAMA3_8B_PRETRAIN_CONFIG_GB300_NVFP4
        precision_config = get_precision_config(precision)

    cfg = llama3_8b_pretrain_config(mock=mock, precision_config=precision_config)
    set_llama3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1))
    cfg.comm_overlap.tp_comm_overlap = False if precision == "nvfp4" else cfg.comm_overlap.tp_comm_overlap

    return cfg


def llama3_8b_pretrain_config_gb200(precision: str = "bf16", mock: bool = True) -> ConfigContainer:
    """GB200, baseline config."""
    if precision == "bf16":
        base_cfg = LLAMA3_8B_PRETRAIN_CONFIG_GB200_BF16
        precision_config = get_precision_config(precision)
    else:
        base_cfg = LLAMA3_8B_PRETRAIN_CONFIG_GB200_FP8_CS
        if precision == "fp8_mx":
            base_cfg = LLAMA3_8B_PRETRAIN_CONFIG_GB200_FP8_MX
        elif precision == "nvfp4":
            base_cfg = LLAMA3_8B_PRETRAIN_CONFIG_GB200_NVFP4
        precision_config = get_precision_config(precision)

    cfg = llama3_8b_pretrain_config(mock=mock, precision_config=precision_config)
    set_llama3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1))
    cfg.comm_overlap.tp_comm_overlap = False if precision == "nvfp4" else cfg.comm_overlap.tp_comm_overlap

    return cfg


def llama3_8b_pretrain_config_b300(precision: str = "bf16", mock: bool = True) -> ConfigContainer:
    """B300, baseline config."""
    if precision == "bf16":
        base_cfg = LLAMA3_8B_PRETRAIN_CONFIG_B300_BF16
        precision_config = get_precision_config(precision)
    else:
        base_cfg = LLAMA3_8B_PRETRAIN_CONFIG_B300_FP8_CS
        if precision == "fp8_mx":
            base_cfg = LLAMA3_8B_PRETRAIN_CONFIG_B300_FP8_MX
        elif precision == "nvfp4":
            base_cfg = LLAMA3_8B_PRETRAIN_CONFIG_B300_NVFP4
        precision_config = get_precision_config(precision)

    cfg = llama3_8b_pretrain_config(mock=mock, precision_config=precision_config)
    set_llama3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1))
    cfg.comm_overlap.tp_comm_overlap = False if precision == "nvfp4" else cfg.comm_overlap.tp_comm_overlap

    return cfg


def llama3_8b_pretrain_config_b200(precision: str = "bf16", mock: bool = True) -> ConfigContainer:
    """B200, baseline config."""
    if precision == "bf16":
        base_cfg = LLAMA3_8B_PRETRAIN_CONFIG_B200_BF16
        precision_config = get_precision_config(precision)
    else:
        base_cfg = LLAMA3_8B_PRETRAIN_CONFIG_B200_FP8_CS
        if precision == "fp8_mx":
            base_cfg = LLAMA3_8B_PRETRAIN_CONFIG_B200_FP8_MX
        elif precision == "nvfp4":
            base_cfg = LLAMA3_8B_PRETRAIN_CONFIG_B200_NVFP4
        precision_config = get_precision_config(precision)

    cfg = llama3_8b_pretrain_config(mock=mock, precision_config=precision_config)
    set_llama3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1))
    cfg.comm_overlap.tp_comm_overlap = False if precision == "nvfp4" else cfg.comm_overlap.tp_comm_overlap

    return cfg


def llama3_8b_pretrain_config_h100(precision: str = "bf16", mock: bool = True) -> ConfigContainer:
    """H100, baseline config."""
    if precision == "bf16":
        base_cfg = LLAMA3_8B_PRETRAIN_CONFIG_H100_BF16
        precision_config = get_precision_config(precision)
    else:
        base_cfg = LLAMA3_8B_PRETRAIN_CONFIG_H100_FP8_CS
        precision_config = get_precision_config(precision)

    cfg = llama3_8b_pretrain_config(mock=mock, precision_config=precision_config)
    set_llama3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1))

    if cfg.ddp.use_megatron_fsdp:
        cfg.ddp.nccl_ub = True
        cfg.model.gradient_accumulation_fusion = False  # Disabled to avoid functional errors
        cfg.ddp.keep_fp8_transpose_cache = True

    return cfg
