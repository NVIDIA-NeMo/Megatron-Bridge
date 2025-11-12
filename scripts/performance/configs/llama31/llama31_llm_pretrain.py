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

from megatron.bridge.recipes.llama import llama31_405b_pretrain_config
from megatron.bridge.training.comm_overlap import (
    userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192,
    userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192,
    userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192,
    userbuffers_fp8_h100_h16384_tp8_cp2_mbs1_seqlen8192,
)
from megatron.bridge.training.config import ConfigContainer

from . import workload_base_configs as base_cfgs


logger = logging.getLogger(__name__)


def set_llama31_common_configs(cfg: ConfigContainer) -> None:
    """Set common performance configurations for all Llama3.1 configs."""
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False


def llama31_405b_gb300_config(precision: str = "bf16") -> ConfigContainer:
    """GB300, baseline config."""
    if precision == "bf16":
        base_cfg = base_cfgs.LLAMA31_405B_GB300_BF16_BASE_CONFIG
        precision_config = get_precision_config(precision)
        comm_overlap_cfg = userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192
    else:
        base_cfg = base_cfgs.LLAMA31_405B_GB300_FP8_CS_BASE_CONFIG
        if precision == "fp8_mx":
            base_cfg = base_cfgs.LLAMA31_405B_GB300_FP8_MX_BASE_CONFIG
        precision_config = get_precision_config(precision)
        comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192

    cfg = llama31_405b_pretrain_config(mock=True, precision_config=precision_config)
    set_llama31_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    if cfg.ddp.use_megatron_fsdp:
        cfg.ddp.fsdp_double_buffer = True

    cfg.comm_overlap.tp_comm_overlap_cfg = comm_overlap_cfg

    return cfg


def llama31_405b_gb200_config(precision: str = "bf16") -> ConfigContainer:
    """GB200, baseline config."""
    if precision == "bf16":
        base_cfg = base_cfgs.LLAMA31_405B_GB200_BF16_BASE_CONFIG
        precision_config = get_precision_config(precision)
        comm_overlap_cfg = userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192
    else:
        base_cfg = base_cfgs.LLAMA31_405B_GB200_FP8_CS_BASE_CONFIG
        if precision == "fp8_mx":
            base_cfg = base_cfgs.LLAMA31_405B_GB200_FP8_MX_BASE_CONFIG
        precision_config = get_precision_config(precision)
        comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192

    cfg = llama31_405b_pretrain_config(mock=True, precision_config=precision_config)
    set_llama31_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    if cfg.ddp.use_megatron_fsdp:
        cfg.ddp.fsdp_double_buffer = True

    cfg.comm_overlap.tp_comm_overlap_cfg = comm_overlap_cfg

    return cfg


def llama31_405b_b200_config(precision: str = "bf16") -> ConfigContainer:
    """B200, baseline config."""
    if precision == "bf16":
        base_cfg = base_cfgs.LLAMA31_405B_B200_BF16_BASE_CONFIG
        precision_config = get_precision_config(precision)
        comm_overlap_cfg = userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192
    else:
        base_cfg = base_cfgs.LLAMA31_405B_B200_FP8_CS_BASE_CONFIG
        if precision == "fp8_mx":
            base_cfg = base_cfgs.LLAMA31_405B_B200_FP8_MX_BASE_CONFIG
        precision_config = get_precision_config(precision)
        comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192

    cfg = llama31_405b_pretrain_config(mock=True, precision_config=precision_config)
    set_llama31_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap.tp_comm_overlap_cfg = comm_overlap_cfg

    return cfg


def llama31_405b_h100_config(precision: str = "bf16") -> ConfigContainer:
    """H100, baseline config."""
    if precision == "bf16":
        base_cfg = base_cfgs.LLAMA31_405B_H100_BF16_BASE_CONFIG
        precision_config = get_precision_config(precision)
        comm_overlap_cfg = userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192
    else:
        base_cfg = base_cfgs.LLAMA31_405B_H100_FP8_CS_BASE_CONFIG
        precision_config = get_precision_config(precision)
        comm_overlap_cfg = userbuffers_fp8_h100_h16384_tp8_cp2_mbs1_seqlen8192

    cfg = llama31_405b_pretrain_config(mock=True, precision_config=precision_config)
    set_llama31_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap.tp_comm_overlap_cfg = comm_overlap_cfg

    return cfg
