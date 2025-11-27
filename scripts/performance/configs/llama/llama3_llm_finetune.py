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

from megatron.bridge.recipes.llama import llama3_8b_finetune_config, llama3_70b_finetune_config
from megatron.bridge.training.comm_overlap import (
    CommOverlapConfig,
)
from megatron.bridge.training.config import ConfigContainer

from . import workload_base_configs as base_cfgs


logger = logging.getLogger(__name__)


# Llama3 8B Finetune configs ---------------------------------------------------------


def set_llama3_common_peft_configs(cfg: ConfigContainer) -> None:
    """Set common performance configurations for all Llama3 8B PEFT configs."""
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.model.disable_parameter_transpose_cache = True

    cfg.ddp.use_distributed_optimizer = True
    cfg.optimizer.use_distributed_optimizer = True


def llama3_8b_gb200_sft_config(precision: str = "bf16") -> ConfigContainer:
    """GB200, SFT config."""
    if precision == "bf16":
        base_cfg = base_cfgs.LLAMA3_8B_GB200_SFT_BF16_BASE_CONFIG
        precision_config = get_precision_config(precision)
    else:
        base_cfg = base_cfgs.LLAMA3_8B_GB200_SFT_FP8_CS_BASE_CONFIG
        if precision == "fp8_mx":
            base_cfg = base_cfgs.LLAMA3_8B_GB200_SFT_FP8_MX_BASE_CONFIG
        precision_config = get_precision_config(precision)

    cfg = llama3_8b_finetune_config(
        peft="none",
        precision_config=precision_config,
        packed_sequence=True,
        seq_length=16384,
    )
    set_llama3_common_peft_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    if precision == "fp8_mx":  # keeping this eanbled causes NaN grad norm
        cfg.ddp.overlap_param_gather = False
        cfg.optimizer.overlap_param_gather = False

    return cfg


def llama3_8b_h100_sft_config(precision: str = "bf16") -> ConfigContainer:
    """H100, SFT config."""
    if precision == "bf16":
        base_cfg = base_cfgs.LLAMA3_8B_H100_SFT_BF16_BASE_CONFIG
        precision_config = get_precision_config(precision)
    else:
        base_cfg = base_cfgs.LLAMA3_8B_H100_SFT_FP8_CS_BASE_CONFIG
        if precision == "fp8_mx":
            base_cfg = base_cfgs.LLAMA3_8B_H100_SFT_FP8_MX_BASE_CONFIG
        precision_config = get_precision_config(precision)

    cfg = llama3_8b_finetune_config(
        peft="none",
        precision_config=precision_config,
        packed_sequence=True,
        seq_length=4096,
    )
    set_llama3_common_peft_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def llama3_70b_gb200_sft_config(precision: str = "bf16") -> ConfigContainer:
    """GB200, SFT config."""
    if precision == "bf16":
        base_cfg = base_cfgs.LLAMA3_70B_GB200_SFT_BF16_BASE_CONFIG
        precision_config = get_precision_config(precision)
    else:
        base_cfg = base_cfgs.LLAMA3_70B_GB200_SFT_FP8_CS_BASE_CONFIG
        if precision == "fp8_mx":
            base_cfg = base_cfgs.LLAMA3_70B_GB200_SFT_FP8_MX_BASE_CONFIG
        precision_config = get_precision_config(precision)

    cfg = llama3_70b_finetune_config(
        peft="none",
        precision_config=precision_config,
        packed_sequence=True,
        seq_length=4096,
    )
    set_llama3_common_peft_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1),
        defer_embedding_wgrad_compute=True,
        wgrad_deferral_limit=22,
    )

    if precision == "fp8_mx":  # keeping this eanbled causes NaN grad norm
        cfg.comm_overlap.overlap_param_gather = False
        cfg.ddp.overlap_param_gather = False
        cfg.optimizer.overlap_param_gather = False

    return cfg


def llama3_70b_h100_sft_config(precision: str = "bf16") -> ConfigContainer:
    """H100, SFT config."""
    if precision == "bf16":
        base_cfg = base_cfgs.LLAMA3_70B_H100_SFT_BF16_BASE_CONFIG
        precision_config = get_precision_config(precision)
    else:
        base_cfg = base_cfgs.LLAMA3_70B_H100_SFT_FP8_CS_BASE_CONFIG
        if precision == "fp8_mx":
            base_cfg = base_cfgs.LLAMA3_70B_H100_SFT_FP8_MX_BASE_CONFIG
        precision_config = get_precision_config(precision)

    cfg = llama3_70b_finetune_config(
        peft="none",
        precision_config=precision_config,
        packed_sequence=True,
        seq_length=4096,
    )
    set_llama3_common_peft_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1),
        defer_embedding_wgrad_compute=True,
        wgrad_deferral_limit=22,
    )

    return cfg


def llama3_70b_gb200_lora_config(precision: str = "bf16") -> ConfigContainer:
    """GB200, LORA config."""
    if precision == "bf16":
        base_cfg = base_cfgs.LLAMA3_70B_GB200_LORA_BF16_BASE_CONFIG
        precision_config = get_precision_config(precision)
    else:
        base_cfg = base_cfgs.LLAMA3_70B_GB200_LORA_FP8_CS_BASE_CONFIG
        if precision == "fp8_mx":
            base_cfg = base_cfgs.LLAMA3_70B_GB200_LORA_FP8_MX_BASE_CONFIG
        precision_config = get_precision_config(precision)

    cfg = llama3_70b_finetune_config(
        peft="lora",
        precision_config=precision_config,
        packed_sequence=True,
        seq_length=2048,
    )
    set_llama3_common_peft_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    if precision == "fp8_mx":  # keeping this eanbled causes NaN grad norm
        if cfg.comm_overlap is not None and isinstance(cfg.comm_overlap, CommOverlapConfig):
            cfg.comm_overlap.overlap_param_gather = False
        cfg.ddp.overlap_param_gather = False
        cfg.optimizer.overlap_param_gather = False

    return cfg


def llama3_70b_h100_lora_config(precision: str = "bf16") -> ConfigContainer:
    """H100, LORA config."""
    if precision == "bf16":
        base_cfg = base_cfgs.LLAMA3_70B_H100_LORA_BF16_BASE_CONFIG
        precision_config = get_precision_config(precision)
    else:
        base_cfg = base_cfgs.LLAMA3_70B_H100_LORA_FP8_CS_BASE_CONFIG
        if precision == "fp8_mx":
            base_cfg = base_cfgs.LLAMA3_70B_H100_LORA_FP8_MX_BASE_CONFIG
        precision_config = get_precision_config(precision)

    cfg = llama3_70b_finetune_config(
        peft="lora",
        precision_config=precision_config,
        packed_sequence=True,
        seq_length=4096,
    )
    set_llama3_common_peft_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg
