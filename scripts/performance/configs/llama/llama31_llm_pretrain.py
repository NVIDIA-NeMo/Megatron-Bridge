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

from megatron.bridge.recipes.llama import llama31_405b_pretrain_config
from megatron.bridge.training.comm_overlap import (
    userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192,
    userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192,
    userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192,
    userbuffers_fp8_h100_h16384_tp8_cp2_mbs1_seqlen8192,
)
from megatron.bridge.training.config import ConfigContainer


logger = logging.getLogger(__name__)


def set_llama31_common_configs(cfg: ConfigContainer) -> None:
    """Set common performance configurations for all Llama3.1 configs."""
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False


def llama31_405b_pretrain_config_gb300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB300, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="llama",
        model_recipe_name="llama31_405b",
        gpu="gb300",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    if precision == "bf16":
        comm_overlap_cfg = userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192
    else:
        comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192

    cfg = llama31_405b_pretrain_config(mock=mock, precision_config=precision_config)
    set_llama31_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    if cfg.ddp.use_megatron_fsdp:
        cfg.ddp.fsdp_double_buffer = True
        cfg.model.gradient_accumulation_fusion = False  # Disabled to avoid functional errors

    cfg.comm_overlap.tp_comm_overlap_cfg = comm_overlap_cfg
    cfg.comm_overlap.tp_comm_overlap = False if precision == "nvfp4" else cfg.comm_overlap.tp_comm_overlap

    if precision == "fp8_mx":  # keeping this eanbled causes NaN grad norm
        cfg.comm_overlap.overlap_param_gather = False
        cfg.ddp.overlap_param_gather = False
        cfg.optimizer.overlap_param_gather = False

    return cfg


def llama31_405b_pretrain_config_gb200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB200, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="llama",
        model_recipe_name="llama31_405b",
        gpu="gb200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    if precision == "bf16":
        comm_overlap_cfg = userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192
    else:
        comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192

    cfg = llama31_405b_pretrain_config(mock=mock, precision_config=precision_config)
    set_llama31_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    if cfg.ddp.use_megatron_fsdp:
        cfg.ddp.fsdp_double_buffer = True
        cfg.model.gradient_accumulation_fusion = False  # Disabled to avoid functional errors
        cfg.ddp.num_distributed_optimizer_instances = 2

    cfg.comm_overlap.tp_comm_overlap_cfg = comm_overlap_cfg
    cfg.comm_overlap.tp_comm_overlap = False if precision == "nvfp4" else cfg.comm_overlap.tp_comm_overlap

    if precision == "fp8_mx":  # keeping this eanbled causes NaN grad norm
        cfg.comm_overlap.overlap_param_gather = False
        cfg.ddp.overlap_param_gather = False
        cfg.optimizer.overlap_param_gather = False

    return cfg


def llama31_405b_pretrain_config_b300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B300, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="llama",
        model_recipe_name="llama31_405b",
        gpu="b300",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    if precision == "bf16":
        comm_overlap_cfg = userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192
    else:
        comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192

    cfg = llama31_405b_pretrain_config(mock=mock, precision_config=precision_config)
    set_llama31_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap.tp_comm_overlap_cfg = comm_overlap_cfg
    cfg.comm_overlap.tp_comm_overlap = False if precision == "nvfp4" else cfg.comm_overlap.tp_comm_overlap

    if precision == "fp8_mx":  # keeping this eanbled causes NaN grad norm
        cfg.comm_overlap.overlap_param_gather = False
        cfg.ddp.overlap_param_gather = False
        cfg.optimizer.overlap_param_gather = False

    return cfg


def llama31_405b_pretrain_config_b200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B200, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="llama",
        model_recipe_name="llama31_405b",
        gpu="b200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    if precision == "bf16":
        comm_overlap_cfg = userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192
    else:
        comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192

    cfg = llama31_405b_pretrain_config(mock=mock, precision_config=precision_config)
    set_llama31_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap.tp_comm_overlap_cfg = comm_overlap_cfg
    cfg.comm_overlap.tp_comm_overlap = False if precision == "nvfp4" else cfg.comm_overlap.tp_comm_overlap

    if precision == "fp8_mx":  # keeping this eanbled causes NaN grad norm
        cfg.comm_overlap.overlap_param_gather = False
        cfg.ddp.overlap_param_gather = False
        cfg.optimizer.overlap_param_gather = False

    return cfg


def llama31_405b_pretrain_config_h100(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """H100, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="llama",
        model_recipe_name="llama31_405b",
        gpu="h100",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    if precision == "bf16":
        comm_overlap_cfg = userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192
    else:
        comm_overlap_cfg = userbuffers_fp8_h100_h16384_tp8_cp2_mbs1_seqlen8192

    cfg = llama31_405b_pretrain_config(mock=mock, precision_config=precision_config)
    set_llama31_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap.tp_comm_overlap_cfg = comm_overlap_cfg

    return cfg
