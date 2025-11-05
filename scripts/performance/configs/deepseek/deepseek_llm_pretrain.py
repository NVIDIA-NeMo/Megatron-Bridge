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
    set_moe_a2a_overlap_overrides,
    set_parallelism_and_batch_configs,
)

from megatron.bridge.recipes.deepseek.deepseek_v3 import deepseek_v3_pretrain_config as pretrain_config
from megatron.bridge.training.config import ConfigContainer

from . import parallelism_configs as parallelism_cfg


logger = logging.getLogger(__name__)


def set_deepseek_v3_common_configs(cfg: ConfigContainer) -> None:
    """Set common performance configurations for all DeepSeek-V3 configs."""
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = "selective"
    cfg.dist.enable_megatron_core_experimental = True

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.model.moe_router_force_load_balancing = True


def deepseek_v3_gb200_256gpus_bf16_config() -> ConfigContainer:
    """GB200, 256xGPU, BF16 baseline config."""
    pp = parallelism_cfg.DEEPSEEK_V3_GB200_256GPUS_BF16_PARALLEL_CONFIG.pipeline_model_parallel_size
    vp = parallelism_cfg.DEEPSEEK_V3_GB200_256GPUS_BF16_PARALLEL_CONFIG.virtual_pipeline_model_parallel_size
    cfg = pretrain_config(
        mock=True,
        precision_config=get_precision_config("bf16"),
        pipeline_parallelism=pp,
        virtual_pipeline_parallelism=vp,
        enable_deepep=False,
        layout=None,
    )
    set_deepseek_v3_common_configs(cfg)

    set_parallelism_and_batch_configs(cfg, parallelism_cfg.DEEPSEEK_V3_GB200_256GPUS_BF16_PARALLEL_CONFIG)

    cfg.model.recompute_modules = ["mla_up_proj"]
    cfg.comm_overlap.overlap_grad_reduce = True

    # Setting num_workers and pin_memory to 0 and False respectively gives better performance.
    # we are debugging this and might change this in the future.
    cfg.dataset.num_workers = 0
    cfg.dataset.pin_memory = False

    return cfg


def deepseek_v3_gb200_256gpus_fp8_config(fp8_recipe: str = "cs") -> ConfigContainer:
    """B200, 256xGPU, FP8 baseline config."""
    parallelism_and_batch_cfg = parallelism_cfg.DEEPSEEK_V3_GB200_256GPUS_FP8_MX_PARALLEL_CONFIG
    if fp8_recipe == "cs":
        parallelism_and_batch_cfg = parallelism_cfg.DEEPSEEK_V3_GB200_256GPUS_FP8_CS_PARALLEL_CONFIG

    cfg = pretrain_config(
        mock=True,
        precision_config=get_precision_config("fp8", fp8_recipe),
        pipeline_parallelism=parallelism_and_batch_cfg.pipeline_model_parallel_size,
        virtual_pipeline_parallelism=parallelism_and_batch_cfg.virtual_pipeline_model_parallel_size,
        enable_deepep=False,
        layout=None,
    )
    set_deepseek_v3_common_configs(cfg)

    set_parallelism_and_batch_configs(cfg, parallelism_and_batch_cfg)

    cfg.model.recompute_modules = ["mla_up_proj"]
    cfg.comm_overlap.overlap_grad_reduce = True

    # Setting num_workers and pin_memory to 0 and False respectively gives better performance.
    # we are debugging this and might change this in the future.
    cfg.dataset.num_workers = 0
    cfg.dataset.pin_memory = False

    return cfg


def deepseek_v3_b200_256gpus_bf16_config() -> ConfigContainer:
    """B200, 256xGPU, BF16 baseline config."""
    pp = parallelism_cfg.DEEPSEEK_V3_B200_256GPUS_BF16_PARALLEL_CONFIG.pipeline_model_parallel_size
    vp = parallelism_cfg.DEEPSEEK_V3_B200_256GPUS_BF16_PARALLEL_CONFIG.virtual_pipeline_model_parallel_size
    cfg = pretrain_config(
        mock=True,
        precision_config=get_precision_config("bf16"),
        pipeline_parallelism=pp,
        virtual_pipeline_parallelism=vp,
        enable_deepep=False,
        layout=None,
    )
    set_deepseek_v3_common_configs(cfg)

    set_parallelism_and_batch_configs(cfg, parallelism_cfg.DEEPSEEK_V3_B200_256GPUS_BF16_PARALLEL_CONFIG)

    cfg.model.recompute_modules = ["mla_up_proj"]
    cfg.comm_overlap.overlap_grad_reduce = True

    return cfg


def deepseek_v3_b200_256gpus_fp8_config(fp8_recipe: str = "cs") -> ConfigContainer:
    """B200, 256xGPU, FP8 baseline config."""
    parallelism_and_batch_cfg = parallelism_cfg.DEEPSEEK_V3_B200_256GPUS_FP8_MX_PARALLEL_CONFIG
    if fp8_recipe == "cs":
        parallelism_and_batch_cfg = parallelism_cfg.DEEPSEEK_V3_B200_256GPUS_FP8_CS_PARALLEL_CONFIG

    cfg = pretrain_config(
        mock=True,
        precision_config=get_precision_config("fp8", fp8_recipe),
        pipeline_parallelism=parallelism_and_batch_cfg.pipeline_model_parallel_size,
        virtual_pipeline_parallelism=parallelism_and_batch_cfg.virtual_pipeline_model_parallel_size,
        enable_deepep=False,
        layout=None,
    )
    set_deepseek_v3_common_configs(cfg)

    set_parallelism_and_batch_configs(cfg, parallelism_and_batch_cfg)

    cfg.model.recompute_modules = ["mla_up_proj"]
    cfg.comm_overlap.overlap_grad_reduce = True

    return cfg


def deepseek_v3_h100_1024gpus_bf16_config() -> ConfigContainer:
    """H100, 1024xGPU, BF16 baseline config."""
    pp = parallelism_cfg.DEEPSEEK_V3_H100_1024GPUS_BF16_PARALLEL_CONFIG.pipeline_model_parallel_size
    vp = parallelism_cfg.DEEPSEEK_V3_H100_1024GPUS_BF16_PARALLEL_CONFIG.virtual_pipeline_model_parallel_size
    cfg = pretrain_config(
        mock=True,
        precision_config=get_precision_config("bf16"),
        pipeline_parallelism=pp,
        virtual_pipeline_parallelism=vp,
        enable_deepep=True,
        layout="Et|(tt|)*30mL",
    )
    set_deepseek_v3_common_configs(cfg)

    set_parallelism_and_batch_configs(cfg, parallelism_cfg.DEEPSEEK_V3_H100_1024GPUS_BF16_PARALLEL_CONFIG)

    cfg.model.recompute_modules = ["mla_up_proj", "mlp"]

    set_moe_a2a_overlap_overrides(cfg)

    # Disabling to avoid functional errors. TODO: Test with it enabled and keep it enabled if it works.
    cfg.comm_overlap.overlap_grad_reduce = False

    return cfg


def deepseek_v3_h100_1024gpus_fp8_config(fp8_recipe: str = "cs") -> ConfigContainer:
    """H100, 1024xGPU, FP8 baseline config."""
    parallelism_and_batch_cfg = parallelism_cfg.DEEPSEEK_V3_H100_1024GPUS_FP8_CS_PARALLEL_CONFIG
    if fp8_recipe == "sc":
        parallelism_and_batch_cfg = parallelism_cfg.DEEPSEEK_V3_H100_1024GPUS_FP8_SC_PARALLEL_CONFIG

    cfg = pretrain_config(
        mock=True,
        precision_config=get_precision_config("fp8", fp8_recipe),
        pipeline_parallelism=parallelism_and_batch_cfg.pipeline_model_parallel_size,
        virtual_pipeline_parallelism=parallelism_and_batch_cfg.virtual_pipeline_model_parallel_size,
        enable_deepep=True,
        layout="Et|(tt|)*30mL",
    )
    set_deepseek_v3_common_configs(cfg)

    set_parallelism_and_batch_configs(cfg, parallelism_and_batch_cfg)

    cfg.model.recompute_modules = ["mla_up_proj", "mlp"]

    set_moe_a2a_overlap_overrides(cfg)

    # Disabling to avoid functional errors. TODO: Test with it enabled and keep it enabled if it works.
    cfg.comm_overlap.overlap_grad_reduce = False

    return cfg
