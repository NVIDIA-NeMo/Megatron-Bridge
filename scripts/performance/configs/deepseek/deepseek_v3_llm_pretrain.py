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
    set_moe_a2a_1f1b_overrides,
)

from megatron.bridge.recipes.deepseek.deepseek_v3 import deepseek_v3_pretrain_config as pretrain_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.utils.moe_token_drop import apply_moe_token_drop


logger = logging.getLogger(__name__)


def deepseek_v3_gb200_256gpus_bf16_config(**kwargs) -> ConfigContainer:
    """GB200, 256xGPU, BF16 baseline config."""
    cfg = pretrain_config(
        mock=True,
        precision_config=get_precision_config("bf16"),
        pipeline_parallelism=4,
        virtual_pipeline_parallelism=4,
        enable_deepep=False,
        layout=None,
    )

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.expert_model_parallel_size = 64
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 1
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = "selective"
    cfg.dist.enable_megatron_core_experimental = True

    cfg.model.recompute_modules = ["mla_up_proj"]
    cfg.comm_overlap.overlap_grad_reduce = True

    cfg.model = apply_moe_token_drop(cfg.model)

    cfg.dataset.num_workers = 0
    cfg.dataset.pin_memory = False

    return cfg


def deepseek_v3_gb200_256gpus_fp8_config(fp8_recipe: str = "cs") -> ConfigContainer:
    """B200, 256xGPU, FP8 baseline config."""
    cfg = pretrain_config(
        mock=True,
        precision_config=get_precision_config("fp8", fp8_recipe),
        pipeline_parallelism=4,
        virtual_pipeline_parallelism=4,
        enable_deepep=False,
        layout=None,
    )

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.expert_model_parallel_size = 64
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 1
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = "selective"
    cfg.dist.enable_megatron_core_experimental = True

    cfg.model.recompute_modules = ["mla_up_proj"]
    cfg.comm_overlap.overlap_grad_reduce = True

    cfg.model = apply_moe_token_drop(cfg.model)

    cfg.dataset.num_workers = 0
    cfg.dataset.pin_memory = False

    return cfg


def deepseek_v3_b200_256gpus_bf16_config(**kwargs) -> ConfigContainer:
    """B200, 256xGPU, BF16 baseline config."""
    cfg = pretrain_config(
        mock=True,
        precision_config=get_precision_config("bf16"),
        pipeline_parallelism=16,
        virtual_pipeline_parallelism=1,
        enable_deepep=False,
        layout=None,
    )

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 16
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 1
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = "selective"
    cfg.dist.enable_megatron_core_experimental = True

    cfg.model.recompute_modules = ["mla_up_proj"]
    cfg.comm_overlap.overlap_grad_reduce = True

    cfg.model = apply_moe_token_drop(cfg.model)

    return cfg


def deepseek_v3_b200_256gpus_fp8_config(fp8_recipe: str = "cs") -> ConfigContainer:
    """B200, 256xGPU, FP8 baseline config."""
    cfg = pretrain_config(
        mock=True,
        precision_config=get_precision_config("fp8", fp8_recipe),
        pipeline_parallelism=16,
        virtual_pipeline_parallelism=1,
        enable_deepep=False,
        layout=None,
    )

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 16
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 1
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = "selective"
    cfg.dist.enable_megatron_core_experimental = True

    cfg.model.recompute_modules = ["mla_up_proj"]
    cfg.comm_overlap.overlap_grad_reduce = True

    cfg.model = apply_moe_token_drop(cfg.model)

    return cfg


def deepseek_v3_h100_1024gpus_bf16_config(**kwargs) -> ConfigContainer:
    """H100, 1024xGPU, BF16 baseline config."""
    cfg = pretrain_config(
        mock=True,
        precision_config=get_precision_config("bf16"),
        pipeline_parallelism=8,
        virtual_pipeline_parallelism=4,
        enable_deepep=True,
        layout="Et|(tt|)*30mL",
    )

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.expert_model_parallel_size = 64
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 8192
    cfg.train.micro_batch_size = 1
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = "selective"
    cfg.dist.enable_megatron_core_experimental = True

    cfg.model.recompute_modules = ["mla_up_proj", "mlp"]

    cfg.model.moe_router_force_load_balancing = True

    set_moe_a2a_1f1b_overrides(cfg)

    cfg.comm_overlap.overlap_grad_reduce = False

    return cfg


def deepseek_v3_h100_1024gpus_fp8_config(fp8_recipe: str = "cs") -> ConfigContainer:
    """H100, 1024xGPU, FP8 baseline config."""
    cfg = pretrain_config(
        mock=True,
        precision_config=get_precision_config("fp8", fp8_recipe),
        pipeline_parallelism=8,
        virtual_pipeline_parallelism=4,
        enable_deepep=True,
        layout="Et|(tt|)*30mL",
    )

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.expert_model_parallel_size = 64
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 8192
    cfg.train.micro_batch_size = 1
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = "selective"
    cfg.dist.enable_megatron_core_experimental = True

    cfg.model.recompute_modules = ["mla_up_proj", "mlp"]

    cfg.model.moe_router_force_load_balancing = True

    set_moe_a2a_1f1b_overrides(cfg)

    cfg.comm_overlap.overlap_grad_reduce = False

    return cfg
