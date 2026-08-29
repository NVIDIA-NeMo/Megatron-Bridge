# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Nemotron 3.5 Super VL GB200 training recipes."""

import torch

from megatron.bridge.recipes.nemotron_omni.h100.nemotron_35_super_vl import (
    nemotron_35_super_vl_peft_16gpu_h100_bf16_config,
    nemotron_35_super_vl_pretrain_64gpu_h100_bf16_config,
    nemotron_35_super_vl_sft_64gpu_h100_bf16_config,
)
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.utils.cuda_graph import set_cuda_graph_modules


def nemotron_35_super_vl_pretrain_64gpu_gb200_bf16_config() -> ConfigContainer:
    """Return the 64-GB200 BF16 pretraining config for Super VL.

    The recipe preserves the H100 data, objective, optimizer schedule, natural
    routing, and numerical-safety contract. It uses the 8192-token workload
    shape of the corresponding Nemotron 3 Super GB200 recipe. Other
    GB200-specific changes are limited to the parallel layout, full-precision
    optimizer state, communication overlap, scoped CUDA graphs, and the
    single-NVL72 HybridEP environment.

    Returns:
        The Super-VL GB200 pretraining configuration.
    """
    cfg = nemotron_35_super_vl_pretrain_64gpu_h100_bf16_config()

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.num_layers_in_first_pipeline_stage = None
    cfg.model.num_layers_in_last_pipeline_stage = None
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.seq_length = 8192
    cfg.dataset.seq_length = 8192
    cfg.model.overlap_p2p_comm = False
    cfg.model.batch_p2p_comm = False
    cfg.model.batch_p2p_sync = True

    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1
    cfg.dataset.micro_batch_size = 1

    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_vision = False
    cfg.model.cuda_graph_impl = "transformer_engine"
    set_cuda_graph_modules(cfg.model, ["attn", "mamba", "moe_router", "moe_preprocess"])
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True

    cfg.model.moe_router_force_load_balancing = False
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_hybridep_num_sms = 32
    cfg.model.moe_hybridep_num_sms_preprocessing = None
    cfg.model.moe_permute_fusion_into_hybridep = False

    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    # Modality-specific embedders can be unused on a batch, which is not
    # compatible with DDP overlap's fixed first-batch gradient-hook contract.
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.checkpoint.async_save = True

    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 64,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
    }
    return cfg


def nemotron_35_super_vl_sft_64gpu_gb200_bf16_config() -> ConfigContainer:
    """Return the 64-GB200 BF16 SFT configuration for Super VL.

    This preserves the H100 SFT data, objective, optimizer, schedule, batch,
    and trainable-parameter contract while mapping execution to one NVL72
    domain. The ordinary all-to-all dispatcher and eager execution keep this
    support-verification recipe independent of topology-sensitive benchmark
    transport and graph capture.

    Returns:
        The Super-VL GB200 SFT configuration.
    """
    cfg = nemotron_35_super_vl_sft_64gpu_h100_bf16_config()
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.num_layers_in_first_pipeline_stage = None
    cfg.model.num_layers_in_last_pipeline_stage = None
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_vision = False
    cfg.model.cuda_graph_impl = "none"
    set_cuda_graph_modules(cfg.model, [])
    cfg.model.use_te_rng_tracker = False
    cfg.rng.te_rng_tracker = False
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.checkpoint.async_save = False
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
    }
    return cfg


def nemotron_35_super_vl_peft_16gpu_gb200_bf16_config() -> ConfigContainer:
    """Return the 16-GB200 BF16 LoRA configuration for Super VL.

    Returns:
        The Super-VL GB200 PEFT configuration.
    """
    cfg = nemotron_35_super_vl_peft_16gpu_h100_bf16_config()
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 16
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.cuda_graph_impl = "none"
    set_cuda_graph_modules(cfg.model, [])
    cfg.model.use_te_rng_tracker = False
    cfg.rng.te_rng_tracker = False
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.checkpoint.async_save = False
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
    }
    return cfg


__all__ = [
    "nemotron_35_super_vl_peft_16gpu_gb200_bf16_config",
    "nemotron_35_super_vl_pretrain_64gpu_gb200_bf16_config",
    "nemotron_35_super_vl_sft_64gpu_gb200_bf16_config",
]
