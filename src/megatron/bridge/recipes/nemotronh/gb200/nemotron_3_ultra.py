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

"""GB200 functional pretraining recipe for Nemotron 3 Ultra."""

import torch

from megatron.bridge.recipes.nemotronh.h100.nemotron_3_ultra import (
    NEMOTRON_3_ULTRA_PRETRAIN_SEQ_LENGTH,
    _nemotron_3_ultra_large_scale_bf16_config,
)
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.config import ConfigContainer


def _nemotron_3_ultra_pretrain_256gpu_gb200_bf16_ep16_config() -> ConfigContainer:
    """Build the verified Nemotron 3 Ultra BF16 config for 256 GB200 GPUs."""
    cfg = _nemotron_3_ultra_large_scale_bf16_config()

    # TP4/PP4/EP16 fits naturally routed real-data batches while keeping each
    # HybridEP group inside one NVLink domain.
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 16
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.seq_length = NEMOTRON_3_ULTRA_PRETRAIN_SEQ_LENGTH
    cfg.dataset.seq_length = NEMOTRON_3_ULTRA_PRETRAIN_SEQ_LENGTH
    cfg.train.global_batch_size = 256

    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_hybridep_pad_uneven_dispatch_inputs = True
    cfg.model.moe_hybridep_num_sms = None
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.model.fine_grained_activation_offloading = True
    cfg.model.min_offloaded_tensor_size = 350_000_000
    cfg.model.offload_modules = ["fused_group_mlp"]
    cfg.model.fine_grained_offloading_max_inflight_offloads = 1
    # Flash Attention recomputes its largest intermediates. Layernorm
    # recompute adds enough headroom for natural-routing expert hotspots;
    # weighted-SReLU cannot use Transformer Engine's BF16 moe_act recompute.
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = ["core_attn", "layernorm"]

    cfg.dist.use_megatron_fsdp = False
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.num_distributed_optimizer_instances = 1
    cfg.ddp.num_buckets = 48
    cfg.ddp.average_in_collective = False
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.overlap_param_gather = True
    cfg.checkpoint.ckpt_format = "torch_dist"

    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 16,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_CPU_OFFLOAD_V1": 1,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


def nemotron_3_ultra_pretrain_256gpu_gb200_bf16_ep16_config() -> ConfigContainer:
    """Return a verified Nemotron 3 Ultra BF16 pretrain config for 256 GB200 GPUs.

    The recipe uses TP4/PP4/EP16 natural-routing HybridEP with targeted expert
    activation offload and selective attention/layernorm recompute. It retains
    numerical checks and the library recipe's optimizer, scheduler, and
    training objective.

    Returns:
        GB200 BF16 distributed-optimizer pretraining configuration.
    """
    return _nemotron_3_ultra_pretrain_256gpu_gb200_bf16_ep16_config()


__all__ = ["nemotron_3_ultra_pretrain_256gpu_gb200_bf16_ep16_config"]
