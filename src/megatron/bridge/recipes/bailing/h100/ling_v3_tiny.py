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
"""Bridge recipe for a full-size Ling 3.0 Tiny fresh-initialization smoke."""

from __future__ import annotations

import os

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.recipes.utils.optimizer_utils import (
    distributed_fused_adam_with_cosine_annealing,
)
from megatron.bridge.training.config import ConfigContainer


LING_V3_TINY_HF_MODEL = "inclusionAI/Ling-3.0-tiny"
LING_V3_TINY_MTP_PATTERN = "+E"


def _ling_v3_tiny_hf_path() -> str:
    """Resolve an optional local HF reference for offline recipe construction."""
    return os.environ.get("LING_V3_TINY_HF_PATH", LING_V3_TINY_HF_MODEL)


def ling_v3_tiny_pretrain_8gpu_h100_bf16_config() -> ConfigContainer:
    """Return a full-size eight-GPU Ling 3.0 Tiny training smoke recipe.

    This is a fresh-initialization recipe using mock data, TP=1, PP=1, EP=8,
    CP=2, and one MTP depth. It is separate from HF checkpoint conversion,
    where the public Tiny artifact has no MTP tensors.
    """
    cfg = _pretrain_common()
    hf_model_path = _ling_v3_tiny_hf_path()
    cfg.model = AutoBridge.from_hf_pretrained(
        hf_model_path,
        trust_remote_code=True,
    ).to_megatron_provider(load_weights=False)

    # The public Tiny checkpoint has no MTP tensors. Add one MLA+MoE depth only
    # for training-path coverage; all main-model architecture remains sourced
    # from AutoBridge and the public HF config.
    cfg.model.mtp_hybrid_override_pattern = LING_V3_TINY_MTP_PATTERN
    cfg.model.mtp_num_layers = 1
    cfg.model.mtp_loss_scaling_factor = 0.1

    # Mock-data training uses synthetic tokens rather than an HF tokenizer.
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = cfg.model.vocab_size
    cfg.tokenizer.make_vocab_size_divisible_by = 1
    cfg.tokenizer.tensor_model_parallel_size = 1
    cfg.model.seq_length = 128

    # Training precision and runtime backend.
    cfg.model.params_dtype = torch.bfloat16
    cfg.model.bf16 = True
    cfg.model.fp16 = False
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = None

    # One-node parallelism for the eight-GPU smoke.
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 8
    cfg.model.context_parallel_size = 2
    cfg.model.sequence_parallel = False
    cfg.model.cp_comm_type = "p2p"
    cfg.model.linear_cp_mode = "headwise"

    # MoE training policy and runtime implementation.
    cfg.model.moe_router_bias_update_rate = 0.0
    cfg.model.moe_z_loss_coeff = 2.9e-6
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_grouped_gemm = True

    # Bound the activation-memory footprint of the full-size smoke.
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1

    # Keep the fresh-init smoke short and deterministic.
    cfg.dataset.blend = None
    cfg.dataset.seq_length = 128
    cfg.train.train_iters = 2
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 1_000_000_000
    cfg.validation.eval_iters = 0
    cfg.logger.log_interval = 1
    cfg.checkpoint.save = None
    cfg.checkpoint.load = None
    cfg.checkpoint.save_interval = 1_000_000_000

    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=0,
        lr_decay_iters=None,
        max_lr=1.0e-4,
        min_lr=1.0e-4,
        weight_decay=0.1,
        clip_grad=1.0,
        lr_decay_style="constant",
    )
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
    }
    return cfg


ling_v3_tiny_pretrain_config = ling_v3_tiny_pretrain_8gpu_h100_bf16_config

__all__ = [
    "LING_V3_TINY_HF_MODEL",
    "LING_V3_TINY_MTP_PATTERN",
    "ling_v3_tiny_pretrain_8gpu_h100_bf16_config",
    "ling_v3_tiny_pretrain_config",
]
