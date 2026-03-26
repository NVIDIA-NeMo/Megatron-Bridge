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

"""Qwen3-Omni thinker training recipes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.data.vlm_datasets.preloaded_provider import PreloadedVLMConversationProvider
from megatron.bridge.recipes.common import _sft_common_vlm
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing

if TYPE_CHECKING:
    from megatron.bridge.training.config import ConfigContainer


def _qwen3_omni_apply_common(
    cfg: "ConfigContainer",
    hf_path: str,
    *,
    tp: int,
    pp: int,
    max_lr: float,
    min_lr: float,
    gbs: int = 32,
) -> None:
    """Apply shared thinker-only training settings for Qwen3-Omni."""

    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096
    cfg.model.tensor_model_parallel_size = tp
    cfg.model.pipeline_model_parallel_size = pp
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 1
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False

    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_audio_model = False

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.attention_backend = "auto"

    cfg.train.train_iters = 1000
    cfg.train.global_batch_size = gbs
    cfg.train.micro_batch_size = 1

    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=50,
        lr_decay_iters=1000,
        max_lr=max_lr,
        min_lr=min_lr,
        adam_beta2=0.98,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg

    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = hf_path
    cfg.dataset.pack_sequences_in_batch = False

    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    cfg.mixed_precision = "bf16_mixed"


def qwen3_omni_30b_a3b_sft_config(
    hf_path: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
) -> "ConfigContainer":
    """Return a minimal thinker-only SFT config for Qwen3-Omni 30B-A3B."""

    cfg = _sft_common_vlm()
    _qwen3_omni_apply_common(cfg, hf_path, tp=1, pp=1, max_lr=5e-6, min_lr=5e-7)
    return cfg


def qwen3_omni_30b_a3b_sft_preloaded_config(
    hf_path: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
) -> "ConfigContainer":
    """Return a thinker-only SFT config backed by preloaded local JSON/JSONL data."""

    cfg = _sft_common_vlm()
    _qwen3_omni_apply_common(cfg, hf_path, tp=1, pp=1, max_lr=5e-6, min_lr=5e-7)
    cfg.dataset = PreloadedVLMConversationProvider(
        seq_length=cfg.model.seq_length,
        hf_processor_path=hf_path,
        train_data_path=None,
        valid_data_path=None,
        test_data_path=None,
        dataloader_type="single",
        num_workers=2,
    )
    cfg.dataset.pack_sequences_in_batch = False
    return cfg
