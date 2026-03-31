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

"""Qwen3.5 text-only (LLM) SFT recipes for Qwen3.5-VL checkpoints.

Loads **language-model weights only** from a HuggingFace Qwen3.5-VL checkpoint via
``skip_megatron_param_globs`` (vision tower keeps initialization values). Uses
``_sft_common`` defaults: SQuAD, HuggingFace tokenizer, and ``vlm_step``-compatible
text batches (no ``hf_processor_path``).

Train with ``megatron.bridge.training.vlm_step.forward_step``.
"""

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _sft_common
from megatron.bridge.recipes.qwen_vl.qwen35_vl import (
    _qwen35_vl_apply_moe,
    _qwen35_vl_enable_recompute,
)
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.config import ConfigContainer


# Megatron parameter name substrings for Qwen3-VL-shaped models (vision tower).
QWEN35_LLM_SFT_SKIP_VISION_GLOBS: tuple[str, ...] = ("*vision_model*",)


def _qwen35_llm_sft_apply_common(
    cfg: ConfigContainer,
    hf_path: str,
    *,
    tp: int,
    pp: int,
    max_lr: float,
    min_lr: float,
    train_iters: int = 1000,
    global_batch_size: int = 128,
    micro_batch_size: int = 1,
) -> None:
    """Apply Qwen3.5 LLM-only SFT settings on top of ``_sft_common``."""
    seq_len = cfg.dataset.seq_length

    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(
        load_weights=True,
        skip_megatron_param_globs=list(QWEN35_LLM_SFT_SKIP_VISION_GLOBS),
    )
    cfg.model.seq_length = seq_len

    cfg.model.tensor_model_parallel_size = tp
    cfg.model.pipeline_model_parallel_size = pp
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_vision_projection = True

    cfg.model.mtp_num_layers = None

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.attention_backend = "auto"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    cfg.train.train_iters = train_iters
    cfg.train.global_batch_size = global_batch_size
    cfg.train.micro_batch_size = micro_batch_size
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=min(50, train_iters),
        lr_decay_iters=train_iters,
        max_lr=max_lr,
        min_lr=min_lr,
        adam_beta2=0.98,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg

    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    cfg.tokenizer.tokenizer_model = hf_path

    cfg.dataset.pack_sequences_in_batch = False

    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    cfg.mixed_precision = "bf16_mixed"


def qwen35_llm_800m_sft_config(hf_path: str = "Qwen/Qwen3.5-0.8B") -> ConfigContainer:
    """SQuAD-style SFT: Qwen3.5 800M, LLM weights only from HF, TP=1 PP=1."""
    cfg = _sft_common()
    _qwen35_llm_sft_apply_common(cfg, hf_path, tp=1, pp=1, max_lr=5e-6, min_lr=5e-7)
    return cfg


def qwen35_llm_2b_sft_config(hf_path: str = "Qwen/Qwen3.5-2B") -> ConfigContainer:
    """SQuAD-style SFT: Qwen3.5 2B, LLM weights only from HF, TP=1 PP=1."""
    cfg = _sft_common()
    _qwen35_llm_sft_apply_common(cfg, hf_path, tp=1, pp=1, max_lr=5e-6, min_lr=5e-7)
    return cfg


def qwen35_llm_4b_sft_config(hf_path: str = "Qwen/Qwen3.5-4B") -> ConfigContainer:
    """SQuAD-style SFT: Qwen3.5 4B, LLM weights only from HF, TP=2 PP=1."""
    cfg = _sft_common()
    _qwen35_llm_sft_apply_common(cfg, hf_path, tp=2, pp=1, max_lr=5e-6, min_lr=5e-7)
    return cfg


def qwen35_llm_9b_sft_config(hf_path: str = "Qwen/Qwen3.5-9B") -> ConfigContainer:
    """SQuAD-style SFT: Qwen3.5 9B, LLM weights only from HF, TP=4 PP=1."""
    cfg = _sft_common()
    _qwen35_llm_sft_apply_common(cfg, hf_path, tp=4, pp=1, max_lr=5e-6, min_lr=5e-7)
    return cfg


def qwen35_llm_27b_sft_config(hf_path: str = "Qwen/Qwen3.5-27B") -> ConfigContainer:
    """SQuAD-style SFT: Qwen3.5 27B, LLM weights only from HF, TP=4 PP=4."""
    cfg = _sft_common()
    _qwen35_llm_sft_apply_common(cfg, hf_path, tp=4, pp=4, max_lr=5e-6, min_lr=5e-7)
    return cfg


def qwen35_llm_35b_a3b_sft_config(hf_path: str = "Qwen/Qwen3.5-35B-A3B") -> ConfigContainer:
    """SQuAD-style SFT: Qwen3.5 35B-A3B MoE, LLM weights only, TP=2 PP=1 EP=16."""
    cfg = _sft_common()
    _qwen35_llm_sft_apply_common(
        cfg, hf_path, tp=2, pp=1, max_lr=2e-5, min_lr=2e-6, global_batch_size=32, micro_batch_size=1
    )
    _qwen35_vl_apply_moe(cfg, ep=16)
    return cfg


def qwen35_llm_122b_a10b_sft_config(hf_path: str = "Qwen/Qwen3.5-122B-A10B") -> ConfigContainer:
    """SQuAD-style SFT: Qwen3.5 122B-A10B MoE, LLM weights only."""
    cfg = _sft_common()
    _qwen35_llm_sft_apply_common(
        cfg, hf_path, tp=2, pp=6, max_lr=2e-5, min_lr=2e-6, global_batch_size=36, micro_batch_size=1
    )
    _qwen35_vl_apply_moe(cfg, ep=8)
    _qwen35_vl_enable_recompute(cfg)
    return cfg


def qwen35_llm_397b_a17b_sft_config(hf_path: str = "Qwen/Qwen3.5-397B-A17B") -> ConfigContainer:
    """SQuAD-style SFT: Qwen3.5 397B-A17B MoE, LLM weights only."""
    cfg = _sft_common()
    _qwen35_llm_sft_apply_common(
        cfg, hf_path, tp=2, pp=4, max_lr=2e-5, min_lr=2e-6, global_batch_size=32, micro_batch_size=1
    )
    _qwen35_vl_apply_moe(cfg, ep=32)
    _qwen35_vl_enable_recompute(cfg)
    return cfg
