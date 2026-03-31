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

"""Qwen3.5-VL recipes.

This module provides pretrain, SFT, and PEFT configurations for Qwen3.5-VL models:

- **Dense**: 800M, 2B, 4B, 9B, 27B
- **MoE**: 35B-A3B, 122B-A10B, 397B-A17B
"""

from __future__ import annotations

import torch
from typing_extensions import Unpack

from megatron.bridge import AutoBridge
from megatron.bridge.data.energon.energon_provider import EnergonProvider
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _peft_common_vlm, _sft_common_vlm
from megatron.bridge.recipes.qwen_vl.qwen3_vl import (
    Qwen3VLCommonKwargs,
    _make_energon_dataset,
    _qwen3_vl_common,
)
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.config import ConfigContainer


# =============================================================================
# Internal helpers — shared configuration for all Qwen3.5-VL recipes
# =============================================================================


def _qwen35_vl_apply_common(
    cfg: ConfigContainer,
    hf_path: str,
    *,
    tp: int,
    pp: int,
    max_lr: float,
    min_lr: float,
    gbs: int = 32,
) -> None:
    """Apply settings shared across all Qwen3.5-VL SFT and PEFT recipes.

    Sets model, parallelism (except EP/SP for MoE), VLM freeze, MTP, TE,
    CUDA graphs, kernels, memory-saving defaults, training, optimizer,
    dataset, DDP, and mixed-precision options.
    """
    # Model configuration
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096

    # Parallel settings (dense defaults; MoE overrides EP/SP via _apply_moe)
    cfg.model.tensor_model_parallel_size = tp
    cfg.model.pipeline_model_parallel_size = pp
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # VLM-specific settings
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False

    # MTP (Multi-Token Prediction) — auto-detected from HF config (mtp_num_hidden_layers=1).
    # Set to None to finetune without MTP loss.
    cfg.model.mtp_num_layers = 1
    cfg.model.mtp_loss_scaling_factor = 0.1

    # TE / Transformer implementation
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph settings
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = "auto"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving (disabled by default)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Training config
    cfg.train.train_iters = 300000
    cfg.train.global_batch_size = gbs
    cfg.train.micro_batch_size = 4  # tested on Blackwell GPUs; reduce for smaller VRAM
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    # Optimizer
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=200,
        lr_decay_iters=300000,
        max_lr=max_lr,
        min_lr=min_lr,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Dataset configuration
    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = hf_path
    cfg.dataset.pack_sequences_in_batch = False

    # DDP settings
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    # Mixed precision
    cfg.mixed_precision = "bf16_mixed"


def _qwen35_vl_apply_moe(cfg: ConfigContainer, *, ep: int, etp: int = 1) -> None:
    """Apply MoE-specific settings on top of the common configuration.

    Enables expert parallelism, sequence parallelism, token dispatcher,
    MoE kernels, and sets MoE-specific overlap / balance / FP8-padding defaults.
    """
    cfg.model.expert_model_parallel_size = ep
    cfg.model.expert_tensor_parallel_size = etp
    cfg.model.sequence_parallel = True

    # MoE dispatcher (alltoall is the standard choice for EP>1)
    cfg.model.moe_token_dispatcher_type = "alltoall"

    # MoE kernel selections
    cfg.model.moe_router_fusion = True
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True

    # MoE overlap
    cfg.model.moe_shared_expert_overlap = False

    # MoE force balance
    cfg.model.moe_router_force_load_balancing = False

    # MoE FP8 padding
    cfg.model.moe_router_padding_for_fp8 = False

    # Comm overlap settings
    cfg.comm_overlap = None


def _qwen35_vl_enable_recompute(cfg: ConfigContainer) -> None:
    """Enable activation recomputation for large models."""
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1


def _qwen35_vl_apply_peft_scheme(cfg: ConfigContainer, peft_scheme: str | PEFT) -> None:
    """Resolve and apply the PEFT scheme (LoRA, DoRA, or custom)."""
    if isinstance(peft_scheme, str) and peft_scheme.lower() in ["lora", "dora"]:
        cfg.peft = default_peft_config(peft_scheme)
    else:
        cfg.peft = peft_scheme


# =============================================================================
# Qwen3.5-VL Pretrain Configurations (mock dataset)
# =============================================================================
# Qwen3.5-VL reuses the Qwen3-VL _qwen3_vl_common helper for pretrain configs
# since both families share the same VLM architecture and mock-dataset pipeline.


def qwen35_vl_9b_pretrain_mock_config(**user_kwargs: Unpack[Qwen3VLCommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Qwen3.5-VL 9B (dense).

    See `_qwen3_vl_common` for the full list of parameters.
    """
    recommended_kwargs: Qwen3VLCommonKwargs = {
        "hf_path": "Qwen/Qwen3.5-9B",
        "tensor_model_parallel_size": 4,
        "pipeline_model_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "freeze_language_model": True,
        "freeze_vision_model": True,
        "freeze_vision_projection": False,
    }
    combined_kwargs: Qwen3VLCommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen3_vl_common(**combined_kwargs)


def qwen35_vl_35b_a3b_pretrain_mock_config(**user_kwargs: Unpack[Qwen3VLCommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Qwen3.5-VL 35B-A3B (MoE).

    See `_qwen3_vl_common` for the full list of parameters.
    """
    recommended_kwargs: Qwen3VLCommonKwargs = {
        "hf_path": "Qwen/Qwen3.5-35B-A3B",
        "tensor_model_parallel_size": 4,
        "pipeline_model_parallel_size": 2,
        "expert_model_parallel_size": 4,
        "sequence_parallel": True,
        "freeze_language_model": True,
        "freeze_vision_model": True,
        "freeze_vision_projection": False,
    }
    combined_kwargs: Qwen3VLCommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen3_vl_common(**combined_kwargs)


def qwen35_vl_122b_a10b_pretrain_mock_config(**user_kwargs: Unpack[Qwen3VLCommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Qwen3.5-VL 122B-A10B (MoE).

    See `_qwen3_vl_common` for the full list of parameters.
    """
    recommended_kwargs: Qwen3VLCommonKwargs = {
        "hf_path": "Qwen/Qwen3.5-122B-A10B",
        "tensor_model_parallel_size": 4,
        "pipeline_model_parallel_size": 8,
        "expert_model_parallel_size": 8,
        "context_parallel_size": 2,
        "sequence_parallel": True,
        "freeze_language_model": True,
        "freeze_vision_model": True,
        "freeze_vision_projection": False,
    }
    combined_kwargs: Qwen3VLCommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen3_vl_common(**combined_kwargs)


def qwen35_vl_397b_a17b_pretrain_mock_config(**user_kwargs: Unpack[Qwen3VLCommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Qwen3.5-VL 397B-A17B (MoE).

    See `_qwen3_vl_common` for the full list of parameters.
    """
    recommended_kwargs: Qwen3VLCommonKwargs = {
        "hf_path": "Qwen/Qwen3.5-397B-A17B",
        "tensor_model_parallel_size": 4,
        "pipeline_model_parallel_size": 16,
        "expert_model_parallel_size": 16,
        "context_parallel_size": 2,
        "sequence_parallel": True,
        "freeze_language_model": True,
        "freeze_vision_model": True,
        "freeze_vision_projection": False,
    }
    combined_kwargs: Qwen3VLCommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen3_vl_common(**combined_kwargs)


# =============================================================================
# Qwen3.5-VL Dense SFT Configurations (800M, 2B, 4B, 9B, 27B)
# =============================================================================


def qwen35_vl_800m_sft_config(hf_path: str = "Qwen/Qwen3.5-0.8B") -> ConfigContainer:
    """Return a full SFT config for Qwen3.5-VL 800M (dense).

    Default configuration: 1 node, 8 GPUs
    - TP=1, PP=1
    - LR=5e-6 (full SFT)
    - Sequence length: 4096

    Note: num_kv_heads=2, so max TP=2.

    Args:
        hf_path: HuggingFace model ID or local path to model directory.
    """
    cfg = _sft_common_vlm()
    _qwen35_vl_apply_common(cfg, hf_path, tp=1, pp=1, max_lr=5e-6, min_lr=5e-7)
    return cfg


def qwen35_vl_2b_sft_config(hf_path: str = "Qwen/Qwen3.5-2B") -> ConfigContainer:
    """Return a full SFT config for Qwen3.5-VL 2B (dense).

    Default configuration: 1 node, 8 GPUs
    - TP=1, PP=1
    - LR=5e-6 (full SFT)
    - Sequence length: 4096

    Note: num_kv_heads=2, so max TP=2.

    Args:
        hf_path: HuggingFace model ID or local path to model directory.
    """
    cfg = _sft_common_vlm()
    _qwen35_vl_apply_common(cfg, hf_path, tp=1, pp=1, max_lr=5e-6, min_lr=5e-7)
    return cfg


def qwen35_vl_4b_sft_config(hf_path: str = "Qwen/Qwen3.5-4B") -> ConfigContainer:
    """Return a full SFT config for Qwen3.5-VL 4B (dense).

    Default configuration: 1 node, 8 GPUs
    - TP=2, PP=1
    - LR=5e-6 (full SFT)
    - Sequence length: 4096

    Note: num_kv_heads=4, so max TP=4.

    Args:
        hf_path: HuggingFace model ID or local path to model directory.
    """
    cfg = _sft_common_vlm()
    _qwen35_vl_apply_common(cfg, hf_path, tp=2, pp=1, max_lr=5e-6, min_lr=5e-7)
    return cfg


def qwen35_vl_9b_sft_config(hf_path: str = "Qwen/Qwen3.5-9B") -> ConfigContainer:
    """Return a full SFT config for Qwen3.5-VL 9B (dense).

    Default configuration: 1 node, 8 GPUs
    - TP=4, PP=1
    - LR=5e-6 (full SFT)
    - Sequence length: 4096

    Note: num_kv_heads=4, so max TP=4.

    Args:
        hf_path: HuggingFace model ID or local path to model directory.
    """
    cfg = _sft_common_vlm()
    _qwen35_vl_apply_common(cfg, hf_path, tp=4, pp=1, max_lr=5e-6, min_lr=5e-7)
    return cfg


def qwen35_vl_27b_sft_config(hf_path: str = "Qwen/Qwen3.5-27B") -> ConfigContainer:
    """Return a full SFT config for Qwen3.5-VL 27B (dense).

    Default configuration: 2 nodes, 16 GPUs total
    - TP=4, PP=4
    - LR=5e-6 (full SFT)
    - Sequence length: 4096

    Args:
        hf_path: HuggingFace model ID or local path to model directory.
    """
    cfg = _sft_common_vlm()
    _qwen35_vl_apply_common(cfg, hf_path, tp=4, pp=4, max_lr=5e-6, min_lr=5e-7)
    return cfg


# =============================================================================
# Qwen3.5-VL MoE SFT Configurations (35B-A3B, 122B-A10B, 397B-A17B)
# =============================================================================


def qwen35_vl_35b_a3b_sft_config(hf_path: str = "Qwen/Qwen3.5-35B-A3B") -> ConfigContainer:
    """Return a full SFT config for Qwen3.5-VL 35B-A3B (MoE).

    Default configuration: 2 nodes, 16 GPUs
    - TP=2, PP=1, EP=16
    - LR=2e-5 (full SFT)
    - Sequence length: 4096

    Args:
        hf_path: HuggingFace model ID or local path to model directory.
    """
    cfg = _sft_common_vlm()
    _qwen35_vl_apply_common(cfg, hf_path, tp=2, pp=1, max_lr=2e-5, min_lr=2e-6)
    _qwen35_vl_apply_moe(cfg, ep=16)
    return cfg


def qwen35_vl_35b_a3b_sft_energon_config(hf_path: str = "Qwen/Qwen3.5-35B-A3B") -> ConfigContainer:
    """Return a full SFT config for Qwen3.5-VL 35B-A3B (MoE) with Energon dataset.

    Same as qwen35_vl_35b_a3b_sft_config but uses EnergonProvider instead of HF dataset.
    Set the dataset path via CLI override: dataset.path=/path/to/energon/dataset
    """
    cfg = qwen35_vl_35b_a3b_sft_config(hf_path=hf_path)
    cfg.dataset = _make_energon_dataset(
        hf_path, cfg.dataset.seq_length, cfg.train.micro_batch_size, cfg.train.global_batch_size
    )
    return cfg


def qwen35_vl_35b_a3b_fsdp_sft_config(hf_path: str = "Qwen/Qwen3.5-35B-A3B") -> ConfigContainer:
    """Return a full SFT config for Qwen3.5-VL 35B-A3B (MoE) with Megatron FSDP.

    Uses Megatron FSDP for memory-efficient training with AG/RS overlap.
    Requires fsdp_dtensor checkpoint format (convert offline with
    checkpoint_inspector.py convert-torch-dist-to-fsdp-dtensor).

    Default configuration: 2 nodes, 16 GPUs
    - TP=1, PP=1, EP=2
    - Megatron FSDP with double buffering
    - NCCL UB disabled (heterogeneous FSDP units cause hangs)
    - LR=2e-5 (full SFT)
    - Sequence length: 4096

    Args:
        hf_path: HuggingFace model ID or local path to model directory.
    """
    cfg = _sft_common_vlm()
    _qwen35_vl_apply_common(cfg, hf_path, tp=1, pp=1, max_lr=2e-5, min_lr=2e-6)
    _qwen35_vl_apply_moe(cfg, ep=2)

    # _apply_moe enables SP, but SP requires TP>1
    cfg.model.sequence_parallel = False

    # Megatron FSDP settings
    cfg.ddp.use_megatron_fsdp = True
    cfg.ddp.fsdp_double_buffer = True
    cfg.ddp.nccl_ub = False
    cfg.ddp.fsdp_db_use_persist_buf_on_alloc_fail = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.num_distributed_optimizer_instances = 1

    return cfg


def qwen35_vl_122b_a10b_sft_config(hf_path: str = "Qwen/Qwen3.5-122B-A10B") -> ConfigContainer:
    """Return a full SFT config for Qwen3.5-VL 122B-A10B (MoE).

    Default configuration: 4 nodes, 32 GPUs
    - TP=2, PP=6, EP=8
    - LR=2e-5 (full SFT)
    - Sequence length: 4096

    Args:
        hf_path: HuggingFace model ID or local path to model directory.
    """
    cfg = _sft_common_vlm()
    _qwen35_vl_apply_common(cfg, hf_path, tp=2, pp=6, max_lr=2e-5, min_lr=2e-6, gbs=36)
    _qwen35_vl_apply_moe(cfg, ep=8)
    _qwen35_vl_enable_recompute(cfg)
    return cfg


def qwen35_vl_397b_a17b_sft_config(hf_path: str = "Qwen/Qwen3.5-397B-A17B") -> ConfigContainer:
    """Return a full SFT config for Qwen3.5-VL 397B-A17B (MoE).

    Default configuration: 16 nodes, 128 GPUs
    - TP=2, PP=4, EP=32
    - LR=2e-5 (full SFT)
    - Sequence length: 4096

    Args:
        hf_path: HuggingFace model ID or local path to model directory.
    """
    cfg = _sft_common_vlm()
    _qwen35_vl_apply_common(cfg, hf_path, tp=2, pp=4, max_lr=2e-5, min_lr=2e-6)
    _qwen35_vl_apply_moe(cfg, ep=32)
    _qwen35_vl_enable_recompute(cfg)
    return cfg


# =============================================================================
# Qwen3.5-VL Dense PEFT Configurations (800M, 2B, 4B, 9B, 27B)
# =============================================================================


def qwen35_vl_800m_peft_config(
    peft_scheme: str | PEFT = "lora", hf_path: str = "Qwen/Qwen3.5-0.8B"
) -> ConfigContainer:
    """Return a PEFT config for Qwen3.5-VL 800M (dense).

    Default configuration: 1 node, 8 GPUs
    - TP=1, PP=1
    - LR=1e-4 (PEFT)
    - Sequence length: 4096

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.
        hf_path: HuggingFace model ID or local path to model directory.
    """
    cfg = _peft_common_vlm()
    _qwen35_vl_apply_peft_scheme(cfg, peft_scheme)
    _qwen35_vl_apply_common(cfg, hf_path, tp=1, pp=1, max_lr=1e-4, min_lr=3e-5)
    return cfg


def qwen35_vl_2b_peft_config(peft_scheme: str | PEFT = "lora", hf_path: str = "Qwen/Qwen3.5-2B") -> ConfigContainer:
    """Return a PEFT config for Qwen3.5-VL 2B (dense).

    Default configuration: 1 node, 8 GPUs
    - TP=1, PP=1
    - LR=1e-4 (PEFT)
    - Sequence length: 4096

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.
        hf_path: HuggingFace model ID or local path to model directory.
    """
    cfg = _peft_common_vlm()
    _qwen35_vl_apply_peft_scheme(cfg, peft_scheme)
    _qwen35_vl_apply_common(cfg, hf_path, tp=1, pp=1, max_lr=1e-4, min_lr=3e-5)
    return cfg


def qwen35_vl_4b_peft_config(peft_scheme: str | PEFT = "lora", hf_path: str = "Qwen/Qwen3.5-4B") -> ConfigContainer:
    """Return a PEFT config for Qwen3.5-VL 4B (dense).

    Default configuration: 1 node, 8 GPUs
    - TP=1, PP=1
    - LR=1e-4 (PEFT)
    - Sequence length: 4096

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.
        hf_path: HuggingFace model ID or local path to model directory.
    """
    cfg = _peft_common_vlm()
    _qwen35_vl_apply_peft_scheme(cfg, peft_scheme)
    _qwen35_vl_apply_common(cfg, hf_path, tp=1, pp=1, max_lr=1e-4, min_lr=3e-5)
    return cfg


def qwen35_vl_9b_peft_config(peft_scheme: str | PEFT = "lora", hf_path: str = "Qwen/Qwen3.5-9B") -> ConfigContainer:
    """Return a PEFT config for Qwen3.5-VL 9B (dense).

    Default configuration: 1 node, 8 GPUs
    - TP=1, PP=1
    - LR=1e-4 (PEFT)
    - Sequence length: 4096

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.
        hf_path: HuggingFace model ID or local path to model directory.
    """
    cfg = _peft_common_vlm()
    _qwen35_vl_apply_peft_scheme(cfg, peft_scheme)
    _qwen35_vl_apply_common(cfg, hf_path, tp=1, pp=1, max_lr=1e-4, min_lr=3e-5)
    return cfg


def qwen35_vl_27b_peft_config(peft_scheme: str | PEFT = "lora", hf_path: str = "Qwen/Qwen3.5-27B") -> ConfigContainer:
    """Return a PEFT config for Qwen3.5-VL 27B (dense).

    Default configuration: 1 node, 8 GPUs
    - TP=2, PP=1
    - LR=1e-4 (PEFT)
    - Sequence length: 4096

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.
        hf_path: HuggingFace model ID or local path to model directory.
    """
    cfg = _peft_common_vlm()
    _qwen35_vl_apply_peft_scheme(cfg, peft_scheme)
    _qwen35_vl_apply_common(cfg, hf_path, tp=2, pp=1, max_lr=1e-4, min_lr=3e-5)
    return cfg


# =============================================================================
# Qwen3.5-VL MoE PEFT Configurations (35B-A3B, 122B-A10B, 397B-A17B)
# =============================================================================


def qwen35_vl_35b_a3b_peft_config(
    peft_scheme: str | PEFT = "lora", hf_path: str = "Qwen/Qwen3.5-35B-A3B"
) -> ConfigContainer:
    """Return a PEFT config for Qwen3.5-VL 35B-A3B (MoE).

    Default configuration: 1 node, 8 GPUs
    - TP=2, PP=1, EP=4
    - LR=2e-4 (PEFT)
    - Sequence length: 4096

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.
        hf_path: HuggingFace model ID or local path to model directory.
    """
    cfg = _peft_common_vlm()
    _qwen35_vl_apply_peft_scheme(cfg, peft_scheme)
    _qwen35_vl_apply_common(cfg, hf_path, tp=2, pp=1, max_lr=2e-4, min_lr=3e-5)
    _qwen35_vl_apply_moe(cfg, ep=4)
    return cfg


def qwen35_vl_122b_a10b_peft_config(
    peft_scheme: str | PEFT = "lora", hf_path: str = "Qwen/Qwen3.5-122B-A10B"
) -> ConfigContainer:
    """Return a PEFT config for Qwen3.5-VL 122B-A10B (MoE).

    Default configuration: 2 nodes, 16 GPUs
    - TP=2, PP=1, EP=8
    - LR=2e-4 (PEFT)
    - Sequence length: 4096

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.
        hf_path: HuggingFace model ID or local path to model directory.
    """
    cfg = _peft_common_vlm()
    _qwen35_vl_apply_peft_scheme(cfg, peft_scheme)
    _qwen35_vl_apply_common(cfg, hf_path, tp=2, pp=1, max_lr=2e-4, min_lr=3e-5, gbs=36)
    _qwen35_vl_apply_moe(cfg, ep=8)
    return cfg


def qwen35_vl_397b_a17b_peft_config(
    peft_scheme: str | PEFT = "lora", hf_path: str = "Qwen/Qwen3.5-397B-A17B"
) -> ConfigContainer:
    """Return a PEFT config for Qwen3.5-VL 397B-A17B (MoE).

    Default configuration: 4 nodes, 32 GPUs
    - TP=2, PP=1, EP=32
    - LR=2e-4 (PEFT)
    - Sequence length: 4096

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.
        hf_path: HuggingFace model ID or local path to model directory.
    """
    cfg = _peft_common_vlm()
    _qwen35_vl_apply_peft_scheme(cfg, peft_scheme)
    _qwen35_vl_apply_common(cfg, hf_path, tp=2, pp=1, max_lr=2e-4, min_lr=3e-5)
    _qwen35_vl_apply_moe(cfg, ep=32)
    return cfg
