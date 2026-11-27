# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Llama 3.1 training configurations for VR NVL72."""

from megatron.bridge.recipes.llama.gb200.llama3 import _llama31_405b_pretrain_64gpu_fp8mx_common
from megatron.bridge.training.config import ConfigContainer


def llama31_405b_pretrain_64gpu_vrnvl72_fp8mx_config() -> ConfigContainer:
    """Configure 405B MXFP8 pretraining at 8K on 64 VR NVL72 GPUs.

    Uses TP4/PP8/VPP4/CP1 with sequence parallelism, tensor-parallel
    communication overlap disabled, no activation recomputation, and no
    CUDA graphs. Place all sixteen four-GPU nodes in one NVLink domain.
    Global batch 1536 and microbatch one give 768 microbatches per optimizer
    update at DP2.

    MXFP8 parameter gather reuses the gradient buffer. Gradient accumulation
    is BF16; optimizer master parameters and Adam moments remain FP32.
    The training schedule, checkpointing, and numerical checks retain their
    training defaults. The default dataset is synthetic; configure a tokenizer
    and indexed dataset for real-text training. Causal attention is provided
    by the attention backend without materializing a dense dataset mask.
    """
    cfg = _llama31_405b_pretrain_64gpu_fp8mx_common()
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.comm_overlap.tp_comm_overlap = False
    cfg.dataset.create_attention_mask = False
    return cfg
