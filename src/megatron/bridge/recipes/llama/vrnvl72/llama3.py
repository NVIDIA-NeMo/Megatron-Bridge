# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Llama 3.1 training configurations for VR NVL72."""

from megatron.bridge.recipes.llama.gb200.llama3 import _llama31_405b_pretrain_64gpu_fp8mx_common
from megatron.bridge.training.config import ConfigContainer


def llama31_405b_pretrain_64gpu_vrnvl72_fp8mx_config() -> ConfigContainer:
    """Configure 405B MXFP8 pretraining at 8K on 64 VR NVL72 GPUs.

    Shares the GB200 64-GPU training configuration: TP8/PP8/VPP8/CP1,
    sequence parallelism, tensor-parallel communication overlap, no activation
    recomputation, and no CUDA graphs. Place all sixteen four-GPU nodes in
    one NVLink domain. Global batch 1536 and microbatch one give 1536
    microbatches per optimizer update at DP1.

    MXFP8 parameter gather reuses the gradient buffer. Gradient accumulation
    is BF16; optimizer master parameters and Adam moments remain FP32.
    The training schedule, checkpointing, and numerical checks retain their
    training defaults. The default dataset is synthetic; configure a tokenizer
    and indexed dataset for real-text training.
    """
    return _llama31_405b_pretrain_64gpu_fp8mx_common()
