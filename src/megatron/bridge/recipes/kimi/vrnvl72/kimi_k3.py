# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Early Kimi K3 training configurations for VR NVL72."""

from megatron.bridge.recipes.kimi.kimi_k3 import _kimi_k3_pretrain_common
from megatron.bridge.training.config import ConfigContainer


def kimi_k3_pretrain_512gpu_vrnvl72_bf16_config() -> ConfigContainer:
    """Configure early, perf not optimized Kimi K3 BF16 training on 512 VR GPUs.

    Preserves the GB200 recipe's TP4/PP4/EP128, 24/23/23/23 pipeline,
    full activation recomputation, 8K sequences, and global batch 128.
    CP must remain one and VPP must remain disabled. This is an unvalidated
    Rubin port; the GB200 validation does not establish Rubin performance.

    Use scripts/training/run_recipe.py with this recipe name. Set
    KIMI_K3_HF_PATH to a local snapshot when offline, supply K3-tokenized
    indexed data, and configure checkpoint.pretrained_checkpoint for HF
    initialization. Preserve the learning-rate horizon in shortened runs.
    """
    return _kimi_k3_pretrain_common()
