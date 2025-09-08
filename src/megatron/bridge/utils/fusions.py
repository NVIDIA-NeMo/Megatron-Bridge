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

"""Fusion capability checks for Megatron models.

This module provides functions to check if various fusion optimizations
can be enabled based on the current environment and dependencies.
"""

import logging
import os

from megatron.core.transformer.transformer_config import TransformerConfig


logger = logging.getLogger(__name__)

# Control whether to log warnings when fusions are disabled
# Set environment variable MEGATRON_SUPPRESS_FUSION_WARNINGS=1 to disable warnings
LOG_FUSION_DISABLE = os.environ.get("MEGATRON_SUPPRESS_FUSION_WARNINGS", "0") != "1"


def can_enable_apply_rope_fusion() -> bool:
    """Check if RoPE (Rotary Position Embedding) fusion can be enabled.

    Returns:
        bool: True if RoPE fusion is available and compatible.
    """
    # Check for Transformer Engine availability
    try:
        import transformer_engine  # noqa: F401
        from megatron.core.utils import get_te_version, is_te_min_version

        if not is_te_min_version("2.2.0.dev0"):
            if LOG_FUSION_DISABLE:
                logger.warning(
                    "apply_rope_fusion requires Transformer Engine >= 2.2.0.dev0. "
                    f"Current version: {get_te_version()}. Fusion disabled."
                )
            return False
    except ImportError:
        if LOG_FUSION_DISABLE:
            logger.warning("apply_rope_fusion requires Transformer Engine but it is not installed. Fusion disabled.")
        return False

    # Check for RoPE fusion kernel availability
    try:
        from megatron.core.models.common.embeddings.rope_utils import (
            fused_apply_rotary_pos_emb,
            fused_apply_rotary_pos_emb_thd,
        )

        if fused_apply_rotary_pos_emb is None and fused_apply_rotary_pos_emb_thd is None:
            if LOG_FUSION_DISABLE:
                logger.warning("apply_rope_fusion kernels are not available in megatron.core. Fusion disabled.")
            return False
        return True
    except ImportError:
        if LOG_FUSION_DISABLE:
            logger.warning(
                "apply_rope_fusion requires RoPE fusion kernels from megatron.core but they are not available. "
                "Fusion disabled."
            )
        return False


def can_enable_gradient_accumulation_fusion() -> bool:
    """Check if gradient accumulation fusion can be enabled.

    Returns:
        bool: True if gradient accumulation fusion is available.
    """
    try:
        import fused_weight_gradient_mlp_cuda  # noqa: F401

        return True
    except ImportError:
        if LOG_FUSION_DISABLE:
            logger.warning(
                "gradient_accumulation_fusion requires FusedLayerNorm from megatron.core.fusions "
                "but it is not available. Fusion disabled."
            )
        return False


def can_enable_bias_dropout_fusion() -> bool:
    """Check if bias dropout fusion can be enabled.

    Returns:
        bool: True if bias dropout fusion is available.
    """
    try:
        from megatron.core.fusions.fused_bias_dropout import bias_dropout_add_fused_train  # noqa: F401

        return True
    except ImportError:
        if LOG_FUSION_DISABLE:
            logger.warning(
                "bias_dropout_fusion requires fused_bias_dropout from megatron.core.fusions "
                "but it is not available. Fusion disabled."
            )
        return False


def can_enable_masked_softmax_fusion() -> bool:
    """Check if masked softmax fusion can be enabled.

    Returns:
        bool: True if masked softmax fusion kernels are available.
    """
    try:
        # Try to import the CUDA kernels that are required for masked softmax fusion
        import scaled_masked_softmax_cuda  # noqa: F401
        import scaled_upper_triang_masked_softmax_cuda  # noqa: F401

        return True
    except ImportError:
        if LOG_FUSION_DISABLE:
            logger.warning(
                "masked_softmax_fusion requires CUDA kernels (scaled_masked_softmax_cuda) "
                "but they are not available. This typically happens when Megatron-Core is not "
                "built with CUDA extensions. Fusion disabled."
            )
        return False


def validate_rope_fusion_compatibility(config: TransformerConfig) -> bool:
    """Validate if RoPE fusion is compatible with the current model configuration.

    Args:
        model_provider: The GPTModelProvider instance to validate.

    Returns:
        bool: True if RoPE fusion is compatible, False otherwise.
    """
    if not config.apply_rope_fusion:
        return True

    # Check for multi_latent_attention incompatibility
    if getattr(config, "multi_latent_attention", False):
        if LOG_FUSION_DISABLE:
            logger.warning(
                "apply_rope_fusion for multi-latent attention only supports training. "
                "It is experimental and may change in future versions."
            )
        return True

    # Check TE version for rotary_interleaved
    if getattr(config, "rotary_interleaved", False):
        try:
            from megatron.core.utils import get_te_version, is_te_min_version

            if not is_te_min_version("2.2.0.dev0"):
                if LOG_FUSION_DISABLE:
                    logger.warning(
                        "apply_rope_fusion with rotary_interleaved requires TE >= 2.2.0.dev0. "
                        f"Current TE version: {get_te_version()}. Consider disabling apply_rope_fusion."
                    )
                return False
        except ImportError:
            if LOG_FUSION_DISABLE:
                logger.warning(
                    "apply_rope_fusion with rotary_interleaved requires Transformer Engine. "
                    "Consider disabling apply_rope_fusion."
                )
            return False

    return True
