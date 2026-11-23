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

"""GLU row-layout conversions shared by checkpointing and model conversion."""

import os
import re
from typing import Literal

import torch


GLUFC1Kind = Literal["routed_expert", "shared_expert", "dense"]

_FC1_PARAM_RE = re.compile(r"(?:weight|bias)\d*\Z")
_BIAS_PARAM_RE = re.compile(r"bias\d*\Z")


def classify_glu_fc1(param_name: str) -> GLUFC1Kind | None:
    """Classify a Megatron FC1 parameter that may use GLU interleaving.

    Dense FC1 parameters use the routed-expert interleave configuration only
    when dense activation fusion is explicitly enabled, matching Megatron
    checkpoint loading and saving. Scale metadata, extra state and adapter
    parameters are not FC1 weights or biases and are excluded.

    Args:
        param_name: Fully resolved Megatron parameter name.

    Returns:
        The FC1 kind, or ``None`` when the parameter is not interleaved.
    """
    path = param_name.split(".")
    # LoRA wrappers keep the original parameter under ``linear_fc1.to_wrap``.
    fc1_index = -3 if len(path) >= 3 and path[-2] == "to_wrap" else -2
    if len(path) < -fc1_index or path[fc1_index] != "linear_fc1" or _FC1_PARAM_RE.fullmatch(path[-1]) is None:
        return None
    path_segments = path[:fc1_index]
    if "shared_experts" in path_segments:
        return "shared_expert"
    if "experts" in path_segments:
        return "routed_expert"
    if os.environ.get("USE_ACT_FUSION_FOR_DENSE", "0") == "1" and "mlp" in path_segments:
        return "dense"
    return None


def get_glu_interleave_size(model_config: object, param_name: str) -> int | None:
    """Return the physical GLU interleave size for a Megatron FC1 parameter.

    Args:
        model_config: Finalized Megatron model configuration.
        param_name: Fully resolved Megatron parameter name.

    Returns:
        A positive interleave size when the parameter uses the physical
        block-interleaved layout, otherwise ``None``.

    Raises:
        ValueError: If the selected configuration value is not a positive
            integer or ``None``.
    """
    # This is the same activation guard used by MCore's expert implementations.
    # Some non-GLU configurations retain a tuning value without changing FC1 layout.
    if getattr(model_config, "gated_linear_unit", False) is not True:
        return None
    fc1_kind = classify_glu_fc1(param_name)
    if fc1_kind is None:
        return None

    config_name = (
        "moe_shared_expert_glu_interleave_size" if fc1_kind == "shared_expert" else "moe_mlp_glu_interleave_size"
    )
    if fc1_kind == "shared_expert" and not getattr(model_config, "use_grouped_gemm_for_shared_expert", False):
        return None

    interleave_size = getattr(model_config, config_name, None)
    if interleave_size is None:
        return None
    if isinstance(interleave_size, bool) or not isinstance(interleave_size, int) or interleave_size <= 0:
        raise ValueError(f"{config_name} must be a positive integer or None, got {interleave_size!r}")
    return interleave_size


def _validate_glu_layout(tensor: torch.Tensor, interleave_size: int, row_dim: int) -> int:
    if isinstance(interleave_size, bool) or not isinstance(interleave_size, int) or interleave_size <= 0:
        raise ValueError(f"GLU interleave size must be a positive integer, got {interleave_size!r}")
    if tensor.ndim == 0:
        raise ValueError("GLU interleaving requires a tensor with at least one dimension")
    if not -tensor.ndim <= row_dim < tensor.ndim:
        raise ValueError(f"GLU projection row dimension {row_dim} is invalid for shape {tuple(tensor.shape)}")
    row_dim %= tensor.ndim
    block_pair_size = 2 * interleave_size
    if tensor.shape[row_dim] % block_pair_size:
        raise ValueError(
            f"GLU projection rows ({tensor.shape[row_dim]}) must be divisible by "
            f"2 * interleave_size ({block_pair_size})"
        )
    return row_dim


def interleave_glu_tensor(tensor: torch.Tensor, interleave_size: int, *, row_dim: int = 0) -> torch.Tensor:
    """Convert a logical contiguous GLU tensor to Megatron's physical layout.

    The logical projection-row dimension is ``[gate; up]``. The physical
    projection-row dimension alternates fixed-size gate and up blocks.

    Args:
        tensor: GLU FC1 weight or bias in logical ``[gate; up]`` order.
        interleave_size: Number of gate or up rows in each physical block.
        row_dim: Projection-row dimension. Defaults to the leading dimension.

    Returns:
        A contiguous tensor in the block-interleaved physical layout.
    """
    row_dim = _validate_glu_layout(tensor, interleave_size, row_dim)
    shape = tensor.shape
    return (
        tensor.reshape(
            *shape[:row_dim],
            2,
            shape[row_dim] // (2 * interleave_size),
            interleave_size,
            *shape[row_dim + 1 :],
        )
        .transpose(row_dim, row_dim + 1)
        .contiguous()
        .reshape(shape)
    )


def deinterleave_glu_tensor(tensor: torch.Tensor, interleave_size: int, *, row_dim: int = 0) -> torch.Tensor:
    """Convert a physical block-interleaved GLU tensor to logical order.

    Args:
        tensor: GLU FC1 weight or bias in Megatron's physical layout.
        interleave_size: Number of gate or up rows in each physical block.
        row_dim: Projection-row dimension. Defaults to the leading dimension.

    Returns:
        A contiguous tensor in logical ``[gate; up]`` order.
    """
    row_dim = _validate_glu_layout(tensor, interleave_size, row_dim)
    shape = tensor.shape
    return (
        tensor.reshape(
            *shape[:row_dim],
            shape[row_dim] // (2 * interleave_size),
            2,
            interleave_size,
            *shape[row_dim + 1 :],
        )
        .transpose(row_dim, row_dim + 1)
        .contiguous()
        .reshape(shape)
    )


def _get_glu_projection_row_dim(tensor: torch.Tensor, param_name: str) -> int:
    """Return the FC1 projection-row dimension for ordinary or grouped storage."""
    if tensor.ndim == 0:
        return 0
    is_bias = _BIAS_PARAM_RE.fullmatch(param_name.rsplit(".", 1)[-1]) is not None
    return max(tensor.ndim - (1 if is_bias else 2), 0)


def logical_to_physical_glu_tensor(
    tensor: torch.Tensor,
    *,
    model_config: object,
    param_name: str,
) -> torch.Tensor:
    """Apply the configured logical-to-physical GLU conversion when needed.

    Args:
        tensor: Logical GLU FC1 weight or bias.
        model_config: Finalized Megatron model configuration.
        param_name: Fully resolved Megatron parameter name.

    Returns:
        The original tensor when interleaving is disabled, otherwise a
        contiguous tensor in the configured physical layout.
    """
    interleave_size = get_glu_interleave_size(model_config, param_name)
    if interleave_size is None:
        return tensor
    return interleave_glu_tensor(
        tensor,
        interleave_size,
        row_dim=_get_glu_projection_row_dim(tensor, param_name),
    )


def physical_to_logical_glu_tensor(
    tensor: torch.Tensor,
    *,
    model_config: object,
    param_name: str,
) -> torch.Tensor:
    """Apply the configured physical-to-logical GLU conversion when needed.

    Quantized tensors must be dequantized before calling this function so that
    physical FP8 data remains paired with its scale metadata. Callers with a
    tensor containing multiple TP-local layouts must convert each shard before
    concatenating across TP.
    Leading expert/batch axes are preserved for grouped weights and biases.
    This is a one-time layout conversion, not an idempotent normalization.

    Args:
        tensor: GLU FC1 weight or bias in Megatron's physical layout.
        model_config: Finalized Megatron model configuration.
        param_name: Fully resolved Megatron parameter name.

    Returns:
        The original tensor when interleaving is disabled, otherwise a
        contiguous tensor in logical ``[gate; up]`` order.
    """
    interleave_size = get_glu_interleave_size(model_config, param_name)
    if interleave_size is None:
        return tensor
    return deinterleave_glu_tensor(
        tensor,
        interleave_size,
        row_dim=_get_glu_projection_row_dim(tensor, param_name),
    )
