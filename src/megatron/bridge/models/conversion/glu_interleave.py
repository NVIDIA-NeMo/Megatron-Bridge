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

"""Utilities for converting fused GLU tensors between contiguous and interleaved layouts."""

import torch


def _validate_glu_tensor(tensor: torch.Tensor, interleave_size: int) -> None:
    if tensor.ndim == 0:
        raise ValueError("GLU tensor rank must be at least 1")
    if not isinstance(interleave_size, int) or isinstance(interleave_size, bool) or interleave_size <= 0:
        raise ValueError(f"interleave_size must be positive, got {interleave_size!r}")

    block_size = 2 * interleave_size
    if tensor.shape[0] % block_size != 0:
        raise ValueError(
            f"GLU tensor dim 0 size {tensor.shape[0]} must be divisible by 2 * interleave_size ({block_size})"
        )


def interleave_glu_tensor(
    tensor: torch.Tensor,
    interleave_size: int,
) -> torch.Tensor:
    """Convert a fused GLU tensor from contiguous to block-interleaved layout."""
    _validate_glu_tensor(tensor, interleave_size)
    shape = tensor.shape
    tensor = tensor.reshape(2, shape[0] // (2 * interleave_size), interleave_size, *shape[1:])
    return tensor.transpose(0, 1).contiguous().reshape(shape)


def deinterleave_glu_tensor(
    tensor: torch.Tensor,
    interleave_size: int,
) -> torch.Tensor:
    """Convert a fused GLU tensor from block-interleaved to contiguous layout."""
    _validate_glu_tensor(tensor, interleave_size)
    shape = tensor.shape
    tensor = tensor.reshape(shape[0] // (2 * interleave_size), 2, interleave_size, *shape[1:])
    return tensor.transpose(0, 1).contiguous().reshape(shape)
