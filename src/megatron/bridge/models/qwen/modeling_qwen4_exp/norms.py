# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Grouped RMSNorm with bounded FP32 reduction workspaces."""

from typing import cast

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx, once_differentiable


_WORKSPACE_BYTES = 64 * 1024 * 1024


def norm_chunk_rows(width: int, workspace_bytes: int) -> int:
    """Budget eight row-width FP32 intermediates, excluding required output gradients.

    A single row is the minimum allocation when its width exceeds the budget.
    """
    if width < 1 or workspace_bytes < 1:
        raise ValueError("Normalization width and workspace budget must be positive.")
    return max(1, workspace_bytes // (8 * width * 4))


class _GroupedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor, weight: Tensor, groups: int, eps: float, workspace_bytes: int) -> Tensor:
        width = x.shape[-1]
        rows = x.reshape(-1, width)
        channels = width // groups
        chunk = norm_chunk_rows(width, workspace_bytes)
        output = torch.empty_like(rows)
        rstd = torch.empty(rows.shape[0], groups, 1, dtype=torch.float32, device=x.device)
        gain = (1.0 + weight.float()).reshape(groups, channels)
        for start in range(0, rows.shape[0], chunk):
            end = min(start + chunk, rows.shape[0])
            values = rows[start:end].float().reshape(-1, groups, channels)
            scale = torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
            rstd[start:end] = scale
            output[start:end] = (values * scale * gain).flatten(-2).to(x.dtype)
        ctx.save_for_backward(x, weight, rstd)
        ctx.groups = groups
        ctx.chunk = chunk
        return output.reshape(x.shape)

    @staticmethod
    @once_differentiable
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> tuple[Tensor, Tensor, None, None, None]:
        x, weight, rstd = ctx.saved_tensors
        width = x.shape[-1]
        rows = x.reshape(-1, width)
        gradients = grad_output.reshape(-1, width)
        channels = width // ctx.groups
        dx = torch.empty_like(rows)
        dw = torch.zeros_like(weight, dtype=torch.float32)
        gain = (1.0 + weight.float()).reshape(ctx.groups, channels)
        for start in range(0, rows.shape[0], ctx.chunk):
            end = min(start + ctx.chunk, rows.shape[0])
            normalized = rows[start:end].float().reshape(-1, ctx.groups, channels) * rstd[start:end]
            gradient = gradients[start:end].float().reshape(-1, ctx.groups, channels)
            dw.add_((gradient * normalized).flatten(-2).sum(0))
            scaled_gradient = gradient * gain
            projection = (scaled_gradient * normalized).mean(-1, keepdim=True)
            dx[start:end] = (rstd[start:end] * (scaled_gradient - normalized * projection)).flatten(-2).to(x.dtype)
        return dx.reshape(x.shape), dw.to(weight.dtype), None, None, None


def grouped_rmsnorm(
    x: Tensor,
    weight: Tensor,
    groups: int,
    eps: float,
    *,
    workspace_bytes: int = _WORKSPACE_BYTES,
) -> Tensor:
    """Normalize streams with zero-centered gains and chunked backward reductions.

    Saves the activation and one FP32 inverse RMS per stream instead of the
    token-sized FP32 products that eager autograd retains. Weight gradients
    accumulate in FP32; chunking changes only their reduction order.
    """
    if x.ndim < 1 or groups < 1 or x.shape[-1] < 1 or x.shape[-1] % groups:
        raise ValueError("Input width must be positive and divisible by the residual stream count.")
    if weight.ndim != 1 or weight.numel() != x.shape[-1] or eps <= 0:
        raise ValueError("Grouped RMSNorm requires one gain per channel and positive epsilon.")
    # Function.apply has an untyped return in PyTorch; forward returns one Tensor.
    return cast(Tensor, _GroupedRMSNorm.apply(x, weight, groups, eps, workspace_bytes))
