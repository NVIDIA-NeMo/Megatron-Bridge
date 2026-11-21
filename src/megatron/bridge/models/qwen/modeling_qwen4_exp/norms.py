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

# The adapted Megatron-LM kernels retain the upstream BSD-3-Clause notice:
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


"""Grouped RMSNorm with optional fused kernels and bounded FP32 backward."""

from typing import cast

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx, once_differentiable


try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


_WORKSPACE_BYTES = 64 * 1024 * 1024


# Specialization of _gr_fwd_norm_kernel (HAS_GATE=False, ZERO_CENTERED=True):
# NVIDIA/Megatron-LM PR #7553, commit 53ff95854b024e36ce7b51a61e65d43d3ced8dac.
# https://github.com/NVIDIA/Megatron-LM/pull/7553
# Preserve its fixed tiles; no autotuning or write-gate/full-GR fusion is involved.
_NORM_BM, _NORM_BC, _NORM_WARPS = 8, 128, 8

if triton is not None:

    @triton.jit
    def _grouped_rmsnorm_fwd_kernel(
        x_ptr,
        weight_ptr,
        rstd_ptr,
        output_ptr,
        rows_count,
        eps,
        GROUPS: tl.constexpr,
        CHANNELS: tl.constexpr,
        BM: tl.constexpr,
        BC: tl.constexpr,
    ):
        width: tl.constexpr = GROUPS * CHANNELS
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        group = tl.program_id(1)
        row_mask = rows < rows_count
        base = rows[:, None] * width + group * CHANNELS
        squares = tl.zeros([BM, BC], dtype=tl.float32)
        for start in range(0, CHANNELS, BC):
            channels = start + tl.arange(0, BC)
            mask = row_mask[:, None] & (channels[None, :] < CHANNELS)
            values = tl.load(x_ptr + base + channels[None, :], mask=mask, other=0.0).to(tl.float32)
            squares += values * values
        rstd = tl.rsqrt(tl.sum(squares, axis=1) / CHANNELS + eps)
        tl.store(rstd_ptr + rows * GROUPS + group, rstd, mask=row_mask)
        for start in range(0, CHANNELS, BC):
            channels = start + tl.arange(0, BC)
            channel_mask = channels < CHANNELS
            mask = row_mask[:, None] & channel_mask[None, :]
            values = tl.load(x_ptr + base + channels[None, :], mask=mask, other=0.0).to(tl.float32)
            gain = tl.load(weight_ptr + group * CHANNELS + channels, mask=channel_mask, other=0.0).to(tl.float32)
            gain = gain + 1.0
            normalized = values * rstd[:, None] * gain[None, :]
            tl.store(output_ptr + base + channels[None, :], normalized.to(output_ptr.dtype.element_ty), mask=mask)

    # Norm-only specialization of PR #7553's contiguous row-split backward
    # and fixed-order parameter reduction (same commit/link as above). Compute
    # the projection directly from existing dY; no gates, residual add or
    # main_grad writes. Each program owns one group and a contiguous row split.
    @triton.jit
    def _grouped_rmsnorm_bwd_kernel(
        x_ptr,
        weight_ptr,
        rstd_ptr,
        gradient_ptr,
        dx_ptr,
        partial_ptr,
        rows_count,
        GROUPS: tl.constexpr,
        CHANNELS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        partition = tl.program_id(0)
        partitions = tl.num_programs(0)
        group = tl.program_id(1)
        channels = tl.arange(0, BLOCK)
        mask = channels < CHANNELS
        width: tl.constexpr = GROUPS * CHANNELS
        columns = group * CHANNELS + channels
        gain = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32) + 1.0
        partial = tl.zeros([BLOCK], dtype=tl.float32)
        rows_per_split = tl.cdiv(rows_count, partitions)
        row_begin = partition * rows_per_split
        row_end = tl.minimum(row_begin + rows_per_split, rows_count)
        for row in range(row_begin, row_end):
            offsets = row * width + columns
            values = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            gradient = tl.load(gradient_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            rstd = tl.load(rstd_ptr + row * GROUPS + group)
            normalized = values * rstd
            scaled = gradient * gain
            projection = tl.sum(scaled * normalized, axis=0) * (1.0 / CHANNELS)
            dx = rstd * (scaled - normalized * projection)
            tl.store(dx_ptr + offsets, dx.to(dx_ptr.dtype.element_ty), mask=mask)
            partial += gradient * normalized
        tl.store(partial_ptr + partition * width + columns, partial, mask=mask)

    @triton.jit
    def _grouped_rmsnorm_dw_kernel(partial_ptr, dw_ptr, partitions, width, BLOCK: tl.constexpr):
        # Fixed-order finalizer specialized from PR #7553's
        # _gr_param_reduce_kernel at the attributed commit above. No atomics or
        # main_grad updates: autograd owns parameter-gradient accumulation.
        columns = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = columns < width
        total = tl.zeros([BLOCK], dtype=tl.float32)
        for partition in range(partitions):
            total += tl.load(partial_ptr + partition * width + columns, mask=mask, other=0.0)
        tl.store(dw_ptr + columns, total.to(dw_ptr.dtype.element_ty), mask=mask)


def _can_use_triton(x: Tensor, weight: Tensor) -> bool:
    return (
        triton is not None
        and x.is_cuda
        and torch.version.hip is None
        and 0 < x.numel() < 2**31  # Kernel offsets use signed 32-bit indexing.
        and x.is_contiguous()
        and weight.is_contiguous()
        and x.device == weight.device
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and weight.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


def norm_chunk_rows(width: int, workspace_bytes: int) -> int:
    """Budget eight row-width FP32 intermediates, excluding required output gradients.

    A single row is the minimum allocation when its width exceeds the budget.
    """
    if width < 1 or workspace_bytes < 1:
        raise ValueError("Normalization width and workspace budget must be positive.")
    return max(1, workspace_bytes // (8 * width * 4))


def _backward_partitions(rows: int, width: int, groups: int, workspace_bytes: int, multiprocessors: int) -> int:
    """Limit FP32 partials to the budget and about two resident programs per SM."""
    return min(rows, (2 * multiprocessors + groups - 1) // groups, workspace_bytes // (width * 4))


class _GroupedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor, weight: Tensor, groups: int, eps: float, workspace_bytes: int) -> Tensor:
        width = x.shape[-1]
        rows = x.reshape(-1, width)
        channels = width // groups
        chunk = norm_chunk_rows(width, workspace_bytes)
        output = torch.empty_like(rows)
        rstd = torch.empty(rows.shape[0], groups, 1, dtype=torch.float32, device=x.device)
        if _can_use_triton(x, weight):
            # Keep eager's separate FP32 product/add rounding. The tiled sum order
            # still differs from torch.mean; both save FP32 rstd for the same backward.
            with torch.cuda.device(x.device):
                _grouped_rmsnorm_fwd_kernel[(triton.cdiv(rows.shape[0], _NORM_BM), groups)](
                    rows,
                    weight,
                    rstd,
                    output,
                    rows.shape[0],
                    eps,
                    GROUPS=groups,
                    CHANNELS=channels,
                    BM=_NORM_BM,
                    BC=_NORM_BC,
                    num_warps=_NORM_WARPS,
                    enable_fp_fusion=False,
                )
        else:
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
        ctx.workspace_bytes = workspace_bytes
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
        if (
            _can_use_triton(x, weight)
            and grad_output.is_contiguous()
            and grad_output.device == x.device
            and grad_output.dtype == x.dtype
            and channels <= 16384
            and ctx.workspace_bytes >= width * 4
        ):
            with torch.cuda.device(x.device):
                partitions = _backward_partitions(
                    rows.shape[0],
                    width,
                    ctx.groups,
                    ctx.workspace_bytes,
                    torch.cuda.get_device_properties(x.device).multi_processor_count,
                )
                partials = torch.empty(partitions, width, dtype=torch.float32, device=x.device)
                dw = torch.empty_like(weight)
                _grouped_rmsnorm_bwd_kernel[(partitions, ctx.groups)](
                    rows,
                    weight,
                    rstd,
                    gradients,
                    dx,
                    partials,
                    rows.shape[0],
                    GROUPS=ctx.groups,
                    CHANNELS=channels,
                    BLOCK=triton.next_power_of_2(channels),
                    num_warps=8,
                    enable_fp_fusion=False,
                )
                _grouped_rmsnorm_dw_kernel[(triton.cdiv(width, 256),)](
                    partials,
                    dw,
                    partitions,
                    width,
                    BLOCK=256,
                    num_warps=8,
                    enable_fp_fusion=False,
                )
            return dx.reshape(x.shape), dw, None, None, None
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
    accumulate in FP32; chunking changes only their reduction order. Contiguous
    CUDA inputs optionally use Triton kernels, whose FP32 reduction orders can
    differ from the eager fallback. Backward always uses the saved rstd, with
    deterministic FP32 weight-gradient partials bounded by the workspace budget.
    """
    if x.ndim < 1 or groups < 1 or x.shape[-1] < 1 or x.shape[-1] % groups:
        raise ValueError("Input width must be positive and divisible by the residual stream count.")
    if weight.ndim != 1 or weight.numel() != x.shape[-1] or eps <= 0:
        raise ValueError("Grouped RMSNorm requires one gain per channel and positive epsilon.")
    # Function.apply has an untyped return in PyTorch; forward returns one Tensor.
    return cast(Tensor, _GroupedRMSNorm.apply(x, weight, groups, eps, workspace_bytes))
