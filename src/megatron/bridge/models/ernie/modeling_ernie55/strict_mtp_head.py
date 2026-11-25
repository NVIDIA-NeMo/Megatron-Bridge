# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP1 BF16 MTP backward with the reference's softmax and projection rounding boundaries."""

import torch
from megatron.core.jit import jit_fuser

from .output_layer import ernie55_multimax


@jit_fuser
def _exp_and_sum(values, row_max):
    values -= row_max.unsqueeze(-1)
    torch.exp(values, out=values)
    return values, values.sum(dim=-1)


@jit_fuser
def _normalize(values, denominator):
    return values.div_(denominator.unsqueeze(-1))


@jit_fuser
def _scale_ce_gradient(probabilities, labels, loss_grad):
    vocab = probabilities.size(-1)
    valid = (labels >= 0) & (labels < vocab)
    safe_labels = labels.masked_fill(~valid, 0)
    rows = torch.arange(labels.numel(), device=labels.device)
    probabilities[rows, safe_labels] -= valid.to(probabilities.dtype)
    return probabilities.mul_(loss_grad.unsqueeze(-1)).to(torch.bfloat16)


def _corrected_logits_gradient(corrected, labels, loss_grad):
    # Materialize FP32 exp, row denominator, then normalized values in the same order as
    # the reference TP1 native CE. Its multimax derivative consumes an intervening BF16 cast.
    values = corrected.flatten(0, -2).float()
    values, denominator = _exp_and_sum(values, values.max(dim=-1).values)
    probabilities = _normalize(values, denominator)
    return _scale_ce_gradient(probabilities, labels, loss_grad).view_as(corrected)


def accumulate_strict_mtp_gradients(ctx, loss_grad):
    """Compute dHidden, dRanges, dScales and accumulate the projection's fp32 main_grad."""
    import fused_weight_gradient_mlp_cuda

    hidden, weight, ranges, scales, labels, _, _, raw = ctx.saved_tensors
    ce_rows, projection_rows = 512, 1024
    dx = torch.empty_like(hidden)
    d_ranges = torch.zeros_like(ranges, dtype=torch.float32)
    d_scales = torch.zeros_like(scales, dtype=torch.float32)
    buffer = hidden.new_empty(min(len(hidden), projection_rows), weight.size(0))

    # When forward did not retain raw logits, repeat each forward GEMM at its original shape.
    forward_rows = max(len(hidden), 1) if ctx.save_projection else ctx.rows
    for block in range(0, len(hidden), forward_rows):
        stop = min(block + forward_rows, len(hidden))
        logits = raw[block:stop] if ctx.save_projection else hidden[block:stop] @ weight.T
        for start in range(block, stop, projection_rows):
            end = min(start + projection_rows, stop)
            for i in range(start, end, ce_rows):
                j = min(i + ce_rows, end)
                with torch.enable_grad():
                    x = logits[i - block : j - block].detach().requires_grad_(True)
                    r = ranges.detach().float().requires_grad_(True)
                    t = scales.detach().float().requires_grad_(True)
                    corrected = ernie55_multimax(x, r, t, fused=True)
                gradient = _corrected_logits_gradient(corrected.detach(), labels[i:j], loss_grad[i:j])
                draw, dr, dt = torch.autograd.grad(corrected, (x, r, t), gradient)
                buffer[i - start : j - start].copy_(draw)
                d_ranges.add_(dr)
                d_scales.add_(dt)
                del corrected, gradient, draw
            d_raw = buffer[: end - start]
            torch.mm(d_raw, weight, out=dx[start:end])
            fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(hidden[start:end], d_raw, weight.main_grad)
    return dx, d_ranges.to(ranges.dtype), d_scales.to(scales.dtype)
