# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token-block BF16 linear, multimax and cross entropy for the TP1 ERNIE training head."""

from __future__ import annotations

import logging

import torch
from torch import Tensor


_LOGGER = logging.getLogger(__name__)


class _MultimaxTrainingLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden,
        weight,
        ranges,
        scales,
        labels,
        rows,
        save_projection,
        return_argmax,
        backend,
        reference_wgrad=False,
    ):
        if backend == "cuda":
            from .cuda_head import forward as head_forward
        else:
            from .chunked_multimax_kernels import forward as head_forward

        n = hidden.size(0)
        max_logits = hidden.new_empty(n, dtype=torch.float32)
        denominators = torch.empty_like(max_logits)
        losses = torch.empty_like(max_logits)
        predictions = torch.empty_like(labels) if return_argmax else labels.new_empty(0)
        raw = hidden @ weight.T if save_projection else hidden.new_empty(0)
        rt = torch.cat((ranges, scales)).float()
        for start in range(0, n, rows):
            group = slice(start, min(start + rows, n))
            logits = raw[group] if save_projection else hidden[group] @ weight.T
            head_forward(
                logits,
                rt,
                labels[group],
                max_logits[group],
                denominators[group],
                losses[group],
                predictions[group] if return_argmax else None,
            )
            del logits
        ctx.save_for_backward(hidden, weight, ranges, scales, labels, max_logits, denominators, raw)
        ctx.reference_wgrad = reference_wgrad
        ctx.rows = rows
        ctx.save_projection = save_projection
        ctx.mark_non_differentiable(predictions)
        return losses, predictions

    @staticmethod
    def backward(ctx, loss_grad, unused_predictions_grad):
        import fused_weight_gradient_mlp_cuda

        from . import chunked_multimax_kernels as kernels

        hidden, weight, ranges, scales, labels, max_logits, denominators, raw = ctx.saved_tensors
        dx = torch.empty_like(hidden)
        rt = torch.cat((ranges, scales)).float()
        grad_rt = torch.zeros_like(rt)
        full_dlogits = hidden.new_empty((hidden.size(0), weight.size(0))) if ctx.reference_wgrad else None
        for start in range(0, hidden.size(0), ctx.rows):
            group = slice(start, min(start + ctx.rows, hidden.size(0)))
            logits = raw[group] if ctx.save_projection else hidden[group] @ weight.T
            dlogits, partial = kernels.backward(
                logits,
                rt,
                labels[group],
                max_logits[group],
                denominators[group],
                loss_grad[group],
            )
            grad_rt.add_(partial)
            dx[group] = dlogits @ weight
            if full_dlogits is None:
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(hidden[group], dlogits, weight.main_grad)
            else:
                full_dlogits[group].copy_(dlogits)
            del dlogits, logits
        dw = (
            full_dlogits.T @ hidden if full_dlogits is not None else _weight_gradient_hook_tensor(weight, hidden.dtype)
        )
        return (
            dx,
            dw,
            grad_rt[:4].to(ranges.dtype),
            grad_rt[4:].to(scales.dtype),
            *(None,) * (len(ctx.needs_input_grad) - 4),
        )


def _weight_gradient_hook_tensor(weight, dtype):
    if not hasattr(weight, "grad_added_to_main_grad"):
        return None
    from transformer_engine.pytorch.module.base import get_dummy_wgrad

    dummy = get_dummy_wgrad(list(weight.shape), dtype, zero=getattr(weight, "zero_out_wgrad", False))
    weight.grad_added_to_main_grad = True
    return dummy


class _StrictMtpTrainingLoss(_MultimaxTrainingLoss):
    """Native CE rounding and BF16 projection VJP shared by either head when requested."""

    @staticmethod
    def backward(ctx, loss_grad, unused_predictions_grad):
        from .strict_mtp_head import accumulate_strict_mtp_gradients

        hidden, weight = ctx.saved_tensors[:2]
        full_dlogits = hidden.new_empty((hidden.size(0), weight.size(0))) if ctx.reference_wgrad else None
        dx, d_ranges, d_scales = accumulate_strict_mtp_gradients(ctx, loss_grad, full_dlogits=full_dlogits)
        dw = (
            full_dlogits.T @ hidden if full_dlogits is not None else _weight_gradient_hook_tensor(weight, hidden.dtype)
        )
        return dx, dw, d_ranges, d_scales, *(None,) * (len(ctx.needs_input_grad) - 4)


def chunked_training_loss(
    layer: torch.nn.Module,
    hidden: Tensor,
    labels: Tensor,
    weight: Tensor | None,
    *,
    return_argmax: bool = False,
    head_kind: str = "main",
) -> tuple[Tensor, Tensor | None] | None:
    """Return per-token losses [B,S] and optional MTP predictions [S,B] for the D2-Z envelope."""
    if not layer.training or not torch.is_grad_enabled():
        return None
    backend = layer.config.ernie_chunked_head_backend
    weight = layer.weight if weight is None else weight
    if not (
        hidden.is_cuda
        and hidden.dtype == weight.dtype == torch.bfloat16
        and layer.output_dtype in (None, torch.bfloat16)
        and weight.requires_grad
        and getattr(weight, "main_grad", None) is not None
        and weight.main_grad.dtype == torch.float32
        and not hasattr(weight, "__fsdp_param__")
        and not layer.sequence_parallel
    ):
        if backend == "cuda":
            raise RuntimeError(
                "The explicit ERNIE CUDA training head needs unsharded CUDA BF16 hidden/weights, "
                "BF16 logits, a trainable projection with an FP32 main_grad, and sequence_parallel=False."
            )
        return None
    reference_wgrad = getattr(layer.config, "ernie_head_reference_wgrad", False)
    owners = None
    if reference_wgrad and layer.config.context_parallel_size > 1:
        from .head_owners import reference_head_owners

        owners = reference_head_owners(layer.config)
        if owners is not None:
            hidden = owners.to_contiguous(hidden)
            labels = owners.to_contiguous(labels, seq_dim=1)
    seq, batch, width = hidden.shape
    strict = (head_kind == "mtp" and getattr(layer.config, "ernie_mtp_strict_head_backward", False)) or (
        head_kind == "main" and getattr(layer.config, "ernie_main_strict_head_backward", False)
    )
    loss_function = _StrictMtpTrainingLoss if strict else _MultimaxTrainingLoss
    losses, predictions = loss_function.apply(
        hidden.reshape(-1, width),
        weight,
        layer.multimax_ranges,
        layer.multimax_ts,
        labels.transpose(0, 1).contiguous().view(-1),
        layer.config.ernie_chunked_head_rows,
        layer.config.ernie_head_save_projection,
        return_argmax,
        backend,
        reference_wgrad,
    )
    if backend == "cuda":
        seen = getattr(layer, "_ernie_cuda_head_seen", None)
        if seen is None:
            seen = layer._ernie_cuda_head_seen = set()
        if head_kind not in seen:
            seen.add(head_kind)
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            _LOGGER.warning(
                "ERNIE55_CUDA_HEAD dispatch=%s rank=%s device=%s tokens=%s vocab=%s rows=%s argmax=%s",
                head_kind,
                rank,
                hidden.device,
                seq * batch,
                weight.shape[0],
                layer.config.ernie_chunked_head_rows,
                return_argmax,
            )
    losses = losses.view(seq, batch).T.contiguous()
    predictions = predictions.view(seq, batch) if return_argmax else None
    if owners is not None:
        losses = owners.to_zigzag(losses, seq_dim=1)
        predictions = owners.to_zigzag(predictions)
    return losses, predictions
