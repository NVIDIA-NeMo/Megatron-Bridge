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

"""ERNIE 5.5 multimax output head.

ERNIE 5.5 passes every logit ``x`` of the language-model head through a learned piecewise
correction with ranges ``r`` and scales ``t`` (four of each):

``multimax(x) = x + t0 relu(r0 - x) + t1 relu(x - r1) + t2 relu(r2 - x)^2 + t3 relu(x - r3)^2``

computed in fp32 and returned in the logits dtype. With ``r = t = 0`` it is the identity. The same
parameters apply to the main logits and to the logits of every MTP depth, because both come out of
the one output layer.

Contract:

* ``multimax_ranges`` and ``multimax_ts`` are ``[4]`` parameters of the output layer in
  ``params_dtype`` (not fp32-kept), zero-initialized (Hugging Face ``model.lm_head.multimax_*``).
* Forward over ``[rows, vocab]`` chunks of :data:`MULTIMAX_CHUNK_ROWS` rows; only the logits and
  the parameters are saved for backward.
* Backward recomputes each chunk with ``r`` and ``t`` as fp32 leaves; the logit gradient comes back
  in the logits dtype, and the ``r`` and ``t`` gradients are summed over chunks in fp32 and cast to
  the parameter dtype once.
* With ``ernie_fused_ops`` (the default) and CUDA logits, the reference implementation's Triton
  kernels replace the chunked forward and backward
  (:mod:`~megatron.bridge.models.ernie.modeling_ernie55.fused_dispatch`): one fp32 pass each, still
  saving only the logits; the ``r`` and ``t`` gradients are summed per 4096-logit block and then over
  the blocks in fp32. CPU logits and ``ernie_fused_ops=False`` keep the chunked path.
* The parameters form their own gradient-norm group, :data:`MULTIMAX_GRAD_NORM_GROUP`: Megatron-Core
  leaves them out of the main (logged) gradient norm and clips them by their own norm with the same
  ``clip_grad``. The group is registered with Megatron-Core lazily
  (:func:`register_multimax_grad_norm_group`), so importing this module changes nothing. Every rank
  must register it, including pipeline stages without an output layer: the optimizer issues one
  flag all-reduce per registered group over a process group that spans the pipeline stages, so
  ranks with different group lists would issue mismatched collectives.
  :class:`~megatron.bridge.models.ernie.modeling_ernie55.model.Ernie55HybridModel` registers it on
  every stage; :func:`install_ernie55_multimax` registers it as well.
* The logits must exist: fused linear cross-entropy (``cross_entropy_fusion_impl="linear"``)
  never materializes them and is rejected.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from megatron.bridge.models.ernie.modeling_ernie55 import fused_dispatch


__all__ = [
    "MULTIMAX_CHUNK_ROWS",
    "MULTIMAX_GRAD_NORM_GROUP",
    "MULTIMAX_PARAMETER_NAMES",
    "ernie55_multimax",
    "install_ernie55_multimax",
    "register_multimax_grad_norm_group",
    "validate_ernie55_multimax_config",
]

MULTIMAX_GRAD_NORM_GROUP = "multimax"
"""Megatron-Core gradient-norm group of the multimax parameters (``param.grad_norm_group``)."""

MULTIMAX_PARAMETER_NAMES = ("multimax_ranges", "multimax_ts")
"""Output-layer parameter names; the Hugging Face keys are ``model.lm_head.<name>``."""

MULTIMAX_CHUNK_ROWS = 512
"""Logit rows per multimax chunk in forward and backward (the reference implementation's chunk)."""

_NUM_TERMS = 4


def _multimax_fp32(x: Tensor, ranges: Tensor, scales: Tensor) -> Tensor:
    """The multimax correction in fp32: ``x + sum_k t_k f_k(x, r_k)``, for fp32 ``x``, ``ranges`` and ``scales``."""
    r0, r1, r2, r3 = ranges.unbind()
    t0, t1, t2, t3 = scales.unbind()
    return (
        x
        + t0 * torch.relu(r0 - x)
        + t1 * torch.relu(x - r1)
        + t2 * torch.relu(r2 - x).square()
        + t3 * torch.relu(x - r3).square()
    )


class _Ernie55MultimaxFunction(torch.autograd.Function):
    """Multimax over ``[rows, vocab]`` chunks; backward recomputes each chunk under autograd.

    Outputs are freshly allocated tensors, not views: the cross-entropy that consumes the logits may
    modify an fp32 input in place.
    """

    @staticmethod
    def forward(ctx, logits: Tensor, ranges: Tensor, scales: Tensor) -> Tensor:
        ctx.save_for_backward(logits, ranges, scales)
        vocab = logits.shape[-1]
        rows = logits.reshape(-1, vocab)
        output = torch.empty(logits.shape, dtype=logits.dtype, device=logits.device)
        output_rows = output.view(-1, vocab)
        ranges_fp32, scales_fp32 = ranges.detach().float(), scales.detach().float()
        with torch.autocast(device_type=logits.device.type, enabled=False):
            for start in range(0, rows.shape[0], MULTIMAX_CHUNK_ROWS):
                end = start + MULTIMAX_CHUNK_ROWS
                output_rows[start:end] = _multimax_fp32(rows[start:end].float(), ranges_fp32, scales_fp32)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        logits, ranges, scales = ctx.saved_tensors
        vocab = logits.shape[-1]
        rows = logits.reshape(-1, vocab)
        grad_rows = grad_output.reshape(-1, vocab)
        ranges_fp32 = ranges.detach().float().requires_grad_(True)
        scales_fp32 = scales.detach().float().requires_grad_(True)
        grad_logits = torch.empty(logits.shape, dtype=logits.dtype, device=logits.device)
        grad_logits_rows = grad_logits.view(-1, vocab)
        grad_ranges = torch.zeros_like(ranges_fp32)
        grad_scales = torch.zeros_like(scales_fp32)
        with torch.enable_grad(), torch.autocast(device_type=logits.device.type, enabled=False):
            for start in range(0, rows.shape[0], MULTIMAX_CHUNK_ROWS):
                end = start + MULTIMAX_CHUNK_ROWS
                chunk = rows[start:end].detach().requires_grad_(True)
                output = _multimax_fp32(chunk.float(), ranges_fp32, scales_fp32).to(logits.dtype)
                d_chunk, d_ranges, d_scales = torch.autograd.grad(
                    output, (chunk, ranges_fp32, scales_fp32), grad_rows[start:end]
                )
                grad_logits_rows[start:end] = d_chunk
                grad_ranges += d_ranges
                grad_scales += d_scales
        return grad_logits, grad_ranges.to(ranges.dtype), grad_scales.to(scales.dtype)


def ernie55_multimax(logits: Tensor, ranges: Tensor, scales: Tensor, *, fused: bool = False) -> Tensor:
    """Apply the ERNIE 5.5 multimax correction to ``logits`` (fp32 compute, output in the logits dtype).

    Args:
        logits: ``[..., vocab]`` logits.
        ranges: ``[4]`` ranges ``r``.
        scales: ``[4]`` scales ``t``.
        fused: Run the Triton forward and backward when the logits are on a CUDA device in a float dtype the
            kernel takes and Triton imports (``ernie_fused_ops``); otherwise, and always on CPU, the chunked
            fp32 path runs.

    Returns:
        Corrected logits, same shape and dtype as ``logits``, in a new tensor.
    """
    if ranges.shape != (_NUM_TERMS,) or scales.shape != (_NUM_TERMS,):
        raise ValueError(
            f"ERNIE 5.5 multimax needs [{_NUM_TERMS}] ranges and scales, got {tuple(ranges.shape)} and "
            f"{tuple(scales.shape)}."
        )
    if fused and fused_dispatch.fused_kernels_support(logits):
        return fused_dispatch.fused_multimax(logits, ranges, scales)
    return _Ernie55MultimaxFunction.apply(logits, ranges, scales)


class _Ernie55MultimaxOutputLayer(torch.nn.Module):
    """Output-layer mixin (see :func:`install_ernie55_multimax`): logits pass through multimax."""

    def forward(self, input_: Tensor, *args: Any, **kwargs: Any) -> tuple[Tensor, Tensor | None]:
        if kwargs.get("output_cross_entropy_loss"):
            raise RuntimeError(
                "ERNIE 5.5 multimax needs the logits, but fused linear cross-entropy never materializes them; "
                "do not use cross_entropy_fusion_impl='linear'."
            )
        logits, bias = super().forward(input_, *args, **kwargs)
        if bias is not None:
            raise RuntimeError("ERNIE 5.5 multimax expects an output layer without a separate bias.")
        fused = getattr(self, "ernie55_fused_multimax", False)
        return ernie55_multimax(logits, self.multimax_ranges, self.multimax_ts, fused=fused), bias


def register_multimax_grad_norm_group() -> None:
    """Register :data:`MULTIMAX_GRAD_NORM_GROUP` in Megatron-Core's ``SEPARATE_GRAD_NORM_GROUPS`` (idempotent).

    Megatron-Core reads that module-level tuple each time it computes or clips gradient norms, so
    registering once per process before the optimizer is built is enough. It is process-wide, and
    it must happen on every rank that shares an optimizer's gradient-statistics group, whether or
    not the rank owns multimax parameters: for each registered group the optimizer all-reduces a
    flag over that group (once; Megatron-Core caches the answer), so an unused group costs one reduction.
    """
    from megatron.core.optimizer import optimizer as mcore_optimizer

    if MULTIMAX_GRAD_NORM_GROUP not in mcore_optimizer.SEPARATE_GRAD_NORM_GROUPS:
        mcore_optimizer.SEPARATE_GRAD_NORM_GROUPS = (
            *mcore_optimizer.SEPARATE_GRAD_NORM_GROUPS,
            MULTIMAX_GRAD_NORM_GROUP,
        )


def validate_ernie55_multimax_config(config: object) -> None:
    """Raise ``ValueError`` if ``config`` cannot run the multimax head.

    The logits must be materialized (no fused linear cross-entropy), the head unsharded
    (``tensor_model_parallel_size=1``; the multimax gradient would otherwise be a per-shard partial
    sum), and not scaled after the output layer (no muP).

    Args:
        config: Transformer config.

    Raises:
        ValueError: On the first violated requirement.
    """

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(f"ERNIE 5.5 multimax: {message}")

    require(
        not (
            getattr(config, "cross_entropy_loss_fusion", False)
            and getattr(config, "cross_entropy_fusion_impl", None) == "linear"
        ),
        "cross_entropy_fusion_impl='linear' never materializes the logits; use the native or TE fusion.",
    )
    require(config.tensor_model_parallel_size == 1, "tensor_model_parallel_size must be 1.")
    require(not getattr(config, "use_mup", False), "muP output scaling is not supported.")


def install_ernie55_multimax(output_layer: torch.nn.Module, config: object) -> None:
    """Give ``output_layer`` the multimax parameters and correction (in place, idempotent).

    Registers zero-initialized ``multimax_ranges`` and ``multimax_ts`` (``[4]``, ``params_dtype``,
    tagged with :data:`MULTIMAX_GRAD_NORM_GROUP`), registers that group with Megatron-Core, and
    makes the layer's ``forward`` return multimax-corrected logits. Call it before the model is
    wrapped for training (DDP, optimizer). Every call, the first included, sets the layer's
    ``ernie55_fused_multimax`` attribute from ``config.ernie_fused_ops`` (``False`` for configs without
    the field), which selects the Triton kernels for CUDA logits (:func:`ernie55_multimax`).

    Args:
        output_layer: The model's output layer (Megatron-Core's ``LinearCrossEntropyModule``, a
            ``ColumnParallelLinear`` subclass).
        config: Transformer config.

    Raises:
        ValueError: If ``config`` fails :func:`validate_ernie55_multimax_config`.
    """
    validate_ernie55_multimax_config(config)
    register_multimax_grad_norm_group()
    output_layer.ernie55_fused_multimax = fused_dispatch.ernie55_fused_ops_enabled(config)
    if isinstance(output_layer, _Ernie55MultimaxOutputLayer):
        return
    weight = getattr(output_layer, "weight", None)
    if isinstance(weight, torch.Tensor):
        device = weight.device
    else:
        device = torch.device("cpu") if config.use_cpu_initialization else torch.cuda.current_device()
    for name in MULTIMAX_PARAMETER_NAMES:
        param = torch.nn.Parameter(torch.zeros(_NUM_TERMS, dtype=config.params_dtype, device=device))
        param.grad_norm_group = MULTIMAX_GRAD_NORM_GROUP
        output_layer.register_parameter(name, param)
    # The same per-instance mixin as Bridge's gemma ``extend_instance``, without importing that package:
    # the output layer class differs between Megatron-Core versions.
    base = type(output_layer)
    output_layer.__class__ = type(base.__name__, (_Ernie55MultimaxOutputLayer, base), {})
