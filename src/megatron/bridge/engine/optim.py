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
"""Optimizer-step helpers: caller-supplied AdamParams, gradient scaling, single finalization, MCore step."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from megatron.core.transformer import MegatronModule

from megatron.bridge.engine.datum import AdamParams


def apply_adam_params(optimizer: Any, params: AdamParams) -> None:
    """Write lr, weight decay, betas and eps into every param group.

    Mirrors ``OptimizerParamScheduler.step``: the recipe's per-group ``lr_mult`` / ``wd_mult`` multipliers
    (for example no weight decay on biases and norms) are preserved; the schedule itself is the caller's.
    """
    for group in optimizer.param_groups:
        lr = params.learning_rate * group.get("lr_mult", 1.0)
        if isinstance(group.get("lr"), torch.Tensor):
            group["lr"].fill_(lr)
        else:
            group["lr"] = lr
        group["weight_decay"] = params.weight_decay * group.get("wd_mult", 1.0)
        if "betas" in group:
            group["betas"] = (params.beta1, params.beta2)
        if "eps" in group:
            group["eps"] = params.eps


def scale_gradients(model: list[MegatronModule], factor: float) -> None:
    """Scale every DDP gradient buffer (dense and expert) in place."""
    for chunk in model:
        chunk.scale_gradients(factor)


def finalize_and_step(
    model: list[MegatronModule],
    optimizer: Any,
    *,
    finalize_model_grads_func: Callable[..., None],
    scale: float | None,
    adam_params: AdamParams,
) -> tuple[bool, float | None, int | None]:
    """Normalize, finalize once, write AdamParams, and step the MCore optimizer (clipping inside).

    Args:
        model: DDP-wrapped model chunks holding the accumulated FP32 gradients.
        optimizer: The MegatronOptimizer built for the model.
        finalize_model_grads_func: Bridge's installed finalizer, called once with ``num_tokens=None`` so it
            does not divide by a token count. It runs ``finish_grad_sync`` (the synchronous DP reduce-scatter
            when ``overlap_grad_reduce`` is off; only a wait when the reductions were launched during
            backward), then the non-overlappable reductions: sequence-parallel layernorm grads over TP,
            tied embeddings across PP stages, MoE expert-bias updates and aux-loss tracker resets.
        scale: Factor applied to the gradients before finalization, or ``None`` to leave them as sums.
        adam_params: Per-step optimizer hyperparameters.

    Returns:
        ``(update_successful, grad_norm, num_zeros_in_grad)`` as returned by ``optimizer.step()``.
    """
    if scale is not None and scale != 1.0:
        scale_gradients(model, scale)
    finalize_model_grads_func(model, None)
    apply_adam_params(optimizer, adam_params)
    update_successful, grad_norm, num_zeros = optimizer.step()
    return bool(update_successful), None if grad_norm is None else float(grad_norm), num_zeros


def zero_gradients(model: list[MegatronModule], optimizer: Any | None) -> None:
    """Zero the DDP gradient buffers and the optimizer's gradient state (start of a new window)."""
    for chunk in model:
        if hasattr(chunk, "zero_grad_buffer"):
            chunk.zero_grad_buffer()
    if optimizer is not None:
        optimizer.zero_grad()
