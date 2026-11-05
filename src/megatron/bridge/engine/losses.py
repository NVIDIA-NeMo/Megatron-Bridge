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
"""Predefined loss math over the ``(PreparedMicrobatch, ModelOutput)`` contract.

Every loss returns the additive FP32 contribution of one microbatch; the engine divides once at
``optim_step``. Losses select active positions (``loss_mask & ~padding_mask``) before any arithmetic so
padding never produces NaN/Inf that a later multiply-by-zero could not remove.

The RL losses also return a dictionary of additive diagnostics (token counts and sums over the same
active positions). The engine reduces them over DP x CP alongside the loss and returns them on
``ForwardBackwardOutput.metrics``; divide by ``num_loss_tokens`` for per-token means.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from megatron.bridge.engine.datum import ModelOutput, PreparedMicrobatch


LossReturn = tuple[torch.Tensor, dict[str, torch.Tensor]]


def cross_entropy(batch: PreparedMicrobatch, output: ModelOutput) -> torch.Tensor:
    """Negative log-likelihood summed over active positions."""
    active = batch.active_mask
    return -output.logprobs[active].float().sum()


def _active_field(batch: PreparedMicrobatch, name: str, active: torch.Tensor, *, needed_by: str) -> torch.Tensor:
    """The active positions of a per-token Datum field; raises when the Datums did not carry it."""
    field = getattr(batch, name)
    if field is None:
        raise ValueError(f"{needed_by} needs Datum.{name}, but the batch does not carry it.")
    return field[active].float()


def _importance_weights(
    batch: PreparedMicrobatch, old: torch.Tensor, active: torch.Tensor, tis_coef: float | None, *, needed_by: str
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
    """Truncated importance-sampling weights ``min(exp(old - rollout), tis_coef)`` (no gradient), or
    ``None`` when ``tis_coef`` is not set. Corrects for rollouts sampled by a different engine."""
    if tis_coef is None:
        return None, {}
    rollout = _active_field(batch, "rollout_logprobs", active, needed_by=needed_by)
    weights = (old - rollout).exp()
    truncated = (weights > tis_coef).float().sum()
    return weights.clamp(max=tis_coef), {"tis_truncated_tokens": truncated}


def importance_sampling(
    batch: PreparedMicrobatch, output: ModelOutput, *, tis_coef: float | None = None
) -> LossReturn:
    """Importance-weighted policy gradient: ``-(exp(logp - old) * advantage)`` summed over active positions.

    ``tis_coef`` enables truncated importance sampling against ``rollout_logprobs``. Metrics:
    ``abs_logprob_diff_sum`` (``|logp - old|``) and, with TIS, ``tis_truncated_tokens``.
    """
    active = batch.active_mask
    logp = output.logprobs[active].float()
    old = _active_field(batch, "old_logprobs", active, needed_by="importance_sampling")
    advantages = _active_field(batch, "advantages", active, needed_by="importance_sampling")
    ratio = (logp - old).exp()
    weights, metrics = _importance_weights(batch, old, active, tis_coef, needed_by="importance_sampling")
    per_token = ratio * advantages
    if weights is not None:
        per_token = weights * per_token
    metrics["abs_logprob_diff_sum"] = (logp - old).detach().abs().sum()
    return -per_token.sum(), metrics


def ppo_clip(
    batch: PreparedMicrobatch,
    output: ModelOutput,
    *,
    clip_low: float,
    clip_high: float,
    kl_beta: float = 0.0,
    tis_coef: float | None = None,
) -> LossReturn:
    """Clipped PPO objective summed over active positions, with optional KL penalty and TIS correction.

    Per token: ``-w * min(r * A, clamp(r, 1 - clip_low, 1 + clip_high) * A) + kl_beta * k3`` where
    ``r = exp(logp - old)``, ``k3 = exp(ref - logp) - (ref - logp) - 1`` against ``reference_logprobs``
    (only when ``kl_beta > 0``) and ``w`` the truncated importance weight (only with ``tis_coef``).
    Metrics: ``clipped_tokens`` (ratio outside the clip range), ``abs_logprob_diff_sum``, ``kl_sum`` when
    the KL term is on, ``tis_truncated_tokens`` with TIS.
    """
    active = batch.active_mask
    logp = output.logprobs[active].float()
    old = _active_field(batch, "old_logprobs", active, needed_by="ppo_clip")
    advantages = _active_field(batch, "advantages", active, needed_by="ppo_clip")
    ratio = (logp - old).exp()
    clipped_ratio = ratio.clamp(1.0 - clip_low, 1.0 + clip_high)
    surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    weights, metrics = _importance_weights(batch, old, active, tis_coef, needed_by="ppo_clip")
    if weights is not None:
        surrogate = weights * surrogate
    loss = -surrogate.sum()
    if kl_beta > 0.0:
        reference = _active_field(batch, "reference_logprobs", active, needed_by="ppo_clip with kl_beta > 0")
        delta = reference - logp
        kl = delta.exp() - delta - 1.0
        loss = loss + kl_beta * kl.sum()
        metrics["kl_sum"] = kl.detach().sum()
    outside = (ratio < 1.0 - clip_low) | (ratio > 1.0 + clip_high)
    metrics["clipped_tokens"] = outside.detach().float().sum()
    metrics["abs_logprob_diff_sum"] = (logp - old).detach().abs().sum()
    return loss, metrics


def value_loss(batch: PreparedMicrobatch, output: ModelOutput, *, clip: float | None = None) -> LossReturn:
    """Critic regression summed over active positions: ``0.5 * (v - R)^2`` against ``loss_inputs["returns"]``.

    With ``clip``, the PPO clipped value loss ``0.5 * max((v - R)^2, (clamp(v, v_old - clip, v_old + clip)
    - R)^2)`` using ``loss_inputs["old_values"]``. Needs a value-head engine (``output.values``).
    Metrics: ``value_abs_error_sum`` and, with ``clip``, ``value_clipped_tokens``.
    """
    if output.values is None:
        raise ValueError("value_loss needs a value-head engine (from_recipe(output_head='value')).")
    active = batch.active_mask
    values = output.values[active].float()
    if "returns" not in batch.loss_inputs:
        raise ValueError("value_loss needs Datum.loss_inputs['returns'] (per-token return targets).")
    returns = batch.loss_inputs["returns"][active].float()
    error = (values - returns) ** 2
    metrics = {"value_abs_error_sum": (values - returns).detach().abs().sum()}
    if clip is not None:
        if "old_values" not in batch.loss_inputs:
            raise ValueError("value_loss with clip needs Datum.loss_inputs['old_values'].")
        old = batch.loss_inputs["old_values"][active].float()
        clipped = old + (values - old).clamp(-clip, clip)
        clipped_error = (clipped - returns) ** 2
        metrics["value_clipped_tokens"] = (clipped_error > error).detach().float().sum()
        error = torch.maximum(error, clipped_error)
    return 0.5 * error.sum(), metrics


def target_logprobs(batch: PreparedMicrobatch, output: ModelOutput) -> list[torch.Tensor]:
    """This rank's target logprobs per Datum, in the microbatch's segment order, with autograd intact.

    Each entry holds the positions this CP rank owns for that Datum, in ascending Datum position;
    padding and fillers are dropped. Needs the default pipeline's ``MicrobatchMapping``.
    """
    mapping = batch.mapping
    if not (hasattr(mapping, "segments") and hasattr(mapping, "cp_indices")):
        raise ValueError(
            "target_logprobs needs the default MicrobatchMapping; slice output.logprobs yourself otherwise."
        )
    if getattr(mapping, "filler", False):
        return []
    values = output.logprobs[0]
    where = mapping.cp_indices
    slices = []
    for segment in mapping.segments:
        mask = (where >= segment.offset) & (where < segment.offset + segment.length)
        order = torch.argsort(where[mask])
        slices.append(values[mask][order])
    return slices


def chunked_vocab_apply(
    logits: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor], *, chunk_tokens: int
) -> torch.Tensor:
    """Apply a full-vocabulary op per token chunk, preserving gradient flow, to bound peak memory."""
    # TODO(logits): copy megatron/rl vocab_parallel_selective_log_softmax (TP-aware, chunked) with attribution.
    raise NotImplementedError("chunked_vocab_apply is not implemented yet (full-logits path).")
