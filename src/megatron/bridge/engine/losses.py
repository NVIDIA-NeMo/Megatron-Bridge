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

import math
from collections.abc import Callable

import torch

from megatron.bridge.engine.datum import ModelOutput, PreparedMicrobatch


LossReturn = tuple[torch.Tensor, dict[str, torch.Tensor]]

# Above this log-ratio, exponentials continue linearly (slope exp(20), about 5e8) instead of growing
# to inf. Loss and gradient stay finite and keep their sign, so a token the policy drove far from the
# old or reference distribution still receives a corrective (if large) gradient; gradient clipping
# handles the magnitude. Below the bound the exponential is exact; nothing is clamped on the negative
# side, where exp decays to zero on its own.
LOG_RATIO_BOUND = 20.0


def _bounded_exp(x: torch.Tensor) -> torch.Tensor:
    """``exp(x)`` up to ``LOG_RATIO_BOUND``, then its tangent line: finite, monotone, non-zero gradient."""
    capped = x.clamp(max=LOG_RATIO_BOUND)
    return capped.exp() + math.exp(LOG_RATIO_BOUND) * (x - capped)


def cross_entropy(batch: PreparedMicrobatch, output: ModelOutput) -> torch.Tensor:
    """Negative log-likelihood summed over active positions."""
    active = batch.active_mask
    return -output.logprobs[active].float().sum()


def _active_field(batch: PreparedMicrobatch, name: str, active: torch.Tensor, *, needed_by: str) -> torch.Tensor:
    """The active positions of a per-token Datum field; raises when the Datums did not carry it.

    A microbatch with no active position (a filler) never needs the field: an empty tensor is returned
    so the loss evaluates to a graph-connected zero with its usual metric keys."""
    field = getattr(batch, name)
    if field is None:
        if not bool(active.any()):
            return torch.zeros(0, dtype=torch.float32, device=active.device)
        raise ValueError(f"{needed_by} needs Datum.{name}, but the batch does not carry it.")
    return field[active].float()


def _active_loss_input(batch: PreparedMicrobatch, key: str, active: torch.Tensor, *, needed_by: str) -> torch.Tensor:
    """Like ``_active_field`` for a custom ``loss_inputs`` tensor."""
    field = batch.loss_inputs.get(key)
    if field is None:
        if not bool(active.any()):
            return torch.zeros(0, dtype=torch.float32, device=active.device)
        raise ValueError(f"{needed_by} needs Datum.loss_inputs[{key!r}], but the batch does not carry it.")
    return field[active].float()


def _importance_weights(
    batch: PreparedMicrobatch, old: torch.Tensor, active: torch.Tensor, tis_coef: float | None, *, needed_by: str
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
    """Truncated importance-sampling weights ``min(exp(old - rollout), tis_coef)`` (no gradient), or
    ``None`` when ``tis_coef`` is not set. Corrects for rollouts sampled by a different engine."""
    if tis_coef is None:
        return None, {}
    rollout = _active_field(batch, "rollout_logprobs", active, needed_by=needed_by)
    log_weights = old - rollout
    log_cap = math.log(tis_coef)
    truncated = (log_weights > log_cap).float().sum()
    return log_weights.clamp(max=log_cap).exp(), {"tis_truncated_tokens": truncated}  # truncated in log space


def importance_sampling(
    batch: PreparedMicrobatch, output: ModelOutput, *, tis_coef: float | None = None
) -> LossReturn:
    """Importance-weighted policy gradient: ``-(exp(logp - old) * advantage)`` summed over active positions.

    ``tis_coef`` enables truncated importance sampling against ``rollout_logprobs``. The ratio grows
    linearly past ``LOG_RATIO_BOUND`` (see ``_bounded_exp``). Metrics: ``abs_logprob_diff_sum``
    (``|logp - old|``) and, with TIS, ``tis_truncated_tokens``.
    """
    active = batch.active_mask
    logp = output.logprobs[active].float()
    old = _active_field(batch, "old_logprobs", active, needed_by="importance_sampling")
    advantages = _active_field(batch, "advantages", active, needed_by="importance_sampling")
    ratio = _bounded_exp(logp - old)
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
    The clipped branch is formed in log space, so it never overflows and has PPO's zero gradient
    outside the clip range; the branch is chosen in log space too, and the unclipped ratio is only
    evaluated where it is selected, through ``_bounded_exp`` (linear past ``LOG_RATIO_BOUND``), so a
    large gap gives a finite, corrective gradient instead of inf or NaN.
    Metrics: ``clipped_tokens`` (ratio outside the clip range), ``abs_logprob_diff_sum``, ``kl_sum`` when
    the KL term is on, ``tis_truncated_tokens`` with TIS.
    """
    active = batch.active_mask
    logp = output.logprobs[active].float()
    old = _active_field(batch, "old_logprobs", active, needed_by="ppo_clip")
    advantages = _active_field(batch, "advantages", active, needed_by="ppo_clip")
    log_ratio = logp - old
    log_low, log_high = math.log(1.0 - clip_low), math.log(1.0 + clip_high)
    clipped_ratio = log_ratio.clamp(log_low, log_high).exp()
    # min(r*A, clip(r)*A) picks the unclipped branch when A >= 0 and r is below the upper clip, or when
    # A < 0 and r is above the lower clip; decided on log_ratio so no overflowing value is needed.
    use_unclipped = torch.where(advantages >= 0, log_ratio <= log_high, log_ratio >= log_low)
    ratio = _bounded_exp(torch.where(use_unclipped, log_ratio, torch.zeros_like(log_ratio)))
    surrogate = torch.where(use_unclipped, ratio * advantages, clipped_ratio * advantages)
    weights, metrics = _importance_weights(batch, old, active, tis_coef, needed_by="ppo_clip")
    if weights is not None:
        surrogate = weights * surrogate
    loss = -surrogate.sum()
    if kl_beta > 0.0:
        reference = _active_field(batch, "reference_logprobs", active, needed_by="ppo_clip with kl_beta > 0")
        delta = reference - logp
        kl = _bounded_exp(delta) - delta - 1.0  # k3; the linear term keeps its gradient for any delta
        loss = loss + kl_beta * kl.sum()
        metrics["kl_sum"] = kl.detach().sum()
    outside = (log_ratio < log_low) | (log_ratio > log_high)
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
    returns = _active_loss_input(batch, "returns", active, needed_by="value_loss")
    error = (values - returns) ** 2
    metrics = {"value_abs_error_sum": (values - returns).detach().abs().sum()}
    if clip is not None:
        old = _active_loss_input(batch, "old_values", active, needed_by="value_loss with clip")
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
