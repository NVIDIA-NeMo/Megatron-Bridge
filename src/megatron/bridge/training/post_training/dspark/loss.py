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

"""DSpark draft training objective and acceptance metrics.

DSpark (arXiv:2607.05147) trains the draft with three position-weighted terms:

* ``ce``: cross-entropy of the draft logits against the target's next tokens;
* ``tv``: total-variation distance ``0.5 * ||p_draft - p_target||_1`` to the
  target's next-token distribution (the dominant, acceptance-proxy term);
* ``confidence``: binary cross-entropy training the confidence head against the
  analytical acceptance label ``1 - TV``.

Each block position ``k`` is weighted by ``exp(-k / loss_decay_gamma)``.
Acceptance is measured analytically as ``accept_rate = 1 - TV`` and the expected
accepted prefix length of a block as ``tau = sum_k cumprod(accept_rate)_k + 1``.

The per-term losses are computed as ``(numerator, denominator)`` sums so that a
data-parallel training step can all-reduce the denominators over its DP group
before dividing once (avoiding per-micro-batch token-count bias). Pass
``dp_group`` for that reduction; the default (``None``) computes the local loss,
which is what single-process unit tests exercise. This module depends only on
``torch`` and ``torch.distributed``; wiring it to a Megatron-Core forward step and
the correct DP group is a separate integration concern.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F


@dataclass
class DSparkForwardOutput:
    """Outputs of one DSpark training forward, consumed by :func:`dspark_loss`.

    Shape symbols: ``batch``, ``num_blocks`` (sampled anchor blocks per sample),
    ``block_size`` (draft positions per anchor), ``vocab``.

    Attributes:
        draft_logits: Draft logits ``[batch, num_blocks, block_size, vocab]``.
        target_ids: Teacher-forced next token per position
            ``[batch, num_blocks, block_size]``.
        eval_mask: Bool/float supervision mask ``[batch, num_blocks, block_size]``
            (a block is a contiguous, in-bounds, loss-enabled prefix).
        block_keep_mask: Kept-anchor mask ``[batch, num_blocks]``.
        confidence_pred: Optional per-position acceptance logit
            ``[batch, num_blocks, block_size]``.
        aligned_target_logits: Optional target next-token logits
            ``[batch, num_blocks, block_size, vocab]`` (the TV / confidence teacher).
    """

    draft_logits: torch.Tensor
    target_ids: torch.Tensor
    eval_mask: torch.Tensor
    block_keep_mask: torch.Tensor
    confidence_pred: torch.Tensor | None = None
    aligned_target_logits: torch.Tensor | None = None


def _loss_weight_mask(eval_mask: torch.Tensor, block_size: int, loss_decay_gamma: float | None) -> torch.Tensor:
    """``eval_mask`` (float) scaled by the per-position decay ``exp(-k / gamma)``."""
    weight = eval_mask.to(torch.float32)
    if loss_decay_gamma is not None and loss_decay_gamma > 0:
        positions = torch.arange(block_size, device=eval_mask.device).view(1, 1, -1)
        weight = weight * torch.exp(-positions.float() / float(loss_decay_gamma))
    return weight


def _l1_distance_per_token(draft_logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
    """FP32 probability L1 distance ``||softmax(draft) - softmax(target)||_1`` per position.

    Args:
        draft_logits: ``[..., vocab]``.
        target_logits: ``[..., vocab]``.

    Returns:
        Per-position L1 distance ``[...]`` (twice the total-variation distance).
    """
    draft_probs = torch.softmax(draft_logits.float(), dim=-1)
    target_probs = torch.softmax(target_logits.float(), dim=-1)
    return (draft_probs - target_probs).abs().sum(dim=-1)


def _reduce_den(value: torch.Tensor, dp_group) -> torch.Tensor:
    """Sum a denominator over ``dp_group``. No reduction when ``dp_group`` is ``None``.

    The caller must pass the intended data-parallel group explicitly; ``None`` is
    local-only. (Do not fall back to the default process group, whose members are
    not the DP replicas under Megatron-Core parallelism.)
    """
    out = value.detach().clone()
    if dp_group is not None:
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=dp_group)
    return out


def dspark_loss(
    outputs: DSparkForwardOutput,
    *,
    ce_alpha: float = 0.1,
    l1_alpha: float = 0.9,
    confidence_alpha: float = 1.0,
    loss_decay_gamma: float | None = 4.0,
    dp_group=None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the DSpark draft loss and acceptance metrics.

    Args:
        outputs: The draft forward outputs; see :class:`DSparkForwardOutput`. All
            tensors are ``[batch, num_blocks, block_size(, vocab)]``.
        ce_alpha: Cross-entropy weight.
        l1_alpha: Total-variation (``0.5 * L1``) weight.
        confidence_alpha: Confidence-BCE weight (used only when
            ``outputs.confidence_pred`` is set).
        loss_decay_gamma: Per-position decay; ``None`` disables it.
        dp_group: Optional data-parallel process group over which the loss
            denominators are summed before dividing. ``None`` computes the local
            (single-process) loss.

    Returns:
        Tuple ``(loss, metrics)``. ``loss`` is the scalar training loss; ``metrics``
        holds the detached per-term losses (``ce_loss`` / ``l1_loss`` /
        ``confidence_loss``) and acceptance diagnostics (``accept_rate``, ``tau``).
    """
    draft_logits = outputs.draft_logits
    _, _, block_size, vocab_size = draft_logits.shape
    weight = _loss_weight_mask(outputs.eval_mask, block_size, loss_decay_gamma)  # [B, N, L]

    # Cross-entropy against the teacher-forced next tokens.
    ce_per_token = F.cross_entropy(
        draft_logits.reshape(-1, vocab_size), outputs.target_ids.reshape(-1).long(), reduction="none"
    ).reshape_as(weight)
    ce_num = (ce_per_token * weight).sum()
    ce_den = weight.sum()

    has_confidence = outputs.confidence_pred is not None
    target_logits = outputs.aligned_target_logits
    if target_logits is not None:
        target_logits = target_logits.detach()  # never backprop into the (shared) lm_head
    if (l1_alpha > 0 or has_confidence) and target_logits is None:
        raise ValueError("aligned_target_logits is required for the TV loss or the confidence head.")

    zero = ce_num.new_zeros(())
    l1_num = l1_den = zero
    conf_num = conf_den = zero
    accept_rate = None
    if target_logits is not None and (l1_alpha > 0 or has_confidence):
        l1_dist = _l1_distance_per_token(draft_logits, target_logits)  # [B, N, L] == 2 * TV
        accept_rate = (1.0 - 0.5 * l1_dist).clamp_(0.0, 1.0)
        if l1_alpha > 0:
            l1_num = (l1_dist * weight).sum()
            l1_den = weight.sum()
        if has_confidence:
            conf_num = (
                F.binary_cross_entropy_with_logits(
                    outputs.confidence_pred.float(), accept_rate.detach(), reduction="none"
                )
                * weight
            ).sum()
            conf_den = weight.sum()

    # Divide by the (optionally DP-reduced) denominators; the backward gradient is
    # scaled by world_size so the DP-averaged gradient matches the global loss.
    world_size = dist.get_world_size(dp_group) if dp_group is not None else 1
    ce_g, l1_g, conf_g = (_reduce_den(d, dp_group) for d in (ce_den, l1_den, conf_den))
    ce_loss = ce_num / (ce_g + 1e-6)
    l1_loss = l1_num / (l1_g + 1e-6) if l1_g.item() > 0 else zero
    conf_loss = conf_num / (conf_g + 1e-6) if (has_confidence and conf_g.item() > 0) else zero
    loss = (ce_alpha * ce_loss + l1_alpha * l1_loss + confidence_alpha * conf_loss) * world_size

    metrics: dict[str, torch.Tensor] = {
        "ce_loss": (ce_num / (ce_den + 1e-6)).detach(),
        "l1_loss": (l1_num / (l1_den + 1e-6)).detach() if l1_den.item() > 0 else zero,
        "confidence_loss": (conf_num / (conf_den + 1e-6)).detach() if conf_den.item() > 0 else zero,
    }
    with torch.no_grad():
        metrics.update(_acceptance_metrics(outputs, accept_rate))
    return loss, metrics


def _acceptance_metrics(outputs: DSparkForwardOutput, accept_rate: torch.Tensor | None) -> dict[str, torch.Tensor]:
    """Analytical acceptance diagnostics.

    ``accept_rate`` (``= 1 - TV``) is the per-position acceptance probability. A
    draft token survives only if every earlier token in its block is accepted,
    hence ``tau`` is the running product over the block, ``+ 1`` for the verified
    anchor token.

    Args:
        outputs: The forward outputs (for the masks).
        accept_rate: Per-position acceptance ``[batch, num_blocks, block_size]`` or
            ``None`` when no teacher signal is available.

    Returns:
        ``{"accept_rate": scalar, "tau": scalar}`` (zeros when ``accept_rate`` is None).
    """
    zero = outputs.draft_logits.new_zeros((), dtype=torch.float32)
    if accept_rate is None:
        return {"accept_rate": zero, "tau": zero}
    eval_mask = outputs.eval_mask.to(torch.float32)
    valid_accept = accept_rate * eval_mask
    accept_rate_mean = valid_accept.sum() / eval_mask.sum().clamp_min(1.0)
    valid_blocks = (outputs.block_keep_mask.bool() & outputs.eval_mask.bool().any(dim=-1)).to(torch.float32)
    tau_per_block = valid_accept.cumprod(dim=-1).sum(dim=-1) + 1.0
    tau_mean = (tau_per_block * valid_blocks).sum() / valid_blocks.sum().clamp_min(1.0)
    return {"accept_rate": accept_rate_mean, "tau": tau_mean}
