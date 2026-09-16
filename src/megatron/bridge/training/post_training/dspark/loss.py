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
* ``l1``: the **raw** probability L1 distance ``||p_draft - p_target||_1`` to the
  target's next-token distribution (paper Eq. 10, weight 0.9). This is the
  dominant term. The factor ``1/2`` that turns L1 into a total-variation distance
  belongs to the *acceptance label* of Eq. 8 only, never to this training term;
* ``confidence``: binary cross-entropy training the confidence head against the
  analytical acceptance label ``1 - 0.5 * L1``.

Each block position ``k`` is weighted by ``exp(-k / loss_decay_gamma)``.
Acceptance is measured analytically as ``accept_rate = 1 - 0.5 * L1`` and the
expected accepted prefix length of a block as ``tau = sum_k cumprod(accept_rate)_k + 1``.

**Return contract**

:func:`dspark_loss` returns the Megatron-Core 3-tuple ``(loss, num_tokens,
report)``, matching :func:`megatron.bridge.training.losses.masked_next_token_loss`:

* ``loss`` is the **unnormalized** weighted numerator summed over the micro-batch;
* ``num_tokens`` is this micro-batch's supervised token count;
* ``report`` maps each metric to a ``[numerator, denominator]`` pair so the
  training loop can sum both over the log window and the data-parallel group and
  form the global ratio once.

Nothing here divides by a local denominator, because dividing per micro-batch and
averaging afterwards computes a mean of micro-batch means, which is biased
whenever micro-batches carry different supervised-token counts. The unbiased
global mean requires ``calculate_per_token_loss=True`` on the model config, which
makes Megatron-Core accumulate ``num_tokens`` across micro-batches and scale the
gradient once by the data-parallel total in ``finalize_model_grads_func``. With
the default ``calculate_per_token_loss=False``, Megatron-Core divides each
micro-batch by its own ``num_tokens`` and re-introduces exactly that bias, so the
integration that calls this function must set the flag; see
:func:`assert_per_token_loss_config`.

This module depends only on ``torch``. Wiring it to a Megatron-Core forward step
is a separate integration concern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# Rows of the flattened ``[batch * num_blocks * block_size, vocab]`` logits whose
# FP32 distribution is materialized at once. Bounds the softmax temporaries to
# ``chunk * vocab`` elements instead of the full ``[B, N, L, V]`` tensor. Matches
# the NeMo AutoModel reference implementation's chunk size.
_PROBABILITY_CHUNK_TOKENS = 128


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
            ``[batch, num_blocks, block_size, vocab]`` (the L1 / confidence teacher).
    """

    draft_logits: torch.Tensor
    target_ids: torch.Tensor
    eval_mask: torch.Tensor
    block_keep_mask: torch.Tensor
    confidence_pred: torch.Tensor | None = None
    aligned_target_logits: torch.Tensor | None = None


def assert_per_token_loss_config(model_config: Any) -> None:
    """Fail fast unless ``model_config.calculate_per_token_loss`` is set.

    :func:`dspark_loss` returns an unnormalized numerator; only the
    per-token-loss path sums ``num_tokens`` across micro-batches and scales the
    gradient once by the global total. Under the default path Megatron-Core
    normalizes each micro-batch by its own token count, which silently trains a
    mean of micro-batch means.

    When the forward-step wiring lands, this belongs in ``ConfigContainer.validate``
    alongside the existing context-parallel ``calculate_per_token_loss`` rule, so it
    fires before training rather than at the first loss call. It lives here for now
    because this package has no config field to key that rule off yet. The parameter
    is untyped because this module deliberately imports nothing beyond ``torch``;
    the attribute is read without a default, so passing the wrong object raises
    ``AttributeError`` rather than reporting a config error that is not there.

    Args:
        model_config: The Megatron-Core model config used for the run.

    Raises:
        AttributeError: If ``model_config`` has no ``calculate_per_token_loss``.
        ValueError: If ``calculate_per_token_loss`` is not enabled.
    """
    if not model_config.calculate_per_token_loss:
        raise ValueError(
            "DSpark training requires model.calculate_per_token_loss=True: dspark_loss returns an "
            "unnormalized per-micro-batch numerator, and only the per-token-loss path divides once "
            "by the globally reduced token count. Leaving it False averages micro-batch means and "
            "biases the objective whenever micro-batches carry different supervised-token counts."
        )


def _loss_weight_mask(eval_mask: torch.Tensor, loss_decay_gamma: float | None) -> torch.Tensor:
    """``eval_mask`` (float) scaled by the per-position decay ``exp(-k / gamma)``."""
    weight = eval_mask.to(torch.float32)
    if loss_decay_gamma is not None and loss_decay_gamma > 0:
        positions = torch.arange(eval_mask.shape[-1], device=eval_mask.device).view(1, 1, -1)
        weight = weight * torch.exp(-positions.float() / float(loss_decay_gamma))
    return weight


def _chunk_terms(
    draft_logits: torch.Tensor, target_ids: torch.Tensor, target_logits: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row cross-entropy and probability L1 over one ``[rows, vocab]`` chunk.

    Both terms are derived from a single FP32 ``log_softmax`` of the draft logits,
    so the draft distribution is normalized once rather than once per term.
    """
    draft_log_probs = torch.log_softmax(draft_logits, dim=-1, dtype=torch.float32)
    ce = -draft_log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    if target_logits is None:
        return ce, torch.zeros_like(ce)
    target_probs = torch.softmax(target_logits, dim=-1, dtype=torch.float32)
    l1 = (draft_log_probs.exp() - target_probs).abs().sum(dim=-1)
    return ce, l1


def _ce_and_l1_per_token(
    draft_logits: torch.Tensor,
    target_ids: torch.Tensor,
    target_logits: torch.Tensor | None,
    chunk_tokens: int = _PROBABILITY_CHUNK_TOKENS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-position cross-entropy and exact FP32 probability L1, without full-vocab temporaries.

    Both terms need the draft's normalized distribution over the whole vocabulary.
    A single ``[batch, num_blocks, block_size, vocab]`` FP32 tensor is 2.03 GiB at
    reference scale (``num_blocks=512``, ``block_size=7``, ``vocab≈150k``), and
    computing the two terms separately keeps several of them live at once, since
    the cross-entropy's own ``log_softmax`` is saved for its backward. Measured on
    an 80 GB card, an unchunked pass costs 10.15 GiB of forward-plus-backward
    overhead above its inputs, against 2.06 GiB here.

    Chunking the flattened rows caps the temporaries at ``chunk_tokens * vocab``
    elements, and each chunk is recomputed in the backward
    (:func:`torch.utils.checkpoint.checkpoint`) so no chunk keeps its distribution
    alive for autograd. Chunking does not change the result: every row is reduced
    independently over the vocab axis.

    Args:
        draft_logits: ``[..., vocab]``.
        target_ids: Teacher-forced next token per position ``[...]``.
        target_logits: ``[..., vocab]`` already detached by the caller, or ``None``
            to compute cross-entropy only (the L1 output is then zeros).
        chunk_tokens: Rows of the flattened logits per chunk.

    Returns:
        ``(ce, l1)``, both ``[...]``. ``l1`` is twice the total-variation distance.
    """
    output_shape = draft_logits.shape[:-1]
    vocab_size = draft_logits.shape[-1]
    flat_draft = draft_logits.reshape(-1, vocab_size)
    flat_ids = target_ids.reshape(-1).long()
    flat_target = None if target_logits is None else target_logits.reshape(-1, vocab_size)
    use_checkpoint = torch.is_grad_enabled() and flat_draft.requires_grad
    ce_chunks, l1_chunks = [], []
    for start in range(0, flat_draft.shape[0], chunk_tokens):
        rows = slice(start, start + chunk_tokens)
        args = (flat_draft[rows], flat_ids[rows], None if flat_target is None else flat_target[rows])
        if use_checkpoint:
            ce, l1 = checkpoint(_chunk_terms, *args, use_reentrant=False, preserve_rng_state=False)
        else:
            ce, l1 = _chunk_terms(*args)
        ce_chunks.append(ce)
        l1_chunks.append(l1)
    return torch.cat(ce_chunks).reshape(output_shape), torch.cat(l1_chunks).reshape(output_shape)


def _pair(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Pack a reducible ``[numerator, denominator]`` reporting pair."""
    return torch.cat([numerator.detach().reshape(1).float(), denominator.detach().reshape(1).float()])


def dspark_loss(
    outputs: DSparkForwardOutput,
    *,
    ce_alpha: float = 0.1,
    l1_alpha: float = 0.9,
    confidence_alpha: float = 1.0,
    loss_decay_gamma: float | None = 4.0,
    chunk_tokens: int = _PROBABILITY_CHUNK_TOKENS,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the DSpark draft loss and acceptance metrics.

    Follows the Megatron-Core 3-tuple loss contract: the returned loss is the
    unnormalized micro-batch numerator, and normalization happens once against the
    globally reduced ``num_tokens``. See the module docstring for why, and
    :func:`assert_per_token_loss_config` for the config this requires.

    Args:
        outputs: The draft forward outputs; see :class:`DSparkForwardOutput`. All
            tensors are ``[batch, num_blocks, block_size(, vocab)]``.
        ce_alpha: Cross-entropy weight.
        l1_alpha: Weight of the raw probability-L1 term (paper Eq. 10; no ``1/2``).
        confidence_alpha: Confidence-BCE weight (used only when
            ``outputs.confidence_pred`` is set).
        loss_decay_gamma: Per-position decay; ``None`` disables it.
        chunk_tokens: Rows per chunk of the FP32 probability L1; see
            :func:`_l1_distance_per_token`.

    Returns:
        Tuple ``(loss, num_tokens, report)``. ``loss`` is the unnormalized weighted
        numerator for this micro-batch; ``num_tokens`` is its supervised token
        count as an int tensor; ``report`` maps each metric name to a detached
        ``[numerator, denominator]`` pair that the caller sums over the log window
        and the data-parallel group before dividing once.

    Raises:
        ValueError: If the L1 term or the confidence head is requested without
            ``aligned_target_logits``.
    """
    draft_logits = outputs.draft_logits
    weight = _loss_weight_mask(outputs.eval_mask, loss_decay_gamma)  # [B, N, L]
    weight_sum = weight.sum()
    eval_mask = outputs.eval_mask.to(torch.float32)
    num_tokens = eval_mask.sum().detach().to(torch.int)

    has_confidence = outputs.confidence_pred is not None
    target_logits = outputs.aligned_target_logits
    if target_logits is not None:
        target_logits = target_logits.detach()  # never backprop into the (shared) lm_head
    if (l1_alpha > 0 or has_confidence) and target_logits is None:
        raise ValueError("aligned_target_logits is required for the L1 loss or the confidence head.")
    needs_teacher = l1_alpha > 0 or has_confidence

    # One FP32 normalization of the draft distribution serves both terms.
    ce_per_token, l1_dist = _ce_and_l1_per_token(
        draft_logits, outputs.target_ids, target_logits if needs_teacher else None, chunk_tokens
    )
    ce_num = (ce_per_token * weight).sum()

    zero = ce_num.new_zeros(())
    l1_num = conf_num = zero
    accept_rate = None
    if needs_teacher:
        accept_rate = (1.0 - 0.5 * l1_dist).clamp(0.0, 1.0)
        if l1_alpha > 0:
            # Raw weighted L1 (paper Eq. 10). The 1/2 lives in accept_rate above.
            l1_num = (l1_dist * weight).sum()
        if has_confidence:
            conf_num = (
                F.binary_cross_entropy_with_logits(
                    outputs.confidence_pred.float(), accept_rate.detach(), reduction="none"
                )
                * weight
            ).sum()

    loss = ce_alpha * ce_num + l1_alpha * l1_num + confidence_alpha * conf_num

    # Every term shares the weighted denominator, so a disabled term reports 0 and
    # never a 0/0 that the reducer in ``training/train.py`` would turn into NaN.
    report = {
        "dspark loss": _pair(loss, weight_sum),
        "dspark ce loss": _pair(ce_num, weight_sum),
        "dspark l1 loss": _pair(l1_num, weight_sum),
        "dspark confidence loss": _pair(conf_num, weight_sum),
    }
    if accept_rate is not None:
        with torch.no_grad():
            report.update(_acceptance_report(outputs, eval_mask, accept_rate))
    return loss, num_tokens, report


def _acceptance_report(
    outputs: DSparkForwardOutput, eval_mask: torch.Tensor, accept_rate: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Analytical acceptance diagnostics as reducible ``[numerator, denominator]`` pairs.

    ``accept_rate`` (``= 1 - 0.5 * L1``) is the per-position acceptance probability.
    A draft token survives only if every earlier token in its block is accepted,
    hence ``tau`` is the running product over the block, ``+ 1`` for the verified
    anchor token.

    Args:
        outputs: The forward outputs (for ``block_keep_mask``).
        eval_mask: The float supervision mask ``[batch, num_blocks, block_size]``.
        accept_rate: Per-position acceptance ``[batch, num_blocks, block_size]``.

    Returns:
        ``{"dspark accept rate": pair, "dspark tau": pair}``.
    """
    valid_accept = accept_rate * eval_mask
    valid_blocks = (outputs.block_keep_mask.bool() & (eval_mask > 0).any(dim=-1)).to(torch.float32)
    tau_per_block = valid_accept.cumprod(dim=-1).sum(dim=-1) + 1.0
    return {
        "dspark accept rate": _pair(valid_accept.sum(), eval_mask.sum()),
        "dspark tau": _pair((tau_per_block * valid_blocks).sum(), valid_blocks.sum()),
    }
